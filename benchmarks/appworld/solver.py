"""AppWorld solver: the ReAct-in-REPL loop of the paper's runs over the official `appworld` package.

One `solve` call = one round: a FRESH `AppWorld(task_id, experiment_name, random_seed=123)` is opened,
the instruction messages are rendered from the arm's template (jinja2, the same template parameters as
the paper's runs: task, supervisor, app descriptions, the whole playbook), the model writes one
```python block per step, the world executes it and the output comes back as the next user turn, until
`world.task_completed()` or `max_steps` steps. Templates: prompts/solver/appworld.txt (the paper's
generator prompt, shows the playbook) for the memory arms; prompts/solver/appworld_react.txt (AppWorld's
official ReAct prompt, no playbook) for the no-memory arms, as the paper's ReAct / refinement-only runs.

    Trajectory.completed = task submitted AND the last non-empty execution output shows neither
                           "Execution failed" nor "Traceback" (the objective env_clean signal)
    Trajectory.text      = the "=== FULL CONVERSATION HISTORY ===" block over the trimmed messages,
                           i.e. the trajectory part of the critic's and the consolidator's input
    Trajectory.failed    = the model call failed after the client's retries

A retry (critique given) injects the previous round's full reflection as three messages after the
instruction messages, exactly as the paper's runs did. `appworld` and `jinja2` are imported lazily.
"""
import copy
import json
import os
import random
import re
import shutil
import signal
import time
from pathlib import Path
from typing import Any

from remo.interfaces import Trajectory

ROOT = Path(__file__).resolve().parents[2]
SOLVER_PROMPT_PATH = ROOT / "prompts" / "solver" / "appworld.txt"
REACT_PROMPT_PATH = ROOT / "prompts" / "solver" / "appworld_react.txt"

EMPTY_PLAYBOOK = "(empty)"                 # the playbook placeholder value when the arm has no memory
MAX_OUTPUT_LENGTH = 400000                 # chars of post-instruction text kept in the model's context (reflect loop)
REACT_MAX_OUTPUT_LENGTH = 50000            # the same limit in AppWorld's plain ReAct scaffold (the paper's ReAct runs)
OUTPUT_CAP = 20000                         # chars of one execution output shown to the model (reflect loop; ReAct: none)
ERROR_MARKERS = ("Execution failed", "Traceback")
RETRY_INTRO = ("Your previous attempt did not fully solve the task. Below is a reflection diagnosing what "
               "went wrong, based only on the task requirements and the environment feedback.")
RETRY_USE_WITH_PLAYBOOK = ("Use the reflection above, along with the playbook, to avoid repeating these "
                           "mistakes and solve the task correctly this time.")
RETRY_USE = "Use the reflection above to avoid repeating these mistakes and solve the task correctly this time."
NOT_SHOWN = "Output:\n```\n[NOT SHOWN FOR BREVITY]```\n\n"
TRIMMED = "[TRIMMED HISTORY]\n\n"

# Anchored at a line start: the template's role markers always begin a line, but the playbook injected
# into that template is model-written text, and a lesson ending a line with e.g. "...to the file system:"
# otherwise matches mid-word and splits a spurious message out of the middle of the prompt (a "system"
# one makes the server reject the request, a "user"/"assistant" one corrupts the turn structure silently).
_ROLE = re.compile("^(USER|ASSISTANT|SYSTEM):\n", re.IGNORECASE | re.MULTILINE)
_FULL_CODE = re.compile(r"```python\n(.*?)```", re.DOTALL)
_PARTIAL_CODE = re.compile(r".*```python\n(.*)", re.DOTALL)


def read_prompt(path: Path) -> str:
    with open(path, encoding="utf-8", newline="") as f:
        return f.read()


# -- message construction (pure; unit-tested without a model or a world) ----------------------------
def text_to_messages(text: str) -> list[dict]:
    """USER:/ASSISTANT: turns of the rendered template -> chat messages (no system message)."""
    messages, last_start = [], 0
    for m in _ROLE.finditer(text):
        last_end = m.span()[0]
        if not messages:
            if last_end != 0:
                raise ValueError(f"Start of the prompt has no assigned role: {text[:last_end]}")
        else:
            messages[-1]["content"] = text[last_start:last_end]
        messages.append({"role": m.group(1).lower(), "content": None})
        last_start = m.span()[1]
    messages[-1]["content"] = text[last_start:]
    return messages


def messages_to_text(messages: list[dict]) -> str:
    out = ""
    for message in messages:
        role = message["role"]
        if role == "assistant":
            out += "ASSISTANT:\n" + message["content"]
        elif role == "user":
            out += "USER:\n" + message["content"]
        else:
            raise ValueError(f"Unknown message role {role} in: {message}")
    return out


def instruction_messages(template: str, instruction: str, supervisor: Any, app_descriptions: dict,
                         playbook: str) -> list[dict]:
    """The rendered generator template as chat messages (`supervisor` needs first_name / last_name /
    email / phone_number as attributes or keys)."""
    from jinja2 import Template
    descriptions = json.dumps([{"name": k, "description": v} for k, v in app_descriptions.items()], indent=1)
    text = Template(template.lstrip()).render({"input_str": instruction, "main_user": supervisor,
                                               "app_descriptions": descriptions, "playbook": playbook})
    return text_to_messages(text + "\n\n")


def retry_messages(reflection: str, has_playbook: bool) -> list[dict]:
    """The three messages a retry appends after the instruction messages: the previous round's whole
    reflection as an assistant turn between two fixed user turns."""
    return [{"role": "user", "content": RETRY_INTRO},
            {"role": "assistant", "content": reflection + "\n\n"},
            {"role": "user", "content": RETRY_USE_WITH_PLAYBOOK if has_playbook else RETRY_USE}]


def extract_code(text: str | None) -> tuple[str, str]:
    """(code, assistant content kept in context): the FIRST ```python block, the content cut right after
    its closing fence; a block without a closing fence is taken to the end and the fence appended;
    no block -> ("", text)."""
    if text is None:
        return "", ""
    m = next(_FULL_CODE.finditer(text), None)
    if m:
        return m.group(1).strip(), text[:m.end()]
    partial = _PARTIAL_CODE.match(text)
    if partial:
        code = partial.group(1).strip()
        text = text + ("" if text.endswith("\n") else "\n") + "```"
        return ("", text) if not code else (code, text)
    return "", text


def truncate_output(output: str, cap: int | None = OUTPUT_CAP) -> str:
    if cap is not None and len(output) > cap:
        output = output[:cap] + "\n[REST NOT SHOWN FOR BREVITY]"
    return output


def output_message(output: str, cap: int | None = OUTPUT_CAP) -> dict:
    return {"role": "user", "content": "Output:\n```\n" + truncate_output(output, cap) + "```\n\n"}


def trimmed_messages(messages: list[dict], num_instruction_messages: int,
                     max_output_length: int = MAX_OUTPUT_LENGTH) -> list[dict]:
    """The messages sent to the model: while the text after "Task: " of the last instruction message
    exceeds `max_output_length`, blank one observation (oldest first, never among the last 5 messages),
    then drop whole messages after the task message (marking it [TRIMMED HISTORY])."""
    messages = copy.deepcopy(messages)
    pre, post = messages[:num_instruction_messages - 1], messages[num_instruction_messages - 1:]
    text = messages_to_text(post)
    prefix = text[:text.index("Task: ") + 6]
    text = text.removeprefix(prefix)
    observation_index = 0
    while len(text) > max_output_length:
        found = False
        if observation_index < len(post) - 5:
            for i, message in enumerate(post[observation_index:]):
                if message["role"] == "user" and message["content"].startswith("Output:"):
                    message["content"] = NOT_SHOWN
                    found = True
                    observation_index += i + 1
                    break
            if not found:
                observation_index = len(post)
        if not found and post:
            first = copy.deepcopy(post[0])
            if not first["content"].endswith(TRIMMED):
                first["content"] += TRIMMED
            post = [first] + post[2:]
            found = True
        if not found:
            raise ValueError(f"No blocks found to be removed!\n{post}")
        text = messages_to_text(post).removeprefix(prefix)
    return pre + post


def conversation_history(messages: list[dict]) -> str:
    """The trajectory as the critic and the consolidator read it."""
    out = "\n\n=== FULL CONVERSATION HISTORY ===\n"
    for i, m in enumerate(messages):
        out += f"[{i}] {m.get('role', 'unknown').upper()}: {m.get('content', '')}\n\n"
    return out


def env_clean(task_completed: bool, last_output: str) -> bool:
    return bool(task_completed) and not any(k in (last_output or "") for k in ERROR_MARKERS)


# -- model calls -----------------------------------------------------------------------------------
class ChatLLM:
    """OpenAI-compatible chat calls as the paper's runs made them (messages, max_tokens, temperature;
    nothing else). A failed call is retried every `retry_after_s` seconds until `attempts` calls were
    made — except a context-length overflow, which is deterministic and fails at once — then the
    exception propagates."""

    def __init__(self, base_url: str, model: str, timeout: float = 600.0, attempts: int = 50,
                 retry_after_s: float = 10.0, log=print):
        import openai
        assert attempts >= 1
        self.client = openai.OpenAI(api_key=os.environ.get("REMO_API_KEY", "EMPTY"), base_url=base_url,
                                    timeout=timeout, max_retries=0)
        self.model, self.attempts, self.retry_after_s, self.log = model, attempts, retry_after_s, log
        self.calls = self.failures = 0
        self.prompt_tokens = self.completion_tokens = 0

    def served_models(self) -> list[str]:
        return [m.id for m in self.client.models.list().data]

    def chat(self, messages: list[dict], max_tokens: int, temperature: float) -> str:
        for attempt in range(1, self.attempts + 1):
            self.calls += 1
            try:
                r = self.client.chat.completions.create(model=self.model, messages=messages, max_tokens=max_tokens,
                                                        temperature=temperature)
            except Exception as e:                       # noqa: BLE001 — transport / server errors
                self.failures += 1
                if attempt == self.attempts or "maximum context length" in str(e):
                    raise
                self.log(f"[llm] {type(e).__name__}: {str(e)[:200]} — retrying in {self.retry_after_s}s")
                time.sleep(self.retry_after_s)
                continue
            usage = getattr(r, "usage", None)
            self.prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
            self.completion_tokens += getattr(usage, "completion_tokens", 0) or 0
            return r.choices[0].message.content or ""


# -- the solver -----------------------------------------------------------------------------------
class AppWorldSolver:
    """`task` is a dict with `task_id`. `llm.chat(messages, max_tokens, temperature) -> str`.

    template_path     the generator template (SOLVER_PROMPT_PATH or REACT_PROMPT_PATH)
    max_output_length / output_cap
                      context limit and per-output cap of the arm's scaffold (see the module docstring;
                      output_cap None = outputs are shown whole)
    experiment_name   AppWorld writes the world's end state and logs to
                      $APPWORLD_ROOT/experiments/outputs/<experiment_name>/tasks/<task_id>/
    round1_experiment if given, the end state of every task's FIRST round is copied there so the
                      first attempt can be scored post hoc as well.
    exec_timeout_s    AppWorld's own per-execution timeout; `guard_timeout_s` is the outer SIGALRM
                      guard of the paper's runs (an execution that outlives it returns an
                      "Execution failed" output).
    random_seed       the world's seed; Python's RNG is re-seeded with it at the start of every round.
    """

    def __init__(self, llm, experiment_name: str, max_steps: int = 40, max_tokens: int = 8192,
                 temperature: float = 0.0, random_seed: int | None = 123, exec_timeout_s: int | None = 100,
                 guard_timeout_s: int = 300, round1_experiment: str | None = None, template_path: Path = SOLVER_PROMPT_PATH,
                 max_output_length: int = MAX_OUTPUT_LENGTH, output_cap: int | None = OUTPUT_CAP, log=print):
        self.llm, self.experiment_name = llm, experiment_name
        self.max_steps, self.max_tokens, self.temperature = max_steps, max_tokens, temperature
        self.random_seed, self.exec_timeout_s, self.guard_timeout_s = random_seed, exec_timeout_s, guard_timeout_s
        self.round1_experiment, self.log = round1_experiment, log
        self.template = read_prompt(template_path)
        self.max_output_length, self.output_cap = max_output_length, output_cap
        self.round_metas: list[dict] = []          # metas of the rounds of the CURRENT task (reset by the runner)

    def _execute(self, world, code: str) -> str:
        def handler(signum, frame):
            raise TimeoutError(f"world.execute exceeded {self.guard_timeout_s}s")
        old = signal.signal(signal.SIGALRM, handler)
        signal.alarm(self.guard_timeout_s)
        try:
            return world.execute(code)
        except TimeoutError as e:
            self.log(f"[solver] WARN: {e}; returning an execution failure to the agent")
            return f"Execution failed. Traceback:\n  TimeoutError: {e}"
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)

    def solve(self, task: dict, memory_text: str, critique: str | None) -> Trajectory:
        from appworld import AppWorld                   # lazy: needs the installed package + data

        task_id, round_idx, t0 = task["task_id"], len(self.round_metas) + 1, time.time()
        steps: list[dict] = []
        last_output, error, output = "", "", None
        meta: dict[str, Any] = {"task_id": task_id, "round": round_idx}
        with AppWorld(task_id=task_id, experiment_name=self.experiment_name, random_seed=self.random_seed,
                      timeout_seconds=self.exec_timeout_s, load_ground_truth=False) as world:
            if self.random_seed is not None:
                random.seed(self.random_seed)      # the paper's runs re-seeded Python's RNG here, after the world's own seeding
            sup = world.task.supervisor
            meta.update(instruction=world.task.instruction, output_dir=world.output_directory,
                        misc_dir=world.output_misc_directory,
                        supervisor={k: getattr(sup, k, "") for k in ("first_name", "last_name", "email", "phone_number")})
            messages = instruction_messages(self.template, world.task.instruction, sup, world.task.app_descriptions,
                                            memory_text or EMPTY_PLAYBOOK)
            n_instruction = len(messages)
            if critique:
                messages += retry_messages(critique, bool(memory_text))
            for step in range(1, self.max_steps + 1):
                if output is not None:
                    messages.append(output_message(output, self.output_cap))
                try:
                    reply = self.llm.chat(trimmed_messages(messages, n_instruction, self.max_output_length),
                                          self.max_tokens, self.temperature)
                except Exception as e:                  # noqa: BLE001 — the round is a failed attempt
                    error = f"{type(e).__name__}: {e}"[:500]
                    self.log(f"[solver] {task_id} r{round_idx} step {step}: model call failed: {error}")
                    break
                code, kept = extract_code(reply)
                messages.append({"role": "assistant", "content": kept + "\n\n"})
                output = self._execute(world, code)
                if output.strip():
                    last_output = output
                steps.append({"step": step, "reply": reply, "code": code, "output": output})
                if world.task_completed():
                    break
            completed = world.task_completed()
        if round_idx == 1 and self.round1_experiment:
            self._snapshot_round1(meta["output_dir"], task_id)
        history = trimmed_messages(messages, n_instruction, self.max_output_length)
        clean = env_clean(completed, last_output)
        meta.update(steps=len(steps), task_completed=completed, env_clean=clean, last_output=last_output[-2000:],
                    error=error, elapsed_s=round(time.time() - t0, 1), memory_chars=len(memory_text),
                    critique_chars=len(critique or ""), messages=history, num_instruction_messages=n_instruction,
                    steps_full=steps)
        self.round_metas.append(meta)
        self.log(f"[solver] {task_id} r{round_idx}: steps={len(steps)} submitted={completed} env_clean={clean}"
                 + (f" error={error}" if error else "") + f" {meta['elapsed_s']}s")
        submit = next((s["code"] for s in reversed(steps) if "complete_task" in s["code"]), "")
        return Trajectory(text=conversation_history(history), answer=submit, completed=clean, failed=bool(error),
                          meta=meta)

    def _snapshot_round1(self, output_dir: str, task_id: str) -> None:
        """Copy tasks/<task_id>/dbs of the round-1 world to the round-1 experiment (a fresh world wipes
        the task directory, so this is the only way to score the first attempt later)."""
        outputs_root = os.path.dirname(os.path.dirname(os.path.dirname(output_dir)))   # .../experiments/outputs
        dst = os.path.join(outputs_root, self.round1_experiment, "tasks", task_id)
        try:
            shutil.rmtree(dst, ignore_errors=True)
            shutil.copytree(os.path.join(output_dir, "dbs"), os.path.join(dst, "dbs"))
        except OSError as e:
            self.log(f"[solver] WARN could not snapshot the round-1 state of {task_id}: {e}")
