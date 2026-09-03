"""AppWorld solver: a ReAct agent in the AppWorld Python REPL, driven through the official
`appworld` package (pip appworld==0.1.3.post1).

One `solve` call = one round of the paper's loop: a FRESH `AppWorld(task_id, experiment_name)` is
opened, the model writes one python code block per step, the world executes it and returns the
output, until the task is submitted (`world.task_completed()`) or `max_steps` steps are spent.

    Trajectory.completed = task submitted AND the last execution output shows no error
                           ("Execution failed" / "Traceback"), i.e. the objective env_clean signal
    Trajectory.text      = trimmed transcript of the round (what the critic reads)
    Trajectory.answer    = the code block that called complete_task (for inspection; scoring is
                           post hoc with AppWorld's unit tests, see run_appworld.py)

A reply without a code block spends a step: it is recorded for the critic, dropped from the model's
context and a reminder is appended to the last user message; three in a row end the round.

Memory `M` is prepended to the task as a playbook; the previous round's critique `ρ` is injected at
the start of the next round (the world is fresh, so the retry starts from the initial state).
No ground truth is loaded into the world (`load_ground_truth=False`).

`appworld` is imported lazily inside `solve`, so this module (and the unit tests) import without it.
"""
import os
import re
import shutil
import time
from typing import Any

from remo.interfaces import Trajectory

# ---------------------------------------------------------------------------------------------
# prompt (our own wording of the standard AppWorld ReAct protocol)
# ---------------------------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an autonomous assistant that completes everyday tasks for your supervisor by \
programming against the APIs of simulated apps (e-mail, phone, payments, shopping, music, files, ...).

You work in a persistent Python REPL. At every step you write ONE python code block; the environment runs \
it and shows you the output; you then write the next block, and so on, until the task is done. Variables \
persist between steps. Apps are reached through the `apis` object, e.g. `apis.venmo.login(...)`.

Discovering the environment (always available):
```python
print(apis.api_docs.show_app_descriptions())                        # which apps exist
print(apis.api_docs.show_api_descriptions(app_name='venmo'))         # which APIs an app has
print(apis.api_docs.show_api_doc(app_name='venmo', api_name='login'))  # full spec of one API
```

Rules:
1. Reply with a short thought and then exactly one code block fenced as ```python ... ``` (close the fence \
with ``` and a newline). Never write more than one block per step; never invent an output.
2. Keep each block small and check the output before doing anything irreversible (sending, paying, deleting).
3. Read the spec of an API (`show_api_doc`) before you call it; do not guess parameter names or response fields.
4. Account passwords, addresses and payment cards of the supervisor come from the `supervisor` app; \
information about friends, family and other people comes from the `phone` app (contacts). Take identities and \
relationships from these sources, never from free-text guesses.
5. Most list APIs are paginated (`page_index`); loop until a page comes back empty so nothing is missed.
6. Answer exactly what is asked (units, type, direction of a transfer, time window, who sent / received).
7. When the task is complete call `apis.supervisor.complete_task()`. If the task asks for information, pass \
it as `apis.supervisor.complete_task(answer=<value>)` with a value of the right type; otherwise pass no argument. \
Calling complete_task ends the task, so call it only once, at the very end.
8. There is no separate code-execution tool: the code must be written as text inside your reply, in the \
```python fence. A reply without such a block does nothing."""

PLAYBOOK_HEADER = ("Playbook — lessons learned on earlier tasks in this environment. Apply the ones that fit "
                   "the task at hand:")
CRITIQUE_HEADER = ("A reviewer examined your previous attempt at this very task (the environment has been reset "
                   "to its initial state, so nothing you did before persists) and reported the following. "
                   "Fix these problems this time:")
NO_CODE_MESSAGE = ("No python code block was found in your reply. Reply with a short thought followed by exactly one "
                   "```python ... ``` block.")
# A reply without a block is NOT kept in the context: with gpt-oss it usually means the code went to a
# tool channel the server drops, and the model stays in that mode if the preamble is echoed back (nudges
# never recovered it). The failed turn is discarded and this reminder appended once to the last user message.
NO_CODE_REMINDER = ("\n\n(Your previous reply contained no ```python code block and was discarded. There is no separate "
                    "code-execution tool: write the code as text inside a ```python fence in this reply.)")
ERROR_MARKERS = ("Execution failed", "Traceback")
TRIM_MARKER = "[earlier steps of this attempt were trimmed to fit the context]"

_FENCED = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.S)
_OPEN_FENCE = re.compile(r"```(?:python|py)?[ \t]*\n(.*)$", re.S)


# ---------------------------------------------------------------------------------------------
# pure helpers (unit-tested without a model or the appworld package)
# ---------------------------------------------------------------------------------------------
def extract_code(text: str) -> str:
    """The FIRST fenced code block of a reply (the protocol is one block per step). A block whose
    closing fence is missing (the model stopped early) is taken to the end of the text. Returns ""
    when the reply contains no block."""
    if not text:
        return ""
    m = _FENCED.search(text)
    if m:
        return m.group(1).strip()
    m = _OPEN_FENCE.search(text)
    return m.group(1).strip() if m else ""


def build_task_prompt(instruction: str, supervisor: dict, memory_text: str, critique: str | None,
                      critique_cap: int = 4000) -> str:
    """User message that opens a round: playbook (if any), previous critique (if any), the task."""
    parts = []
    if memory_text and memory_text.strip():
        parts.append(f"{PLAYBOOK_HEADER}\n{memory_text.strip()}")
    if critique and critique.strip():
        parts.append(f"{CRITIQUE_HEADER}\n{critique.strip()[:critique_cap]}")
    parts.append(f"My name is {supervisor.get('first_name', '')} {supervisor.get('last_name', '')}. "
                 f"My personal e-mail is {supervisor.get('email', '')} and my phone number is "
                 f"{supervisor.get('phone_number', '')}.\n\nTask: {instruction}\n\n"
                 "Start now. Remember: one python code block per step.")
    return "\n\n".join(parts)


def format_output(output: str, cap: int) -> str:
    """Environment output as shown to the model; very long outputs are cut in the middle."""
    out = output if output.strip() else "Execution successful."
    if len(out) > cap:
        half = cap // 2
        out = out[:half] + f"\n[... {len(out) - cap} characters cut ...]\n" + out[-half:]
    return f"Output:\n```\n{out}\n```"


def trim_history(messages: list[dict], keep_head: int, max_chars: int) -> list[dict]:
    """Keep the first `keep_head` messages (system + task prompt) and drop the OLDEST step pairs
    (assistant reply, environment output) after them until the total is under `max_chars`; one marker
    replaces the dropped part. The marker of an earlier trim is removed first so the pairs stay aligned."""
    total = sum(len(m["content"]) for m in messages)
    if total <= max_chars:
        return messages
    head = messages[:keep_head]
    tail = [m for m in messages[keep_head:] if m["content"] != TRIM_MARKER]
    while tail and sum(len(m["content"]) for m in head) + sum(len(m["content"]) for m in tail) > max_chars:
        tail = tail[2:] if len(tail) >= 2 else []
    return head + [{"role": "user", "content": TRIM_MARKER}] + tail


def transcript(steps: list[dict], step_cap: int = 2500, total_cap: int = 60000) -> str:
    """What the critic reads: every step's reply and output, each capped, then the whole capped by
    keeping the beginning and the end (the final steps carry the completion call)."""
    blocks = []
    for s in steps:
        reply = s.get("reply", "")
        out = s.get("output", "")
        blocks.append(f"[step {s['step']}] AGENT:\n{reply[:step_cap]}"
                      + (" [...]" if len(reply) > step_cap else "")
                      + f"\n[step {s['step']}] OUTPUT:\n{out[:step_cap]}"
                      + (" [...]" if len(out) > step_cap else ""))
    text = "\n\n".join(blocks)
    if len(text) > total_cap:
        half = total_cap // 2
        text = text[:half] + "\n\n[... middle of the transcript omitted ...]\n\n" + text[-half:]
    return text


def env_clean(task_completed: bool, last_output: str) -> bool:
    """Objective environment signal: task submitted and the last execution output shows no error."""
    return bool(task_completed) and not any(mk in (last_output or "") for mk in ERROR_MARKERS)


# ---------------------------------------------------------------------------------------------
# the solver
# ---------------------------------------------------------------------------------------------
class AppWorldSolver:
    """ReAct agent over the official AppWorld REPL. `task` is a dict with at least `task_id`.

    experiment_name   AppWorld writes the world's end state and logs to
                      $APPWORLD_ROOT/experiments/outputs/<experiment_name>/tasks/<task_id>/;
                      the post-hoc evaluation reads it from there.
    round1_experiment if given, the end state of the FIRST round of every task is copied to this
                      experiment name so round-1 TGC/SGC can be evaluated post hoc as well.
    """

    def __init__(self, client, model: str, experiment_name: str, max_steps: int = 40, max_tokens: int = 8192,
                 temperature: float = 0.0, random_seed: int | None = 100, exec_timeout_s: int | None = 100,
                 output_cap_chars: int = 12000, max_prompt_chars: int = 320000, round1_experiment: str | None = None,
                 log=print):
        self.client, self.model, self.experiment_name = client, model, experiment_name
        self.max_steps, self.max_tokens, self.temperature = max_steps, max_tokens, temperature
        self.random_seed, self.exec_timeout_s = random_seed, exec_timeout_s
        self.output_cap_chars, self.max_prompt_chars = output_cap_chars, max_prompt_chars
        self.round1_experiment, self.log = round1_experiment, log
        self.round_metas: list[dict] = []          # metas of the rounds of the CURRENT task (reset by the runner)
        self.calls = self.call_failures = 0

    # -- model call -------------------------------------------------------------------------------
    def _chat(self, messages: list[dict]) -> tuple[str, dict]:
        self.calls += 1
        r = self.client.chat.completions.create(model=self.model, messages=messages, max_tokens=self.max_tokens,
                                                temperature=self.temperature)
        usage = getattr(r, "usage", None)
        return (r.choices[0].message.content or ""), {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(usage, "completion_tokens", 0) or 0}

    # -- one round --------------------------------------------------------------------------------
    def solve(self, task: dict, memory_text: str, critique: str | None) -> Trajectory:
        from appworld import AppWorld                   # lazy: needs the installed package + data

        task_id = task["task_id"]
        round_idx = len(self.round_metas) + 1
        t0 = time.time()
        steps: list[dict] = []
        usage = {"prompt_tokens": 0, "completion_tokens": 0}
        last_output, completed, error, no_code = "", False, "", 0
        meta: dict[str, Any] = {"task_id": task_id, "round": round_idx}
        with AppWorld(task_id=task_id, experiment_name=self.experiment_name, random_seed=self.random_seed,
                      timeout_seconds=self.exec_timeout_s, load_ground_truth=False) as world:
            sup = world.task.supervisor
            supervisor = {k: getattr(sup, k, "") for k in ("first_name", "last_name", "email", "phone_number")}
            meta.update(instruction=world.task.instruction, supervisor=supervisor,
                        output_dir=world.output_directory, misc_dir=world.output_misc_directory)
            messages = [{"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": build_task_prompt(world.task.instruction, supervisor,
                                                                       memory_text, critique)}]
            for step in range(1, self.max_steps + 1):
                messages = trim_history(messages, keep_head=2, max_chars=self.max_prompt_chars)
                try:
                    reply, u = self._chat(messages)
                except Exception as e:                  # noqa: BLE001 — the round ends as a failed attempt
                    self.call_failures += 1
                    error = f"{type(e).__name__}: {e}"[:500]
                    self.log(f"[solver] {task_id} r{round_idx} step {step}: model call failed: {error}")
                    break
                usage["prompt_tokens"] += u["prompt_tokens"]; usage["completion_tokens"] += u["completion_tokens"]
                code = extract_code(reply)
                if not code:                            # counts as a step; the failed turn is dropped (see NO_CODE_REMINDER)
                    no_code += 1
                    steps.append({"step": step, "reply": reply, "code": "", "output": NO_CODE_MESSAGE})
                    if not messages[-1]["content"].endswith(NO_CODE_REMINDER):
                        messages[-1]["content"] += NO_CODE_REMINDER
                    if no_code >= 3:                    # three in a row: the model is not following the protocol
                        error = "no_code_block"
                        break
                    continue
                no_code = 0
                output = world.execute(code)
                last_output = output if output.strip() else last_output
                steps.append({"step": step, "reply": reply, "code": code, "output": output})
                messages.append({"role": "assistant", "content": reply})
                messages.append({"role": "user", "content": format_output(output, self.output_cap_chars)})
                if world.task_completed():
                    break
            completed = world.task_completed()
        clean = env_clean(completed, last_output)
        if round_idx == 1 and self.round1_experiment:
            self._snapshot_round1(meta["output_dir"], task_id)
        submit = next((s["code"] for s in reversed(steps) if "complete_task" in s.get("code", "")), "")
        meta.update(steps=len(steps), task_completed=completed, env_clean=clean, last_output=last_output[-2000:],
                    error=error, no_code_steps=sum(1 for x in steps if not x["code"]), elapsed_s=round(time.time() - t0, 1),
                    memory_chars=len(memory_text), critique_chars=len(critique or ""), **usage)
        meta["steps_full"] = steps
        self.round_metas.append(meta)
        self.log(f"[solver] {task_id} r{round_idx}: steps={len(steps)} submitted={completed} env_clean={clean}"
                 + (f" error={error}" if error else "") + f" {meta['elapsed_s']}s")
        return Trajectory(text=transcript(steps), answer=submit, completed=clean, meta=meta)

    def _snapshot_round1(self, output_dir: str, task_id: str) -> None:
        """Copy tasks/<task_id>/dbs of the round-1 world to the round-1 experiment (same layout, so
        the post-hoc evaluation can score the first attempt of every task as well)."""
        outputs_root = os.path.dirname(os.path.dirname(os.path.dirname(output_dir)))   # .../experiments/outputs
        dst_task = os.path.join(outputs_root, self.round1_experiment, "tasks", task_id)
        try:
            shutil.rmtree(dst_task, ignore_errors=True)
            shutil.copytree(os.path.join(output_dir, "dbs"), os.path.join(dst_task, "dbs"))
        except OSError as e:
            self.log(f"[solver] WARN could not snapshot the round-1 state of {task_id}: {e}")
