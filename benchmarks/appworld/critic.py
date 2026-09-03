"""AppWorld critic (the paper's `Reflect`): one chat call over the prompt used for the paper's runs
(prompts/critic/appworld_remo.txt for ReMo, prompts/critic/appworld_adaremo.txt for AdaReMo) with the
whole playbook, the previous round's reflection (or "N/A") and the conversation history of the attempt.
No ground truth, no test result, no code execution.

The reply is parsed as the runs parsed it: JSON located by remo.critic.extract_json_balanced,
`trajectory_verdict` compared case-insensitively to "no_errors", a verdict that is neither parseable
nor quoted in the text follows the objective env_clean signal; `key_insight` is the lesson; refine
defaults to True and store to False; the whole reply is the critique the next round receives and the
text the consolidator stores. The AdaReMo prompt also asks for `confidence`, and the runs honoured a
`store` only with confidence >= store_conf (0.7): below it the reflection reaches the policy with
store=False and no cited entry, so nothing is written or reinforced and the saturation window records
no memory demand (the runs' `skipped_lowconf`); the record keeps what the critic said.

EnvCritic is the ReAct arm: the paper's ReAct runs made no critic call, the verdict is the environment
signal alone.
"""
import time
from pathlib import Path

from remo.critic import extract_json_balanced, parse_reflection
from remo.interfaces import Reflection, Trajectory

from benchmarks.appworld.solver import EMPTY_PLAYBOOK, read_prompt

ROOT = Path(__file__).resolve().parents[2]
CRITIC_PROMPT_PATHS = {False: ROOT / "prompts" / "critic" / "appworld_remo.txt",
                       True: ROOT / "prompts" / "critic" / "appworld_adaremo.txt"}
SEE_HISTORY = "See full conversation history below"
NO_PRIOR = "N/A"


def build_critic_input(prompt: str, playbook_text: str, previous_reflection: str | None, history: str) -> str:
    """The one user message of the critic call: the prompt with its placeholders bound, then the
    conversation-history block of the attempt (Trajectory.text)."""
    return (prompt.replace("{{generated_code}}", SEE_HISTORY)
                  .replace("{{generated_rationale}}", SEE_HISTORY)
                  .replace("{{playbook}}", playbook_text or NO_PRIOR)
                  .replace("{{previous_reflection}}", previous_reflection or NO_PRIOR)) + history


def parse_critic_reply(text: str, adaptive: bool, env_clean: bool) -> tuple[Reflection, float]:
    """Reply -> (Reflection, confidence). The verdict is "no_errors" iff the parsed
    `trajectory_verdict` equals it case-insensitively; without the key, iff the quoted literal
    "no_errors" occurs in the text, else "errors_found" if that literal occurs, else `env_clean`."""
    text = text or ""
    refl = parse_reflection(text, adaptive, verdict_key="trajectory_verdict", lesson_keys=("key_insight",),
                            extract=extract_json_balanced)
    refl.cited_id = ""                                   # the runs took cited ids from novelty_reason only
    d = extract_json_balanced(text)
    d = d if isinstance(d, dict) else {}
    if "trajectory_verdict" in d:
        no_errors = str(d["trajectory_verdict"]).strip().lower() == "no_errors"
    elif '"no_errors"' in text:
        no_errors = True
    elif '"errors_found"' in text:
        no_errors = False
    else:
        no_errors = env_clean
    refl.verdict = "correct" if no_errors else "incorrect"
    refl.critique = text
    try:
        confidence = max(0.0, min(1.0, float(d.get("confidence", 0.0))))
    except (TypeError, ValueError):
        confidence = 0.0
    return refl, confidence


class AppWorldCritic:
    """One chat call per round. A transport failure (after the client's retries) returns
    Reflection(failed=True): the episode stops, nothing is admitted or written. `store_conf` is the
    confidence a `store` needs (AdaReMo; the ReMo prompt has neither field)."""

    def __init__(self, llm, adaptive: bool, max_tokens: int = 8192, temperature: float = 0.0,
                 store_conf: float = 0.7, log=print):
        self.llm, self.adaptive, self.max_tokens, self.temperature, self.log = llm, adaptive, max_tokens, temperature, log
        self.store_conf = store_conf
        self.prompt = read_prompt(CRITIC_PROMPT_PATHS[adaptive])
        self.calls = self.parse_failures = self.call_failures = 0
        # one entry per call: confidence, parsed, elapsed_s and, for a store below store_conf, low_confidence =
        # the store / novelty_reason / cited_id fields as the critic gave them (cleared in the Reflection)
        self.history: list[dict] = []

    def build_input(self, traj: Trajectory, memory_text: str, prior_critique: str | None) -> str:
        return build_critic_input(self.prompt, memory_text or EMPTY_PLAYBOOK, prior_critique, traj.text)

    def reflect(self, task, traj: Trajectory, memory_text: str, prior_critique: str | None,
                round_idx: int, K: int) -> Reflection:
        t0 = time.time()
        self.calls += 1
        try:
            text = self.llm.chat([{"role": "user", "content": self.build_input(traj, memory_text, prior_critique)}],
                                 self.max_tokens, self.temperature)
        except Exception as e:                       # noqa: BLE001 — transport / server failure
            self.call_failures += 1
            self.log(f"[critic] call failed: {type(e).__name__}: {e}")
            self.history.append({"confidence": None, "parsed": False, "elapsed_s": round(time.time() - t0, 2)})
            return Reflection(verdict="none", critique=f"(critic call failed: {type(e).__name__}: {e})"[:500],
                              parsed=False, failed=True)
        refl, confidence = parse_critic_reply(text, self.adaptive, traj.completed)
        if not refl.parsed:
            self.parse_failures += 1
        entry = {"confidence": confidence, "parsed": refl.parsed, "elapsed_s": round(time.time() - t0, 2)}
        if refl.store and confidence < self.store_conf:
            entry["low_confidence"] = {"store": True, "novelty_reason": refl.novelty_reason, "cited_id": refl.cited_id}
            refl.store, refl.novelty_reason, refl.cited_id = False, "", ""
        self.history.append(entry)
        return refl

    def drain(self) -> list[dict]:
        h, self.history = self.history, []
        return h


class EnvCritic:
    """No model call: the verdict is `Trajectory.completed` (the ReAct arm)."""
    calls = parse_failures = call_failures = 0

    def reflect(self, task, traj: Trajectory, memory_text: str, prior_critique: str | None,
                round_idx: int, K: int) -> Reflection:
        return Reflection(verdict="correct" if traj.completed else "incorrect", parsed=False)

    def drain(self) -> list[dict]:
        return []
