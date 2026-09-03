"""Formula critic (the paper's `Reflect`): one chat call per round with the prompts used for the paper's runs.

prompts/critic/formula_remo.txt (ReMo) takes {q} {trace} {answer}; prompts/critic/formula_adaremo.txt (AdaReMo)
also {rnd} {K} {prior} (the previous round's critique, "(none — first attempt)" without one) and {playbook} (the
whole playbook text). {trace} is the solver's full reply, {answer} the extracted answer. The reply is parsed by
remo.critic.parse_reflection with the greedy JSON locator, no re-ask; cited ids are read from novelty_reason
only. A failed call yields Reflection(failed=True): the episode stops, not admitted.
"""
import time

from benchmarks.formula.solver import read_prompt
from remo.critic import parse_reflection
from remo.interfaces import Reflection, Trajectory

CRITIC_MAX_TOKENS = 8192
NO_PRIOR = "(none — first attempt)"
EMPTY_PLAYBOOK = "(empty)"
CRITIC_PROMPTS = {False: read_prompt("critic/formula_remo.txt"), True: read_prompt("critic/formula_adaremo.txt")}


def build_critic_prompt(task: dict, traj: Trajectory, memory_text: str, prior_critique: str | None, round_idx: int,
                        K: int, adaptive: bool) -> str:
    if adaptive:
        return CRITIC_PROMPTS[True].format(q=task["question"], trace=traj.text, answer=traj.answer, rnd=round_idx, K=K,
                                           prior=prior_critique or NO_PRIOR, playbook=memory_text or EMPTY_PLAYBOOK)
    return CRITIC_PROMPTS[False].format(q=task["question"], trace=traj.text, answer=traj.answer)


class FormulaCritic:
    def __init__(self, llm, adaptive: bool, max_tokens: int = CRITIC_MAX_TOKENS):
        self.llm, self.adaptive, self.max_tokens = llm, adaptive, max_tokens
        self.calls = self.parse_failures = self.call_failures = 0
        self.history: list[dict] = []

    def reflect(self, task: dict, traj: Trajectory, memory_text: str, prior_critique: str | None,
                round_idx: int, K: int) -> Reflection:
        prompt = build_critic_prompt(task, traj, memory_text, prior_critique, round_idx, K, self.adaptive)
        t0 = time.time()
        self.calls += 1
        rec = {"calls": 1, "prompt_tokens": 0, "completion_tokens": 0, "error": ""}
        try:
            text, usage = self.llm(prompt, self.max_tokens)
        except Exception as e:                       # noqa: BLE001 — the paper's runs stopped the episode here
            self.call_failures += 1
            rec.update(error=f"{type(e).__name__}: {e}"[:300], elapsed_s=round(time.time() - t0, 2))
            self.history.append(rec)
            return Reflection(verdict="none", critique=rec["error"], parsed=False, failed=True)
        refl = parse_reflection(text, self.adaptive, lesson_keys=("key_insight",))
        refl.cited_id = ""
        self.parse_failures += int(not refl.parsed)
        rec.update(prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
                   completion_tokens=int(usage.get("completion_tokens", 0) or 0), elapsed_s=round(time.time() - t0, 2))
        self.history.append(rec)
        return refl

    def drain(self) -> list[dict]:
        h, self.history = self.history, []
        return h
