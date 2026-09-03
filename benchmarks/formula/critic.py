"""Formula critic (the paper's `Reflect`) and the optional LLM consolidator.

The critic re-derives the computation from the question alone — no answer key exists inside the loop — and checks
inputs, scale/units, rounding and the question's stated requirements. The domain checks are this file's; the
decision fields (verdict / critique / lesson, plus refine / store / novelty for AdaReMo) come verbatim from
remo.critic.fields_spec and are parsed by remo.critic.parse_reflection, so the policy sees the same Reflection on
every benchmark.
"""
import time

from remo.critic import fields_spec, parse_reflection
from remo.interfaces import Reflection, Trajectory

TRACE_CAP_CHARS = 12000
CRITIC_MEMORY_CAP_CHARS = 40000

CRITIC_TEMPLATE = """You are reviewing another analyst's answer to a quantitative finance question. There is no answer key: verify the work with your own computation.

Question: {question}

Analyst's reply (the last Finish[...] line is the submitted answer):
{trace}

Submitted answer: {answer}

Review procedure:
1. Name the formula the question calls for and list every input it needs; confirm each input is taken from the question (right quantity, right period, right sign, nothing invented).
2. Re-derive the result independently, step by step, and compare it with the submitted number.
3. Check scale and units: percent versus fraction as the formula defines the quantity, "million" / "thousand" / basis-point conversions, currency and time units, annualization.
4. Check the format: one plain number, rounded to the nearest hundredth when the question sets no precision, no symbols or words inside Finish[...].
5. If the question states a requirement (a precision, a unit, "as a percentage"), confirm the answer meets it.
The verdict is errors_found ONLY when the submitted number is wrong or breaks a stated requirement; an equal value presented differently (15 vs 15.00) is not an error. Do not judge style or the length of the reasoning.
{prior_block}
The lesson must be ONE general, imperative sentence about procedure (which quantity to use, a unit or scale convention, a rounding rule, a pitfall) that would help on OTHER questions; never repeat this question's numbers or its answer.

"""

PRIOR_TEMPLATE = """
This is attempt {round_idx} of {K}. The reviewer of the previous attempt reported:
{prior_critique}
Judge THIS attempt on its own; say whether that issue is fixed.
"""

MEMORY_TEMPLATE = """

CURRENT MEMORY (playbook entries the solver was shown; used for the novelty check only):
{memory}

Reply with ONLY the JSON object."""

NO_ANSWER_CRITIQUE = ("The reply contains no Finish[<number>] line (it may have been cut off or failed). Answer more "
                      "concisely and end with a single line Finish[<number>].")

CONSOLIDATE_TEMPLATE = """A reviewer wrote the lesson below after verifying one financial-formula calculation. Rewrite it as ONE entry for a playbook that will be shown before solving OTHER questions: a single imperative sentence of at most 40 words about procedure (which quantity to use, unit or scale conventions, rounding, intermediate steps). No company names, no numbers taken from the specific question, no reference to "this question". Output the sentence only.

Lesson: {lesson}"""


def _cap_lines(text: str, cap: int) -> str:
    """Whole lines up to `cap` chars (Playbook.render already ranks the most reinforced first under a cap)."""
    out, total = [], 0
    for line in (text or "").splitlines():
        if total + len(line) + 1 > cap:
            break
        out.append(line)
        total += len(line) + 1
    return "\n".join(out)


def build_critic_prompt(task: dict, traj: Trajectory, memory_text: str, prior_critique: str | None, round_idx: int,
                        K: int, adaptive: bool, memory_cap: int = CRITIC_MEMORY_CAP_CHARS) -> str:
    template = CRITIC_TEMPLATE + fields_spec(adaptive)
    if adaptive:
        template += MEMORY_TEMPLATE
    prior = ""
    if prior_critique and prior_critique.strip():
        prior = PRIOR_TEMPLATE.format(round_idx=round_idx, K=K, prior_critique=prior_critique.strip()[:3000])
    trace = traj.text or ""
    if len(trace) > TRACE_CAP_CHARS:
        trace = trace[:TRACE_CAP_CHARS] + "\n[... reply truncated for review ...]"
    mem = _cap_lines(memory_text, memory_cap) or "(empty)"
    # .format runs once on the template; question / trace / memory are values, so braces in them are safe
    return template.format(question=task["question"], trace=trace, answer=traj.answer or "(none)",
                           prior_block=prior, memory=mem)


class FormulaCritic:
    """One chat call per round. No call when the attempt produced no answer (nothing to verify: verdict
    errors_found with a fixed critique). Unparseable output is retried `retries` times, then parsed
    conservatively (errors_found, refine=true, store=false). A transport failure yields no_errors with parsed=false
    and no lesson, so a critic outage neither burns solver rounds nor writes memory."""

    def __init__(self, llm, adaptive: bool, max_tokens: int = 8192, temperature: float = 0.0,
                 memory_cap: int = CRITIC_MEMORY_CAP_CHARS, retries: int = 1):
        self.llm, self.adaptive = llm, adaptive
        self.max_tokens, self.temperature, self.memory_cap, self.retries = max_tokens, temperature, memory_cap, retries
        self.calls = self.parse_failures = self.call_failures = self.skipped_no_answer = 0
        self.history: list[dict] = []

    def build_prompt(self, task, traj, memory_text, prior_critique, round_idx, K) -> str:
        return build_critic_prompt(task, traj, memory_text, prior_critique, round_idx, K, self.adaptive, self.memory_cap)

    def reflect(self, task: dict, traj: Trajectory, memory_text: str, prior_critique: str | None,
                round_idx: int, K: int) -> Reflection:
        t0 = time.time()
        usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
        if not (traj.answer or "").strip():
            self.skipped_no_answer += 1
            self.history.append({**usage, "elapsed_s": 0.0, "skipped": "no_answer"})
            return Reflection(verdict="incorrect", critique=NO_ANSWER_CRITIQUE, refine=True, store=False, parsed=True)
        prompt = self.build_prompt(task, traj, memory_text, prior_critique, round_idx, K)
        refl = None
        for _ in range(self.retries + 1):
            self.calls += 1
            usage["calls"] += 1
            try:
                txt, u = self.llm(prompt, max_tokens=self.max_tokens, temperature=self.temperature)
            except Exception as e:
                self.call_failures += 1
                refl = Reflection(verdict="correct", critique=f"(critic call failed: {type(e).__name__}: {e})"[:500],
                                  refine=False, store=False, parsed=False, raw="")
                break
            usage["prompt_tokens"] += int(u.get("prompt_tokens", 0) or 0)
            usage["completion_tokens"] += int(u.get("completion_tokens", 0) or 0)
            refl = parse_reflection(txt, self.adaptive)
            if refl.parsed:
                break
        else:
            self.parse_failures += 1
        self.history.append({**usage, "elapsed_s": round(time.time() - t0, 2)})
        return refl

    def drain(self) -> list[dict]:
        h, self.history = self.history, []
        return h


class LLMConsolidator:
    """`--consolidator llm`: one call turns the admitted lesson into one general playbook line (falls back to the
    raw lesson when the call fails or returns nothing). The default AppendConsolidator stores the lesson as is."""

    def __init__(self, llm, max_tokens: int = 512, temperature: float = 0.0):
        self.llm, self.max_tokens, self.temperature = llm, max_tokens, temperature
        self.calls = self.failures = 0
        self.history: list[dict] = []

    def rewrite(self, lesson: str) -> str:
        self.calls += 1
        usage = {"calls": 1, "prompt_tokens": 0, "completion_tokens": 0}
        try:
            text, u = self.llm(CONSOLIDATE_TEMPLATE.format(lesson=lesson.strip()), max_tokens=self.max_tokens,
                               temperature=self.temperature)
            usage["prompt_tokens"] = int(u.get("prompt_tokens", 0) or 0)
            usage["completion_tokens"] = int(u.get("completion_tokens", 0) or 0)
        except Exception:
            self.failures += 1
            text = ""
        lines = [l.strip().strip("\"'“”") for l in (text or "").splitlines() if l.strip()]
        entry = lines[-1] if lines else ""
        self.history.append({**usage, "fallback": not entry})
        return entry or lesson.strip()

    def consolidate(self, playbook, lesson: str, task, traj) -> str:
        return playbook.add(self.rewrite(lesson))

    def drain(self) -> list[dict]:
        h, self.history = self.history, []
        return h
