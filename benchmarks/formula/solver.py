"""Formula solver: one chat call per attempt.

Prompt = memory (playbook lines, when the arm uses memory) + question + the reviewer's critique of the previous
attempt (retries only); the reply is short reasoning followed by a last line `Finish[<number>]`. The answer is
the last `Finish[...]` in the reply (fallback: the last number in the reply). `completed` is the benchmark's
objective signal — an answer was produced; a failed or truncated generation yields completed=False and still
spends a round (mean rounds count it).
"""
import os
import re
import time

from remo.interfaces import Trajectory

DEFAULT_MODEL = "GPT-OSS-120B"
DEFAULT_BASE_URL = os.environ.get("REMO_BASE_URL", "http://localhost:8125/v1")

SOLVER_INTRO = ("You are answering a quantitative finance question. Use the formula the question calls for with the "
                "quantities it gives: name the inputs you use, compute in a few short steps, and check units, scale "
                "and rounding before you answer.")
MEMORY_INTRO = ("Lessons from earlier questions (one per line: [id] helpful=<votes> :: lesson). Apply the ones that "
                "fit this question and ignore the rest.")
CRITIQUE_INTRO = "A reviewer checked your previous attempt at this question and found a problem:"
FINAL_INSTRUCTION = ("Reply with brief reasoning (a few lines), then end with one line of the form Finish[<number>] "
                     "that contains only the final number: no currency symbol, percent sign, thousands separators "
                     "or words.")

_FINISH = re.compile(r"Finish\[(.*?)\]", re.S)
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?|-?\.\d+")
_WRAP = "\"'`“”‘’[]"


class ChatLLM:
    """OpenAI-compatible chat completion (vLLM). Called as llm(prompt, max_tokens, temperature) -> (text, usage).
    Reasoning models may return an empty content when the budget is spent on hidden reasoning; the caller treats
    that as "no answer"."""

    def __init__(self, base_url: str = DEFAULT_BASE_URL, model: str = DEFAULT_MODEL, timeout: float = 600.0,
                 max_retries: int = 2):
        import openai
        self.client = openai.OpenAI(api_key=os.environ.get("REMO_API_KEY", "EMPTY"), base_url=base_url,
                                    timeout=timeout, max_retries=max_retries)
        self.model = model

    def served_models(self) -> list[str]:
        return [m.id for m in self.client.models.list().data]

    def __call__(self, prompt: str, max_tokens: int = 8192, temperature: float = 0.0) -> tuple[str, dict]:
        r = self.client.chat.completions.create(model=self.model, messages=[{"role": "user", "content": prompt}],
                                                max_tokens=max_tokens, temperature=temperature)
        choice = r.choices[0]
        usage = {"prompt_tokens": int(getattr(r.usage, "prompt_tokens", 0) or 0),
                 "completion_tokens": int(getattr(r.usage, "completion_tokens", 0) or 0),
                 "finish_reason": choice.finish_reason}
        return choice.message.content or "", usage


def build_solver_prompt(question: str, memory_text: str, critique: str | None) -> str:
    parts = [SOLVER_INTRO]
    if memory_text and memory_text.strip():
        parts.append(f"{MEMORY_INTRO}\n{memory_text.strip()}")
    parts.append(f"Question: {question.strip()}")
    if critique and critique.strip():
        parts.append(f"{CRITIQUE_INTRO}\n{critique.strip()}\nFix this in your new attempt.")
    parts.append(FINAL_INSTRUCTION)
    return "\n\n".join(parts)


def _clean(s: str) -> str:
    s = s.strip()
    while len(s) >= 1 and (s[0] in _WRAP or s[-1] in _WRAP):
        s = s.strip(_WRAP).strip()
    return s


def extract_answer(text: str) -> str:
    """Content of the last Finish[...]; without one, the last number in the text; "" if neither."""
    found = _FINISH.findall(text or "")
    if found:
        return _clean(found[-1])
    nums = _NUMBER.findall(text or "")
    return nums[-1].rstrip(",") if nums else ""


class FormulaSolver:
    """Solver of the paper's `Solve(task, M, rho)`. `history` keeps one record per attempt (full reply, answer,
    usage) until the runner drains it into the run directory."""

    def __init__(self, llm, max_tokens: int = 8192, temperature: float = 0.0):
        self.llm, self.max_tokens, self.temperature = llm, max_tokens, temperature
        self.calls = self.failures = 0
        self.history: list[dict] = []

    def solve(self, task: dict, memory_text: str, critique: str | None) -> Trajectory:
        prompt = build_solver_prompt(task["question"], memory_text, critique)
        t0 = time.time()
        self.calls += 1
        try:
            text, usage = self.llm(prompt, max_tokens=self.max_tokens, temperature=self.temperature)
            error = ""
        except Exception as e:                       # transport / server failure: the round is spent
            self.failures += 1
            text, usage, error = "", {}, f"{type(e).__name__}: {e}"[:300]
        answer = extract_answer(text)
        meta = {"answer": answer, "prompt_chars": len(prompt), "elapsed_s": round(time.time() - t0, 2),
                "error": error, "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
                "finish_reason": usage.get("finish_reason")}
        self.history.append({**meta, "text": text, "critique_in": critique or ""})
        view = text if text.strip() else (f"(the solver produced no text: "
                                          f"{error or 'empty output, finish_reason=' + str(meta['finish_reason'])})")
        return Trajectory(text=view, answer=answer, completed=bool(answer), meta=meta)

    def drain(self) -> list[dict]:
        h, self.history = self.history, []
        return h
