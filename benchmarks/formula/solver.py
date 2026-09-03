"""Formula solver: one chat call per attempt with the prompt used for the paper's runs.

prompts/solver/formula.txt takes four positional values: the playbook text (whole; "" on the arms without
memory), the reflection ("(empty)" on the first attempt, the reviewer's critique afterwards), the question and
the (empty) context. The reply is asked for as JSON; `extract_answer` reads the answer back with the fallbacks
of the paper's runs, ending in the "No final answer found" sentinel (scored as wrong). A failed call yields
Trajectory(failed=True): no critic runs and the episode stops.
"""
import json
import os
import random
import re
import time
from pathlib import Path

from remo.interfaces import Trajectory

DEFAULT_MODEL = "GPT-OSS-120B"
DEFAULT_BASE_URL = os.environ.get("REMO_BASE_URL", "http://localhost:8125/v1")
PROMPTS = Path(__file__).resolve().parents[2] / "prompts"
SOLVER_MAX_TOKENS = 8192
FIRST_REFLECTION = "(empty)"
RETRY_REFLECTION = "A reviewer found problems with your previous attempt. Reviewer critique:\n{critique}"
NO_ANSWER = "No final answer found"


def read_prompt(rel: str) -> str:
    with open(PROMPTS / rel, encoding="utf-8", newline="") as f:
        return f.read()


SOLVER_PROMPT = read_prompt("solver/formula.txt")


class ChatLLM:
    """OpenAI-compatible chat completion as the paper's runs made it: one user message, temperature 0, a
    max_tokens per role. Called as llm(prompt, max_tokens) -> (text, usage). A reply without content is an
    error (a failed call); timeouts, rate limits, connection and 5xx errors are retried after a pause."""

    def __init__(self, base_url: str = DEFAULT_BASE_URL, model: str = DEFAULT_MODEL, timeout: float = 600.0,
                 transient_retries: int = 5, pause_s: float = 15.0):
        import openai
        self.openai = openai
        self.client = openai.OpenAI(api_key=os.environ.get("REMO_API_KEY", "EMPTY"), base_url=base_url, timeout=timeout)
        self.model, self.transient_retries, self.pause_s = model, transient_retries, pause_s

    def served_models(self) -> list[str]:
        return [m.id for m in self.client.models.list().data]

    def __call__(self, prompt: str, max_tokens: int) -> tuple[str, dict]:
        transient = (self.openai.APITimeoutError, self.openai.APIConnectionError, self.openai.RateLimitError,
                     self.openai.InternalServerError)
        for attempt in range(self.transient_retries + 1):
            try:
                r = self.client.chat.completions.create(model=self.model, messages=[{"role": "user", "content": prompt}],
                                                        temperature=0.0, max_tokens=max_tokens)
                break
            except transient:
                if attempt == self.transient_retries:
                    raise
                time.sleep(self.pause_s * random.uniform(0.5, 1.5))
        if not r.choices:
            raise RuntimeError("Empty response from API")
        content = r.choices[0].message.content
        if content is None:
            raise RuntimeError("API returned None content")
        usage = {"prompt_tokens": int(getattr(r.usage, "prompt_tokens", 0) or 0),
                 "completion_tokens": int(getattr(r.usage, "completion_tokens", 0) or 0),
                 "finish_reason": r.choices[0].finish_reason}
        return content, usage


def build_solver_prompt(task: dict, memory_text: str, critique: str | None) -> str:
    reflection = FIRST_REFLECTION if critique is None else RETRY_REFLECTION.format(critique=critique)
    return SOLVER_PROMPT.format(memory_text, reflection, task["question"], task["context"])


def _boxed(text: str) -> str | None:
    m = re.search(r"\\boxed\{", text)
    if not m:
        return None
    start, depth = m.end() - 1, 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1:i]
    return None


def extract_answer(response: str) -> str:
    """The answer of a reply, in the order the paper's runs tried: whole reply as JSON -> its final_answer; the
    last Finish[...]; "final_answer": "..." / '...' / unquoted; "the final answer is \\boxed{...}"; "the final
    answer is X"; otherwise the sentinel NO_ANSWER."""
    try:
        parsed = json.loads(response)
        return str(parsed.get("final_answer", NO_ANSWER))
    except (json.JSONDecodeError, KeyError, AttributeError):
        pass
    m = re.findall(r"Finish\[(.*?)\]", response)
    if m:
        return m[-1]
    m = re.findall(r'"final_answer"\s*:\s*"([^"]*)"', response)
    if m:
        return m[-1]
    m = re.findall(r"'final_answer'\s*:\s*'([^']*)'", response)
    if m:
        return m[-1]
    m = re.findall(r'[\'"]final_answer[\'"]\s*:\s*([^,}]+)', response)
    if m:
        return re.sub(r"[,}]*$", "", m[-1].strip())
    m = re.search(r"[Tt]he final answer is:?\s*\$?\\boxed\{", response)
    if m:
        boxed = _boxed(response[m.start():])
        if boxed:
            return boxed
    m = re.findall(r"[Tt]he final answer is:?\s*([^\n.]+)", response)
    if m:
        a = re.sub(r"^\$?\\boxed\{([^}]+)\}\$?$", r"\1", m[-1].strip()).replace("$", "").strip()
        if a:
            return a
    return NO_ANSWER


class FormulaSolver:
    """The paper's `Solve(task, M, rho)`. `history` keeps one record per attempt (reflection sent, full reply,
    answer, usage) until the runner drains it into the run directory."""

    def __init__(self, llm, max_tokens: int = SOLVER_MAX_TOKENS):
        self.llm, self.max_tokens = llm, max_tokens
        self.calls = self.failures = 0
        self.history: list[dict] = []

    def solve(self, task: dict, memory_text: str, critique: str | None) -> Trajectory:
        prompt = build_solver_prompt(task, memory_text, critique)
        reflection = FIRST_REFLECTION if critique is None else RETRY_REFLECTION.format(critique=critique)
        t0 = time.time()
        self.calls += 1
        rec = {"reflection": reflection, "prompt_chars": len(prompt), "answer": "", "text": "", "error": "",
               "prompt_tokens": 0, "completion_tokens": 0, "finish_reason": None}
        try:
            text, usage = self.llm(prompt, self.max_tokens)
        except Exception as e:                       # noqa: BLE001 — the paper's runs stopped the episode here
            self.failures += 1
            rec.update(error=f"{type(e).__name__}: {e}"[:300], elapsed_s=round(time.time() - t0, 2))
            self.history.append(rec)
            return Trajectory(text="", completed=False, failed=True, meta={"error": rec["error"]})
        answer = extract_answer(text)
        rec.update(answer=answer, text=text, elapsed_s=round(time.time() - t0, 2),
                   prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
                   completion_tokens=int(usage.get("completion_tokens", 0) or 0), finish_reason=usage.get("finish_reason"))
        self.history.append(rec)
        return Trajectory(text=text, answer=answer, completed=True, meta={k: v for k, v in rec.items() if k != "text"})

    def drain(self) -> list[dict]:
        h, self.history = self.history, []
        return h
