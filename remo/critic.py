"""Critic output parsing shared by the benchmark adapters. Each adapter ships the critic prompt used
for the paper's runs and turns the reply into the one Reflection the policy understands."""
import json
import re
from typing import Callable

from .interfaces import Reflection


def extract_json(text: str) -> dict:
    """Greedy locator (Formula / FinanceGym runs): first "{" to last "}", or {}."""
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return {}
    try:
        d = json.loads(m.group(0))
        return d if isinstance(d, dict) else {}
    except json.JSONDecodeError:
        return {}


def _balanced_objects(text: str) -> list[str]:
    out, i = [], 0
    while i < len(text):
        if text[i] == "{":
            depth, start, i = 1, i, i + 1
            while i < len(text) and depth > 0:
                c = text[i]
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                elif c == '"':
                    i += 1
                    while i < len(text) and text[i] != '"':
                        if text[i] == "\\":
                            i += 1
                        i += 1
                i += 1
            if depth == 0:
                out.append(text[start:i])
        else:
            i += 1
    return out


def extract_json_balanced(text: str):
    """Locator of the AppWorld critic and of the consolidator replies: the whole text, then each
    ```json fence, then each balanced {...} block, the first that parses (any JSON type) or None."""
    text = text or ""
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    for m in re.findall(r"```json\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE):
        try:
            return json.loads(m.strip())
        except json.JSONDecodeError:
            continue
    for cand in _balanced_objects(text):
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            continue
    return None


def _bool(v, default: bool) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("true", "yes", "1")
    return default


def parse_reflection(text: str, adaptive: bool, *, verdict_key: str = "verdict",
                     lesson_keys: tuple[str, ...] = ("lesson", "key_insight"), no_errors: str = "no_errors",
                     errors: str = "errors_found", extract: Callable[[str], object] = extract_json) -> Reflection:
    """Reply text -> Reflection. verdict: the parsed value when it is one of the two literals; any other
    present value counts as errors; when absent, `no_errors` iff its quoted literal occurs in the text.
    critique defaults to the first 2000 chars of the reply, lesson to "" (first non-empty of lesson_keys).
    Adaptive fields default to the conservative side: refine=True (retry as ReMo would), store=False."""
    text = text or ""
    d = extract(text)
    d = d if isinstance(d, dict) else {}
    v = d.get(verdict_key)
    if v in (no_errors, errors):
        pass
    elif v:
        v = errors
    else:
        v = no_errors if f'"{no_errors}"' in text else errors
    refl = Reflection(verdict="correct" if v == no_errors else "incorrect",
                      critique=str(d.get("critique", text[:2000])),
                      lesson=next((str(d[k]) for k in lesson_keys if d.get(k)), ""),
                      parsed=bool(d), raw=text)
    if adaptive:
        refl.refine = _bool(d.get("refine"), True)
        refl.store = _bool(d.get("store"), False)
        refl.novelty_reason = str(d.get("novelty_reason", ""))
        refl.cited_id = str(d.get("cited_id", "")).strip("[] ")
    return refl
