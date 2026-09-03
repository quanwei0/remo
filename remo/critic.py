"""Shared critic output contract. Benchmark adapters own the domain part of the prompt (what
counts as an error on Formula / FinanceGym / AppWorld); this module owns the decision fields and
their parsing, so every adapter feeds the policy the same Reflection."""
import json
import re

from .interfaces import Reflection

REMO_FIELDS = """Output ONLY a JSON object:
{{
  "checks": "your independent verification work",
  "verdict": "no_errors" or "errors_found",
  "critique": "if errors_found: exactly what is wrong and the concrete fix; if no_errors: one sentence on why it is trustworthy",
  "lesson": "the single reusable lesson, stated generally"
}}"""

ADAREMO_FIELDS = """Then make two more decisions:
- refine (only matters if errors_found): true ONLY if you can state ONE concrete, actionable correction the solver can apply in a retry. If the failure comes from missing information, or a previous reviewer already gave the same correction and the retry still failed, answer false: a retry would be a lottery.
- store (only matters if no_errors): true ONLY if ALL hold: (a) the lesson is backed by verified evidence in THIS attempt, not speculation; (b) it generalizes beyond this single task; (c) the CURRENT MEMORY below does NOT already contain an equivalent entry. If an equivalent entry exists, answer false and cite its id in novelty_reason.

Output ONLY a JSON object:
{{
  "checks": "your independent verification work",
  "verdict": "no_errors" or "errors_found",
  "critique": "if errors_found: exactly what is wrong and the concrete fix; if no_errors: one sentence on why it is trustworthy",
  "refine": true or false,
  "store": true or false,
  "novelty_reason": "why the lesson is (not) already covered by the memory; cite the entry id like [les-00012] if covered",
  "lesson": "the single reusable lesson, stated generally"
}}"""


def fields_spec(adaptive: bool) -> str:
    return ADAREMO_FIELDS if adaptive else REMO_FIELDS


def extract_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return {}
    try:
        d = json.loads(m.group(0))
        return d if isinstance(d, dict) else {}
    except json.JSONDecodeError:
        return {}


def _bool(v, default: bool) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("true", "yes", "1")
    return default


def parse_reflection(text: str, adaptive: bool) -> Reflection:
    """Conservative defaults on unparseable output: refine=True (behave like ReMo), store=False."""
    d = extract_json(text)
    v = str(d.get("verdict", "")).strip().lower()
    if v not in ("no_errors", "errors_found"):
        v = "no_errors" if '"no_errors"' in (text or "") else "errors_found"
    refl = Reflection(
        verdict="correct" if v == "no_errors" else "incorrect",
        critique=str(d.get("critique", "") or (text or "")[:2000]),
        lesson=str(d.get("lesson", "") or d.get("key_insight", "")),
        parsed=bool(d), raw=text or "")
    if adaptive:
        refl.refine = _bool(d.get("refine"), True)
        refl.store = _bool(d.get("store"), False)
        refl.novelty_reason = str(d.get("novelty_reason", ""))
        refl.cited_id = str(d.get("cited_id", "")).strip("[] ")
    return refl
