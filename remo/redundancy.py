"""Structural redundancy check (redundant_mode "structural"): is an admitted episode's lesson already in
the memory? Two stages, both benchmark-agnostic:

1. LexicalRetriever — the entries most similar to the lesson by a weighted Jaccard over the words, where
   identifier-like tokens (API paths, snake_case names, back-ticked spans, numbers) weigh more than
   ordinary words. Entries in `principle_ids` (the seed / general-principle bullets) are never candidates:
   a principle such as "look at the API docs" does not cover a specific, API-level lesson.
2. a judge — LLMRedundancyChecker asks the model one narrow question about the retrieved candidates
   ("does one of these state the SAME specific rule?") and accepts only an id from the candidate list;
   LexicalRedundancyChecker needs no model and declares coverage above a similarity threshold.

No candidates -> no model call -> the lesson is novel. The checker never writes to the playbook; the
caller reinforces the covering entry (attaching the lesson as an example) or consolidates.
"""
import re
import time
from dataclasses import dataclass, field

from .critic import extract_json_balanced
from .interfaces import RedundancyResult

_TOKEN = re.compile(r"`[^`]+`|apis\.[\w.]+|[A-Za-z_][\w.]*_[\w.]*|\d{3,}|[A-Za-z]{3,}")
_STOP = {"the", "and", "for", "with", "that", "this", "when", "then", "than", "from", "into", "not", "are", "was",
         "were", "has", "have", "had", "use", "using", "used", "always", "never", "before", "after", "should", "must",
         "ensure", "make", "sure", "all", "any", "each", "every", "its", "their", "them", "they", "you", "your", "can",
         "will", "which", "what", "where", "while", "only", "also", "such", "like", "e.g", "i.e", "etc", "via", "per",
         "one", "two", "first", "last", "next", "new", "same", "other", "instead", "rather", "given", "task", "tasks",
         "agent", "model", "call", "calls", "calling", "api", "apis", "app", "apps", "step", "steps", "data", "value",
         "values", "field", "fields", "list", "result", "results", "return", "returns", "returned", "check", "verify"}


def specific_tokens(text: str) -> dict[str, float]:
    """token -> weight. Identifier-like tokens (contain '_', '.', digits, or were back-ticked) weigh 3,
    other words 1; stop words are dropped; case-insensitive."""
    out: dict[str, float] = {}
    for m in _TOKEN.finditer(text or ""):
        tok = m.group(0)
        ident = tok.startswith("`") or "_" in tok or "." in tok or any(c.isdigit() for c in tok)
        tok = tok.strip("`").lower().strip(".")
        if not tok or (tok in _STOP and not ident):
            continue
        out[tok] = max(out.get(tok, 0.0), 3.0 if ident else 1.0)
    return out


def weighted_jaccard(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    keys = set(a) | set(b)
    num = sum(min(a.get(k, 0.0), b.get(k, 0.0)) for k in keys)
    den = sum(max(a.get(k, 0.0), b.get(k, 0.0)) for k in keys)
    return num / den if den else 0.0


@dataclass
class LexicalRetriever:
    k: int = 3
    min_sim: float = 0.15

    def top(self, lesson: str, entries: list[tuple[str, str]], exclude: set[str] = frozenset()) -> list[tuple[str, str, float]]:
        """The up to `k` entries with similarity >= min_sim, best first: (id, text, similarity)."""
        q = specific_tokens(lesson)
        scored = [(eid, text, weighted_jaccard(q, specific_tokens(text))) for eid, text in entries if eid not in exclude]
        scored = [x for x in scored if x[2] >= self.min_sim]
        scored.sort(key=lambda x: -x[2])
        return scored[:self.k]


JUDGE_PROMPT = """You maintain a playbook of lessons for a coding agent. A new lesson was learned from a successful attempt.
Decide whether one of the EXISTING entries below already states the SAME rule, so that an agent following that entry would
avoid the exact mistake the new lesson is about.

Rules:
- "Covered" means the same app / API / field / condition and the same instruction. Being about a related topic, or being a
  more general principle that the new lesson is an instance of, is NOT coverage.
- If no entry states the same rule, answer null.

New lesson:
{lesson}

Existing entries (id in brackets):
{candidates}

Answer with one JSON object and nothing else:
{{"covered_by": "<entry id from the list, or null>", "reason": "<one sentence quoting the covering entry's key words, or why none covers it>"}}"""


@dataclass
class LLMRedundancyChecker:
    """Retrieve, then let the model judge the candidates. `llm.chat(messages, max_tokens, temperature) -> str`.
    A judge reply that names an id outside the candidate list, is unparseable, or fails to arrive counts
    as "not covered" (the lesson is stored; memory growth is the safer error here) and is logged."""
    llm: object
    retriever: LexicalRetriever = field(default_factory=LexicalRetriever)
    principle_ids: set[str] = field(default_factory=set)
    max_tokens: int = 1024
    temperature: float = 0.0
    log: object = print
    calls: int = 0
    failures: int = 0
    covered: int = 0
    novel: int = 0
    no_lesson: int = 0

    def check(self, lesson: str, playbook, episode=None, task=None, traj=None) -> RedundancyResult:
        lesson = " ".join((lesson or "").split())
        if not lesson or lesson.lower().strip(".") in ("none", "n/a", "na", "nothing"):
            self.no_lesson += 1
            return RedundancyResult(reason="no lesson")
        cands = self.retriever.top(lesson, playbook.entries_text(), exclude=self.principle_ids)
        if not cands:
            self.novel += 1
            return RedundancyResult(reason="no similar entry")
        ids = [c[0] for c in cands]
        prompt = JUDGE_PROMPT.format(lesson=lesson, candidates="\n".join(f"[{i}] {t}" for i, t, _ in cands))
        self.calls += 1
        t0 = time.time()
        try:
            reply = self.llm.chat([{"role": "user", "content": prompt}], self.max_tokens, self.temperature)
        except Exception as e:                        # noqa: BLE001 — transport / server failure
            self.failures += 1
            self.novel += 1
            self.log(f"[redundancy] judge call failed: {type(e).__name__}: {e}; treating the lesson as novel")
            return RedundancyResult(reason=f"judge failed: {type(e).__name__}"[:200], candidates=ids)
        d = extract_json_balanced(reply)
        d = d if isinstance(d, dict) else {}
        cid = str(d.get("covered_by") or "").strip("[] ")
        reason = str(d.get("reason", ""))[:300]
        if cid and cid not in ids:
            self.log(f"[redundancy] judge named {cid!r}, not a candidate {ids}; treating the lesson as novel")
            cid = ""
        if cid:
            self.covered += 1
        else:
            self.novel += 1
        return RedundancyResult(covered_by=cid, reason=reason or ("covered" if cid else "not covered"), candidates=ids)

    def stats(self) -> dict:
        return {"calls": self.calls, "failures": self.failures, "covered": self.covered, "novel": self.novel,
                "no_lesson": self.no_lesson}


@dataclass
class LexicalRedundancyChecker:
    """No model call: the best candidate covers the lesson when its similarity is >= `threshold`."""
    retriever: LexicalRetriever = field(default_factory=LexicalRetriever)
    principle_ids: set[str] = field(default_factory=set)
    threshold: float = 0.6
    calls: int = 0
    failures: int = 0
    covered: int = 0
    novel: int = 0
    no_lesson: int = 0

    def check(self, lesson: str, playbook, episode=None, task=None, traj=None) -> RedundancyResult:
        lesson = " ".join((lesson or "").split())
        if not lesson or lesson.lower().strip(".") in ("none", "n/a", "na", "nothing"):
            self.no_lesson += 1
            return RedundancyResult(reason="no lesson")
        cands = self.retriever.top(lesson, playbook.entries_text(), exclude=self.principle_ids)
        ids = [c[0] for c in cands]
        if cands and cands[0][2] >= self.threshold:
            self.covered += 1
            return RedundancyResult(covered_by=cands[0][0], reason=f"lexical similarity {cands[0][2]:.2f}", candidates=ids)
        self.novel += 1
        return RedundancyResult(reason="no similar entry" if not cands else f"best similarity {cands[0][2]:.2f} < {self.threshold}",
                                candidates=ids)

    def stats(self) -> dict:
        return {"calls": 0, "failures": 0, "covered": self.covered, "novel": self.novel, "no_lesson": self.no_lesson}
