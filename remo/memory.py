"""The memory M of Algorithms 1 and 2, in the two shapes the paper's runs used.

Playbook            flat, one entry per line  "[fin-00012] helpful=3 <text>"  (FinanceGym)
SectionedPlaybook   markdown sections holding bullet lines, written by ADD operations of a
                    consolidator model; style "counts" = "[calc-00012] helpful=0 harmful=0 :: <text>"
                    (Formula), style "plain" = "[shr-00012] <text>" (AppWorld)
Both are append-only: an entry's text is never rewritten; reinforcement edits its counter / tag.
"""
import os
import re
from dataclasses import dataclass, field

from .critic import extract_json_balanced

_ID = re.compile(r"([a-z]+-\d{5})")


def find_cited_ids(*texts: str) -> list[str]:
    """Every entry id mentioned in the texts, in order of first appearance, without duplicates."""
    out = []
    for t in texts:
        for i in _ID.findall(t or ""):
            if i not in out:
                out.append(i)
    return out


# -- flat playbook --------------------------------------------------------------------------------
_ENTRY = re.compile(r"^\[([a-z]+-\d{5})\] helpful=(\d+) (.*)$")


@dataclass
class Entry:
    id: str
    text: str
    helpful: int = 1


@dataclass
class Playbook:
    prefix: str = "les"
    entries: list[Entry] = field(default_factory=list)
    _next: int = 1

    find_cited_ids = staticmethod(find_cited_ids)

    def add(self, text: str) -> str:
        eid = f"{self.prefix}-{self._next:05d}"
        self._next += 1
        self.entries.append(Entry(eid, " ".join(text.strip().splitlines())))   # the run wrote lesson.strip(); only line breaks are folded
        return eid

    def reinforce(self, eid: str) -> bool:
        """Redundancy as a vote: helpful+1 on the cited entry. Zero LLM calls."""
        e = self.get(eid)
        if e is None:
            return False
        e.helpful += 1
        return True

    def get(self, eid: str) -> Entry | None:
        return next((e for e in self.entries if e.id == eid), None)

    def ids(self) -> list[str]:
        return [e.id for e in self.entries]

    @staticmethod
    def _line(e: Entry) -> str:
        return f"[{e.id}] helpful={e.helpful} {e.text}"

    def render(self, cap_chars: int | None = None) -> str:
        """Text prepended to the solver. Under a cap the lines are ranked by (-helpful, line) and
        taken until the first one that would overflow."""
        if cap_chars is None:
            return "\n".join(self._line(e) for e in self.entries)
        ranked = sorted(((e.helpful, self._line(e)) for e in self.entries), key=lambda p: (-p[0], p[1]))
        out, total = [], 0
        for _, line in ranked:
            if total + len(line) > cap_chars:
                break
            out.append(line)
            total += len(line)
        return "\n".join(out)

    def __len__(self):
        return len(self.entries)

    def chars(self) -> int:
        return len(self.render())

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.render() + ("\n" if self.entries else ""))

    @classmethod
    def load(cls, path: str, prefix: str = "les") -> "Playbook":
        pb = cls(prefix=prefix)
        if not os.path.exists(path):
            return pb
        with open(path, encoding="utf-8") as f:
            for line in f:
                m = _ENTRY.match(line.rstrip("\n"))
                if m:
                    pb.entries.append(Entry(m.group(1), m.group(3), int(m.group(2))))
        if pb.entries:
            pb._next = max(int(e.id.split("-")[1]) for e in pb.entries) + 1
        return pb


# -- sectioned playbook ---------------------------------------------------------------------------
_SKELETON = {
    "counts": """## STRATEGIES & INSIGHTS

## FORMULAS & CALCULATIONS

## CODE SNIPPETS & TEMPLATES

## COMMON MISTAKES TO AVOID

## PROBLEM-SOLVING HEURISTICS

## CONTEXT CLUES & INDICATORS

## OTHERS""",
    "plain": """## STRATEGIES AND HARD RULES

## APIs TO USE FOR SPECIFIC INFORMATION

## USEFUL CODE SNIPPETS AND TEMPLATES

## COMMON MISTAKES AND CORRECT STRATEGIES

## PROBLEM-SOLVING HEURISTICS AND WORKFLOWS

## VERIFICATION CHECKLIST

## TROUBLESHOOTING AND PITFALLS:

## OTHERS""",
}
# normalised section name -> id slug; a name outside the table gets the initials of its words
_SLUGS = {
    "counts": {
        "financial_strategies_and_insights": "fin",
        "formulas_and_calculations": "calc",
        "code_snippets_and_templates": "code",
        "common_mistakes_to_avoid": "err",
        "problem_solving_heuristics": "prob",
        "context_clues_and_indicators": "ctx",
        "others": "misc",
        "meta_strategies": "meta",
    },
    "plain": {
        "strategies_and_hard_rules": "shr",
        "hard_rules": "hr",
        "strategies_and_insights": "si",
        "apis_to_use_for_specific_information": "api",
        "useful_code_snippets_and_templates": "code",
        "code_snippets_and_templates": "code",
        "common_mistakes_and_correct_strategies": "cms",
        "common_mistakes_to_avoid": "err",
        "problem_solving_heuristics_and_workflows": "psw",
        "problem_solving_heuristics": "prob",
        "verification_checklist": "vc",
        "troubleshooting_and_pitfalls": "ts",
        "others": "misc",
        "meta_strategies": "meta",
    },
}
# style "plain" only: ADD operations naming any other section are dropped before application
_ALLOWED_PLAIN = {"strategies_and_hard_rules", "apis_to_use_for_specific_information",
                  "useful_code_snippets_and_templates", "common_mistakes_and_correct_strategies",
                  "problem_solving_heuristics_and_workflows", "verification_checklist",
                  "troubleshooting_and_pitfalls", "others"}
_BULLET_COUNTS = re.compile(r"\[([^\]]+)\]\s*helpful=(\d+)\s*harmful=(\d+)\s*::\s*(.*)")
_BULLET_PLAIN = re.compile(r"\[([^\]]+)\]\s*(.*)")


def _parse_bullet(line: str, style: str) -> dict | None:
    text = line.strip()
    m = _BULLET_COUNTS.match(text)
    if m:
        return {"id": m.group(1), "helpful": int(m.group(2)), "harmful": int(m.group(3)), "content": m.group(4)}
    if style == "plain":
        m = _BULLET_PLAIN.match(text)
        if m:
            return {"id": m.group(1), "helpful": 0, "harmful": 0, "content": m.group(2).strip()}
    return None


def _format_bullet(bid: str, content, style: str) -> str:
    return f"[{bid}] helpful=0 harmful=0 :: {content}" if style == "counts" else f"[{bid}] {content}"


def _norm(name: str, style: str) -> str:
    s = name.lower().replace(" ", "_").replace("&", "and")
    return s.rstrip(":") if style == "plain" else s


def _slug(section: str, style: str) -> str:
    clean = _norm(section.lower().strip(), style)
    if clean in _SLUGS[style]:
        return _SLUGS[style][clean]
    words = clean.split("_")
    return words[0][:4] if len(words) == 1 else "".join(w[0] for w in words[:5])


def _header(line: str) -> str:
    return line.strip()[2:].strip()


def _apply_add_ops(text: str, ops: list[dict], next_id: int, style: str) -> tuple[str, list[str]]:
    """ADD operations as the paper's runs applied them: a new bullet goes to the end of its section
    (an unknown section means OTHERS), the whole text is stripped and rejoined with "\\n"."""
    lines = text.strip().split("\n")
    sections, current = {}, "general"
    for line in lines:
        if line.strip().startswith("##"):
            current = _norm(_header(line), style)
            sections.setdefault(current, [])
        elif line.strip():
            sections[current].append(line)
    to_add, new_ids = [], []
    for op in ops:
        if op["type"] != "ADD":
            continue
        section = _norm(op.get("section", "general"), style)      # a non-string section raises, as in the runs
        if section not in sections and section != "general":
            section = "others"
        bid = f"{_slug(section, style)}-{next_id:05d}"
        next_id += 1
        to_add.append((section, _format_bullet(bid, op.get("content", ""), style)))
        new_ids.append(bid)
    final, current = [], None
    for line in lines:
        if line.strip().startswith("##"):
            if current:
                final.extend(b for s, b in to_add if s == current)
                to_add = [(s, b) for s, b in to_add if s != current]
            current = _norm(_header(line), style)
        final.append(line)
    if current:
        final.extend(b for s, b in to_add if s == current)
        to_add = [(s, b) for s, b in to_add if s != current]
    if to_add:
        rest = [b for _, b in to_add]
        idx = next((i for i, l in enumerate(final) if l.strip() == "## OTHERS"), -1)
        if idx >= 0:
            final[idx + 1:idx + 1] = rest
        else:
            final.extend(rest)
    return "\n".join(final), new_ids


class SectionedPlaybook:
    """Markdown-section playbook; the text itself is the memory and is injected whole."""

    find_cited_ids = staticmethod(find_cited_ids)

    def __init__(self, text: str, style: str):
        assert style in _SKELETON, style
        self._text, self.style = text, style

    @classmethod
    def from_skeleton(cls, style: str) -> "SectionedPlaybook":
        return cls(_SKELETON[style], style)

    @property
    def text(self) -> str:
        return self._text

    def render(self, cap_chars: int | None = None) -> str:
        return self._text if cap_chars is None else self._text[:cap_chars]

    def _bullets(self) -> list[dict]:
        return [p for line in self._text.split("\n") if (p := _parse_bullet(line, self.style))]

    def ids(self) -> list[str]:
        return [p["id"] for p in self._bullets()]

    def __len__(self):
        return len(self._bullets())

    def chars(self) -> int:
        return len(self._text)

    @property
    def next_id(self) -> int:
        """Highest numeric id in the text + 1 (the counter is global across sections)."""
        nums = [int(m.group(1)) for p in self._bullets() if (m := re.search(r"-(\d+)$", p["id"]))]
        return max(nums, default=0) + 1

    def apply_add_ops(self, ops: list[dict]) -> list[str]:
        """Applies the ADD operations (other types are ignored); returns the new ids."""
        self._text, new_ids = _apply_add_ops(self._text, ops, self.next_id, self.style)
        return new_ids

    def parse_curator_response(self, text: str) -> list[dict] | None:
        """The ADD operations of a consolidator reply {"reasoning": str, "operations": [...]} —
        validated as in the paper's runs — or None when the reply is not usable. Style "counts"
        tolerates other operation types (never applied); style "plain" rejects the reply on any
        non-ADD operation and drops ADDs naming a section outside its section list."""
        info = extract_json_balanced(text)
        if not isinstance(info, dict) or not info:
            return None
        if not isinstance(info.get("reasoning"), str) or not isinstance(info.get("operations"), list):
            return None
        ops = []
        for op in info["operations"]:
            if not isinstance(op, dict) or "type" not in op:
                return None
            if op["type"] != "ADD":
                if self.style == "plain":
                    return None
                continue
            if {"section", "content"} - set(op):
                return None
            if self.style == "plain" and _norm(op.get("section", "").strip(), "plain") not in _ALLOWED_PLAIN:
                continue
            ops.append(op)
        return ops

    def stats(self) -> dict:
        counts = self.style == "counts"
        st = {"total_bullets": 0}
        if counts:
            st.update({"high_performing": 0, "problematic": 0, "unused": 0})
        st["by_section"] = {}
        current = "general"
        for line in self._text.strip().split("\n"):
            if line.strip().startswith("##"):
                current = _header(line)
                continue
            p = _parse_bullet(line, self.style)
            if not p:
                continue
            st["total_bullets"] += 1
            if counts:
                if p["helpful"] > 5 and p["harmful"] < 2:
                    st["high_performing"] += 1
                elif p["harmful"] >= p["helpful"] and p["harmful"] > 0:
                    st["problematic"] += 1
                elif p["helpful"] + p["harmful"] == 0:
                    st["unused"] += 1
            sec = st["by_section"].setdefault(current, {"count": 0, "helpful": 0, "harmful": 0} if counts else {"count": 0})
            sec["count"] += 1
            if counts:
                sec["helpful"] += p["helpful"]
                sec["harmful"] += p["harmful"]
        return st

    def reinforce(self, entry_id: str) -> bool:
        """Redundancy as a vote on the cited bullet line: style "counts" adds 1 to its helpful
        counter; style "plain" appends " [confirmed x2]" and afterwards raises the number."""
        bid = re.escape(entry_id)
        if self.style == "counts":
            self._text, k = re.subn(rf"(\[{bid}\] helpful=)(\d+)", lambda m: f"{m.group(1)}{int(m.group(2)) + 1}", self._text)
        else:
            def bump(m):
                n = int(m.group(2)) + 1 if m.group(2) else 2
                return f"{m.group(1)} [confirmed x{n}]"
            self._text, k = re.subn(rf"(\[{bid}\][^\n]*?)(?: \[confirmed x(\d+)\])?$", bump, self._text, count=1, flags=re.M)
        return bool(k)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(self._text)

    @classmethod
    def load(cls, path: str, style: str) -> "SectionedPlaybook":
        with open(path, encoding="utf-8") as f:
            return cls(f.read(), style)
