"""Append-only playbook with per-entry helpful counters (the memory M of Algorithms 1 and 2).

Entry line format (shared with the FinanceGym/Formula runs):  [<prefix>-00012] helpful=3 :: <text>
"""
import os
import re
from dataclasses import dataclass, field

_ENTRY = re.compile(r"^\[([a-z]+-\d{5})\] helpful=(\d+) :: (.*)$")
_ID = re.compile(r"([a-z]+-\d{5})")


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

    # -- writes (append-only: text of an existing entry is never rewritten) --------------------
    def add(self, text: str) -> str:
        eid = f"{self.prefix}-{self._next:05d}"
        self._next += 1
        self.entries.append(Entry(eid, " ".join(text.split())))
        return eid

    def reinforce(self, eid: str) -> bool:
        """Redundancy as a vote: helpful+1 on the cited entry. Zero LLM calls."""
        for e in self.entries:
            if e.id == eid:
                e.helpful += 1
                return True
        return False

    # -- reads ---------------------------------------------------------------------------------
    def get(self, eid: str) -> Entry | None:
        return next((e for e in self.entries if e.id == eid), None)

    def render(self, cap_chars: int | None = None) -> str:
        """Text prepended to the solver. With a cap, keep the most-reinforced entries first."""
        lines = [f"[{e.id}] helpful={e.helpful} :: {e.text}" for e in self.entries]
        if cap_chars is None:
            return "\n".join(lines)
        ranked = sorted(zip(self.entries, lines), key=lambda p: (-p[0].helpful, p[0].id))
        out, total = [], 0
        for _, line in ranked:
            if total + len(line) > cap_chars:
                break
            out.append(line); total += len(line)
        return "\n".join(out)

    @staticmethod
    def find_cited_id(*texts: str) -> str:
        """First entry id mentioned in any of the texts (critics often cite in novelty_reason)."""
        for t in texts:
            m = _ID.search(t or "")
            if m:
                return m.group(1)
        return ""

    def __len__(self):
        return len(self.entries)

    def chars(self) -> int:
        return len(self.render())

    # -- persistence ---------------------------------------------------------------------------
    def save(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(self.render() + ("\n" if self.entries else ""))

    @classmethod
    def load(cls, path: str, prefix: str = "les") -> "Playbook":
        pb = cls(prefix=prefix)
        if not os.path.exists(path):
            return pb
        for line in open(path):
            m = _ENTRY.match(line.rstrip("\n"))
            if m:
                pb.entries.append(Entry(m.group(1), m.group(3), int(m.group(2))))
        if pb.entries:
            pb._next = max(int(e.id.split("-")[1]) for e in pb.entries) + 1
        return pb
