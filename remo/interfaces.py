"""Contracts between the core loop and a benchmark adapter."""
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class Trajectory:
    """One solver attempt. `completed` is the objective environment signal (task submitted / answer
    produced); on benchmarks without an environment it is simply True."""
    text: str                       # what the critic reads (trace, report, ...)
    answer: str = ""                # final answer / report used for scoring or submission
    completed: bool = True
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Reflection:
    """Critic output. verdict in {"correct", "incorrect"}. Adaptive fields default to the
    conservative side when absent or unparseable: refine=True (retry, as in ReMo), store=False."""
    verdict: str
    critique: str = ""
    lesson: str = ""
    refine: bool = True
    store: bool = False
    novelty_reason: str = ""
    cited_id: str = ""
    parsed: bool = True
    raw: str = ""

    @property
    def correct(self) -> bool:
        return self.verdict == "correct"


class Solver(Protocol):
    def solve(self, task: Any, memory_text: str, critique: str | None) -> Trajectory: ...


class Critic(Protocol):
    def reflect(self, task: Any, traj: Trajectory, memory_text: str, prior_critique: str | None,
                round_idx: int, K: int) -> Reflection: ...


class Consolidator(Protocol):
    """Writes ONE admitted lesson into the playbook (append-only). Returns the new entry id or ""."""
    def consolidate(self, playbook: "Playbook", lesson: str, task: Any, traj: Trajectory) -> str: ...
