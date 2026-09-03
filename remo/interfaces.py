"""Contracts between the core loop and a benchmark adapter."""
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from .memory import Playbook, SectionedPlaybook
    from .policy import EpisodeState


@dataclass
class Trajectory:
    """One solver attempt. `completed` is the objective environment signal (task submitted / answer
    produced); on benchmarks without an environment it is simply True. `failed` means the solver
    call itself failed: the critic is not called and the episode stops, not admitted."""
    text: str                       # what the critic reads (trace, report, ...)
    answer: str = ""                # final answer / report used for scoring or submission
    completed: bool = True
    failed: bool = False
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Reflection:
    """Critic output. verdict in {"correct", "incorrect"} ("none" when no critic ran). Adaptive
    fields default to the conservative side when absent or unparseable: refine=True (retry, as in
    ReMo), store=False. `failed` means the critic call itself failed: the episode stops, not admitted."""
    verdict: str
    critique: str = ""
    lesson: str = ""
    refine: bool = True
    store: bool = False
    novelty_reason: str = ""
    cited_id: str = ""
    parsed: bool = True
    raw: str = ""
    failed: bool = False

    @property
    def correct(self) -> bool:
        return self.verdict == "correct"


class Solver(Protocol):
    def solve(self, task: Any, memory_text: str, critique: str | None) -> Trajectory: ...


class Critic(Protocol):
    def reflect(self, task: Any, traj: Trajectory, memory_text: str, prior_critique: str | None,
                round_idx: int, K: int) -> Reflection: ...


class Consolidator(Protocol):
    """Writes what an admitted episode taught into the playbook (append-only). `episode` carries every
    round, so the adapter builds the consolidator input exactly as its original run did; `traj` is
    the accepted attempt. Returns the new entry id(s) joined by "," or "" when nothing was added."""
    def consolidate(self, playbook: "Playbook | SectionedPlaybook", episode: "EpisodeState", task: Any,
                    traj: Trajectory) -> str: ...
