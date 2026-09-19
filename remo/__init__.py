"""ReMo / AdaReMo core: benchmark-agnostic refinement + memory control (paper Algorithms 1 and 2)."""
from .config import RemoConfig
from .interfaces import Trajectory, Reflection, Solver, Critic, Consolidator, RedundancyChecker, RedundancyResult
from .policy import RemoPolicy, EpisodeState, RoundRecord
from .memory import Playbook, SectionedPlaybook
from .agent import ReMoAgent
from .redundancy import LexicalRetriever, LLMRedundancyChecker, LexicalRedundancyChecker

__all__ = ["RemoConfig", "Trajectory", "Reflection", "Solver", "Critic", "Consolidator",
           "RedundancyChecker", "RedundancyResult", "RemoPolicy", "EpisodeState", "RoundRecord", "Playbook", "SectionedPlaybook",
           "ReMoAgent", "LexicalRetriever", "LLMRedundancyChecker", "LexicalRedundancyChecker"]
