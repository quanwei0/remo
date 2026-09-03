"""Pure decision logic of Algorithms 1 (ReMo) and 2 (AdaReMo). No I/O, no LLM calls, so a
benchmark harness with its own solve loop (e.g. FinanceGym) can drive it step by step.

Per task:   after every round call `after_round` -> "accept" | "retry" | "stop"
            when the loop ends call `gate` and then `memory_decision`
Across tasks: the saturation window (`store_window`, `frozen`) lives on the policy object.
"""
from dataclasses import dataclass, field

from .config import RemoConfig
from .interfaces import Reflection


@dataclass
class RoundRecord:
    round: int
    completed: bool
    reflection: Reflection


@dataclass
class EpisodeState:
    rounds: list[RoundRecord] = field(default_factory=list)
    admitted: bool = False
    stop_reason: str = "max_rounds"        # accepted | critic_stop | max_rounds | solver_error | critic_error

    @property
    def last(self) -> Reflection:
        return self.rounds[-1].reflection

    def lesson(self) -> str:
        """The lesson of the accepting attempt; falls back to the most recent non-empty one (a
        critic that passes a report often leaves `lesson` empty while the round that flipped the
        task carried it)."""
        for rec in reversed(self.rounds):
            if rec.reflection.lesson.strip():
                return rec.reflection.lesson.strip()
        return ""


class RemoPolicy:
    def __init__(self, cfg: RemoConfig):
        self.cfg = cfg
        self.store_window: list[bool] = []
        self.frozen = False
        self.freeze_events: list[dict] = []

    # -- inner loop -------------------------------------------------------------------------------
    def after_round(self, st: EpisodeState, traj_failed: bool, completed: bool, refl: Reflection) -> str:
        """Records the round and answers "accept" | "retry" | "stop". A failed solver call
        (traj_failed; `refl` is the placeholder recorded for the round, no critic ran) stops with
        "solver_error", a failed critic call (refl.failed) with "critic_error"; neither is admitted."""
        if traj_failed:
            completed = False
        st.rounds.append(RoundRecord(len(st.rounds) + 1, completed, refl))
        if traj_failed:
            st.stop_reason = "solver_error"
            return "stop"
        if refl.failed:
            st.stop_reason = "critic_error"
            return "stop"
        if completed and refl.correct:
            st.admitted, st.stop_reason = True, "accepted"
            return "accept"
        if self.cfg.adaptive and not refl.refine:      # Alg. 2: no actionable fix -> no lottery retry
            st.stop_reason = "critic_stop"
            return "stop"
        if len(st.rounds) >= self.cfg.K:
            st.stop_reason = "max_rounds"
            return "stop"
        return "retry"

    # -- outcome gate (Alg. 1 and 2) ---------------------------------------------------------------
    @staticmethod
    def gate(st: EpisodeState) -> str:
        if st.admitted:
            return "round1_clean" if len(st.rounds) == 1 else "cross_round_validated"
        return "critic_stop" if st.stop_reason == "critic_stop" else "never_clean"

    # -- memory decision -------------------------------------------------------------------------
    def memory_decision(self, st: EpisodeState, task_index: int, cited_present: list[str]) -> str:
        """Returns one of: skipped (not admitted), stored, reinforced, discarded, skipped_frozen.
        `cited_present` are the entry ids the critic cited that exist in the playbook. The caller
        performs the actual write for "stored" (consolidator, which may add nothing) and
        "reinforced" (playbook.reinforce on every cited id); this method only decides and does the
        saturation bookkeeping."""
        if not st.admitted:
            return "skipped"
        if not self.cfg.adaptive:                        # Alg. 1: every admitted episode is consolidated
            return "stored"
        refl = st.last
        probe = (task_index + 1) % self.cfg.probe_p == 0
        want = refl.store or self.cfg.redundant_mode == "off"
        decision = "discarded"
        if want:
            if self.frozen and not probe:
                decision = "skipped_frozen"
            else:
                if self.frozen:
                    self.frozen = False
                    self.freeze_events.append({"task_index": task_index, "event": "unfreeze(probe)"})
                decision = "stored"
        elif self.cfg.redundant_mode == "reinforce" and cited_present:
            decision = "reinforced"
        # saturation window records memory DEMAND (the critic wanted to store, or reinforced), not
        # whether the write happened: counting only actual writes would make a freeze self-sustaining
        # (frozen -> no writes -> window stays empty -> stays frozen). Same semantics as the Formula/AppWorld runs.
        self.store_window = (self.store_window + [want or decision == "reinforced"])[-self.cfg.freeze_w:]
        if (not self.frozen and len(self.store_window) >= self.cfg.freeze_w
                and sum(self.store_window) < self.cfg.freeze_rho * self.cfg.freeze_w):
            self.frozen = True
            self.freeze_events.append({"task_index": task_index, "event": "freeze",
                                       "window_rate": sum(self.store_window) / len(self.store_window)})
        return decision

    def state(self) -> dict:
        return {"frozen": self.frozen, "store_window": list(self.store_window),
                "freeze_events": list(self.freeze_events)}

    def load_state(self, d: dict) -> None:
        self.frozen = bool(d.get("frozen", False))
        self.store_window = list(d.get("store_window", []))
        self.freeze_events = list(d.get("freeze_events", []))
