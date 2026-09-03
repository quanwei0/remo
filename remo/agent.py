"""Algorithms 1 and 2 as one loop over pluggable Solver / Critic / Consolidator."""
import json
import os
from dataclasses import asdict
from typing import Any

from .config import RemoConfig
from .interfaces import Consolidator, Critic, Solver, Trajectory
from .memory import Playbook
from .policy import EpisodeState, RemoPolicy


class AppendConsolidator:
    """Default consolidator: the lesson itself becomes the entry (no LLM call)."""
    def consolidate(self, playbook: Playbook, lesson: str, task: Any, traj: Trajectory) -> str:
        return playbook.add(lesson)


class ReMoAgent:
    def __init__(self, cfg: RemoConfig, solver: Solver, critic: Critic, playbook: Playbook | None = None,
                 consolidator: Consolidator | None = None, run_dir: str | None = None):
        self.cfg = cfg
        self.solver, self.critic = solver, critic
        self.playbook = playbook if playbook is not None else Playbook()   # an EMPTY Playbook is falsy (__len__), keep its prefix
        self.consolidator = consolidator if consolidator is not None else AppendConsolidator()
        self.policy = RemoPolicy(cfg)
        self.run_dir = run_dir
        if run_dir:
            os.makedirs(run_dir, exist_ok=True)
            self._load()

    # -- one task ---------------------------------------------------------------------------------
    def run_task(self, task: Any, task_index: int) -> dict:
        st = EpisodeState()
        memory_text = self.playbook.render(self.cfg.inject_cap_chars) if self.cfg.use_memory else ""
        critique, traj = None, None
        for r in range(1, self.cfg.K + 1):
            traj = self.solver.solve(task, memory_text, critique)
            refl = self.critic.reflect(task, traj, memory_text, critique, r, self.cfg.K)
            action = self.policy.after_round(st, traj.completed, refl)
            if action != "retry":
                break
            critique = refl.critique
        gate = self.policy.gate(st)
        cited = st.last.cited_id or Playbook.find_cited_id(st.last.novelty_reason)
        decision = self.policy.memory_decision(st, task_index, cited) if self.cfg.use_memory else "no_memory"
        entry_id = ""
        if decision == "stored":
            entry_id = self.consolidator.consolidate(self.playbook, st.lesson(), task, traj)
        elif decision == "reinforced":
            if not self.playbook.reinforce(cited):
                decision = "discarded"
            entry_id = cited
        rec = {"task_index": task_index, "gate": gate, "stop_reason": st.stop_reason,
               "rounds": [{"round": x.round, "completed": x.completed, **asdict(x.reflection)} for x in st.rounds],
               "store_decision": decision, "entry_id": entry_id, "frozen": self.policy.frozen,
               "memory_chars_at_start": len(memory_text), "final_answer": traj.answer if traj else "",
               "final_completed": traj.completed if traj else False}
        if self.run_dir:
            self._save(rec)
        return rec

    # -- persistence (resumable: one jsonl line per task + playbook + policy state) ---------------
    def _save(self, rec: dict) -> None:
        with open(os.path.join(self.run_dir, "episodes.jsonl"), "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        self.playbook.save(os.path.join(self.run_dir, "playbook.txt"))
        json.dump(self.policy.state(), open(os.path.join(self.run_dir, "policy_state.json"), "w"))

    def _load(self) -> None:
        pb = os.path.join(self.run_dir, "playbook.txt")
        if os.path.exists(pb):
            self.playbook = Playbook.load(pb, prefix=self.playbook.prefix)
        ps = os.path.join(self.run_dir, "policy_state.json")
        if os.path.exists(ps):
            self.policy.load_state(json.load(open(ps)))

    def done_indices(self) -> set[int]:
        p = os.path.join(self.run_dir or "", "episodes.jsonl")
        if not self.run_dir or not os.path.exists(p):
            return set()
        return {json.loads(l)["task_index"] for l in open(p)}
