"""Formula consolidation: the curator call of the paper's runs (default) or a plain append.

The lesson is built from the accepting round as the runs built it; prompts/consolidator/formula.txt is formatted
with token_budget, current_step (task index + 1), total_samples (tasks in the run), the playbook stats as indented
JSON, the lesson, the whole playbook text and the question. The reply's ADD operations go through
SectionedPlaybook.parse_curator_response / apply_add_ops. An unusable reply, a text the operations cannot be
applied to or a failed call leaves the playbook unchanged; the runner then records store_decision "curator_error".
"""
import json
import time
from typing import Any

from benchmarks.formula.solver import read_prompt
from remo.interfaces import Trajectory
from remo.memory import SectionedPlaybook
from remo.policy import EpisodeState, RemoPolicy

TOKEN_BUDGET = 80000
CURATOR_MAX_TOKENS = 8192
CURATOR_PROMPT = read_prompt("consolidator/formula.txt")


def lesson_text(episode: EpisodeState, adaptive: bool) -> str:
    """The reflection handed to the curator: the gate, the number of rounds, the accepting round's critique and
    key insight; AdaReMo adds the critic's novelty reason."""
    n, last = len(episode.rounds), episode.last
    text = (f"[validated: {RemoPolicy.gate(episode)}] The final answer passed independent review"
            f"{' after ' + str(n) + ' rounds of refinement' if n > 1 else ''}. "
            f"Reviewer assessment: {last.critique}\nKey reusable insight: {last.lesson}")
    if adaptive:
        text += f"\nWhy this is new to the playbook: {last.novelty_reason}"
    return text


class CuratorConsolidator:
    """`--consolidator curator` (default). `history` keeps one record per call (usage, outcome, new ids) until
    the runner drains it."""

    def __init__(self, llm, n_tasks: int, adaptive: bool, token_budget: int = TOKEN_BUDGET,
                 max_tokens: int = CURATOR_MAX_TOKENS):
        self.llm, self.n_tasks, self.adaptive = llm, n_tasks, adaptive
        self.token_budget, self.max_tokens = token_budget, max_tokens
        self.calls = self.failures = 0
        self.history: list[dict] = []

    def build_prompt(self, playbook: SectionedPlaybook, episode: EpisodeState, task: dict) -> str:
        return CURATOR_PROMPT.format(token_budget=self.token_budget, current_step=task["task_index"] + 1,
                                     total_samples=self.n_tasks, playbook_stats=json.dumps(playbook.stats(), indent=2),
                                     recent_reflection=lesson_text(episode, self.adaptive),
                                     current_playbook=playbook.text, question_context=task["question"])

    def consolidate(self, playbook: SectionedPlaybook, episode: EpisodeState, task: dict, traj: Trajectory) -> str:
        prompt = self.build_prompt(playbook, episode, task)
        t0 = time.time()
        self.calls += 1
        rec = {"calls": 1, "prompt_tokens": 0, "completion_tokens": 0, "outcome": "", "new_ids": [], "error": ""}
        try:
            text, usage = self.llm(prompt, self.max_tokens)
        except Exception as e:                       # noqa: BLE001 — the playbook stays as it is
            rec.update(outcome="llm_error", error=f"{type(e).__name__}: {e}"[:300])
        else:
            rec.update(prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
                       completion_tokens=int(usage.get("completion_tokens", 0) or 0))
            ops = playbook.parse_curator_response(text)
            if ops is None:
                rec["outcome"] = "parse_error"
            else:
                try:
                    rec["new_ids"] = playbook.apply_add_ops(ops)
                    rec["outcome"] = "added" if rec["new_ids"] else "no_ops"
                except (KeyError, AttributeError, TypeError) as e:
                    rec.update(outcome="apply_error", error=f"{type(e).__name__}: {e}")
        self.failures += int(rec["outcome"].endswith("_error"))
        rec["elapsed_s"] = round(time.time() - t0, 2)
        self.history.append(rec)
        return ",".join(rec["new_ids"])

    def drain(self) -> list[dict]:
        h, self.history = self.history, []
        return h


class AppendConsolidator:
    """`--consolidator append`: no model call; the accepting round's key insight becomes one bullet in OTHERS."""

    def consolidate(self, playbook: SectionedPlaybook, episode: EpisodeState, task: Any, traj: Trajectory) -> str:
        lesson = " ".join(episode.last.lesson.split())
        if not lesson:
            return ""
        return ",".join(playbook.apply_add_ops([{"type": "ADD", "section": "others", "content": lesson}]))
