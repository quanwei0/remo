"""AppWorld consolidators (the paper's `Consolidate`) for the sectioned playbook.

LLMConsolidator (default, what the paper's runs used): the consolidator prompt
(prompts/consolidator/appworld.txt) with the admitted reflection, the whole playbook, the task
instruction and the conversation history of the accepted attempt; the reply's ADD operations are
appended as new bullets. The reflection stored is chosen as in the runs: for round1_clean the whole
raw reflection of round 1; for cross_round_validated the reflection of the round that flipped the
task (the one before the accepting round) behind the "[VALIDATED BY RETRY: ...]" line.

InsightConsolidator (--consolidator append, alternative only): no model call, the critic's
`key_insight` becomes one bullet of the OTHERS section.
"""
from pathlib import Path

from remo.memory import SectionedPlaybook
from remo.policy import EpisodeState

from benchmarks.appworld.solver import read_prompt

ROOT = Path(__file__).resolve().parents[2]
CONSOLIDATOR_PROMPT_PATH = ROOT / "prompts" / "consolidator" / "appworld.txt"
SEE_HISTORY = "See full conversation history below"
VALIDATED_PREFIX = ("[VALIDATED BY RETRY: after applying this reflection in a retry, the "
                    "previously failing attempt succeeded.]\n")


def select_reflection(episode: EpisodeState) -> str:
    """The reflection text an admitted episode consolidates."""
    if len(episode.rounds) == 1:
        return episode.rounds[0].reflection.raw
    return VALIDATED_PREFIX + episode.rounds[-2].reflection.raw


def build_consolidator_input(prompt: str, reflection: str, playbook_text: str, instruction: str, history: str) -> str:
    return prompt.format(initial_generated_code=SEE_HISTORY, final_generated_code=SEE_HISTORY, guidebook=reflection,
                         current_playbook=playbook_text, question_context=instruction, gt=None) + history


class LLMConsolidator:
    """`last_error` is "" after a successful call (even one adding nothing) and names the failure
    otherwise (transport error, unusable reply, unparseable playbook text); the playbook is then
    unchanged and the runner records store_decision "consolidator_error"."""

    def __init__(self, llm, max_tokens: int = 8192, temperature: float = 0.0, log=print):
        self.llm, self.max_tokens, self.temperature, self.log = llm, max_tokens, temperature, log
        self.prompt = read_prompt(CONSOLIDATOR_PROMPT_PATH)
        self.calls = self.failures = 0
        self.last_error = ""

    def consolidate(self, playbook: SectionedPlaybook, episode: EpisodeState, task, traj) -> str:
        self.last_error = ""
        reflection = select_reflection(episode)
        if not reflection:                            # an empty critic reply: nothing to curate, no call
            return ""
        content = build_consolidator_input(self.prompt, reflection, playbook.text, traj.meta.get("instruction", ""), traj.text)
        self.calls += 1
        try:
            reply = self.llm.chat([{"role": "user", "content": content}], self.max_tokens, self.temperature)
            ops = playbook.parse_ops_response(reply)
            if ops is None:
                raise ValueError("unusable consolidator reply: " + (reply or "")[:200])
            return ",".join(playbook.apply_add_ops(ops))
        except Exception as e:                        # noqa: BLE001 — transport, reply or playbook-text (KeyError) failure
            self.failures += 1
            self.last_error = f"{type(e).__name__}: {e}"[:500]
            self.log(f"[consolidator] {self.last_error}; playbook unchanged")
            return ""


class InsightConsolidator:
    last_error = ""

    def consolidate(self, playbook: SectionedPlaybook, episode: EpisodeState, task, traj) -> str:
        lesson = episode.lesson()
        if not lesson:
            return ""
        return ",".join(playbook.apply_add_ops([{"type": "ADD", "section": "others", "content": lesson}]))
