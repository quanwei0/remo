"""FinanceGym adapter — pieces that do NOT need the FinanceHarness import (usable in any env):
benchmark file, question builder, the critic (LLM), the memory consolidator, the pre-submission quality
floor and the episodes.jsonl reader. The solver (harness + PIT backend) lives in solver.py.

FinanceGym has NO local ground truth (organizers grade); the critic judges only internal
consistency and evidence discipline of the analyst's own record. Prompt, injection strings,
call parameters and parsing are those of the paper's runs.
"""
import json
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:                       # works from a checkout without `pip install -e`
    sys.path.insert(0, _REPO)

from remo.critic import extract_json                        # noqa: E402
from remo.interfaces import Reflection, Trajectory          # noqa: E402
from remo.memory import Entry, Playbook, find_cited_ids     # noqa: E402

# The official FinanceHarness (vendored under third_party/; override with FINHARNESS_ROOT).
FH_ROOT = os.path.abspath(os.path.expanduser(os.environ.get("FINHARNESS_ROOT",
                                                            os.path.join(_REPO, "third_party", "finance_harness"))))
BENCH_FILE = os.environ.get("FH_TASKS", os.path.join(FH_ROOT, "FinanceGym", "data", "benchmark_400_public.jsonl"))
MODES = ("baseline", "react", "refine", "memory", "remo", "adaremo")   # CLI arms; see make_config
EMBED_MODEL = "Qwen/Qwen3-Embedding-4B"
DEFAULT_PIT_URL = "http://127.0.0.1:8889"
DEFAULT_EMBED_URL = "http://127.0.0.1:8888/v1/embeddings"
DEFAULT_MODEL = "GPT-OSS-120B"
PLAYBOOK_PREFIX = "fin"
INJECT_CAP_CHARS = 30000            # memory text prepended to the harness question
CRITIC_MEMORY_CAP_CHARS = 20000     # memory text shown to the critic for the novelty check
CRITIC_MAX_TOKENS = 2048            # the critic call sends max_tokens only: no temperature (server default)
EXTRA_ROUNDS_BUDGET = 550           # run-wide cap on retry rounds (never reached in the paper's runs)
MIN_DOCS = 3                        # below this the episode is NOT saved (redone on the next run)
REPORT_MIN_CHARS = 1500             # pre-submission quality floor
REPORT_CAP_CHARS = 12000            # report chars the critic reads
QUERY_CAP = 20                      # queries the critic reads

with open(os.path.join(_REPO, "prompts", "critic", "financegym.txt"), encoding="utf-8", newline="") as _f:
    CRITIC_PROMPT = _f.read()       # one prompt for both arms; ReMo simply ignores refine / store


# -- data -----------------------------------------------------------------------------------------
def load_tasks(path: str = BENCH_FILE, limit: int = 0) -> list[dict]:
    """Benchmark tasks in file order: {"task_id", "question", "cutoff"}. `limit` keeps the first N."""
    tasks = [json.loads(l) for l in open(path) if l.strip()]
    return tasks[:limit] if limit else tasks


def read_episodes(run_dir: str) -> list[dict]:
    p = os.path.join(run_dir, "episodes.jsonl")
    if not os.path.exists(p):
        return []
    return [json.loads(l) for l in open(p) if l.strip()]


# -- what the solver is asked ---------------------------------------------------------------------
PLAYBOOK_HEADER = "Analyst playbook — lessons from prior research tasks; apply when relevant:"
RETRY_HEADER = ("A reviewer found these issues in a previous attempt — run a fresh, better investigation "
                "that fixes them:")


def build_question(task: dict, memory_text: str, critique: str | None, plain: bool = False) -> str:
    """Playbook (if any) prepended, then the research question + PIT constraint, then (retry rounds:
    `critique` is not None) the previous critique. `plain=True` is the official-harness baseline: the
    question and the PIT sentence only, no header, no memory, no critique (the leaderboard entry)."""
    base_q = (f"{task['question']}\n\n(Point-in-time constraint: use only information published on or "
              f"before {task['cutoff']}. The search environment enforces this cutoff.)")
    if plain:
        return base_q
    q = (f"{PLAYBOOK_HEADER}\n{memory_text}\n\nResearch question: {base_q}" if memory_text
         else f"Research question: {base_q}")
    if critique is not None:
        q += f"\n\n{RETRY_HEADER}\n{critique[:3000]}"
    return q


# -- record-level view: report + queries + doc/citation counts (no messages, no tool log) ----------
def make_trajectory(report: str, queries: list, docs_retrieved: int, citations: list, min_docs: int = MIN_DOCS,
                    **meta) -> Trajectory:
    """completed = report non-empty AND docs_retrieved >= min_docs (the harness leaves the report empty
    on max_rounds / timeout / error)."""
    report = report or ""
    return Trajectory(text=report, answer=report, completed=bool(report.strip()) and docs_retrieved >= min_docs,
                      meta={"docs_retrieved": docs_retrieved, "queries": list(queries or []),
                            "citations": list(citations or []), **meta})


def trajectory_from_record(rec: dict, min_docs: int = MIN_DOCS) -> Trajectory:
    """Rebuild a Trajectory from a saved record (an episodes.jsonl line, or a per-task JSON "record"
    with final_answer/report, queries, docs_retrieved, citations): used by calibrate.py to run the critic
    without solver rollouts."""
    return make_trajectory(rec.get("final_answer", rec.get("report", "")), rec.get("queries"),
                           int(rec.get("docs_retrieved", 0) or 0), rec.get("citations"), min_docs,
                           steps=rec.get("steps", 0), termination=rec.get("termination"))


# -- pre-submission quality floor (report < 1500 chars or starts with a JSON brace) ----------------
def is_defective(report: str, min_chars: int = REPORT_MIN_CHARS) -> str:
    """Returns "" if the report passes, else the reason (`empty`, `short`, `json_fragment`)."""
    r = (report or "").strip()
    if not r:
        return "empty"
    if r.startswith("{"):
        return "json_fragment"          # gpt-oss occasionally emits a tool-call JSON fragment as the final
    if len(r) < min_chars:
        return "short"
    return ""


# -- critic -----------------------------------------------------------------------------------------
def build_critic_prompt(task: dict, traj: Trajectory, memory_text: str) -> str:
    """The prompt of the paper's runs: question, cutoff, the analyst's queries (first 20, JSON), doc and
    citation counts, the report (first 12000 chars) and the memory text (first 20000 chars) as the prior
    playbook for the novelty check."""
    queries, citations = traj.meta.get("queries") or [], traj.meta.get("citations") or []
    return CRITIC_PROMPT.format(
        q=task["question"], cutoff=task["cutoff"], nq=len(queries), queries=json.dumps(queries[:QUERY_CAP]),
        ndocs=traj.meta.get("docs_retrieved", 0), ncit=len(citations),
        report=traj.answer[:REPORT_CAP_CHARS], playbook=memory_text[:CRITIC_MEMORY_CAP_CHARS] or "(empty)")


def parse_critic_reply(text: str) -> Reflection:
    """Reply -> Reflection exactly as the paper's runs read it: the greedy {...} block parsed as JSON;
    verdict defaults to no_errors when the key is missing, refine / store through bool(), the text fields
    through str(); `confidence` is read (float) and ignored. Anything unparseable (no JSON, invalid JSON,
    a non-numeric confidence) is accepted without a lesson: no_errors, refine=False, store=False, parsed=False."""
    d = extract_json(text)
    try:
        if not d:
            raise ValueError("no JSON object")
        float(d.get("confidence", 0.5))
    except (ValueError, TypeError) as e:
        return Reflection(verdict="correct", critique=f"(critic reply unparseable: {e})", refine=False, store=False,
                          parsed=False, raw=text)
    return Reflection(verdict="correct" if d.get("verdict", "no_errors") == "no_errors" else "incorrect",
                      critique=str(d.get("critique", "")), lesson=str(d.get("lesson", "")),
                      refine=bool(d.get("refine", False)), store=bool(d.get("store", False)),
                      novelty_reason=str(d.get("novelty_reason", "")), cited_id=str(d.get("cited_id", "")),
                      parsed=True, raw=text)


def cited_entry(refl: Reflection) -> str:
    """The ONE entry id a reflection cites: `cited_id` without brackets, else the first id mentioned in
    `novelty_reason` (the critic often writes the id there), else ""."""
    cited = refl.cited_id.strip("[] ")
    if cited:
        return cited
    ids = find_cited_ids(refl.novelty_reason)
    return ids[0] if ids else ""


class VerbatimConsolidator:
    """Stores the episode's lesson as the paper's runs wrote it: `lesson.strip()` byte for byte (Playbook.add
    would collapse internal whitespace, e.g. the narrow no-break space of "Form 4"). No model call."""

    def consolidate(self, playbook: Playbook, episode, task, traj) -> str:
        lesson = episode.lesson()
        if not lesson:
            return ""
        eid = f"{playbook.prefix}-{playbook._next:05d}"
        playbook._next += 1
        playbook.entries.append(Entry(eid, lesson))
        return eid


class FinanceGymCritic:
    """One chat call per round (independent of earlier rounds), async over one shared openai.AsyncOpenAI
    client (creating a client per call under concurrency crashed the process in ssl.SSLContext.__new__).
    Sent: model, max_tokens, one user message — no temperature, so the server default applies, as in the
    paper's runs. Only a failure of the call itself is a failed Reflection (the episode stops, not admitted);
    an unparseable reply is accepted without a lesson (see parse_critic_reply)."""

    def __init__(self, client, model: str, max_tokens: int = CRITIC_MAX_TOKENS):
        self.client, self.model, self.max_tokens = client, model, max_tokens
        self.calls = self.parse_failures = self.call_failures = 0

    def build_prompt(self, task, traj: Trajectory, memory_text: str) -> str:
        return build_critic_prompt(task, traj, memory_text)

    async def reflect(self, task, traj: Trajectory, memory_text: str, prior_critique: str | None,
                      round_idx: int, K: int) -> Reflection:
        prompt = self.build_prompt(task, traj, memory_text)
        self.calls += 1
        try:
            r = await self.client.chat.completions.create(model=self.model, max_tokens=self.max_tokens,
                                                          messages=[{"role": "user", "content": prompt}])
            txt = r.choices[0].message.content or ""
        except Exception as e:                      # noqa: BLE001 — transport / server failure
            self.call_failures += 1
            return Reflection(verdict="none", critique=f"(critic call failed: {type(e).__name__}: {e})"[:500],
                              refine=False, store=False, parsed=False, raw="", failed=True)
        refl = parse_critic_reply(txt)
        self.parse_failures += int(not refl.parsed)
        return refl


# -- arms -> RemoConfig -------------------------------------------------------------------------------
def make_config(mode: str, K: int | None, **adaremo_knobs):
    """CLI arm -> (RemoConfig, baseline). baseline = the official harness alone (one attempt, no critic,
    no memory); react = K=1 no memory; refine = K>1 no memory; memory = K=1 with memory; remo / adaremo =
    Algorithms 1 / 2. K defaults to 1 for the single-attempt arms and 3 otherwise."""
    from remo import RemoConfig
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; choose from {MODES}")
    single = mode in ("baseline", "react", "memory")
    K = (1 if single else 3) if K is None else K
    if single and K != 1:
        raise ValueError(f"--mode {mode} is a single-attempt arm (K=1); got --K {K}")
    if mode == "refine" and K < 2:
        raise ValueError("--mode refine needs --K >= 2 (K=1 without memory is --mode react)")
    use_memory = mode in ("memory", "remo", "adaremo")
    cfg = RemoConfig(mode="adaremo" if mode == "adaremo" else "remo", K=K, use_memory=use_memory, **adaremo_knobs)
    return cfg, mode == "baseline"


# -- post-hoc summary (final_results.json) -------------------------------------------------------------
def count_tokens(text: str) -> int | None:
    """tiktoken cl100k_base token count (None if tiktoken is unavailable)."""
    try:
        import tiktoken
        return len(tiktoken.get_encoding("cl100k_base").encode(text or ""))
    except Exception:                                       # noqa: BLE001 — optional dependency / no network
        return None


def summarize_run(run_dir: str, tasks_path: str = BENCH_FILE, min_chars: int = REPORT_MIN_CHARS) -> dict:
    """Everything final_results.json needs, computed from the run dir only (episodes.jsonl, playbook.txt,
    policy_state.json). FinanceGym has no local ground truth, so `accuracy` is None; the round-1 / final
    "clean" rates are the CRITIC's verdicts, not correctness."""
    eps = read_episodes(run_dir)
    n = len(eps)
    tasks = load_tasks(tasks_path) if os.path.exists(tasks_path) else []
    gates, decisions, verdict_rounds = {}, {}, 0
    round1_clean = final_clean = unparsed_rounds = 0
    for e in eps:
        gates[e["gate"]] = gates.get(e["gate"], 0) + 1
        decisions[e["store_decision"]] = decisions.get(e["store_decision"], 0) + 1
        r1 = (e.get("rounds") or [{}])[0]
        if r1.get("verdict") in ("correct", "incorrect"):
            verdict_rounds += 1
            round1_clean += int(r1.get("verdict") == "correct" and bool(r1.get("completed")))
        final_clean += int(e["gate"] in ("round1_clean", "cross_round_validated"))
        unparsed_rounds += sum(1 for r in e.get("rounds") or [] if r.get("verdict") != "none" and not r.get("parsed"))
    critic_errors = sum(e.get("stop_reason") == "critic_error" for e in eps)
    pb = Playbook.load(os.path.join(run_dir, "playbook.txt"), prefix=PLAYBOOK_PREFIX)
    ps_path = os.path.join(run_dir, "policy_state.json")
    policy = json.load(open(ps_path)) if os.path.exists(ps_path) else {}
    cfg_path = os.path.join(run_dir, "run_config.json")
    cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
    mean = lambda xs: round(sum(xs) / len(xs), 3) if xs else None   # noqa: E731
    defective = [e["task_id"] for e in eps if is_defective(e.get("final_answer", ""), min_chars)]
    return {
        "benchmark": "financegym", "mode": cfg.get("mode"), "K": cfg.get("K"),
        "freeze_after": cfg.get("freeze_after"),
        "n_tasks": len(tasks) or None, "n_saved": n,
        "accuracy": None,
        "scoring": "no local ground truth: answers.jsonl is scored by the FinanceGym organizers",
        "round1_clean_rate": round(round1_clean / verdict_rounds, 4) if verdict_rounds else None,
        "final_clean_rate": round(final_clean / n, 4) if n and verdict_rounds else None,
        "gate_distribution": gates, "store_decisions": decisions,
        "critic_error_episodes": critic_errors,          # the critic call failed: episode stopped, not admitted
        "critic_unparsed_rounds": unparsed_rounds,       # accepted without a lesson (the paper's runs did the same)
        "mean_rounds": mean([len(e.get("rounds") or []) for e in eps]),
        "extra_rounds_used": policy.get("extra_rounds_used"),
        "memory_entries": len(pb), "memory_chars": pb.chars(), "memory_tokens_cl100k": count_tokens(pb.render()),
        "frozen": policy.get("frozen"), "freeze_events": policy.get("freeze_events", []),
        "mean_docs_retrieved": mean([e.get("docs_retrieved", 0) for e in eps]),
        "mean_report_chars": mean([len(e.get("final_answer") or "") for e in eps]),
        "mean_elapsed_s": mean([e.get("elapsed_s", 0) for e in eps]),
        "defective_reports": defective,
    }
