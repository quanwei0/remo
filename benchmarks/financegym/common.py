"""FinanceGym adapter — pieces that do NOT need the FinanceHarness import (usable in any env):
benchmark file, record-level view of a research run, the critic (LLM), the pre-submission
quality floor and the episodes.jsonl reader. The solver (harness + PIT backend) lives in solver.py.

FinanceGym has NO local ground truth (organizers grade); the critic judges only internal
consistency and evidence discipline of the analyst's own record.
"""
import json
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:                       # works from a checkout without `pip install -e`
    sys.path.insert(0, _REPO)

from remo.critic import fields_spec, parse_reflection      # noqa: E402
from remo.interfaces import Reflection, Trajectory          # noqa: E402

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
INJECT_CAP_CHARS = 30000            # memory text prepended to the harness question (the paper's runs)
CRITIC_MEMORY_CAP_CHARS = 20000     # memory text shown to the critic for the novelty check
MIN_DOCS = 3                        # below this the episode is NOT saved (redone on the next run)
REPORT_MIN_CHARS = 1500             # pre-submission quality floor
REPORT_CAP_CHARS = 12000            # report chars the critic reads
QUERY_CAP = 20                      # queries the critic reads


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
    """Playbook (if any) prepended, then the research question + PIT constraint, then (retry) the
    critique. `plain=True` is the official-harness baseline: the question and the PIT sentence only,
    no header, no memory, no critique (the format of the leaderboard entry)."""
    base_q = (f"{task['question']}\n\n(Point-in-time constraint: use only information published on or "
              f"before {task['cutoff']}. The search environment enforces this cutoff.)")
    if plain:
        return base_q
    q = (f"{PLAYBOOK_HEADER}\n{memory_text}\n\nResearch question: {base_q}" if memory_text.strip()
         else f"Research question: {base_q}")
    if critique and critique.strip():
        q += f"\n\n{RETRY_HEADER}\n{critique.strip()[:3000]}"
    return q


# -- record-level view (what the critic reads; no messages, no tool log) --------------------------
def record_view(report: str, queries: list[str], docs_retrieved: int, citations: list,
                report_cap: int = REPORT_CAP_CHARS, query_cap: int = QUERY_CAP) -> str:
    qs = [q for q in (queries or []) if isinstance(q, str)]
    rep = report or ""
    body = rep[:report_cap] + ("\n[... report truncated for review ...]" if len(rep) > report_cap else "")
    return "\n".join([
        f"Analyst's search queries ({len(qs)}): {json.dumps(qs[:query_cap], ensure_ascii=False)}",
        f"Documents actually fetched: {docs_retrieved} | Citations listed: {len(citations or [])}",
        "",
        "=== REPORT ===",
        body,
        "=== END REPORT ===",
    ])


def trajectory_from_record(rec: dict, min_docs: int = MIN_DOCS) -> Trajectory:
    """Rebuild a Trajectory from a saved record (an episodes.jsonl line, or a per-task JSON "record"
    with final_answer/report, queries, docs_retrieved, citations): used by calibrate.py to run the critic
    without solver rollouts."""
    report = rec.get("final_answer", rec.get("report", "")) or ""
    queries, cits = rec.get("queries") or [], rec.get("citations") or []
    docs = int(rec.get("docs_retrieved", 0) or 0)
    return Trajectory(text=record_view(report, queries, docs, cits), answer=report,
                      completed=bool(report.strip()) and docs >= min_docs,
                      meta={"docs_retrieved": docs, "steps": rec.get("steps", 0),
                            "termination": rec.get("termination"), "queries": queries, "citations": cits})


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
# Domain part of the critic prompt (five severe-only checks; the same text the paper's runs used). The
# decision fields are appended from remo.critic.fields_spec so the policy always gets the same Reflection.
CRITIC_TEMPLATE = """You are a rigorous financial research reviewer. Review the report against the evidence the analyst actually gathered. There is NO ground truth — judge only internal consistency and evidence discipline.

Research question: {q}
Point-in-time cutoff: {cutoff} (nothing published after this date may be used)

{record_view}

A problem counts as an error ONLY if it is severe enough to make the report's MAIN conclusion unreliable:
1. A key number or quote that the conclusion rests on, attributable to no fetched document (spot-fabrication).
2. Point-in-time leakage in a load-bearing claim: facts only knowable after {cutoff}, presented as fact.
3. The report answers a materially different question than the one asked.
4. The main conclusion contradicts the report's own cited evidence.
5. The evidence base is essentially absent: almost nothing was fetched for a question that requires sources.
Minor imperfections are NOT errors — a few uncited peripheral figures, limited depth, style issues, or clearly-labeled forward-looking analysis. If the main conclusion is adequately supported, the verdict MUST be "no_errors". Expect most professionally-done reports to pass.
{prior_block}
The lesson must be ONE imperative, generalizable sentence (query strategy, source triage, report discipline) that would help on OTHER research questions — never a task-specific fact. If errors_found, the critique must name the exact claim/number and what to do about it (e.g. the missing search query).

"""

PRIOR_TEMPLATE = """
This is attempt {round_idx} of {K}. A reviewer of the previous attempt reported:
{prior_critique}
Judge THIS attempt on its own evidence; say whether those issues were fixed.
"""

MEMORY_TEMPLATE = """

CURRENT MEMORY (analyst playbook entries; used for the novelty check only):
{memory}

Reply with ONLY the JSON object."""


def _cap_lines(text: str, cap: int) -> str:
    """Keep whole lines up to `cap` chars (the render is already ranked most-helpful first)."""
    out, total = [], 0
    for line in (text or "").splitlines():
        if total + len(line) + 1 > cap:
            break
        out.append(line); total += len(line) + 1
    return "\n".join(out)


def build_critic_prompt(task: dict, traj: Trajectory, memory_text: str, prior_critique: str | None,
                        round_idx: int, K: int, adaptive: bool,
                        memory_cap: int = CRITIC_MEMORY_CAP_CHARS) -> str:
    template = CRITIC_TEMPLATE + fields_spec(adaptive)
    if adaptive:
        template += MEMORY_TEMPLATE
    prior = ""
    if prior_critique and prior_critique.strip():
        prior = PRIOR_TEMPLATE.format(round_idx=round_idx, K=K, prior_critique=prior_critique.strip()[:3000])
    mem = _cap_lines(memory_text, memory_cap) or "(empty)"
    # .format once on the template only: the report/queries/memory are VALUES, so their braces are safe
    return template.format(q=task["question"], cutoff=task["cutoff"], record_view=traj.text,
                           prior_block=prior, memory=mem)


CRITIC_FAILED_PREFIX = "(critic call failed"     # critique of the fallback Reflection when the model call itself failed


def is_critic_failure(refl: Reflection) -> bool:
    """True for the fallback FinanceGymCritic.reflect returns when the critic CALL failed (transport / server
    error): no verdict was produced. An unparseable reply is not a failure (it carries `raw` and is parsed
    conservatively by remo.critic.parse_reflection)."""
    return (not refl.parsed) and (not refl.raw) and refl.critique.startswith(CRITIC_FAILED_PREFIX)


class FinanceGymCritic:
    """LLM critic over the record-level view. Async (one shared openai.AsyncOpenAI client — creating a
    client per call under concurrency crashed the process in ssl.SSLContext.__new__). On unparseable
    output the call is retried `retries` times; on a transport failure the verdict is no_errors with
    parsed=False, no lesson and the CRITIC_FAILED_PREFIX critique, so the loop stops without burning
    solver rounds; the driver recognises it (is_critic_failure) and skips the memory write, because an
    acceptance produced by an outage verifies nothing."""

    def __init__(self, client, model: str, adaptive: bool, max_tokens: int = 2048,
                 temperature: float = 0.0, memory_cap: int = CRITIC_MEMORY_CAP_CHARS, retries: int = 1):
        self.client, self.model, self.adaptive = client, model, adaptive
        self.max_tokens, self.temperature, self.memory_cap, self.retries = max_tokens, temperature, memory_cap, retries
        self.calls = self.parse_failures = self.call_failures = 0

    def build_prompt(self, task, traj, memory_text, prior_critique, round_idx, K) -> str:
        return build_critic_prompt(task, traj, memory_text, prior_critique, round_idx, K,
                                   self.adaptive, self.memory_cap)

    async def reflect(self, task, traj: Trajectory, memory_text: str, prior_critique: str | None,
                      round_idx: int, K: int) -> Reflection:
        prompt = self.build_prompt(task, traj, memory_text, prior_critique, round_idx, K)
        refl = None
        for _ in range(self.retries + 1):
            self.calls += 1
            try:
                r = await self.client.chat.completions.create(
                    model=self.model, max_tokens=self.max_tokens, temperature=self.temperature,
                    messages=[{"role": "user", "content": prompt}])
                txt = r.choices[0].message.content or ""
            except Exception as e:                      # transport / server failure
                self.call_failures += 1
                return Reflection(verdict="correct", critique=f"{CRITIC_FAILED_PREFIX}: {type(e).__name__}: {e})"[:500],
                                  refine=False, store=False, parsed=False, raw="")
            refl = parse_reflection(txt, self.adaptive)
            if refl.parsed:
                return refl
        self.parse_failures += 1
        return refl


# -- optional LLM consolidator (--consolidator llm) -------------------------------------------------
CONSOLIDATE_TEMPLATE = """A reviewer distilled this lesson from one finance research task:
{lesson}

Rewrite it as ONE general playbook line for future research questions about other companies, markets and dates:
a single imperative sentence of at most 40 words that keeps only the transferable rule (query strategy, source
triage, evidence checks, report discipline) and drops every task-specific name, number and date.
Output the sentence only."""


class LLMConsolidator:
    """One chat call turns the critic's lesson into one general entry line (`condense`, async, run by the
    driver OUTSIDE its lock). The sync `consolidate` of the Consolidator protocol appends as-is (used only
    if a caller cannot await). On any failure the raw lesson is stored, so a model outage never loses an
    admitted lesson."""

    def __init__(self, client, model: str, max_tokens: int = 256, temperature: float = 0.0):
        self.client, self.model, self.max_tokens, self.temperature = client, model, max_tokens, temperature
        self.calls = self.failures = 0

    @staticmethod
    def _clean(text: str, fallback: str) -> str:
        for line in (text or "").splitlines():
            line = line.strip().lstrip("-*•1234567890. ").strip().strip('"\'')
            if line:
                return " ".join(line.split())
        return fallback

    async def condense(self, lesson: str, task, traj) -> str:
        self.calls += 1
        try:
            r = await self.client.chat.completions.create(
                model=self.model, max_tokens=self.max_tokens, temperature=self.temperature,
                messages=[{"role": "user", "content": CONSOLIDATE_TEMPLATE.format(lesson=lesson.strip())}])
            return self._clean(r.choices[0].message.content or "", lesson.strip())
        except Exception:                                   # noqa: BLE001 — store the raw lesson instead
            self.failures += 1
            return lesson.strip()

    def consolidate(self, playbook, lesson: str, task, traj) -> str:
        return playbook.add(lesson)


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
    from remo import Playbook
    eps = read_episodes(run_dir)
    n = len(eps)
    tasks = load_tasks(tasks_path) if os.path.exists(tasks_path) else []
    gates, decisions, verdict_rounds = {}, {}, 0
    round1_clean = final_clean = 0
    for e in eps:
        gates[e["gate"]] = gates.get(e["gate"], 0) + 1
        decisions[e["store_decision"]] = decisions.get(e["store_decision"], 0) + 1
        r1 = (e.get("rounds") or [{}])[0]
        if r1.get("verdict") in ("correct", "incorrect"):
            verdict_rounds += 1
            round1_clean += int(r1.get("verdict") == "correct" and bool(r1.get("completed")))
        final_clean += int(e["gate"] in ("round1_clean", "cross_round_validated"))
    critic_failed = sum(bool(e.get("critic_failed")) for e in eps)
    pb = Playbook.load(os.path.join(run_dir, "playbook.txt"), prefix=PLAYBOOK_PREFIX)
    ps_path = os.path.join(run_dir, "policy_state.json")
    policy = json.load(open(ps_path)) if os.path.exists(ps_path) else {}
    cfg_path = os.path.join(run_dir, "run_config.json")
    cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
    mean = lambda xs: round(sum(xs) / len(xs), 3) if xs else None   # noqa: E731
    defective = [e["task_id"] for e in eps if is_defective(e.get("final_answer", ""), min_chars)]
    return {
        "benchmark": "financegym", "mode": cfg.get("mode"), "K": cfg.get("K"),
        "freeze_after": cfg.get("freeze_after"), "consolidator": cfg.get("consolidator"),
        "n_tasks": len(tasks) or None, "n_saved": n,
        "accuracy": None,
        "scoring": "no local ground truth: answers.jsonl is scored by the FinanceGym organizers",
        "round1_clean_rate": round(round1_clean / verdict_rounds, 4) if verdict_rounds else None,
        "final_clean_rate": round(final_clean / n, 4) if n and verdict_rounds else None,
        "gate_distribution": gates, "store_decisions": decisions,
        "critic_failed_episodes": critic_failed,     # accepted on a failed critic call: no verdict, no memory write
        "mean_rounds": mean([len(e.get("rounds") or []) for e in eps]),   # failed rounds count too
        "memory_entries": len(pb), "memory_chars": pb.chars(), "memory_tokens_cl100k": count_tokens(pb.render()),
        "frozen": policy.get("frozen"), "freeze_events": policy.get("freeze_events", []),
        "mean_docs_retrieved": mean([e.get("docs_retrieved", 0) for e in eps]),
        "mean_report_chars": mean([len(e.get("final_answer") or "") for e in eps]),
        "mean_elapsed_s": mean([e.get("elapsed_s", 0) for e in eps]),
        "defective_reports": defective,
    }
