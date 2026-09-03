"""ReMo / AdaReMo (and the baseline arms) on FinanceGym: 400 open-ended point-in-time finance research
questions, no local ground truth — the organizers grade the submitted reports.

Arms (--mode): baseline = the official FinanceHarness alone (one attempt, no critic, no memory — the
leaderboard entry), react (K=1, no memory), refine (K>1, no memory), memory (K=1, memory), remo
(Algorithm 1), adaremo (Algorithm 2). --freeze-after A consolidates on the first A tasks and runs the
rest with the memory read-only (they wait until the A learning tasks have finished).

The FinanceHarness solver is async and 8 tasks run concurrently, while remo.ReMoAgent.run_task is
synchronous. So this driver replays ReMoAgent.run_task over remo.RemoPolicy step by step: solve/critic
run concurrently per task (asyncio, semaphore --conc), and every shared-state step — policy.after_round
/ gate / memory_decision, playbook add/reinforce, the retry-round budget, persistence — happens under
ONE asyncio.Lock. Where the paper's runs differ from the core defaults the driver follows the runs:
one cited entry is reinforced, memory bookkeeping only happens when the episode carries a lesson, an
empty report is never retried, retry rounds draw on one run-wide budget, and the submitted report is
the last round's unless it is empty (then the first round's).

Run dir (resumable): episodes.jsonl (one line per finished task; tasks present are skipped),
playbook.txt, policy_state.json, trajs/<task_id>.json (every round's report/queries/critic raw),
run_config.json, answers.jsonl (submission file, rebuilt at the end; see make_answers.py) and
final_results.json (post hoc: gate distribution, store decisions, mean rounds, memory size, ...).

Episodes whose final report has fewer than --min-docs fetched documents are NOT saved (and write no
memory), so a later run redoes them (when the embedding service dies mid-run every task returns 0 docs).

Run from the harness root (its configs/*.json are resolved from there), env `remo-financegym`:
  cd third_party/finance_harness
  export FH_VLLM_BASE_URL=$URL FH_VLLM_READER_BASE_URL=$URL FH_PIT_URL=http://PIT:8889 \\
         FH_EMBED_URL=http://EMBED:8888/v1/embeddings
  python ../../benchmarks/financegym/run_financegym.py --mode adaremo --K 3 --out ../../runs/financegym/adaremo
"""
import argparse
import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import asdict

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from benchmarks.financegym.common import (BENCH_FILE, CRITIC_MAX_TOKENS, DEFAULT_EMBED_URL, DEFAULT_MODEL,   # noqa: E402
                                          DEFAULT_PIT_URL, EMBED_MODEL, EXTRA_ROUNDS_BUDGET, INJECT_CAP_CHARS,
                                          MIN_DOCS, MODES, PLAYBOOK_PREFIX, FinanceGymCritic, VerbatimConsolidator,
                                          cited_entry, load_tasks, make_config, read_episodes, summarize_run)
from remo import EpisodeState, Playbook, Reflection, RemoConfig, RemoPolicy   # noqa: E402


def _log(msg: str) -> None:
    print(time.strftime("%Y-%m-%d %H:%M:%S"), msg, flush=True)


def no_critic_reflection() -> Reflection:
    """--mode baseline: the official harness alone. No critic call; the record carries a placeholder."""
    return Reflection(verdict="none", critique="", lesson="", refine=False, store=False, parsed=False, raw="")


class RunState:
    """Shared, lock-protected state of one run directory."""

    def __init__(self, cfg: RemoConfig, run_dir: str, freeze_after: int | None = None,
                 extra_rounds_budget: int = EXTRA_ROUNDS_BUDGET):
        self.cfg, self.run_dir, self.freeze_after = cfg, run_dir, freeze_after
        self.extra_rounds_budget = extra_rounds_budget
        os.makedirs(os.path.join(run_dir, "trajs"), exist_ok=True)
        self.playbook = Playbook.load(os.path.join(run_dir, "playbook.txt"), prefix=PLAYBOOK_PREFIX)
        self.policy = RemoPolicy(cfg)
        self.extra_rounds_used = 0
        ps = os.path.join(run_dir, "policy_state.json")
        if os.path.exists(ps):
            with open(ps) as f:
                state = json.load(f)
            self.policy.load_state(state)
            self.extra_rounds_used = int(state.get("extra_rounds_used", 0))
        self.consolidator = VerbatimConsolidator()
        self.lock = asyncio.Lock()
        self.done_ids = {e["task_id"] for e in read_episodes(run_dir)}
        self.saved = self.unsaved = 0
        self.learn_pending, self.learn_done = 0, None     # learn-then-freeze barrier (see arm_barrier)

    # -- learn-then-freeze (--freeze-after A) -------------------------------------------------------
    def readonly(self, task_index: int) -> bool:
        return self.freeze_after is not None and task_index >= self.freeze_after

    def arm_barrier(self, todo_indices) -> None:
        """Read-only tasks wait until every learning task (index < A) of this invocation has finished,
        so they all see the same, complete memory."""
        self.learn_done = asyncio.Event()
        self.learn_pending = sum(1 for i in todo_indices if not self.readonly(i)) if self.freeze_after is not None else 0
        if self.learn_pending == 0:
            self.learn_done.set()

    def task_finished(self, task_index: int) -> None:
        if self.freeze_after is not None and not self.readonly(task_index):
            self.learn_pending -= 1
            if self.learn_pending <= 0 and self.learn_done is not None:
                self.learn_done.set()

    def take_extra_round(self) -> bool:
        """Charges one retry round to the run-wide budget; False when it is exhausted."""
        if self.extra_rounds_used >= self.extra_rounds_budget:
            return False
        self.extra_rounds_used += 1
        return True

    def persist(self, rec: dict, rounds_full: list[dict]) -> None:
        with open(os.path.join(self.run_dir, "episodes.jsonl"), "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        self.playbook.save(os.path.join(self.run_dir, "playbook.txt"))
        with open(os.path.join(self.run_dir, "policy_state.json"), "w") as f:
            json.dump({**self.policy.state(), "extra_rounds_used": self.extra_rounds_used}, f)
        with open(os.path.join(self.run_dir, "trajs", f"{rec['task_id']}.json"), "w") as f:
            json.dump({"task_id": rec["task_id"], "rounds": rounds_full}, f, default=str)


async def run_one(task: dict, task_index: int, rs: RunState, solver, critic, sem: asyncio.Semaphore,
                  min_docs: int, log=_log) -> dict | None:
    """Algorithm 1/2 for one task (mirrors remo.agent.ReMoAgent.run_task). `critic=None` is the baseline
    arm (no critic call). Returns the saved record, or None when the episode is not saved (final report
    below the doc floor)."""
    try:
        return await _run_one(task, task_index, rs, solver, critic, sem, min_docs, log)
    finally:
        rs.task_finished(task_index)


async def _run_one(task, task_index, rs, solver, critic, sem, min_docs, log):
    cfg, tid = rs.cfg, task["task_id"]
    readonly = rs.readonly(task_index)
    if readonly and rs.learn_done is not None:
        await rs.learn_done.wait()
    async with sem:
        async with rs.lock:
            memory_text = rs.playbook.render(cfg.inject_cap_chars) if cfg.use_memory else ""
        st, critique, trajs, rounds_full = EpisodeState(), None, [], []
        for r in range(1, cfg.K + 1):
            traj = await solver.solve(task, memory_text, critique)
            refl = (await critic.reflect(task, traj, memory_text, critique, r, cfg.K)) if critic is not None \
                else no_critic_reflection()
            trajs.append(traj)
            rounds_full.append({"round": r, "report": traj.answer, **{k: v for k, v in traj.meta.items()
                                                                     if k not in ("task_id", "cutoff")},
                                "critic_raw": refl.raw})
            async with rs.lock:
                action = rs.policy.after_round(st, traj.failed, traj.completed, refl)
                if action == "retry" and not traj.answer.strip():      # the runs never retried an empty report
                    action, st.stop_reason = "stop", "empty_report"
                elif action == "retry" and not rs.take_extra_round():
                    action, st.stop_reason = "stop", "round_budget"
            log(f"[round] {tid} r{r} verdict={refl.verdict} completed={traj.completed} "
                f"docs={traj.meta.get('docs_retrieved')} steps={traj.meta.get('steps')} "
                f"report={len(traj.answer)}c refine={refl.refine} store={refl.store} -> {action}")
            if action != "retry":
                break
            critique = refl.critique
        # submitted report: the last round's, unless it is empty (then the first round's)
        final = trajs[-1] if trajs[-1].answer.strip() else trajs[0]
        final_round = trajs.index(final) + 1
        if final.meta.get("docs_retrieved", 0) < min_docs:
            log(f"[min-docs] {tid} docs={final.meta.get('docs_retrieved')} term={final.meta.get('termination')} "
                f"— not saved, will rerun")
            rs.unsaved += 1
            return None
        async with rs.lock:
            gate = "no_critic" if critic is None else rs.policy.gate(st)
            cited = cited_entry(st.last)
            cited_present = [cited] if rs.playbook.get(cited) else []
            if not cfg.use_memory:
                decision = "no_memory"
            elif readonly:
                decision = "readonly"                        # learn-then-freeze: no write, no saturation bookkeeping
            elif not st.lesson():
                decision = "skipped"                         # no lesson: no write, no reinforcement, no bookkeeping
            else:
                decision = rs.policy.memory_decision(st, task_index, cited_present)
            entry_id = ""
            if decision == "stored":
                entry_id = rs.consolidator.consolidate(rs.playbook, st, task, final)
            elif decision == "reinforced":
                rs.playbook.reinforce(cited)
                entry_id = cited
            rec = {"task_index": task_index, "task_id": tid, "question": task["question"], "cutoff": task["cutoff"],
                   "mode": cfg.mode, "K": cfg.K, "use_memory": cfg.use_memory, "readonly_memory": readonly,
                   "gate": gate, "stop_reason": "no_critic" if critic is None else st.stop_reason,
                   "rounds": [{"round": x.round, "completed": x.completed, **asdict(x.reflection),
                               "solver": {k: trajs[i].meta.get(k) for k in
                                          ("elapsed_s", "docs_retrieved", "steps", "termination", "empty_attempts",
                                           "report_chars")}}
                              for i, x in enumerate(st.rounds)],
                   "store_decision": decision, "entry_id": entry_id, "frozen": rs.policy.frozen,
                   "memory_chars_at_start": len(memory_text), "playbook_entries": len(rs.playbook),
                   "final_answer": final.answer, "final_completed": final.completed, "final_round": final_round,
                   "elapsed_s": round(sum(t.meta.get("elapsed_s", 0) for t in trajs), 1),
                   "docs_retrieved": final.meta.get("docs_retrieved", 0), "steps": final.meta.get("steps", 0),
                   "termination": final.meta.get("termination"), "citations": final.meta.get("citations", []),
                   "queries": final.meta.get("queries", []), "backend": "financegym-pit"}
            rs.persist(rec, rounds_full)
            rs.saved += 1
            rs.done_ids.add(tid)
        log(f"[done] {tid} rounds={len(st.rounds)} gate={gate} store={decision}{('=' + entry_id) if entry_id else ''} "
            f"docs={rec['docs_retrieved']} report={len(final.answer)}c pb={len(rs.playbook)}e/{rs.playbook.chars()}c "
            f"frozen={rs.policy.frozen} extra_rounds={rs.extra_rounds_used} {rec['elapsed_s']}s")
        return rec


async def run_all(tasks: list[dict], rs: RunState, solver, critic, conc: int, min_docs: int, log=_log) -> dict:
    sem = asyncio.Semaphore(conc)
    todo = [(i, t) for i, t in enumerate(tasks) if t["task_id"] not in rs.done_ids]
    rs.arm_barrier([i for i, _ in todo])
    log(f"[run] {len(todo)} to do / {len(tasks)} listed ({len(tasks) - len(todo)} already in episodes.jsonl); "
        f"mode={rs.cfg.mode} K={rs.cfg.K} memory={rs.cfg.use_memory} critic={critic is not None} "
        f"freeze_after={rs.freeze_after} conc={conc} min_docs={min_docs} playbook={len(rs.playbook)} entries "
        f"extra_rounds={rs.extra_rounds_used}/{rs.extra_rounds_budget}")
    recs = await asyncio.gather(*[run_one(t, i, rs, solver, critic, sem, min_docs, log) for i, t in todo])
    recs = [r for r in recs if r]
    gates, decisions = {}, {}
    for r in recs:
        gates[r["gate"]] = gates.get(r["gate"], 0) + 1
        decisions[r["store_decision"]] = decisions.get(r["store_decision"], 0) + 1
    return {"saved": len(recs), "unsaved_min_docs": rs.unsaved, "gates": gates, "store_decisions": decisions,
            "playbook_entries": len(rs.playbook), "playbook_chars": rs.playbook.chars(), "frozen": rs.policy.frozen,
            "extra_rounds_used": rs.extra_rounds_used}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", required=True, choices=MODES,
                   help="baseline = official harness alone; react / refine / memory = ablation arms; "
                        "remo = Alg. 1; adaremo = Alg. 2")
    p.add_argument("--K", type=int, default=None, help="round budget per task (default 1 for baseline/react/memory, else 3)")
    p.add_argument("--out", required=True, help="run dir (resumable)")
    p.add_argument("--limit", type=int, default=0, help="first N benchmark tasks (0 = all 400)")
    p.add_argument("--base-url", default=None,
                   help="OpenAI-compatible base url for the harness backbone, the page reader (unless "
                        "FH_VLLM_READER_BASE_URL is set) and the critic "
                        "(default: $FH_VLLM_BASE_URL, then $REMO_BASE_URL, then http://localhost:8125/v1)")
    p.add_argument("--model", default=os.environ.get("FIN_MODEL", DEFAULT_MODEL),
                   help="served model name, used for the harness backbone + reader and for the critic")
    p.add_argument("--redundant-mode", default="reinforce", choices=["reinforce", "gate", "off"],
                   help="AdaReMo: what to do with a lesson the critic judges already covered")
    p.add_argument("--freeze-after", type=int, default=None, metavar="A",
                   help="learn-then-freeze: consolidate on the first A tasks, then run with the memory read-only")
    # FinanceGym-specific
    p.add_argument("--tasks", default=BENCH_FILE, help="benchmark jsonl (task_id/question/cutoff)")
    p.add_argument("--pit-url", default=os.environ.get("FH_PIT_URL", DEFAULT_PIT_URL), help="FinanceGym PIT search service")
    p.add_argument("--embed-url", default=os.environ.get("FH_EMBED_URL", DEFAULT_EMBED_URL), help="query-embedding endpoint")
    p.add_argument("--embed-model", default=EMBED_MODEL)
    p.add_argument("--conc", type=int, default=int(os.environ.get("FH_CONC", "8")), help="concurrent tasks")
    p.add_argument("--min-docs", type=int, default=int(os.environ.get("FH_MIN_DOCS", str(MIN_DOCS))),
                   help="final report below this many fetched docs is not saved (redone later)")
    p.add_argument("--max-empty-retries", type=int, default=int(os.environ.get("FH_MAX_EMPTY_RETRIES", "3")),
                   help="re-run an attempt whose final report is empty up to this many times")
    p.add_argument("--task-timeout-s", type=float, default=float(os.environ.get("FH_TASK_TIMEOUT_S", "3660")))
    p.add_argument("--backend-timeout-s", type=float, default=120.0, help="PIT/embed http timeout")
    p.add_argument("--extra-rounds-budget", type=int, default=EXTRA_ROUNDS_BUDGET,
                   help="run-wide cap on retry rounds (rounds after the first), persisted in policy_state.json")
    p.add_argument("--critic-max-tokens", type=int, default=CRITIC_MAX_TOKENS,
                   help="the critic sends no temperature: the server default applies")
    p.add_argument("--llm-timeout-s", type=float, default=600.0)
    # memory / AdaReMo knobs (defaults = RemoConfig defaults; inject cap 30000 chars as in the paper's runs)
    p.add_argument("--inject-cap", type=int, default=INJECT_CAP_CHARS, help="playbook chars prepended to the question")
    p.add_argument("--freeze-w", type=int, default=20)
    p.add_argument("--freeze-rho", type=float, default=0.1)
    p.add_argument("--probe-p", type=int, default=20)
    p.add_argument("--skip-health", action="store_true", help="do not probe PIT/embed/LLM before starting")
    p.add_argument("--loglevel", default=os.environ.get("FH_LOGLEVEL", "WARNING"), help="harness log level")
    return p


def write_final_results(run_dir: str, tasks_path: str, extra: dict | None = None) -> dict:
    res = summarize_run(run_dir, tasks_path)
    res.update(extra or {})
    with open(os.path.join(run_dir, "final_results.json"), "w") as f:
        json.dump(res, f, indent=1)
    return res


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    base_url = args.base_url or os.environ.get("FH_VLLM_BASE_URL") or os.environ.get("REMO_BASE_URL") \
        or "http://localhost:8125/v1"
    reader_base_url = os.environ.get("FH_VLLM_READER_BASE_URL") or base_url
    try:                                    # the harness loads a CA bundle per client; point it at certifi's
        import certifi
        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    except ImportError:
        pass
    logging.basicConfig(level=getattr(logging, args.loglevel.upper(), logging.WARNING),
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    if args.loglevel.upper() != "DEBUG":
        logging.getLogger("httpx").setLevel(logging.WARNING)     # one INFO line per fetch otherwise

    try:
        cfg, baseline = make_config(args.mode, args.K, redundant_mode=args.redundant_mode, freeze_w=args.freeze_w,
                                    freeze_rho=args.freeze_rho, probe_p=args.probe_p, inject_cap_chars=args.inject_cap)
    except ValueError as e:
        raise SystemExit(str(e))
    args.K = cfg.K
    if args.freeze_after is not None and not cfg.use_memory:
        raise SystemExit(f"--freeze-after needs a memory arm (memory/remo/adaremo), not --mode {args.mode}")

    from benchmarks.financegym.solver import FinanceGymSolver    # applies the harness runtime patches
    import openai

    os.makedirs(args.out, exist_ok=True)
    cfg_path = os.path.join(args.out, "run_config.json")
    prev = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}  # noqa: SIM115 (tiny, read once)
    for k in ("mode", "K", "freeze_after"):
        if prev and prev.get(k) != getattr(args, k):
            raise SystemExit(f"run dir {args.out} was started with {k}={prev.get(k)}; refusing {k}={getattr(args, k)}")
    with open(cfg_path, "w") as f:
        json.dump({**vars(args), "base_url": base_url, "reader_base_url": reader_base_url,
                   "remo_config": asdict(cfg), "baseline": baseline,
                   "harness_env": {k: os.environ.get(k) for k in
                                   ("FH_VLLM_BASE_URL", "FH_VLLM_READER_BASE_URL", "FH_PROFILE", "SSL_CERT_FILE")},
                   "cwd": os.getcwd(), "started": prev.get("started") or time.strftime("%Y-%m-%d %H:%M:%S")},
                  f, indent=1)

    tasks = load_tasks(args.tasks, args.limit)
    client = openai.AsyncOpenAI(api_key=os.environ.get("REMO_API_KEY", "EMPTY"), base_url=base_url,
                                timeout=args.llm_timeout_s, max_retries=2)
    solver = FinanceGymSolver(pit_url=args.pit_url, embed_url=args.embed_url, embed_model=args.embed_model,
                              timeout_s=args.backend_timeout_s, task_timeout_s=args.task_timeout_s,
                              max_empty_retries=args.max_empty_retries, min_docs=args.min_docs,
                              model=args.model, base_url=base_url, reader_base_url=reader_base_url,
                              plain=baseline, log=_log)
    critic = None if baseline else FinanceGymCritic(client, args.model, max_tokens=args.critic_max_tokens)
    rs = RunState(cfg, args.out, freeze_after=args.freeze_after,
                  extra_rounds_budget=args.extra_rounds_budget)

    async def _main():
        if not args.skip_health:
            await solver.health()
            models = [m.id for m in (await client.models.list()).data]
            if args.model not in models:
                raise SystemExit(f"model {args.model!r} not served at {base_url}: {models}")
            _log(f"[health] PIT {args.pit_url} + embed {args.embed_url} + LLM {base_url} ({args.model}) OK")
        return await run_all(tasks, rs, solver, critic, args.conc, args.min_docs)

    summary = asyncio.run(_main())
    if critic is not None:
        summary["critic"] = {"calls": critic.calls, "parse_failures": critic.parse_failures,
                             "call_failures": critic.call_failures}
    from benchmarks.financegym.make_answers import write_answers
    n_written, missing, defective = write_answers(args.out, os.path.join(args.out, "answers.jsonl"), args.tasks,
                                                  quiet=True)
    summary["answers"] = {"written": n_written, "missing": len(missing), "defective": len(defective)}
    final = write_final_results(args.out, args.tasks, {"this_invocation": summary})
    _log("[summary] " + json.dumps(summary))
    _log(f"[final_results] {os.path.join(args.out, 'final_results.json')}: n_saved={final['n_saved']} "
         f"gates={final['gate_distribution']} store={final['store_decisions']} mean_rounds={final['mean_rounds']} "
         f"memory={final['memory_entries']}e/{final['memory_tokens_cl100k']}tok")
    if defective:
        _log(f"[quality] {len(defective)} defective report(s) (see check_answers.py --delete to redo them)")
    return 0


if __name__ == "__main__":     # REQUIRED: the harness parse pool is a spawn ProcessPool that re-imports this module
    sys.exit(main())
