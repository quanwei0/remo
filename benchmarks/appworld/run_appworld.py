"""ReMo / AdaReMo on AppWorld (official `appworld` package; `test_normal` = 168 tasks / 56 scenarios).

Arms (README "Arms"):  --mode react (K=1, no memory) | refine (K>1, no memory) | memory (K=1) | remo | adaremo.
The loop is remo.ReMoAgent (Algorithms 1/2) over AppWorldSolver (a fresh world per round, the previous
critique injected at the start of the retry) and AppWorldCritic; memory is a remo.Playbook with prefix
"aw", 60000 chars injected. Tasks are appworld.load_task_ids(split) in file order; --limit N = first N.

Run dir (--out; resumable — tasks whose task_index is already in episodes.jsonl are skipped):
  episodes.jsonl        one record per task: the core record + task_id, per-round solver stats
  trajs/<task_id>.json  every round's full steps (reply / code / output) and the critic's raw reply
  playbook.txt, policy_state.json, run_config.json, final_results.json (post hoc, see below)
AppWorld writes each world's end state and logs to
  $APPWORLD_ROOT/experiments/outputs/<experiment_name>/tasks/<task_id>/  (dbs/, logs/, misc/remo_rounds.json)
and the runner copies the end state of every task's FIRST round to
  $APPWORLD_ROOT/experiments/outputs/<experiment_name>__round1/tasks/<task_id>/dbs
<experiment_name> defaults to the basename of --out.

Scoring is post hoc with AppWorld's unit tests and never happens inside the loop (the worlds are even
opened with load_ground_truth=False): after the last task the runner runs appworld.evaluator.evaluate_task
on every task it ran — the same tests `appworld evaluate <experiment_name> <split> --root $APPWORLD_ROOT`
runs for a whole split — and writes TGC / SGC (percent) of the final state and of the round-1 snapshot,
the gate distribution, store decisions, mean rounds (len(rounds), failed rounds included) and the memory
size (entries / cl100k_base tokens) into final_results.json. --eval-only redoes just that step.
"""
import argparse
import json
import os
import sys
import time

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:                        # works from a checkout without `pip install -e`
    sys.path.insert(0, _REPO)

from benchmarks.appworld.critic import CRITIC_MEMORY_CAP_CHARS, AppWorldCritic     # noqa: E402
from benchmarks.appworld.solver import AppWorldSolver                              # noqa: E402
from remo import Playbook, ReMoAgent, RemoConfig                                   # noqa: E402
from remo.agent import AppendConsolidator                                          # noqa: E402

PLAYBOOK_PREFIX = "aw"
INJECT_CAP_CHARS = 60000
DEFAULT_MODEL = "GPT-OSS-120B"
# cli mode -> (core mode, use_memory, fixed K or None)
MODES = {"react": ("remo", False, 1), "refine": ("remo", False, None), "memory": ("remo", True, 1),
         "remo": ("remo", True, None), "adaremo": ("adaremo", True, None)}


def _log(msg: str) -> None:
    print(time.strftime("%Y-%m-%d %H:%M:%S"), msg, flush=True)


# -- configuration ------------------------------------------------------------------------------------
def make_config(mode: str, K: int | None, redundant_mode: str = "reinforce", freeze_w: int = 20,
                freeze_rho: float = 0.1, probe_p: int = 20, inject_cap: int = INJECT_CAP_CHARS) -> RemoConfig:
    """CLI arm -> RemoConfig. Fixed-K arms (react, memory) take their K when --K is omitted and refuse
    another one; refine needs K >= 2; the default K otherwise is 3 (paper default)."""
    core_mode, use_memory, fixed_k = MODES[mode]
    if fixed_k is not None:
        if K not in (None, fixed_k):
            raise SystemExit(f"--mode {mode} is defined with K={fixed_k} (got --K {K})")
        K = fixed_k
    K = 3 if K is None else K
    if mode == "refine" and K < 2:
        raise SystemExit("--mode refine (refinement only) needs --K >= 2; K=1 without memory is --mode react")
    return RemoConfig(mode=core_mode, K=K, redundant_mode=redundant_mode, freeze_w=freeze_w, freeze_rho=freeze_rho,
                      probe_p=probe_p, inject_cap_chars=inject_cap, use_memory=use_memory)


class ReadOnlyMemory:
    """Learn-then-freeze (--freeze-after A). Wraps the RemoPolicy after the first A tasks: the inner
    loop, the gate and the state are the real policy's, but memory_decision answers "memory_frozen",
    so ReMoAgent performs no write (the playbook is still injected)."""

    def __init__(self, inner):
        self._inner = inner

    def memory_decision(self, st, task_index: int, cited_id: str = "") -> str:
        return "memory_frozen"

    def __getattr__(self, name):
        return getattr(self._inner, name)


class LLMConsolidator:
    """Optional (--consolidator llm): one model call turns the critic's lesson into one general playbook
    line; on any failure the lesson itself is stored (AppendConsolidator behaviour)."""
    PROMPT = ("Below is a lesson a reviewer drew from one solved task in an environment where an assistant calls "
              "app APIs from a Python REPL. Rewrite it as ONE line of at most 40 words for a playbook shown before "
              "OTHER tasks: imperative, general (no task-specific names, values or dates), keeping the concrete API "
              "behaviour or rule that makes it useful. Output only that line.\n\nTask it came from: {instruction}\n"
              "Lesson: {lesson}")

    def __init__(self, client, model: str, max_tokens: int = 256, temperature: float = 0.0, log=_log):
        self.client, self.model, self.max_tokens, self.temperature, self.log = client, model, max_tokens, temperature, log
        self.calls = self.failures = 0

    def consolidate(self, playbook: Playbook, lesson: str, task, traj) -> str:
        self.calls += 1
        line = ""
        try:
            instruction = (getattr(traj, "meta", None) or {}).get("instruction", "")
            r = self.client.chat.completions.create(
                model=self.model, max_tokens=self.max_tokens, temperature=self.temperature,
                messages=[{"role": "user", "content": self.PROMPT.format(instruction=instruction, lesson=lesson)}])
            line = (r.choices[0].message.content or "").strip().splitlines()
            line = next((l.strip().lstrip("-*• ").strip('"') for l in line if l.strip()), "")
        except Exception as e:                            # noqa: BLE001
            self.failures += 1
            self.log(f"[consolidator] call failed, storing the lesson verbatim: {type(e).__name__}: {e}")
        return playbook.add(line or lesson)


# -- records -------------------------------------------------------------------------------------------
SOLVER_KEYS = ("steps", "task_completed", "env_clean", "error", "no_code_steps", "elapsed_s", "prompt_tokens",
               "completion_tokens", "memory_chars", "critique_chars")


class AppWorldAgent(ReMoAgent):
    """ReMoAgent whose saved record also carries task_id, the CLI arm and per-round solver stats
    (taken from `solver.round_metas`, which the runner resets before every task)."""

    def __init__(self, cfg, solver, critic, cli_mode: str, playbook: Playbook | None = None, **kw):
        super().__init__(cfg, solver, critic, playbook=playbook, **kw)
        if playbook is not None:
            # the core does `playbook or Playbook()`: an EMPTY playbook is falsy (len 0), so a fresh
            # Playbook(prefix="aw") would be replaced by the default prefix. Put ours back, then reload.
            self.playbook = playbook
            if self.run_dir:
                self._load()
        self.cli_mode = cli_mode
        self.memory_readonly = False

    def _save(self, rec: dict) -> None:
        metas = getattr(self.solver, "round_metas", []) or []
        rec["task_id"] = metas[0]["task_id"] if metas else ""
        rec["mode"], rec["K"] = self.cli_mode, self.cfg.K
        for i, rd in enumerate(rec["rounds"]):
            if i < len(metas):
                rd["solver"] = {k: metas[i].get(k) for k in SOLVER_KEYS}
        rec["elapsed_s"] = round(sum(m.get("elapsed_s", 0) or 0 for m in metas), 1)
        rec["memory_readonly"] = self.memory_readonly
        rec["playbook_entries"] = len(self.playbook)
        super()._save(rec)


def rounds_record(rec: dict, memory_frozen: bool) -> dict:
    """misc/remo_rounds.json of a task (written into AppWorld's task output directory)."""
    clean = rec["gate"] in ("round1_clean", "cross_round_validated")
    return {"clean": clean, "curation": rec["gate"],
            "rounds": [{"round": r["round"], "env_clean": bool(r["completed"]),
                        "verdict_no_errors": r["verdict"] == "correct", "refine": bool(r.get("refine", True)),
                        "store": bool(r.get("store", False))} for r in rec["rounds"]],
            "store_decision": rec["store_decision"], "memory_frozen": bool(memory_frozen),
            "task_id": rec.get("task_id", ""), "task_index": rec["task_index"], "stop_reason": rec["stop_reason"],
            "entry_id": rec.get("entry_id", "")}


def run_task(agent: AppWorldAgent, task_id: str, task_index: int, run_dir: str, freeze_after: int | None = None,
             log=_log) -> dict:
    """One task: Algorithm 1/2 via the core agent, then trajs/<task_id>.json and misc/remo_rounds.json."""
    if freeze_after and task_index >= freeze_after and not agent.memory_readonly:
        agent.policy = ReadOnlyMemory(agent.policy)
        agent.memory_readonly = True
        log(f"[freeze-after] task_index {task_index} >= {freeze_after}: memory is read-only from here on")
    agent.solver.round_metas = []
    rec = agent.run_task({"task_id": task_id, "task_index": task_index}, task_index)
    metas = agent.solver.round_metas
    os.makedirs(os.path.join(run_dir, "trajs"), exist_ok=True)
    with open(os.path.join(run_dir, "trajs", f"{task_id}.json"), "w") as f:
        json.dump({"task_id": task_id, "task_index": task_index,
                   "rounds": [{"round": i + 1, **{k: v for k, v in m.items() if k not in ("task_id", "round")},
                               "critic_raw": rec["rounds"][i]["raw"] if i < len(rec["rounds"]) else ""}
                              for i, m in enumerate(metas)]}, f, default=str)
    misc_dir = metas[-1].get("misc_dir") if metas else None
    if misc_dir:
        try:
            os.makedirs(misc_dir, exist_ok=True)
            with open(os.path.join(misc_dir, "remo_rounds.json"), "w") as f:
                json.dump(rounds_record(rec, agent.memory_readonly or agent.policy.frozen), f, indent=1)
        except OSError as e:
            log(f"[warn] could not write {misc_dir}/remo_rounds.json: {e}")
    log(f"[done] {task_id} (#{task_index}) rounds={len(rec['rounds'])} gate={rec['gate']} "
        f"store={rec['store_decision']}{('=' + rec['entry_id']) if rec.get('entry_id') else ''} "
        f"pb={len(agent.playbook)}e frozen={agent.policy.frozen} {rec.get('elapsed_s', 0)}s")
    return rec


# -- post-hoc scoring ---------------------------------------------------------------------------------
def read_episodes(run_dir: str) -> list[dict]:
    p = os.path.join(run_dir, "episodes.jsonl")
    if not os.path.exists(p):
        return []
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def evaluate_experiment(experiment_name: str, task_ids: list[str], log=_log) -> dict:
    """TGC / SGC (percent, AppWorld's Metric) over `task_ids` with AppWorld's unit tests, reading the end
    state from experiments/outputs/<experiment_name>/tasks/<id>/dbs. Per task: evaluate_task (also saves
    tasks/<id>/evaluation/report.md). SGC is over the scenarios of the evaluated tasks only, so it equals
    the official number only for a complete split."""
    import appworld.evaluator as ev
    from appworld import AppWorld
    AppWorld.close_all()
    ev.CachedDBHandler.reset()
    metric, errors = ev.Metric(), {}
    for tid in task_ids:
        try:
            metric(tid, ev.evaluate_task(task_id=tid, experiment_name=experiment_name, suppress_errors=True,
                                         save_report=True))
        except Exception as e:                            # noqa: BLE001 — missing output, broken state ...
            errors[tid] = f"{type(e).__name__}: {e}"[:300]
            log(f"[eval] {experiment_name} {tid}: {errors[tid]}")
        ev.CachedDBHandler.reset()
    if not metric.task_id_to_test_tracker:
        return {"TGC": None, "SGC": None, "n_evaluated": 0, "per_task_success": {}, "errors": errors}
    m = metric.get_metrics(include_details=True, reset=True)
    return {"TGC": m["aggregate"]["task_goal_completion"], "SGC": m["aggregate"]["scenario_goal_completion"],
            "n_evaluated": len(m["individual"]),
            "per_task_success": {tid: bool(d.get("success")) for tid, d in m["individual"].items()},
            "errors": errors}


def memory_tokens(text: str) -> int | None:
    try:
        import tiktoken
        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except Exception:                                     # noqa: BLE001 — offline / not installed
        return None


def summarize(eps: list[dict], playbook: Playbook, final_eval: dict | None, round1_eval: dict | None) -> dict:
    gates, stops, decisions = {}, {}, {}
    for e in eps:
        gates[e["gate"]] = gates.get(e["gate"], 0) + 1
        stops[e["stop_reason"]] = stops.get(e["stop_reason"], 0) + 1
        decisions[e["store_decision"]] = decisions.get(e["store_decision"], 0) + 1
    n = len(eps)
    rounds = [len(e["rounds"]) for e in eps]
    steps = [sum((r.get("solver") or {}).get("steps") or 0 for r in e["rounds"]) for e in eps]
    toks = [sum(((r.get("solver") or {}).get("prompt_tokens") or 0) + ((r.get("solver") or {}).get("completion_tokens") or 0)
                for r in e["rounds"]) for e in eps]
    out = {"n_tasks": n, "gate_distribution": gates, "stop_reasons": stops, "store_decisions": decisions,
           "mean_rounds": round(sum(rounds) / n, 3) if n else None, "total_rounds": sum(rounds),
           "mean_solver_steps_per_task": round(sum(steps) / n, 2) if n else None,
           "mean_solver_tokens_per_task": round(sum(toks) / n) if n else None,
           "final_completed_rate": round(sum(bool(e["final_completed"]) for e in eps) / n, 4) if n else None,
           "round1_env_clean_rate": round(sum(bool(e["rounds"][0]["completed"]) for e in eps if e["rounds"]) / n, 4) if n else None,
           "memory": {"entries": len(playbook), "chars": playbook.chars(), "tokens_cl100k": memory_tokens(playbook.render()),
                      "helpful_total": sum(x.helpful for x in playbook.entries)}}
    if final_eval:
        out.update(TGC=final_eval["TGC"], SGC=final_eval["SGC"], n_evaluated=final_eval["n_evaluated"],
                   eval_errors=final_eval["errors"])
        succ = final_eval["per_task_success"]
        by_gate = {}
        for e in eps:
            if e.get("task_id") in succ:
                g = by_gate.setdefault(e["gate"], [0, 0])
                g[0] += int(succ[e["task_id"]]); g[1] += 1
        out["tgc_by_gate"] = {g: {"passed": p, "n": k, "rate": round(p / k, 4)} for g, (p, k) in by_gate.items()}
        out["per_task"] = {e["task_id"]: {"success": succ.get(e["task_id"]), "gate": e["gate"], "rounds": len(e["rounds"]),
                                          "store_decision": e["store_decision"]} for e in eps if e.get("task_id")}
    if round1_eval:
        out.update(round1_TGC=round1_eval["TGC"], round1_SGC=round1_eval["SGC"], round1_n_evaluated=round1_eval["n_evaluated"],
                   round1_eval_errors=round1_eval["errors"])
    return out


# -- CLI ----------------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", required=True, choices=sorted(MODES), help="paper arm (see README 'Arms')")
    p.add_argument("--K", type=int, default=None, help="round budget per task (default 3; react/memory fix K=1)")
    p.add_argument("--out", required=True, help="run dir (resumable)")
    p.add_argument("--limit", type=int, default=0, help="first N tasks of the split (0 = all)")
    p.add_argument("--split", default="test_normal", help="AppWorld dataset name (data/datasets/<split>.txt)")
    p.add_argument("--base-url", default=os.environ.get("REMO_BASE_URL", "http://localhost:8125/v1"),
                   help="OpenAI-compatible base url (solver, critic, llm consolidator)")
    p.add_argument("--model", default=os.environ.get("REMO_MODEL", DEFAULT_MODEL))
    p.add_argument("--redundant-mode", default="reinforce", choices=["reinforce", "gate", "off"])
    p.add_argument("--freeze-after", type=int, default=None, metavar="A",
                   help="learn-then-freeze: consolidate on the first A tasks, then run with the memory read-only")
    p.add_argument("--consolidator", default="append", choices=["append", "llm"])
    p.add_argument("--max-steps", type=int, default=40, help="REPL steps per round")
    p.add_argument("--root", default=os.environ.get("APPWORLD_ROOT"),
                   help="AppWorld root (data/ inside it; experiment outputs go to experiments/outputs/). "
                        "Default $APPWORLD_ROOT")
    p.add_argument("--experiment-name", default=None, help="AppWorld experiment name (default: basename of --out)")
    p.add_argument("--inject-cap", type=int, default=INJECT_CAP_CHARS, help="playbook chars shown to the solver")
    p.add_argument("--freeze-w", type=int, default=20)
    p.add_argument("--freeze-rho", type=float, default=0.1)
    p.add_argument("--probe-p", type=int, default=20)
    p.add_argument("--max-tokens", type=int, default=8192, help="solver completion budget per step")
    p.add_argument("--critic-max-tokens", type=int, default=4096)
    p.add_argument("--critic-memory-cap", type=int, default=CRITIC_MEMORY_CAP_CHARS)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--random-seed", type=int, default=100, help="AppWorld world seed")
    p.add_argument("--exec-timeout", type=int, default=100, help="seconds per REPL execution (AppWorld)")
    p.add_argument("--llm-timeout-s", type=float, default=600.0)
    p.add_argument("--no-round1-snapshot", action="store_true", help="do not keep the round-1 end state")
    p.add_argument("--no-eval", action="store_true", help="skip the post-hoc evaluation (final_results.json untouched)")
    p.add_argument("--eval-only", action="store_true", help="only (re)evaluate the tasks in episodes.jsonl")
    p.add_argument("--skip-health", action="store_true", help="do not check the model server first")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if not args.root:
        raise SystemExit("set APPWORLD_ROOT (or --root) to the directory holding data/ (appworld download data)")
    import appworld
    root = appworld.update_root(os.path.abspath(args.root))
    if not os.path.isdir(os.path.join(root, "data", "tasks")):
        raise SystemExit(f"{root}/data/tasks not found — run `appworld download data` inside {root}")
    from appworld import load_task_ids

    cfg = make_config(args.mode, args.K, args.redundant_mode, args.freeze_w, args.freeze_rho, args.probe_p, args.inject_cap)
    args.K = cfg.K
    experiment = args.experiment_name or os.path.basename(os.path.normpath(args.out))
    round1_experiment = None if args.no_round1_snapshot else f"{experiment}__round1"
    os.makedirs(args.out, exist_ok=True)
    cfg_path = os.path.join(args.out, "run_config.json")
    prev = {}
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            prev = json.load(f)
    for k in ("mode", "K", "split", "experiment_name"):
        v = experiment if k == "experiment_name" else getattr(args, k)
        if prev and prev.get(k) not in (None, v):
            raise SystemExit(f"run dir {args.out} was started with {k}={prev.get(k)}; refusing {k}={v}")
    with open(cfg_path, "w") as f:
        json.dump({**vars(args), "experiment_name": experiment, "round1_experiment": round1_experiment,
                   "appworld_root": root, "appworld_output_dir": os.path.join(root, "experiments", "outputs", experiment),
                   "core_config": cfg.__dict__, "started": prev.get("started") or time.strftime("%Y-%m-%d %H:%M:%S")},
                  f, indent=1, default=str)

    task_ids = load_task_ids(args.split)
    if args.limit:
        task_ids = task_ids[:args.limit]

    if not args.eval_only:
        import openai
        client = openai.OpenAI(api_key=os.environ.get("REMO_API_KEY", "EMPTY"), base_url=args.base_url,
                               timeout=args.llm_timeout_s, max_retries=2)
        if not args.skip_health:
            served = [m.id for m in client.models.list().data]
            if args.model not in served:
                raise SystemExit(f"model {args.model!r} not served at {args.base_url}: {served}")
            _log(f"[health] {args.base_url} serves {args.model}")
        solver = AppWorldSolver(client, args.model, experiment, max_steps=args.max_steps, max_tokens=args.max_tokens,
                                temperature=args.temperature, random_seed=args.random_seed,
                                exec_timeout_s=args.exec_timeout, round1_experiment=round1_experiment, log=_log)
        critic = AppWorldCritic(client, args.model, adaptive=cfg.adaptive, max_tokens=args.critic_max_tokens,
                                temperature=args.temperature, memory_cap=args.critic_memory_cap, log=_log)
        consolidator = LLMConsolidator(client, args.model) if args.consolidator == "llm" else AppendConsolidator()
        agent = AppWorldAgent(cfg, solver, critic, cli_mode=args.mode, playbook=Playbook(prefix=PLAYBOOK_PREFIX),
                              consolidator=consolidator, run_dir=args.out)
        done = agent.done_indices()
        todo = [(i, t) for i, t in enumerate(task_ids) if i not in done]
        _log(f"[run] {len(todo)} to do / {len(task_ids)} listed ({len(done)} already in episodes.jsonl); "
             f"mode={args.mode} core={cfg.mode} K={cfg.K} memory={'on' if cfg.use_memory else 'off'} "
             f"experiment={experiment} playbook={len(agent.playbook)} entries")
        for i, tid in todo:
            run_task(agent, tid, i, args.out, args.freeze_after)
        _log(f"[run] finished: solver calls={solver.calls} failures={solver.call_failures}; critic calls={critic.calls} "
             f"parse_failures={critic.parse_failures} call_failures={critic.call_failures}")

    if args.no_eval:
        _log("[eval] skipped (--no-eval); final_results.json is left as it is — run again with --eval-only to score")
        return 0
    eps = read_episodes(args.out)
    eps_by_id = {e.get("task_id"): e for e in eps}
    ran = [t for t in task_ids if t in eps_by_id]
    playbook = Playbook.load(os.path.join(args.out, "playbook.txt"), prefix=PLAYBOOK_PREFIX)
    final_eval = round1_eval = None
    if ran:
        _log(f"[eval] {len(ran)} tasks of {experiment} with AppWorld's unit tests")
        final_eval = evaluate_experiment(experiment, ran)
        if round1_experiment:
            round1_eval = evaluate_experiment(round1_experiment, ran)
    results = {"benchmark": "appworld", "split": args.split, "mode": args.mode, "K": cfg.K, "core_mode": cfg.mode,
               "use_memory": cfg.use_memory, "redundant_mode": cfg.redundant_mode, "freeze_after": args.freeze_after,
               "experiment_name": experiment, "round1_experiment": round1_experiment, "appworld_root": root,
               "n_tasks_listed": len(task_ids), "metric_note": "TGC/SGC in percent from appworld.evaluator.Metric; "
               "round1_* score the end state of each task's first round; SGC over evaluated tasks only",
               **summarize(eps, playbook, final_eval, round1_eval),
               "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(os.path.join(args.out, "final_results.json"), "w") as f:
        json.dump(results, f, indent=1, default=str)
    _log("[summary] " + json.dumps({k: results.get(k) for k in ("n_tasks", "TGC", "SGC", "round1_TGC", "round1_SGC",
                                                                   "gate_distribution", "store_decisions", "mean_rounds")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
