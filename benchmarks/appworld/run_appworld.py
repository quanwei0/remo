"""ReMo / AdaReMo on AppWorld (the `appworld` package at the revision pinned in pyproject.toml, data 0.1.0;
`test_normal` = 168 tasks / 56 scenarios).

Arms:  --mode react (K=1, no memory) | refine (K>1, no memory) | memory (K=1) | remo | adaremo.
The loop is remo.ReMoAgent (Algorithms 1/2) over AppWorldSolver (a fresh world per round, the previous
reflection injected into the retry), AppWorldCritic and CuratorConsolidator; memory is a
remo.SectionedPlaybook (style "plain") seeded from benchmarks/appworld/initial_playbook.txt and injected
whole. The memory arms render the paper's generator prompt; the no-memory arms render AppWorld's official
ReAct prompt as the paper's baselines did, and react is AppWorld's plain ReAct scaffold (no critic call,
50 000-char context, uncapped outputs; see solver_settings). Tasks are appworld.load_task_ids(split) in
file order; --limit N = first N.

Run dir (--out; resumable — tasks whose task_index is already in episodes.jsonl are skipped):
  episodes.jsonl        one record per task: the core record + task_id, per-round solver / critic stats
  trajs/<task_id>.json  every round's messages, steps (reply / code / output) and the critic's raw reply
  playbook.txt, policy_state.json, run_config.json, final_results.json (post hoc, see below)
AppWorld writes each world's end state and logs to
  $APPWORLD_ROOT/experiments/outputs/<experiment_name>/tasks/<task_id>/  (dbs/, logs/, misc/remo_rounds.json)
and the runner copies the end state of every task's FIRST round to
  $APPWORLD_ROOT/experiments/outputs/<experiment_name>__round1/tasks/<task_id>/dbs
<experiment_name> defaults to the basename of --out.

Scoring is post hoc with AppWorld's unit tests and never happens inside the loop (the worlds are opened
with load_ground_truth=False): after the last task the runner runs appworld.evaluator.evaluate_task on
every task it ran and writes TGC / SGC (percent) of the final state and of the round-1 snapshot, the gate
distribution, store decisions, mean rounds (len(rounds), failed rounds included) and the memory size
(bullets / chars / cl100k_base tokens) into final_results.json. --eval-only redoes just that step.
"""
import argparse
import json
import os
import sys
import time

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:                        # works from a checkout without `pip install -e`
    sys.path.insert(0, _REPO)

from benchmarks.appworld.consolidator import CuratorConsolidator, InsightConsolidator     # noqa: E402
from benchmarks.appworld.critic import AppWorldCritic, EnvCritic                           # noqa: E402
from benchmarks.appworld.solver import (REACT_MAX_OUTPUT_LENGTH, REACT_PROMPT_PATH,        # noqa: E402
                                        SOLVER_PROMPT_PATH, AppWorldSolver, ChatLLM)
from remo import ReMoAgent, RemoConfig, SectionedPlaybook                                  # noqa: E402

INITIAL_PLAYBOOK_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "initial_playbook.txt")
APPWORLD_VERSION = "0.1.4.dev0"    # the appworld revision pinned in pyproject.toml (the paper's runs)
DATA_VERSION = "0.1.0"             # data/version.txt written by `appworld download data` at that revision
PLAYBOOK_STYLE = "plain"
DEFAULT_MODEL = "GPT-OSS-120B"
# cli mode -> (core mode, use_memory, fixed K or None)
MODES = {"react": ("remo", False, 1), "refine": ("remo", False, None), "memory": ("remo", True, 1),
         "remo": ("remo", True, None), "adaremo": ("adaremo", True, None)}


def _log(msg: str) -> None:
    print(time.strftime("%Y-%m-%d %H:%M:%S"), msg, flush=True)


# -- configuration ------------------------------------------------------------------------------------
def check_versions(root: str, appworld_version: str) -> None:
    """Refuse another appworld package or dataset than the paper's: later versions changed the apps' API
    responses (what the model sees from step 2 on) and the unit tests / base DBs (what gets scored)."""
    path = os.path.join(root, "data", "version.txt")
    data_version = open(path, encoding="utf-8").read().strip() if os.path.exists(path) else None
    if appworld_version != APPWORLD_VERSION or data_version != DATA_VERSION:
        raise SystemExit(f"appworld {appworld_version} with data {data_version} at {root}; the paper's runs need "
                         f"appworld {APPWORLD_VERSION} (pip install -e '.[agents]') and data {DATA_VERSION} "
                         f"(`appworld download data` inside {root}) — see benchmarks/appworld/README.md")


def make_config(mode: str, K: int | None, redundant_mode: str = "reinforce", freeze_w: int = 20,
                freeze_rho: float = 0.1, probe_p: int = 20) -> RemoConfig:
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
                      probe_p=probe_p, inject_cap_chars=None, use_memory=use_memory)


def solver_settings(mode: str) -> dict:
    """Template and context limits of the arm's generator loop, as the paper's runs had them: the memory
    arms ran the reflect loop with the paper's generator prompt (400 000-char context, outputs capped at
    20 000 chars); refine ran that loop with AppWorld's official ReAct prompt; react ran AppWorld's plain
    ReAct scaffold with that prompt (50 000-char context, uncapped outputs)."""
    if MODES[mode][1]:
        return {"template_path": SOLVER_PROMPT_PATH}
    if mode == "refine":
        return {"template_path": REACT_PROMPT_PATH}
    return {"template_path": REACT_PROMPT_PATH, "max_output_length": REACT_MAX_OUTPUT_LENGTH, "output_cap": None}


def load_playbook(path: str | None) -> SectionedPlaybook:
    """The initial memory: the seed playbook file (None = the empty skeleton)."""
    return SectionedPlaybook.load(path, PLAYBOOK_STYLE) if path else SectionedPlaybook.from_skeleton(PLAYBOOK_STYLE)


class ReadOnlyPlaybook:
    """View of the playbook for the read-only phase of --freeze-after: reads pass through, writes are refused."""

    def __init__(self, playbook: SectionedPlaybook):
        self._pb = playbook

    def apply_add_ops(self, ops) -> list[str]:
        return []

    def reinforce(self, entry_id: str) -> bool:
        return False

    def __len__(self):
        return len(self._pb)

    def __getattr__(self, name):
        return getattr(self._pb, name)


class NoWriteConsolidator:
    last_error = ""

    def consolidate(self, playbook, episode, task, traj) -> str:
        return ""


# -- records -------------------------------------------------------------------------------------------
SOLVER_KEYS = ("steps", "task_completed", "env_clean", "error", "elapsed_s", "memory_chars", "critique_chars")


class AppWorldAgent(ReMoAgent):
    """ReMoAgent plus (a) task_id, the CLI arm, per-round solver / critic stats in the saved record (a
    round whose store the critic's confidence gate removed keeps the critic's own store / novelty_reason /
    cited_id), (b) store_decision "curator_error" when the consolidator call failed (playbook unchanged),
    "skipped_lowconf" when the gate removed the store of the accepting round, and (c) the read-only phase
    of --freeze-after (task_index >= A: the playbook is injected but nothing is added or reinforced, no
    consolidator call, and the saturation window stays as it was after A-1)."""

    def __init__(self, cfg, solver, critic, playbook, consolidator, cli_mode: str, run_dir=None, freeze_after=None):
        super().__init__(cfg, solver, critic, playbook=playbook, consolidator=consolidator, run_dir=run_dir)
        self.cli_mode, self.freeze_after = cli_mode, freeze_after
        self.base_consolidator = self.consolidator
        self.memory_readonly, self._policy_snap = False, self.policy.state()

    def run_task(self, task, task_index):
        self.memory_readonly = self.cfg.use_memory and self.freeze_after is not None and task_index >= self.freeze_after
        real = (self.playbook, self.consolidator)
        self._policy_snap = self.policy.state()
        if self.memory_readonly:
            self.playbook, self.consolidator = ReadOnlyPlaybook(real[0]), NoWriteConsolidator()
        try:
            return super().run_task(task, task_index)
        finally:
            self.playbook, self.consolidator = real
            if self.memory_readonly:
                self.policy.load_state(self._policy_snap)

    def _save(self, rec: dict) -> None:
        metas = self.solver.round_metas
        critic = self.critic.drain()
        rec["task_id"] = metas[0]["task_id"] if metas else ""
        rec["mode"], rec["K"] = self.cli_mode, self.cfg.K
        for i, rd in enumerate(rec["rounds"]):
            if i < len(metas):
                rd["solver"] = {k: metas[i].get(k) for k in SOLVER_KEYS}
            if i < len(critic):
                rd["critic"] = critic[i]
                rd.update(critic[i].get("low_confidence", {}))
        rec["elapsed_s"] = round(sum(m.get("elapsed_s", 0) or 0 for m in metas), 1)
        rec["memory_readonly"] = self.memory_readonly
        if self.memory_readonly:
            self.policy.load_state(self._policy_snap)          # policy_state.json keeps the learn-phase state
            rec["frozen"] = self.policy.frozen
            if rec["store_decision"] not in ("skipped", "no_memory"):
                rec["store_decision"], rec["entry_id"] = "skipped_readonly", ""
        elif rec["store_decision"] == "stored" and self.consolidator.last_error:
            rec["store_decision"] = "curator_error"
        elif rec["store_decision"] == "discarded" and len(critic) == len(rec["rounds"]) and "low_confidence" in critic[-1]:
            rec["store_decision"] = "skipped_lowconf"
        rec["playbook_entries"], rec["playbook_chars"] = len(self.playbook), self.playbook.chars()
        super()._save(rec)


def rounds_record(rec: dict, memory_frozen: bool) -> dict:
    """misc/remo_rounds.json of a task (written into AppWorld's task output directory)."""
    clean = rec["gate"] in ("round1_clean", "cross_round_validated")
    return {"clean": clean, "curation": rec["gate"],
            "rounds": [{"round": r["round"], "env_clean": bool(r["completed"]),
                        "verdict_no_errors": r["verdict"] == "correct", "reflection": r.get("raw", ""),
                        "confidence": (r.get("critic") or {}).get("confidence"), "refine": bool(r.get("refine", True)),
                        "store": bool(r.get("store", False)), "novelty_reason": r.get("novelty_reason", ""),
                        "parsed": bool(r.get("parsed", False))} for r in rec["rounds"]],
            "store_decision": rec["store_decision"], "memory_frozen": bool(memory_frozen),
            "task_id": rec.get("task_id", ""), "task_index": rec["task_index"], "stop_reason": rec["stop_reason"],
            "entry_id": rec.get("entry_id", "")}


def run_task(agent: AppWorldAgent, task_id: str, task_index: int, run_dir: str, log=_log) -> dict:
    """One task: Algorithm 1/2 via the core agent, then trajs/<task_id>.json and misc/remo_rounds.json."""
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
        f"pb={len(agent.playbook)}b/{agent.playbook.chars()}c frozen={agent.policy.frozen}"
        f"{' readonly' if agent.memory_readonly else ''} {rec.get('elapsed_s', 0)}s")
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


def summarize(eps: list[dict], playbook: SectionedPlaybook, final_eval: dict | None, round1_eval: dict | None) -> dict:
    gates, stops, decisions = {}, {}, {}
    for e in eps:
        gates[e["gate"]] = gates.get(e["gate"], 0) + 1
        stops[e["stop_reason"]] = stops.get(e["stop_reason"], 0) + 1
        decisions[e["store_decision"]] = decisions.get(e["store_decision"], 0) + 1
    n = len(eps)
    rounds = [len(e["rounds"]) for e in eps]
    steps = [sum((r.get("solver") or {}).get("steps") or 0 for r in e["rounds"]) for e in eps]
    out = {"n_tasks": n, "gate_distribution": gates, "stop_reasons": stops, "store_decisions": decisions,
           "mean_rounds": round(sum(rounds) / n, 3) if n else None, "total_rounds": sum(rounds),
           "mean_solver_steps_per_task": round(sum(steps) / n, 2) if n else None,
           "final_completed_rate": round(sum(bool(e["final_completed"]) for e in eps) / n, 4) if n else None,
           "round1_env_clean_rate": round(sum(bool(e["rounds"][0]["completed"]) for e in eps if e["rounds"]) / n, 4) if n else None,
           "memory": {"entries": len(playbook), "chars": playbook.chars(), "tokens_cl100k": memory_tokens(playbook.text),
                      "by_section": playbook.stats()["by_section"]}}
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
                   help="OpenAI-compatible base url (solver, critic, consolidator)")
    p.add_argument("--model", default=os.environ.get("REMO_MODEL", DEFAULT_MODEL))
    p.add_argument("--redundant-mode", default="reinforce", choices=["reinforce", "gate", "off"])
    p.add_argument("--store-conf", type=float, default=0.7, help="adaremo: a store the critic asserts with lower confidence is skipped")
    p.add_argument("--freeze-after", type=int, default=None, metavar="A",
                   help="learn-then-freeze: consolidate on the first A tasks, then run with the memory read-only")
    p.add_argument("--consolidator", default="curator", choices=["curator", "append"])
    p.add_argument("--initial-playbook", default=INITIAL_PLAYBOOK_PATH, help="seed playbook file")
    p.add_argument("--no-initial-playbook", action="store_true", help="start from the empty section skeleton")
    p.add_argument("--max-steps", type=int, default=40, help="REPL steps per round")
    p.add_argument("--root", default=os.environ.get("APPWORLD_ROOT"),
                   help="AppWorld root (data/ inside it; experiment outputs go to experiments/outputs/). "
                        "Default $APPWORLD_ROOT")
    p.add_argument("--experiment-name", default=None, help="AppWorld experiment name (default: basename of --out)")
    p.add_argument("--freeze-w", type=int, default=20)
    p.add_argument("--freeze-rho", type=float, default=0.1)
    p.add_argument("--probe-p", type=int, default=20)
    p.add_argument("--max-tokens", type=int, default=8192, help="solver completion budget per step")
    p.add_argument("--critic-max-tokens", type=int, default=8192)
    p.add_argument("--consolidator-max-tokens", type=int, default=8192)
    p.add_argument("--temperature", type=float, default=0.0, help="solver, critic and consolidator")
    p.add_argument("--random-seed", type=int, default=123, help="AppWorld world seed")
    p.add_argument("--exec-timeout", type=int, default=100, help="AppWorld's seconds per REPL execution")
    p.add_argument("--guard-timeout", type=int, default=300, help="outer seconds per REPL execution")
    p.add_argument("--llm-timeout-s", type=float, default=600.0)
    p.add_argument("--llm-attempts", type=int, default=50,
                   help="calls attempted per model request, 10 s apart (a context overflow fails at once)")
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
    check_versions(root, appworld.__version__)
    from appworld import load_task_ids

    cfg = make_config(args.mode, args.K, args.redundant_mode, args.freeze_w, args.freeze_rho, args.probe_p)
    args.K = cfg.K
    initial_playbook = None if args.no_initial_playbook else os.path.abspath(args.initial_playbook)
    experiment = args.experiment_name or os.path.basename(os.path.normpath(args.out))
    round1_experiment = None if args.no_round1_snapshot else f"{experiment}__round1"
    os.makedirs(args.out, exist_ok=True)
    cfg_path = os.path.join(args.out, "run_config.json")
    prev = {}
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            prev = json.load(f)
    for k in ("mode", "K", "split", "experiment_name", "consolidator", "initial_playbook", "model"):
        v = experiment if k == "experiment_name" else initial_playbook if k == "initial_playbook" else getattr(args, k)
        if prev and prev.get(k) not in (None, v):
            raise SystemExit(f"run dir {args.out} was started with {k}={prev.get(k)}; refusing {k}={v}")
    with open(cfg_path, "w") as f:
        json.dump({**vars(args), "experiment_name": experiment, "round1_experiment": round1_experiment,
                   "initial_playbook": initial_playbook, "appworld_root": root,
                   "appworld_version": appworld.__version__, "data_version": DATA_VERSION,
                   "appworld_output_dir": os.path.join(root, "experiments", "outputs", experiment),
                   "core_config": cfg.__dict__, "started": prev.get("started") or time.strftime("%Y-%m-%d %H:%M:%S")},
                  f, indent=1, default=str)

    task_ids = load_task_ids(args.split)
    if args.limit:
        task_ids = task_ids[:args.limit]

    if not args.eval_only:
        llm = ChatLLM(args.base_url, args.model, timeout=args.llm_timeout_s, attempts=args.llm_attempts, log=_log)
        if not args.skip_health:
            served = llm.served_models()
            if args.model not in served:
                raise SystemExit(f"model {args.model!r} not served at {args.base_url}: {served}")
            _log(f"[health] {args.base_url} serves {args.model}")
        solver = AppWorldSolver(llm, experiment, max_steps=args.max_steps, max_tokens=args.max_tokens,
                                temperature=args.temperature, random_seed=args.random_seed,
                                exec_timeout_s=args.exec_timeout, guard_timeout_s=args.guard_timeout,
                                round1_experiment=round1_experiment, log=_log, **solver_settings(args.mode))
        critic = (EnvCritic() if args.mode == "react" else
                  AppWorldCritic(llm, adaptive=cfg.adaptive, max_tokens=args.critic_max_tokens,
                                 temperature=args.temperature, store_conf=args.store_conf, log=_log))
        consolidator = (CuratorConsolidator(llm, max_tokens=args.consolidator_max_tokens, temperature=args.temperature, log=_log)
                        if args.consolidator == "curator" else InsightConsolidator())
        agent = AppWorldAgent(cfg, solver, critic, load_playbook(initial_playbook), consolidator, cli_mode=args.mode,
                              run_dir=args.out, freeze_after=args.freeze_after)
        done = agent.done_indices()
        todo = [(i, t) for i, t in enumerate(task_ids) if i not in done]
        _log(f"[run] {len(todo)} to do / {len(task_ids)} listed ({len(done)} already in episodes.jsonl); "
             f"mode={args.mode} core={cfg.mode} K={cfg.K} memory={'on' if cfg.use_memory else 'off'} "
             f"experiment={experiment} playbook={len(agent.playbook)} bullets")
        for i, tid in todo:
            run_task(agent, tid, i, args.out)
        _log(f"[run] finished: llm calls={llm.calls} failures={llm.failures} prompt_tokens={llm.prompt_tokens} "
             f"completion_tokens={llm.completion_tokens}; critic parse_failures={critic.parse_failures} "
             f"call_failures={critic.call_failures}; consolidator failures={getattr(consolidator, 'failures', 0)}")

    if args.no_eval:
        _log("[eval] skipped (--no-eval); final_results.json is left as it is — run again with --eval-only to score")
        return 0
    eps = read_episodes(args.out)
    eps_by_id = {e.get("task_id"): e for e in eps}
    ran = [t for t in task_ids if t in eps_by_id]
    pb_path = os.path.join(args.out, "playbook.txt")
    playbook = SectionedPlaybook.load(pb_path, PLAYBOOK_STYLE) if os.path.exists(pb_path) else load_playbook(initial_playbook)
    final_eval = round1_eval = None
    if ran:
        _log(f"[eval] {len(ran)} tasks of {experiment} with AppWorld's unit tests")
        final_eval = evaluate_experiment(experiment, ran)
        if round1_experiment:
            round1_eval = evaluate_experiment(round1_experiment, ran)
    results = {"benchmark": "appworld", "split": args.split, "mode": args.mode, "K": cfg.K, "core_mode": cfg.mode,
               "use_memory": cfg.use_memory, "redundant_mode": cfg.redundant_mode, "freeze_after": args.freeze_after,
               "consolidator": args.consolidator, "initial_playbook": initial_playbook,
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
