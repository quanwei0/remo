"""ReMo / AdaReMo on Formula (200 numeric financial questions, exact-match accuracy scored post hoc).

Arms are configurations of the one loop (remo.ReMoAgent):
  --mode react    remo, K=1, no memory          --mode refine  remo, K>1, no memory
  --mode memory   remo, K=1, memory             --mode remo    Alg. 1        --mode adaremo  Alg. 2

Run dir (resumable): episodes.jsonl (one line per finished question; present indices are skipped), trajs.jsonl
(full replies), playbook.txt, policy_state.json, run_config.json and, at the end, final_results.json.
The targets are never read inside the loop.

  python benchmarks/formula/run_formula.py --mode adaremo --K 3 --data data/formula_test.jsonl \
         --base-url http://HOST:8125/v1 --model GPT-OSS-120B --out runs/formula/adaremo_k3_r1
"""
import argparse
import json
import os
import sys
import time

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from benchmarks.formula.critic import CRITIC_MEMORY_CAP_CHARS, FormulaCritic, LLMConsolidator      # noqa: E402
from benchmarks.formula.data import (DEFAULT_DATA_PATH, MANIFEST_PATH, check_against_manifest,     # noqa: E402
                                     file_sha256, load_rows, make_task, read_manifest)
from benchmarks.formula.scoring import summary_line, write_final_results                          # noqa: E402
from benchmarks.formula.solver import DEFAULT_BASE_URL, DEFAULT_MODEL, ChatLLM, FormulaSolver    # noqa: E402
from remo import Playbook, ReMoAgent, RemoConfig                                                  # noqa: E402
from remo.agent import AppendConsolidator                                                          # noqa: E402

PLAYBOOK_PREFIX = "calc"
MODES = ("react", "refine", "memory", "remo", "adaremo")
LOCKED = ("mode", "K", "redundant_mode", "freeze_after", "consolidator", "model", "data_sha256")   # fixed for a run dir


def _log(msg: str) -> None:
    print(time.strftime("%Y-%m-%d %H:%M:%S"), msg, flush=True)


def config_for(mode: str, K: int, **knobs) -> RemoConfig:
    """The paper's arms as RemoConfig values (see the table in the README)."""
    if mode == "react":
        return RemoConfig(mode="remo", K=1, use_memory=False, **knobs)
    if mode == "refine":
        if K < 2:
            raise ValueError("--mode refine needs --K >= 2 (K=1 without memory is --mode react)")
        return RemoConfig(mode="remo", K=K, use_memory=False, **knobs)
    if mode == "memory":
        return RemoConfig(mode="remo", K=1, use_memory=True, **knobs)
    if mode in ("remo", "adaremo"):
        return RemoConfig(mode=mode, K=K, use_memory=True, **knobs)
    raise ValueError(f"unknown mode {mode!r}; choose from {MODES}")


class NoWriteConsolidator:
    def consolidate(self, playbook, lesson, task, traj) -> str:
        return ""


class ReadOnlyPlaybook:
    """View of a Playbook for the frozen phase of --freeze-after: reads pass through, writes are refused."""

    def __init__(self, playbook: Playbook):
        self._pb = playbook

    def add(self, text: str) -> str:
        return ""

    def reinforce(self, eid: str) -> bool:
        return False

    def __len__(self):
        return len(self._pb)

    def __getattr__(self, name):
        return getattr(self._pb, name)


class FormulaAgent(ReMoAgent):
    """remo.ReMoAgent plus (a) per-round answers / usage / full replies persisted with the episode and (b) the
    read-only memory phase of --freeze-after (task_index >= A: the playbook is read but never written, and the
    policy's saturation state stays as it was after task A-1; the record says store_decision="skipped_readonly")."""

    def __init__(self, cfg, solver, critic, playbook=None, consolidator=None, run_dir=None, freeze_after=None):
        super().__init__(cfg, solver, critic, playbook, consolidator, run_dir)
        # ReMoAgent tests `playbook or Playbook()`; an EMPTY caller-supplied playbook is falsy (it has __len__) and is
        # replaced by a default-prefix one, and _load() then reuses that prefix. Keep the "calc" prefix either way.
        self.playbook.prefix = (playbook or Playbook(prefix=PLAYBOOK_PREFIX)).prefix if playbook is not None else PLAYBOOK_PREFIX
        self.base_consolidator = self.consolidator
        self.freeze_after = freeze_after
        self._readonly, self._task, self._policy_snap = False, None, self.policy.state()

    def run_task(self, task, task_index):
        self._task = task
        self._readonly = self.cfg.use_memory and self.freeze_after is not None and task_index >= self.freeze_after
        real = (self.playbook, self.consolidator)
        self._policy_snap = self.policy.state()
        if self._readonly:
            self.playbook, self.consolidator = ReadOnlyPlaybook(real[0]), NoWriteConsolidator()
        try:
            return super().run_task(task, task_index)
        finally:
            self.playbook, self.consolidator = real
            if self._readonly:                       # no saturation bookkeeping while the memory is read-only
                self.policy.load_state(self._policy_snap)

    def _save(self, rec: dict) -> None:
        s_hist, c_hist = self.solver.drain(), self.critic.drain()
        for i, rd in enumerate(rec["rounds"]):
            s = s_hist[i] if i < len(s_hist) else {}
            c = c_hist[i] if i < len(c_hist) else {}
            rd["answer"] = s.get("answer", "")
            rd["solver"] = {k: s.get(k) for k in ("elapsed_s", "finish_reason", "error", "prompt_chars")}
            rd["usage"] = {"solver_calls": 1 if s else 0, "critic_calls": int(c.get("calls", 0)),
                           "prompt_tokens": int(s.get("prompt_tokens", 0)) + int(c.get("prompt_tokens", 0)),
                           "completion_tokens": int(s.get("completion_tokens", 0)) + int(c.get("completion_tokens", 0)),
                           "critic_elapsed_s": c.get("elapsed_s")}
        cu = self.base_consolidator.drain() if hasattr(self.base_consolidator, "drain") else []
        rec["consolidator_usage"] = {"calls": sum(int(x.get("calls", 0)) for x in cu),
                                     "prompt_tokens": sum(int(x.get("prompt_tokens", 0)) for x in cu),
                                     "completion_tokens": sum(int(x.get("completion_tokens", 0)) for x in cu)}
        rec["formula"] = (self._task or {}).get("formula", "")
        rec["memory_readonly"] = self._readonly
        if self._readonly:
            self.policy.load_state(self._policy_snap)          # policy_state.json keeps the learn-phase state
            rec["frozen"] = self.policy.frozen
            if rec["store_decision"] not in ("skipped", "no_memory"):
                rec["store_decision"], rec["entry_id"] = "skipped_readonly", ""
        super()._save(rec)
        with open(os.path.join(self.run_dir, "trajs.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps({"task_index": rec["task_index"],
                                "rounds": [{"round": i + 1, "critique_in": s.get("critique_in", ""), "answer": s.get("answer", ""),
                                            "text": s.get("text", "")} for i, s in enumerate(s_hist)]}, default=str) + "\n")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", required=True, choices=MODES)
    p.add_argument("--K", type=int, default=None, help="round budget per question (default 3; react / memory are K=1 arms)")
    p.add_argument("--out", required=True, help="run dir (resumable)")
    p.add_argument("--limit", type=int, default=0, help="first N questions (0 = all)")
    p.add_argument("--data", default=DEFAULT_DATA_PATH, help="jsonl from prepare_data.py")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL, help="OpenAI-compatible endpoint ($REMO_BASE_URL)")
    p.add_argument("--model", default=os.environ.get("REMO_MODEL", DEFAULT_MODEL))
    p.add_argument("--redundant-mode", default="reinforce", choices=["reinforce", "gate", "off"],
                   help="AdaReMo: what happens to a lesson the critic judges already covered")
    p.add_argument("--freeze-after", type=int, default=None, metavar="A",
                   help="consolidate on the first A questions, then run with the memory read-only")
    p.add_argument("--consolidator", default="append", choices=["append", "llm"])
    p.add_argument("--freeze-w", type=int, default=20)
    p.add_argument("--freeze-rho", type=float, default=0.1)
    p.add_argument("--probe-p", type=int, default=20)
    p.add_argument("--inject-cap", type=int, default=None, help="playbook chars shown to the solver (default: all)")
    p.add_argument("--max-tokens", type=int, default=8192, help="solver max_tokens")
    p.add_argument("--critic-max-tokens", type=int, default=8192)
    p.add_argument("--consolidator-max-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.0, help="solver and critic decode temperature")
    p.add_argument("--critic-memory-cap", type=int, default=CRITIC_MEMORY_CAP_CHARS, help="memory chars the critic sees")
    p.add_argument("--timeout-s", type=float, default=600.0, help="per-request timeout")
    p.add_argument("--skip-health", action="store_true", help="do not check that --model is served before starting")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    K_given, args.K = args.K, (3 if args.K is None else args.K)
    try:
        cfg = config_for(args.mode, args.K, redundant_mode=args.redundant_mode, freeze_w=args.freeze_w,
                         freeze_rho=args.freeze_rho, probe_p=args.probe_p, inject_cap_chars=args.inject_cap)
    except ValueError as e:
        raise SystemExit(str(e))
    if args.mode in ("react", "memory") and K_given not in (None, 1):
        _log(f"[note] --mode {args.mode} is a K=1 arm; ignoring --K {K_given}")
    if args.freeze_after is not None and not cfg.use_memory:
        _log("[note] --freeze-after has no effect without memory")

    all_rows = load_rows(args.data)
    rows = all_rows[:args.limit] if args.limit else all_rows
    manifest = check_against_manifest(all_rows, read_manifest(MANIFEST_PATH)) if os.path.exists(MANIFEST_PATH) else {}
    if manifest and not manifest.get("exact"):
        _log(f"[warn] {args.data} is not the paper's 200-row file in evaluation order: {manifest}")

    os.makedirs(args.out, exist_ok=True)
    config = {**vars(args), "K": cfg.K, "use_memory": cfg.use_memory, "core_mode": cfg.mode, "n_tasks": len(rows),
              "data_sha256": file_sha256(args.data), "data_matches_manifest": bool(manifest.get("exact")),
              "playbook_prefix": PLAYBOOK_PREFIX}
    cfg_path = os.path.join(args.out, "run_config.json")
    prev = {}
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            prev = json.load(f)
    for k in LOCKED:
        if prev and prev.get(k) != config.get(k):
            raise SystemExit(f"run dir {args.out} was started with {k}={prev.get(k)!r}; refusing {k}={config.get(k)!r}")
    config["started"] = prev.get("started") or time.strftime("%Y-%m-%d %H:%M:%S")
    with open(cfg_path, "w") as f:
        json.dump(config, f, indent=1)

    llm = ChatLLM(args.base_url, args.model, timeout=args.timeout_s)
    if not args.skip_health:
        try:
            served = llm.served_models()
        except Exception as e:
            raise SystemExit(f"cannot reach {args.base_url}: {type(e).__name__}: {e}")
        if args.model not in served:
            raise SystemExit(f"model {args.model!r} not served at {args.base_url}: {served}")
    solver = FormulaSolver(llm, max_tokens=args.max_tokens, temperature=args.temperature)
    critic = FormulaCritic(llm, adaptive=cfg.adaptive, max_tokens=args.critic_max_tokens, temperature=args.temperature,
                           memory_cap=args.critic_memory_cap)
    consolidator = LLMConsolidator(llm, max_tokens=args.consolidator_max_tokens) if args.consolidator == "llm" \
        else AppendConsolidator()
    agent = FormulaAgent(cfg, solver, critic, Playbook(prefix=PLAYBOOK_PREFIX), consolidator, run_dir=args.out,
                         freeze_after=args.freeze_after)

    done = agent.done_indices()
    todo = [i for i in range(len(rows)) if i not in done]
    _log(f"[run] {len(todo)} to do / {len(rows)} listed ({len(done)} already in episodes.jsonl); arm={args.mode} "
         f"core={cfg.mode} K={cfg.K} memory={cfg.use_memory} redundant={cfg.redundant_mode} "
         f"freeze_after={args.freeze_after} consolidator={args.consolidator} playbook={len(agent.playbook)} entries")
    t_run = time.time()
    interrupted = False
    try:
        for i in todo:
            t0 = time.time()
            rec = agent.run_task(make_task(i, rows[i]), i)
            eid = f"={rec['entry_id']}" if rec.get("entry_id") else ""
            _log(f"[s{i}] gate={rec['gate']} rounds={len(rec['rounds'])} stop={rec['stop_reason']} "
                 f"store={rec['store_decision']}{eid} pb={len(agent.playbook)}e/{agent.playbook.chars()}c "
                 f"frozen={agent.policy.frozen}{' readonly' if rec.get('memory_readonly') else ''} "
                 f"answer={rec['final_answer']!r} {time.time() - t0:.1f}s")
    except KeyboardInterrupt:
        interrupted = True
        _log("[interrupt] stopping after the current question; rerun the same command to resume")

    res = write_final_results(args.out, rows, {**config, "elapsed_s": round(time.time() - t_run, 1)})
    _log("[final] " + summary_line(res) + f" solver_failures={solver.failures} critic_parse_failures={critic.parse_failures} "
         f"critic_call_failures={critic.call_failures} critic_skipped_no_answer={critic.skipped_no_answer}")
    return 130 if interrupted else 0


if __name__ == "__main__":
    sys.exit(main())
