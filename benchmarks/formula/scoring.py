"""Post-hoc scoring of a Formula run directory (the only place the targets are read).

A prediction is correct iff float(pred.replace(",", "")) == float(target.replace(",", "")); when either side is
not a number the strings must be equal (so the solver's "No final answer found" sentinel is wrong).
final_results.json: accuracy, round-1 accuracy, gate distribution, store decisions (incl. consolidator_error), stop
reasons, mean rounds (len(rounds), failed rounds included), memory size (bullets / chars / tiktoken cl100k_base
tokens), LLM call and token totals, per-formula and per-task breakdowns, learn/frozen segments.

    python benchmarks/formula/scoring.py RUN_DIR --data data/formula_test.jsonl      # rescore an existing run
"""
import argparse
import json
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from benchmarks.formula.data import DEFAULT_DATA_PATH, formula_name, load_rows   # noqa: E402
from remo.memory import SectionedPlaybook                                          # noqa: E402

TOKENIZER = "cl100k_base"


def is_correct(pred: str, target: str) -> bool:
    pred, target = (pred or "").strip(), (target or "").strip()
    try:
        return float(pred.replace(",", "")) == float(target.replace(",", ""))
    except (ValueError, OverflowError):
        return pred == target


def count_tokens(text: str) -> int | None:
    """tiktoken cl100k_base token count; None when the encoding is unavailable (offline node without
    TIKTOKEN_CACHE_DIR)."""
    try:
        import tiktoken
        return len(tiktoken.get_encoding(TOKENIZER).encode(text or ""))
    except Exception:
        return None


def read_episodes(run_dir: str) -> list[dict]:
    p = os.path.join(run_dir, "episodes.jsonl")
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _count(values) -> dict:
    out: dict = {}
    for v in values:
        out[str(v)] = out.get(str(v), 0) + 1
    return dict(sorted(out.items()))


def _round1_answer(ep: dict) -> str:
    rounds = ep.get("rounds") or []
    if rounds and "answer" in rounds[0]:
        return rounds[0]["answer"] or ""
    return ep.get("final_answer", "") if len(rounds) <= 1 else ""


def _segment(per_task: list[dict]) -> dict:
    n = len(per_task)
    c = sum(1 for t in per_task if t["correct"])
    return {"n": n, "correct": c, "accuracy": round(c / n, 4) if n else None}


def score_run(run_dir: str, rows: list[dict], config: dict | None = None) -> dict:
    """Score episodes.jsonl against the rows of the data file used for the run (index-aligned)."""
    config = config or {}
    episodes = sorted(read_episodes(run_dir), key=lambda e: e["task_index"])
    n_tasks = int(config.get("n_tasks") or len(rows))
    per_task, per_formula = [], {}
    llm = {"solver_calls": 0, "critic_calls": 0, "consolidator_calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
    reinforced = 0
    for ep in episodes:
        i = ep["task_index"]
        if i >= len(rows):
            continue
        target = rows[i]["target"]
        ok = is_correct(ep.get("final_answer", ""), target)
        r1 = is_correct(_round1_answer(ep), target)
        f = formula_name(rows[i]["context"]) or "(unknown)"
        per_formula.setdefault(f, {"n": 0, "correct": 0})
        per_formula[f]["n"] += 1
        per_formula[f]["correct"] += int(ok)
        per_task.append({"task_index": i, "formula": f, "correct": ok, "round1_correct": r1,
                         "no_answer": not (ep.get("final_answer") or "").strip(),
                         "rounds": len(ep.get("rounds") or []),
                         "gate": ep.get("gate"), "stop_reason": ep.get("stop_reason"),
                         "store_decision": ep.get("store_decision"), "memory_readonly": bool(ep.get("memory_readonly"))})
        for rd in ep.get("rounds") or []:
            u = rd.get("usage") or {}
            llm["solver_calls"] += int(u.get("solver_calls", 0))
            llm["critic_calls"] += int(u.get("critic_calls", 0))
            llm["prompt_tokens"] += int(u.get("prompt_tokens", 0))
            llm["completion_tokens"] += int(u.get("completion_tokens", 0))
        cu = ep.get("consolidator") or {}
        llm["consolidator_calls"] += int(cu.get("calls", 0))
        llm["prompt_tokens"] += int(cu.get("prompt_tokens", 0))
        llm["completion_tokens"] += int(cu.get("completion_tokens", 0))
        reinforced += int(ep.get("store_decision") == "reinforced")

    n = len(per_task)
    correct = sum(1 for t in per_task if t["correct"])
    r1_correct = sum(1 for t in per_task if t["round1_correct"])
    rounds_total = sum(t["rounds"] for t in per_task)

    pb_path = os.path.join(run_dir, "playbook.txt")
    pb_text, policy = "", {}
    if os.path.exists(pb_path):
        with open(pb_path, encoding="utf-8") as f:
            pb_text = f.read()
    entries = len(SectionedPlaybook(pb_text, "counts")) if pb_text else 0
    tokens = count_tokens(pb_text) if pb_text else 0
    ps_path = os.path.join(run_dir, "policy_state.json")
    if os.path.exists(ps_path):
        with open(ps_path, encoding="utf-8") as f:
            policy = json.load(f)

    res = {
        "benchmark": "formula",
        **{k: config.get(k) for k in ("mode", "K", "use_memory", "redundant_mode", "freeze_after", "consolidator",
                                       "model", "base_url", "data", "data_sha256", "data_matches_manifest")},
        "n_tasks": n_tasks, "n_scored": n, "complete": n >= n_tasks,
        "accuracy": round(correct / n, 4) if n else None, "correct": correct, "no_answer": sum(t["no_answer"] for t in per_task),
        "round1_accuracy": round(r1_correct / n, 4) if n else None, "round1_correct": r1_correct,
        "gates": _count(t["gate"] for t in per_task),
        "store_decisions": _count(t["store_decision"] for t in per_task),
        "stop_reasons": _count(t["stop_reason"] for t in per_task),
        "mean_rounds": round(rounds_total / n, 4) if n else None, "rounds_total": rounds_total,
        "memory": {"entries": entries, "tokens": tokens, "tokenizer": TOKENIZER if tokens is not None else "unavailable",
                   "chars": len(pb_text), "reinforced": reinforced, "frozen": policy.get("frozen", False),
                   "freeze_events": policy.get("freeze_events", [])},
        "llm": llm,
        "per_formula": {k: {**v, "accuracy": round(v["correct"] / v["n"], 4)} for k, v in sorted(per_formula.items())},
        "per_task": per_task,
    }
    fa = config.get("freeze_after")
    if fa is not None:
        res["segments"] = {"learn": _segment([t for t in per_task if t["task_index"] < fa]),
                           "frozen": _segment([t for t in per_task if t["task_index"] >= fa])}
    return res


def write_final_results(run_dir: str, rows: list[dict], config: dict | None = None) -> dict:
    res = score_run(run_dir, rows, config)
    with open(os.path.join(run_dir, "final_results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    return res


def summary_line(res: dict) -> str:
    mem = res["memory"]
    return (f"accuracy={res['accuracy']} ({res['correct']}/{res['n_scored']} of {res['n_tasks']}) "
            f"round1={res['round1_accuracy']} gates={res['gates']} stores={res['store_decisions']} "
            f"mean_rounds={res['mean_rounds']} memory={mem['entries']} entries/{mem['tokens']} tok "
            f"llm_calls={res['llm']['solver_calls']}+{res['llm']['critic_calls']}+{res['llm']['consolidator_calls']}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Rescore a Formula run directory (writes final_results.json).")
    p.add_argument("run_dir")
    p.add_argument("--data", default=None, help="data jsonl used for the run (default: the path in run_config.json)")
    a = p.parse_args(argv)
    cfg_path = os.path.join(a.run_dir, "run_config.json")
    cfg = {}
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
    data = a.data or cfg.get("data") or DEFAULT_DATA_PATH
    rows = load_rows(data)
    res = write_final_results(a.run_dir, rows, cfg)
    print(summary_line(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
