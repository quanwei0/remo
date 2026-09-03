"""Build the FinanceGym submission file answers.jsonl from a run dir's episodes.jsonl.

Submission spec (FinanceGym/docs/participate.md): required `question` (VERBATIM — it is the key
that matches the answer to its rubric), `cutoff`, `report`; optional `searches`, `docs_retrieved`,
`steps`, `elapsed_s`. Rows are written in benchmark order and `question`/`cutoff` are copied from
the benchmark file, never from the episode. Never include rubrics (there are none locally anyway).

Usage: make_answers.py RUN_DIR [--out RUN_DIR/answers.jsonl] [--tasks benchmark.jsonl] [--min-chars 1500]
"""
import argparse
import json
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from benchmarks.financegym.common import BENCH_FILE, REPORT_MIN_CHARS, is_defective, load_tasks, read_episodes  # noqa: E402


def write_answers(run_dir: str, out_path: str | None = None, tasks_path: str = BENCH_FILE,
                  min_chars: int = REPORT_MIN_CHARS, quiet: bool = False):
    """Returns (n_written, missing_task_ids, defective_task_ids). Defective rows ARE written (so a
    partial file is still complete) but reported — run check_answers.py --delete and rerun to fix."""
    out_path = out_path or os.path.join(run_dir, "answers.jsonl")
    rows = {e["task_id"]: e for e in read_episodes(run_dir)}      # last line wins if a task repeats
    order = load_tasks(tasks_path)
    missing, defective, n = [], [], 0
    tmp = out_path + ".tmp"
    with open(tmp, "w") as f:
        for t in order:
            e = rows.get(t["task_id"])
            if e is None:
                missing.append(t["task_id"]); continue
            report = e.get("final_answer", "") or ""
            if is_defective(report, min_chars):
                defective.append(t["task_id"])
            f.write(json.dumps({"question": t["question"], "cutoff": t["cutoff"], "report": report,
                                "searches": [q for q in (e.get("queries") or []) if isinstance(q, str)],
                                "docs_retrieved": e.get("docs_retrieved", 0), "steps": e.get("steps", 0),
                                "elapsed_s": e.get("elapsed_s", 0)}, ensure_ascii=False) + "\n")
            n += 1
    os.replace(tmp, out_path)
    if not quiet:
        print(f"wrote {n}/{len(order)} answers -> {out_path}")
        if missing:
            print(f"missing {len(missing)}: {missing[:5]}{'...' if len(missing) > 5 else ''}")
        if defective:
            print(f"WARNING {len(defective)} defective report(s) (< {min_chars} chars or JSON fragment): "
                  f"{defective[:5]}{'...' if len(defective) > 5 else ''} -> check_answers.py --delete, then rerun")
    return n, missing, defective


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    p.add_argument("--out", default=None)
    p.add_argument("--tasks", default=BENCH_FILE)
    p.add_argument("--min-chars", type=int, default=REPORT_MIN_CHARS)
    a = p.parse_args(argv)
    n, missing, defective = write_answers(a.run_dir, a.out, a.tasks, a.min_chars)
    return 0 if n and not defective else 1


if __name__ == "__main__":
    sys.exit(main())
