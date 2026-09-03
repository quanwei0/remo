"""Pre-submission quality floor for a FinanceGym run dir.

A saved report is DEFECTIVE if it is empty, shorter than --min-chars (1500) or starts with a JSON
brace (gpt-oss occasionally emits a tool-call fragment as its final message). `--delete` removes the
defective episodes from episodes.jsonl (atomic rewrite) and renames their trajs/<task_id>.json to
*.json.defective, so the next run_financegym.py invocation on the same run dir redoes exactly those tasks.

Playbook and policy state are NOT rolled back (memory is append-only) — the redo simply happens with
the memory as it stands. Stop any driver writing to the run
dir before using --delete.

Usage: check_answers.py RUN_DIR [--min-chars 1500] [--delete]
"""
import argparse
import json
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from benchmarks.financegym.common import REPORT_MIN_CHARS, is_defective, read_episodes    # noqa: E402


def find_defective(run_dir: str, min_chars: int = REPORT_MIN_CHARS) -> list[dict]:
    out = []
    for e in read_episodes(run_dir):
        report = e.get("final_answer", "") or ""
        why = is_defective(report, min_chars)
        if why:
            out.append({"task_id": e["task_id"], "task_index": e.get("task_index"), "reason": why,
                        "chars": len(report.strip()), "gate": e.get("gate"), "docs": e.get("docs_retrieved"),
                        "head": report.strip()[:80].replace("\n", " ")})
    return out


def delete_episodes(run_dir: str, task_ids: set[str]) -> int:
    p = os.path.join(run_dir, "episodes.jsonl")
    lines = [l for l in open(p) if l.strip()]
    keep = [l for l in lines if json.loads(l)["task_id"] not in task_ids]
    tmp = p + ".tmp"
    with open(tmp, "w") as f:
        f.writelines(keep)
    os.replace(tmp, p)
    for tid in task_ids:
        tp = os.path.join(run_dir, "trajs", f"{tid}.json")
        if os.path.exists(tp):
            os.rename(tp, tp + ".defective")
    return len(lines) - len(keep)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--min-chars", type=int, default=REPORT_MIN_CHARS)
    ap.add_argument("--delete", action="store_true", help="remove defective episodes so a rerun redoes them")
    a = ap.parse_args(argv)
    eps = read_episodes(a.run_dir)
    bad = find_defective(a.run_dir, a.min_chars)
    print(f"{len(eps)} episodes in {a.run_dir}; {len(bad)} defective (min_chars={a.min_chars})")
    for b in bad:
        print(f"  {b['task_id']} idx={b['task_index']} {b['reason']:13s} chars={b['chars']:5d} gate={b['gate']} "
              f"docs={b['docs']} | {b['head']}")
    if bad and a.delete:
        n = delete_episodes(a.run_dir, {b["task_id"] for b in bad})
        print(f"deleted {n} episode line(s) (trajs renamed *.json.defective); playbook/policy state untouched. "
              f"Rerun run_financegym.py on this run dir to redo them, then make_answers.py.")
    return 1 if bad and not a.delete else 0


if __name__ == "__main__":
    sys.exit(main())
