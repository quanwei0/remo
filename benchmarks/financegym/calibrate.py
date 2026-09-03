"""Zero-rollout critic calibration: run the FinanceGym critic (the exact prompt run_financegym.py
uses — common.build_critic_prompt + remo.critic.fields_spec) over FINISHED reports, without any
solver rollout. Measures the round-1 flag rate (errors_found), the refine rate among flagged, the
store rate and parse health before any retry GPU-time is spent. Prompt iteration tool: a wide-net
prompt flagged 99% of baseline reports; the severe-only five checks brought it to ~56%.

Inputs (one of):
  --run-dir RUN_DIR    this adapter's episodes.jsonl (final_answer/queries/docs_retrieved/citations)
  --trajs-dir DIR      a directory of per-task JSON files ({"record": {...}} or the record itself)
Optional --playbook playbook.txt is shown to the critic as CURRENT MEMORY (adaremo only; default empty).
Resumable: task_ids already in --out are skipped.

Usage: calibrate.py --mode adaremo (--run-dir D | --trajs-dir D) --out cal.jsonl [--base-url URL --model NAME --conc 8 --limit N]
"""
import argparse
import asyncio
import glob
import json
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from benchmarks.financegym.common import (DEFAULT_MODEL, MIN_DOCS, CRITIC_MEMORY_CAP_CHARS, FinanceGymCritic,  # noqa: E402
                                          read_episodes, trajectory_from_record)
from remo import Playbook                                                          # noqa: E402


def load_records(run_dir: str | None, trajs_dir: str | None) -> list[dict]:
    if run_dir:
        return read_episodes(run_dir)
    recs = []
    for p in sorted(glob.glob(os.path.join(trajs_dir, "*.json"))):
        d = json.load(open(p))
        recs.append(d.get("record", d))
    return recs


async def calibrate(recs, critic: FinanceGymCritic, out_path: str, conc: int, memory_text: str, min_docs: int):
    done = set()
    if os.path.exists(out_path):
        done = {json.loads(l)["task_id"] for l in open(out_path) if l.strip()}
    sem, lock = asyncio.Semaphore(conc), asyncio.Lock()
    outf = open(out_path, "a")

    async def one(rec):
        tid = rec["task_id"]
        if tid in done:
            return
        task = {"task_id": tid, "question": rec["question"], "cutoff": rec["cutoff"]}
        traj = trajectory_from_record(rec, min_docs)
        async with sem:
            refl = await critic.reflect(task, traj, memory_text, None, 1, 1)
        row = {"task_id": tid, "parsed": refl.parsed, "verdict": refl.verdict, "refine": refl.refine,
               "store": refl.store, "lesson": refl.lesson[:300], "critique_len": len(refl.critique),
               "cited_id": refl.cited_id or Playbook.find_cited_id(refl.novelty_reason),
               "docs_retrieved": traj.meta["docs_retrieved"], "report_chars": len(traj.answer),
               "completed": traj.completed}
        async with lock:
            outf.write(json.dumps(row) + "\n"); outf.flush()
        print(f"[cal] {tid} parsed={refl.parsed} verdict={refl.verdict} refine={refl.refine} store={refl.store}",
              flush=True)

    await asyncio.gather(*[one(r) for r in recs])
    outf.close()


def summarize(out_path: str) -> dict:
    rows = [json.loads(l) for l in open(out_path) if l.strip()]
    n = len(rows)
    pf = sum(not r["parsed"] for r in rows)
    ef = sum(r["verdict"] == "incorrect" for r in rows)
    rf = sum(r["refine"] and r["verdict"] == "incorrect" for r in rows)
    st = sum(r["store"] for r in rows)
    cov = sum(bool(r.get("cited_id")) for r in rows)
    s = {"n": n, "parse_fail": pf, "errors_found": ef, "errors_found_pct": round(100 * ef / max(n, 1), 1),
         "refine_true_of_flagged": rf, "refine_pct_of_flagged": round(100 * rf / max(ef, 1), 1),
         "store_true": st, "store_pct": round(100 * st / max(n, 1), 1), "novelty_cited": cov}
    print(f"CALIBRATION n={n} parse_fail={pf} errors_found={ef} ({s['errors_found_pct']}%) "
          f"refine_true={rf} ({s['refine_pct_of_flagged']}% of flagged) store_true={st} ({s['store_pct']}%) "
          f"cited={cov}")
    return s


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--run-dir")
    src.add_argument("--trajs-dir")
    p.add_argument("--mode", required=True, choices=["remo", "adaremo"])
    p.add_argument("--out", required=True)
    p.add_argument("--playbook", default=None, help="playbook.txt shown as CURRENT MEMORY (adaremo)")
    p.add_argument("--base-url", default=os.environ.get("FH_VLLM_BASE_URL") or os.environ.get("REMO_BASE_URL")
                   or "http://localhost:8125/v1")
    p.add_argument("--model", default=os.environ.get("FIN_MODEL", DEFAULT_MODEL))
    p.add_argument("--conc", type=int, default=int(os.environ.get("CAL_CONC", "8")))
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--min-docs", type=int, default=MIN_DOCS)
    p.add_argument("--critic-max-tokens", type=int, default=2048)
    p.add_argument("--critic-temperature", type=float, default=0.0)
    p.add_argument("--critic-memory-cap", type=int, default=CRITIC_MEMORY_CAP_CHARS)
    a = p.parse_args(argv)

    import openai
    recs = load_records(a.run_dir, a.trajs_dir)
    if a.limit:
        recs = recs[:a.limit]
    memory_text = Playbook.load(a.playbook, prefix="fin").render() if a.playbook else ""
    client = openai.AsyncOpenAI(api_key=os.environ.get("REMO_API_KEY", "EMPTY"), base_url=a.base_url,
                                timeout=600.0, max_retries=2)
    critic = FinanceGymCritic(client, a.model, adaptive=(a.mode == "adaremo"), max_tokens=a.critic_max_tokens,
                              temperature=a.critic_temperature, memory_cap=a.critic_memory_cap)
    asyncio.run(calibrate(recs, critic, a.out, a.conc, memory_text, a.min_docs))
    summarize(a.out)
    print("ALL DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
