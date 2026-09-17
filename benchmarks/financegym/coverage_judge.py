"""Proxy scorer for FinanceGym reports (no rubric is available locally): an LLM judge that (1) drafts, from the
question alone, the items a complete answer must cover — 10 items, each tagged hindsight (facts up to the cutoff)
or foresight (what happens after) — and (2) marks, for every system's report, which items are addressed. The
rubric is drafted once per task (cached in --out) and never sees a report, so all systems are judged against the
same list. Also reports hedge sentences per report ("not available", "could not be confirmed", ...), report
length and, for multi-round runs, how many of the round-1 numbers survive into the submitted report.

This approximates how the organizers grade (rubric items addressed) — it predicted the direction of the ReMo
submission's score (coverage down, per-item quality flat) — but it is NOT the leaderboard score.

  python benchmarks/financegym/coverage_judge.py --runs baseline=runs/financegym/baseline remo=runs/financegym/remo \\
         --base-url http://HOST:8125/v1 --model GPT-OSS-120B --out runs/financegym/judge --limit 50
A run is a run dir (episodes.jsonl + trajs/) or an answers.jsonl (question / report per line); tasks are matched
by task_id when present, else by the question text.
"""
import argparse
import asyncio
import json
import os
import re
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from benchmarks.financegym.common import BENCH_FILE, DEFAULT_MODEL, load_tasks, read_episodes   # noqa: E402
from remo.critic import extract_json                                                           # noqa: E402

RUBRIC_PROMPT = """You are designing a grading rubric for a finance research report. From the question ALONE (you have not
seen any report), list exactly 10 items a complete, professional answer must cover. Tag each item "hindsight" (a fact,
figure, date, actor or mechanism that was knowable on or before the cutoff) or "foresight" (a forward-looking
judgement about what happens after the cutoff, when the question asks for one). Items must be specific and
checkable ("Q3 2024 revenue and YoY change", "management's stated mitigation"), not generic ("provide context").

Research question: {q}
Cutoff: {cutoff}

Reply with ONLY a JSON object: {{"items": [{{"text": "...", "axis": "hindsight" or "foresight"}}, ...]}}"""

JUDGE_PROMPT = """You are grading a finance research report against a fixed rubric. For each rubric item decide whether the
report ADDRESSES it: it states the fact, figure, date, actor, mechanism or forecast the item asks for (cited or not,
approximate is fine). An item is NOT addressed when the report is silent on it or only says the information is
unavailable / could not be confirmed. Judge coverage only, not correctness.

Research question: {q}
Rubric items:
{items}

=== REPORT ===
{report}
=== END REPORT ===

Reply with ONLY a JSON object: {{"addressed": [true/false for item 1, item 2, ... in order]}}"""

HEDGE = re.compile(r"\b(not (publicly )?available|could not be (found|verified|confirmed|located)|no (public )?(data|figures?|"
                   r"information|disclosure) (was|were|is|are) (found|available)|unable to (verify|confirm|locate)|"
                   r"not disclosed|remains? unclear|is not known)\b", re.I)
NUM = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?%?")


def load_run(path: str) -> dict[str, dict]:
    """task key -> {"report", "round1"}; key = task_id when present, else the question text."""
    out = {}
    if os.path.isdir(path):
        for e in read_episodes(path):
            rounds = e.get("rounds") or []
            r1 = None
            tp = os.path.join(path, "trajs", f"{e['task_id']}.json")
            if os.path.exists(tp):
                try:
                    r1 = (json.load(open(tp)).get("rounds") or [{}])[0].get("report")
                except Exception:                                           # noqa: BLE001
                    r1 = None
            out[e.get("task_id") or e.get("question")] = {"report": e.get("final_answer") or "", "round1": r1,
                                                            "rounds": len(rounds), "question": e.get("question")}
    else:
        for l in open(path):
            if not l.strip():
                continue
            r = json.loads(l)
            out[r.get("task_id") or r.get("question")] = {"report": r.get("report") or r.get("final_answer") or "",
                                                           "round1": None, "rounds": 1, "question": r.get("question")}
    return out


def numbers(text: str) -> set[str]:
    return {n.rstrip(",.") for n in NUM.findall(text or "") if len(n.rstrip(",.")) >= 2}


async def ask(client, model: str, prompt: str, sem, max_tokens: int = 2048) -> dict:
    async with sem:
        for attempt in range(4):
            try:
                r = await client.chat.completions.create(model=model, temperature=0, max_tokens=max_tokens,
                                                         messages=[{"role": "user", "content": prompt}])
                d = extract_json(r.choices[0].message.content or "")
                if d:
                    return d
            except Exception:                                               # noqa: BLE001
                await asyncio.sleep(5)
    return {}


async def main_async(args) -> int:
    import openai
    tasks = load_tasks(args.tasks, args.limit)
    runs = {name: load_run(path) for name, path in (kv.split("=", 1) for kv in args.runs)}
    def rep_of(r, t):                      # per run: by task_id when the run has ids, else by the verbatim question
        return r.get(t["task_id"]) or r.get(t["question"])
    common = [t for t in tasks if all((rep_of(r, t) or {}).get("report", "").strip() for r in runs.values())]
    print(f"[judge] {len(common)} tasks with a report in every run ({', '.join(runs)})", flush=True)
    os.makedirs(args.out, exist_ok=True)
    client = openai.AsyncOpenAI(api_key=os.environ.get("REMO_API_KEY", "EMPTY"), base_url=args.base_url, timeout=600, max_retries=0)
    sem = asyncio.Semaphore(args.conc)

    async def rubric(t):
        p = os.path.join(args.out, "rubric", f"{t['task_id']}.json")
        if os.path.exists(p):
            return json.load(open(p))
        d = await ask(client, args.model, RUBRIC_PROMPT.format(q=t["question"], cutoff=t["cutoff"]), sem)
        items = [i for i in d.get("items", []) if isinstance(i, dict) and i.get("text")][:10]
        os.makedirs(os.path.dirname(p), exist_ok=True)
        json.dump(items, open(p, "w"), indent=1)
        return items

    async def judge(t, items, name, rep):
        p = os.path.join(args.out, "judged", name, f"{t['task_id']}.json")
        if os.path.exists(p):
            return json.load(open(p))
        lines = "\n".join(f"{i + 1}. [{it['axis']}] {it['text']}" for i, it in enumerate(items))
        d = await ask(client, args.model, JUDGE_PROMPT.format(q=t["question"], items=lines, report=rep["report"][:14000]), sem)
        a = [bool(x) for x in d.get("addressed", [])][:len(items)] + [False] * max(0, len(items) - len(d.get("addressed", [])))
        os.makedirs(os.path.dirname(p), exist_ok=True)
        json.dump(a, open(p, "w"))
        return a

    async def one(t):
        items = await rubric(t)
        if not items:
            return None
        res = {}
        for name, r in runs.items():
            rep = rep_of(r, t)
            a = await judge(t, items, name, rep)
            hind = [x for x, it in zip(a, items) if it["axis"] == "hindsight"]
            fore = [x for x, it in zip(a, items) if it["axis"] == "foresight"]
            kept = None
            if rep.get("round1") and rep["round1"] != rep["report"]:
                n1 = numbers(rep["round1"]); kept = len(n1 & numbers(rep["report"])) / len(n1) if n1 else None
            res[name] = {"cov": sum(a) / len(a), "hind": (sum(hind) / len(hind)) if hind else None,
                         "fore": (sum(fore) / len(fore)) if fore else None, "hedges": len(HEDGE.findall(rep["report"])),
                         "chars": len(rep["report"]), "numbers": len(numbers(rep["report"])), "kept": kept,
                         "rounds": rep.get("rounds")}
        return res

    results = [r for r in await asyncio.gather(*[one(t) for t in common]) if r]
    mean = lambda xs: (sum(xs) / len(xs)) if xs else float("nan")     # noqa: E731
    print(f"\n{'run':14s} {'coverage':>9s} {'hindsight':>10s} {'foresight':>10s} {'hedges':>7s} {'numbers':>8s} {'chars':>7s} {'r1 kept':>8s} {'rounds':>7s}")
    summary = {}
    for name in runs:
        rows = [r[name] for r in results]
        summary[name] = {k: mean([x[k] for x in rows if x.get(k) is not None]) for k in ("cov", "hind", "fore", "hedges", "chars", "numbers", "kept", "rounds")}
        s = summary[name]
        print(f"{name:14s} {100 * s['cov']:8.1f}% {100 * s['hind']:9.1f}% {100 * s['fore']:9.1f}% {s['hedges']:7.2f} {s['numbers']:8.1f} {s['chars']:7.0f} "
              f"{(100 * s['kept'] if s['kept'] == s['kept'] else float('nan')):7.1f}% {s['rounds']:7.2f}")
    json.dump({"n_tasks": len(results), "summary": summary, "per_task": results}, open(os.path.join(args.out, "judge.json"), "w"), indent=1)
    print(f"\n[judge] n={len(results)} -> {os.path.join(args.out, 'judge.json')}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True, metavar="NAME=PATH", help="run dirs or answers.jsonl files")
    p.add_argument("--tasks", default=BENCH_FILE)
    p.add_argument("--limit", type=int, default=0, help="first N benchmark tasks")
    p.add_argument("--base-url", default=os.environ.get("FH_VLLM_BASE_URL") or os.environ.get("REMO_BASE_URL") or "http://localhost:8125/v1")
    p.add_argument("--model", default=os.environ.get("FIN_MODEL", DEFAULT_MODEL))
    p.add_argument("--out", required=True, help="cache + judge.json")
    p.add_argument("--conc", type=int, default=8)
    return asyncio.run(main_async(p.parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
