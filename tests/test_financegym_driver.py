"""FinanceGym adapter: the async concurrent driver over RemoPolicy (fake solver/critic, no harness,
no LLM) — persistence/resume, the min-docs floor, AdaReMo reinforce under concurrency, the baseline and
ablation arms, learn-then-freeze, the LLM consolidator, the critic prompt assembly, the quality floor and
the post-hoc summary."""
import asyncio
import json
import os
import tempfile
import unittest
from types import SimpleNamespace

from benchmarks.financegym import check_answers, make_answers
from benchmarks.financegym.common import (FinanceGymCritic, LLMConsolidator, build_critic_prompt, build_question,
                                          is_critic_failure, is_defective, make_config, record_view, summarize_run)
from benchmarks.financegym.run_financegym import RunState, run_all, write_final_results
from remo import Reflection, RemoConfig, Trajectory
from remo.critic import ADAREMO_FIELDS, REMO_FIELDS


def _traj(report="R" * 2000, docs=10):
    return Trajectory(text=record_view(report, ["q1"], docs, []), answer=report, completed=bool(report) and docs >= 3,
                      meta={"elapsed_s": 1.0, "docs_retrieved": docs, "steps": 5, "termination": "answer",
                            "citations": [], "queries": ["q1"], "empty_attempts": 0, "report_chars": len(report)})


class FakeSolver:
    def __init__(self, per_task=None): self.per_task, self.calls = per_task or {}, []
    async def solve(self, task, memory_text, critique):
        self.calls.append((task["task_id"], len(memory_text), critique))
        await asyncio.sleep(0.001)
        return self.per_task.get(task["task_id"], _traj())


class FakeCritic:
    def __init__(self, fn): self.fn, self.calls = fn, 0
    async def reflect(self, task, traj, memory_text, prior, r, K):
        self.calls += 1
        await asyncio.sleep(0.001)
        return self.fn(task, traj, memory_text, r)


class FakeChatClient:
    """openai.AsyncOpenAI look-alike returning a fixed completion (or raising)."""
    def __init__(self, text, fail=False):
        self.text, self.fail, self.prompts = text, fail, []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
    async def _create(self, **kw):
        self.prompts.append(kw["messages"][0]["content"])
        if self.fail:
            raise RuntimeError("model down")
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.text))])


def _tasks(n):
    return [{"task_id": f"t{i:03d}", "question": f"Q{i}?", "cutoff": "2025-06-01"} for i in range(n)]


def _run(tasks, rs, solver, critic, conc=4, min_docs=3):
    return asyncio.run(run_all(tasks, rs, solver, critic, conc, min_docs, log=lambda m: None))


def _write_tasks(d, tasks):
    tf = os.path.join(d, "tasks.jsonl")
    with open(tf, "w") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")
    return tf


class TestDriver(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()

    def test_remo_concurrent_persist_and_resume(self):
        tasks = _tasks(6)
        critic = FakeCritic(lambda t, tr, m, r: Reflection("correct", lesson=f"lesson {t['task_id']}"))
        rs = RunState(RemoConfig(mode="remo", K=3), self.d)
        s = _run(tasks, rs, FakeSolver(), critic)
        self.assertEqual(s["saved"], 6); self.assertEqual(s["gates"], {"round1_clean": 6})
        self.assertEqual(s["playbook_entries"], 6)
        eps = [json.loads(l) for l in open(os.path.join(self.d, "episodes.jsonl"))]
        self.assertEqual({e["task_id"] for e in eps}, {t["task_id"] for t in tasks})
        self.assertTrue(all(len(e["rounds"]) == 1 and e["store_decision"] == "stored" for e in eps))
        self.assertTrue(os.path.exists(os.path.join(self.d, "trajs", "t000.json")))
        # resume: nothing to do, playbook reloaded with its 6 entries and ids continue from 7
        rs2 = RunState(RemoConfig(mode="remo", K=3), self.d)
        self.assertEqual(len(rs2.done_ids), 6); self.assertEqual(len(rs2.playbook), 6)
        s2 = _run(tasks + _tasks(7)[6:], rs2, FakeSolver(), critic)
        self.assertEqual(s2["saved"], 1); self.assertEqual(rs2.playbook.entries[-1].id, "fin-00007")

    def test_min_docs_not_saved_and_retry_on_incorrect(self):
        tasks = _tasks(2)
        solver = FakeSolver({"t001": _traj(docs=0)})           # embed service dead for t001
        seen = {}
        def fn(t, tr, m, r):
            seen[t["task_id"]] = r
            return Reflection("incorrect", critique="fix X", lesson="L") if r < 2 else Reflection("correct", lesson="L2")
        rs = RunState(RemoConfig(mode="remo", K=3), self.d)
        s = _run(tasks, rs, solver, FakeCritic(fn))
        self.assertEqual(s["saved"], 1); self.assertEqual(s["unsaved_min_docs"], 1)
        eps = [json.loads(l) for l in open(os.path.join(self.d, "episodes.jsonl"))]
        self.assertEqual(eps[0]["task_id"], "t000"); self.assertEqual(eps[0]["gate"], "cross_round_validated")
        self.assertEqual(len(eps[0]["rounds"]), 2)
        self.assertEqual(rs.playbook.entries[0].text, "L2")
        # the retry carried the critique into the solver
        self.assertIn(("t000", 0, "fix X"), solver.calls)
        # t001: completed=False every round -> 3 rounds, never admitted, not saved, no memory write
        self.assertEqual(seen["t001"], 3); self.assertEqual(len(rs.playbook), 1)

    def test_adaremo_reinforce_and_critic_stop(self):
        tasks = _tasks(3)
        def fn(t, tr, m, r):
            if t["task_id"] == "t000":
                return Reflection("correct", lesson="use average equity", store=True)
            if t["task_id"] == "t001":
                return Reflection("incorrect", critique="no fix", refine=False)
            return Reflection("correct", lesson="dup", store=False, novelty_reason="covered by [fin-00001]")
        rs = RunState(RemoConfig(mode="adaremo", K=3), self.d)
        s = _run(tasks, rs, FakeSolver(), FakeCritic(fn), conc=1)      # conc=1: t000 stored before t002 cites it
        self.assertEqual(s["gates"], {"round1_clean": 2, "critic_stop": 1})
        self.assertEqual(s["store_decisions"], {"stored": 1, "skipped": 1, "reinforced": 1})
        self.assertEqual(rs.playbook.get("fin-00001").helpful, 2); self.assertEqual(len(rs.playbook), 1)
        st = json.load(open(os.path.join(self.d, "policy_state.json")))
        self.assertEqual(st["store_window"], [True, True])

    def test_baseline_no_critic_no_memory(self):
        cfg, baseline = make_config("baseline", None)
        self.assertTrue(baseline); self.assertEqual((cfg.mode, cfg.K, cfg.use_memory), ("remo", 1, False))
        solver = FakeSolver({"t001": _traj(docs=1)})
        rs = RunState(cfg, self.d, baseline=True)
        s = _run(_tasks(2), rs, solver, None)                 # critic=None: no critic call at all
        self.assertEqual(s["saved"], 1); self.assertEqual(s["unsaved_min_docs"], 1)   # doc floor still applies
        self.assertEqual(s["gates"], {"no_critic": 1}); self.assertEqual(s["store_decisions"], {"no_memory": 1})
        e = json.loads(open(os.path.join(self.d, "episodes.jsonl")).readline())
        self.assertEqual(len(e["rounds"]), 1); self.assertEqual(e["rounds"][0]["verdict"], "none")
        self.assertEqual(e["stop_reason"], "no_critic"); self.assertEqual(len(rs.playbook), 0)
        self.assertEqual(solver.calls[0][1:], (0, None))       # empty memory, no critique
        # the official-harness question: bare question + PIT sentence, no playbook header
        t = _tasks(1)[0]
        self.assertEqual(build_question(t, "", None, plain=True),
                         "Q0?\n\n(Point-in-time constraint: use only information published on or before 2025-06-01. "
                         "The search environment enforces this cutoff.)")
        self.assertTrue(build_question(t, "", None).startswith("Research question: Q0?"))
        self.assertIn("Analyst playbook", build_question(t, "[fin-00001] helpful=1 :: L", None))
        self.assertIn("reviewer found these issues", build_question(t, "", "fix X"))

    def test_arm_configs(self):
        self.assertEqual(make_config("react", None)[0].K, 1); self.assertFalse(make_config("react", None)[0].use_memory)
        self.assertEqual(make_config("refine", 3)[0].K, 3); self.assertFalse(make_config("refine", None)[0].use_memory)
        self.assertTrue(make_config("memory", None)[0].use_memory); self.assertEqual(make_config("memory", 1)[0].K, 1)
        self.assertEqual(make_config("remo", None)[0].mode, "remo"); self.assertEqual(make_config("adaremo", 5)[0].K, 5)
        self.assertEqual(make_config("adaremo", None, redundant_mode="gate")[0].redundant_mode, "gate")
        for bad in (("react", 3), ("memory", 2), ("baseline", 2), ("refine", 1), ("nope", None)):
            with self.assertRaises(ValueError):
                make_config(*bad)
        # react: critic runs (its verdict is recorded) but nothing is retried or written
        rs = RunState(make_config("react", None)[0], self.d)
        critic = FakeCritic(lambda t, tr, m, r: Reflection("incorrect", critique="c", lesson="L"))
        s = _run(_tasks(2), rs, FakeSolver(), critic)
        self.assertEqual(critic.calls, 2); self.assertEqual(s["gates"], {"never_clean": 2})
        self.assertEqual(s["store_decisions"], {"no_memory": 2}); self.assertEqual(len(rs.playbook), 0)

    def test_freeze_after_barrier_and_readonly(self):
        tasks = _tasks(6)
        critic = FakeCritic(lambda t, tr, m, r: Reflection("correct", lesson=f"lesson {t['task_id']}"))
        solver = FakeSolver()
        rs = RunState(RemoConfig(mode="remo", K=1), self.d, freeze_after=4)
        s = _run(tasks, rs, solver, critic, conc=6)               # all six start together
        self.assertEqual(s["store_decisions"], {"stored": 4, "readonly": 2}); self.assertEqual(len(rs.playbook), 4)
        eps = {e["task_id"]: e for e in map(json.loads, open(os.path.join(self.d, "episodes.jsonl")))}
        full = rs.playbook.render()
        for tid in ("t004", "t005"):                              # read-only tasks saw the complete learned memory
            self.assertTrue(eps[tid]["readonly_memory"]); self.assertEqual(eps[tid]["memory_chars_at_start"], len(full))
            self.assertIn((tid, len(full), None), solver.calls)
        self.assertEqual(len(eps["t000"]["rounds"]), 1)
        # resume: learning done -> barrier opens immediately, a new read-only task runs and does not write
        rs2 = RunState(RemoConfig(mode="remo", K=1), self.d, freeze_after=4)
        s2 = _run(tasks + _tasks(7)[6:], rs2, FakeSolver(), critic)
        self.assertEqual(s2["store_decisions"], {"readonly": 1}); self.assertEqual(len(rs2.playbook), 4)

    def test_llm_consolidator(self):
        client = FakeChatClient("- \"Always cross-check a headline figure against the underlying filing.\"\n(extra line)")
        cons = LLMConsolidator(client, "M")
        critic = FakeCritic(lambda t, tr, m, r: Reflection("correct", lesson="For ACME, the 2024 margin quote was unsourced", store=True))
        rs = RunState(RemoConfig(mode="adaremo", K=1), self.d, consolidator=cons)
        s = _run(_tasks(2), rs, FakeSolver(), critic, conc=2)
        self.assertEqual(s["store_decisions"], {"stored": 2}); self.assertEqual(cons.calls, 2)
        self.assertEqual(rs.playbook.entries[0].text, "Always cross-check a headline figure against the underlying filing.")
        self.assertIn("ACME", client.prompts[0])
        eps = [json.loads(l) for l in open(os.path.join(self.d, "episodes.jsonl"))]
        self.assertEqual({e["entry_id"] for e in eps}, {"fin-00001", "fin-00002"})
        # model outage: the raw lesson is stored, nothing is lost
        cons2 = LLMConsolidator(FakeChatClient("", fail=True), "M")
        rs2 = RunState(RemoConfig(mode="remo", K=1), tempfile.mkdtemp(), consolidator=cons2)
        _run(_tasks(1), rs2, FakeSolver(), critic)
        self.assertEqual(cons2.failures, 1); self.assertIn("ACME", rs2.playbook.entries[0].text)

    def test_critic_call_failure_accepts_but_never_writes_memory(self):
        # the real critic's fallback on a transport failure: verdict correct, parsed=False, no raw, no lesson
        fb = asyncio.run(FinanceGymCritic(FakeChatClient("", fail=True), "M", adaptive=False)
                         .reflect(_tasks(1)[0], _traj(), "", None, 1, 3))
        self.assertTrue(fb.correct and not fb.parsed and not fb.raw and not fb.lesson); self.assertTrue(is_critic_failure(fb))
        self.assertFalse(is_critic_failure(Reflection("correct", parsed=False, raw="garbage")))   # unparseable != failed
        # ReMo: round 1 flagged with a lesson, round 2's critic call fails -> the fallback accepts (no round burnt)
        # but the round-1 lesson (EpisodeState.lesson's fallback) must NOT be stored
        def fn(t, tr, m, r):
            return Reflection("incorrect", critique="fix X", lesson="L1") if r == 1 else fb
        rs = RunState(RemoConfig(mode="remo", K=3), self.d)
        s = _run(_tasks(1), rs, FakeSolver(), FakeCritic(fn))
        e = json.loads(open(os.path.join(self.d, "episodes.jsonl")).readline())
        self.assertEqual((len(e["rounds"]), e["gate"], e["store_decision"], e["entry_id"]), (2, "cross_round_validated", "skipped", ""))
        self.assertTrue(e["critic_failed"]); self.assertEqual(len(rs.playbook), 0)
        self.assertEqual(s["store_decisions"], {"skipped": 1})
        st = json.load(open(os.path.join(self.d, "policy_state.json")))
        self.assertEqual(st["store_window"], [])                       # no saturation bookkeeping either
        tf = _write_tasks(self.d, _tasks(1))
        self.assertEqual(summarize_run(self.d, tf)["critic_failed_episodes"], 1)
        # a parsed, admitted episode is not marked
        rs2 = RunState(RemoConfig(mode="remo", K=1), tempfile.mkdtemp())
        _run(_tasks(1), rs2, FakeSolver(), FakeCritic(lambda t, tr, m, r: Reflection("correct", lesson="L")))
        self.assertFalse(json.loads(open(os.path.join(rs2.run_dir, "episodes.jsonl")).readline())["critic_failed"])

    def test_quality_floor_and_answers(self):
        tasks = _tasks(3)
        solver = FakeSolver({"t001": _traj(report="too short"), "t002": _traj(report='{"tool": "search"}' + "x" * 2000)})
        rs = RunState(RemoConfig(mode="remo", K=1), self.d)
        _run(tasks, rs, solver, FakeCritic(lambda t, tr, m, r: Reflection("correct")))
        tf = _write_tasks(self.d, tasks)
        n, missing, defective = make_answers.write_answers(self.d, None, tf, quiet=True)
        self.assertEqual((n, missing, sorted(defective)), (3, [], ["t001", "t002"]))
        rows = [json.loads(l) for l in open(os.path.join(self.d, "answers.jsonl"))]
        self.assertEqual([r["question"] for r in rows], ["Q0?", "Q1?", "Q2?"])
        self.assertEqual(set(rows[0]), {"question", "cutoff", "report", "searches", "docs_retrieved", "steps", "elapsed_s"})
        bad = check_answers.find_defective(self.d)
        self.assertEqual({b["task_id"]: b["reason"] for b in bad}, {"t001": "short", "t002": "json_fragment"})
        self.assertEqual(check_answers.delete_episodes(self.d, {b["task_id"] for b in bad}), 2)
        self.assertEqual(len(RunState(RemoConfig(mode="remo", K=1), self.d).done_ids), 1)   # rerun redoes t001/t002

    def test_final_results_post_hoc(self):
        tasks = _tasks(4)
        def fn(t, tr, m, r):
            if t["task_id"] == "t003":
                return Reflection("incorrect", critique="c", lesson="L")          # never clean, 2 rounds
            return Reflection("correct", lesson=f"lesson {t['task_id']}", store=True)
        rs = RunState(RemoConfig(mode="adaremo", K=2), self.d)
        _run(tasks, rs, FakeSolver(), FakeCritic(fn))
        tf = _write_tasks(self.d, tasks)
        with open(os.path.join(self.d, "run_config.json"), "w") as f:
            json.dump({"mode": "adaremo", "K": 2, "freeze_after": None, "consolidator": "append"}, f)
        res = write_final_results(self.d, tf, {"this_invocation": {"saved": 4}})
        self.assertTrue(os.path.exists(os.path.join(self.d, "final_results.json")))
        self.assertIsNone(res["accuracy"]); self.assertEqual((res["n_tasks"], res["n_saved"]), (4, 4))
        self.assertEqual(res["gate_distribution"], {"round1_clean": 3, "never_clean": 1})
        self.assertEqual(res["store_decisions"], {"stored": 3, "skipped": 1})
        self.assertEqual(res["mean_rounds"], 1.25)                                  # failed rounds count
        self.assertEqual(res["round1_clean_rate"], 0.75); self.assertEqual(res["final_clean_rate"], 0.75)
        self.assertEqual(res["memory_entries"], 3); self.assertGreater(res["memory_tokens_cl100k"] or 0, 0)
        self.assertEqual(res["this_invocation"], {"saved": 4}); self.assertEqual(res["mode"], "adaremo")
        base = summarize_run(tempfile.mkdtemp(), tf)                                  # empty run dir is fine
        self.assertEqual((base["n_saved"], base["mean_rounds"], base["round1_clean_rate"]), (0, None, None))


class TestPrompt(unittest.TestCase):
    def test_prompt_assembly(self):
        task = {"task_id": "x", "question": "Why {braces}?", "cutoff": "2025-03-01"}
        tr = _traj(report='Report with {"json": 1} braces')
        p = build_critic_prompt(task, tr, "[fin-00001] helpful=1 :: L", "old critique", 2, 3, adaptive=True)
        self.assertIn("Why {braces}?", p); self.assertIn('{"json": 1}', p)
        self.assertIn("Point-in-time cutoff: 2025-03-01", p); self.assertIn("attempt 2 of 3", p)
        self.assertIn("CURRENT MEMORY", p); self.assertIn("[fin-00001]", p)
        self.assertIn(ADAREMO_FIELDS.format(), p)                  # fields spec verbatim, braces un-escaped
        p2 = build_critic_prompt(task, tr, "[fin-00001] helpful=1 :: L", None, 1, 3, adaptive=False)
        self.assertNotIn("CURRENT MEMORY", p2); self.assertIn(REMO_FIELDS.format(), p2); self.assertNotIn("attempt 1 of", p2)

    def test_is_defective(self):
        self.assertEqual(is_defective(""), "empty"); self.assertEqual(is_defective("x" * 100), "short")
        self.assertEqual(is_defective('{"a": 1}' + "x" * 3000), "json_fragment"); self.assertEqual(is_defective("x" * 1500), "")


if __name__ == "__main__":
    unittest.main()
