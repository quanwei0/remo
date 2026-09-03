"""AppWorld adapter: code-block extraction, prompt / transcript helpers, the critic prompt, arm -> config
mapping, the read-only memory wrapper and the record formats (episodes.jsonl line, trajs/<id>.json,
misc/remo_rounds.json) — fake solver and critic, no model, no appworld package."""
import json
import os
import tempfile
import unittest

from benchmarks.appworld.critic import AppWorldCritic, build_critic_prompt
from benchmarks.appworld.run_appworld import (AppWorldAgent, ReadOnlyMemory, make_config, read_episodes,
                                              rounds_record, run_task, summarize)
from benchmarks.appworld.solver import (CRITIQUE_HEADER, PLAYBOOK_HEADER, TRIM_MARKER, build_task_prompt, env_clean,
                                        extract_code, format_output, transcript, trim_history)
from remo import Playbook, Reflection, RemoConfig, RemoPolicy, Trajectory
from remo.critic import ADAREMO_FIELDS, REMO_FIELDS


class TestExtraction(unittest.TestCase):
    def test_first_python_block(self):
        t = "Thought.\n```python\nprint(1)\n```\nmore\n```python\nprint(2)\n```"
        self.assertEqual(extract_code(t), "print(1)")

    def test_untagged_and_py_tag(self):
        self.assertEqual(extract_code("```\nx = 1\n```"), "x = 1")
        self.assertEqual(extract_code("```py\nx = 2\n```"), "x = 2")

    def test_unterminated_block_taken_to_the_end(self):
        self.assertEqual(extract_code("plan\n```python\nfor p in pages:\n    print(p)"), "for p in pages:\n    print(p)")

    def test_no_block(self):
        self.assertEqual(extract_code("I would call the API now."), "")
        self.assertEqual(extract_code(""), "")
        self.assertEqual(extract_code(None), "")


class TestPromptHelpers(unittest.TestCase):
    SUP = {"first_name": "Ann", "last_name": "Lee", "email": "ann@x.org", "phone_number": "123"}

    def test_task_prompt_blocks(self):
        p = build_task_prompt("Count my playlists.", self.SUP, "", None)
        self.assertNotIn(PLAYBOOK_HEADER, p); self.assertNotIn(CRITIQUE_HEADER, p)
        self.assertIn("Task: Count my playlists.", p); self.assertIn("ann@x.org", p)
        p = build_task_prompt("T", self.SUP, "[aw-00001] helpful=1 :: paginate", "use page_index")
        self.assertLess(p.index(PLAYBOOK_HEADER), p.index(CRITIQUE_HEADER))
        self.assertLess(p.index(CRITIQUE_HEADER), p.index("Task: T"))
        self.assertIn("[aw-00001]", p); self.assertIn("use page_index", p)

    def test_env_clean(self):
        self.assertTrue(env_clean(True, "Marked the active task complete."))
        self.assertFalse(env_clean(True, "Execution failed. Traceback:\n  KeyError"))
        self.assertFalse(env_clean(False, "Execution successful."))

    def test_format_output_caps_in_the_middle(self):
        s = format_output("a" * 100 + "b" * 100, cap=50)
        self.assertTrue(s.startswith("Output:\n```\n" + "a" * 25)); self.assertIn("characters cut", s)
        self.assertEqual(format_output("   ", cap=50), "Output:\n```\nExecution successful.\n```")

    def test_trim_history_drops_oldest_pairs_keeps_head(self):
        msgs = [{"role": "system", "content": "S" * 10}, {"role": "user", "content": "T" * 10}]
        for i in range(5):
            msgs += [{"role": "assistant", "content": f"a{i}" * 50}, {"role": "user", "content": f"o{i}" * 50}]
        out = trim_history(msgs, keep_head=2, max_chars=500)
        self.assertEqual(out[:2], msgs[:2]); self.assertIn("trimmed", out[2]["content"])
        self.assertEqual(out[-1], msgs[-1]); self.assertLessEqual(sum(len(m["content"]) for m in out), 500 + 100)
        self.assertIs(trim_history(msgs, 2, 10 ** 6), msgs)
        # a second trim removes the earlier marker and keeps (assistant, user) pairs aligned
        out += [{"role": "assistant", "content": "a9" * 50}, {"role": "user", "content": "o9" * 50}]
        out2 = trim_history(out, keep_head=2, max_chars=500)
        self.assertEqual(sum(m["content"] == TRIM_MARKER for m in out2), 1); self.assertEqual(out2[2]["content"], TRIM_MARKER)
        self.assertEqual([m["role"] for m in out2[3:]], ["assistant", "user"] * (len(out2[3:]) // 2))
        self.assertEqual(out2[-1]["content"], "o9" * 50)

    def test_transcript(self):
        steps = [{"step": 1, "reply": "r1 {x}", "code": "c", "output": "o1"}, {"step": 2, "reply": "r2", "output": "E" * 50}]
        t = transcript(steps, step_cap=20, total_cap=10 ** 6)
        self.assertIn("[step 1] AGENT:\nr1 {x}", t); self.assertIn("[step 2] OUTPUT:\n" + "E" * 20 + " [...]", t)
        self.assertIn("omitted", transcript(steps, step_cap=10 ** 6, total_cap=40))


class _FakeWorld:
    """Stand-in for appworld.AppWorld: records executed code; complete_task submits."""
    instances = []
    def __init__(self, task_id, experiment_name, **kw):
        self.kw, self.executed, self.done = kw, [], False
        self.task = type("T", (), {"instruction": "Count my playlists.", "supervisor": type("S", (), {
            "first_name": "Ann", "last_name": "Lee", "email": "a@x.org", "phone_number": "1"})()})()
        d = tempfile.mkdtemp(); self.output_directory = os.path.join(d, "outputs", experiment_name, "tasks", task_id)
        self.output_misc_directory = os.path.join(self.output_directory, "misc")
        _FakeWorld.instances.append(self)
    def __enter__(self): return self
    def __exit__(self, *a): pass
    def execute(self, code):
        self.executed.append(code)
        if "complete_task" in code:
            self.done = True
        return "Execution successful."
    def task_completed(self): return self.done


class _ScriptedChat:
    def __init__(self, replies):
        self.replies, self.requests = list(replies), []
        self.chat = type("C", (), {})(); self.chat.completions = type("X", (), {})(); self.chat.completions.create = self._create
    def _create(self, **kw):
        self.requests.append([dict(m) for m in kw["messages"]])
        txt = self.replies[min(len(self.requests), len(self.replies)) - 1]
        msg = type("M", (), {"content": txt})(); ch = type("Ch", (), {"message": msg})()
        return type("R", (), {"choices": [ch], "usage": type("U", (), {"prompt_tokens": 10, "completion_tokens": 5})()})()


class TestSolveLoop(unittest.TestCase):
    """solve() against a fake world and client (the appworld import inside solve is redirected)."""
    def _solve(self, replies, **kw):
        import sys
        from unittest import mock
        from benchmarks.appworld.solver import AppWorldSolver, NO_CODE_REMINDER
        client = _ScriptedChat(replies)
        solver = AppWorldSolver(client, "m", "exp", max_steps=kw.pop("max_steps", 10), round1_experiment=None,
                                log=lambda m: None, **kw)
        with mock.patch.dict(sys.modules, {"appworld": type("A", (), {"AppWorld": _FakeWorld})}):
            traj = solver.solve({"task_id": "t1"}, "[aw-00001] helpful=1 :: paginate", None)
        return traj, client, solver, NO_CODE_REMINDER

    def test_no_code_reply_is_dropped_and_reminded_then_recovers(self):
        traj, client, solver, reminder = self._solve(["Thought only, no block.",
                                                       "```python\nprint(1)\n```",
                                                       "```python\napis.supervisor.complete_task(answer=3)\n```"])
        self.assertTrue(traj.completed); self.assertEqual(traj.answer, "apis.supervisor.complete_task(answer=3)")
        self.assertEqual(len(client.requests), 3); self.assertEqual(traj.meta["steps"], 3); self.assertEqual(traj.meta["no_code_steps"], 1)
        r2 = client.requests[1]                      # the failed turn is gone; the reminder sits on the task prompt
        self.assertEqual([m["role"] for m in r2], ["system", "user"]); self.assertTrue(r2[1]["content"].endswith(reminder))
        self.assertEqual(client.requests[2][1]["content"].count("discarded"), 1)
        self.assertIn("No python code block", traj.text); self.assertIn("[step 1] AGENT:\nThought only", traj.text)
        self.assertEqual(_FakeWorld.instances[-1].executed, ["print(1)", "apis.supervisor.complete_task(answer=3)"])
        self.assertEqual(_FakeWorld.instances[-1].kw["load_ground_truth"], False)      # no ground truth in the loop
        self.assertEqual(traj.meta["prompt_tokens"], 30)

    def test_three_no_code_replies_end_the_round(self):
        traj, client, solver, reminder = self._solve(["no block"])
        self.assertFalse(traj.completed); self.assertEqual(traj.meta["error"], "no_code_block")
        self.assertEqual((traj.meta["steps"], traj.meta["no_code_steps"], len(client.requests)), (3, 3, 3))
        self.assertEqual(client.requests[2][-1]["content"].count("discarded"), 1)      # reminder appended once
        self.assertEqual(_FakeWorld.instances[-1].executed, [])

    def test_error_in_last_output_is_not_clean(self):
        class BrokenWorld(_FakeWorld):
            def execute(self, code):
                super().execute(code); return "Execution failed. Traceback:\nKeyError"
        import sys
        from unittest import mock
        from benchmarks.appworld.solver import AppWorldSolver
        solver = AppWorldSolver(_ScriptedChat(["```python\napis.supervisor.complete_task()\n```"]), "m", "exp",
                                round1_experiment=None, log=lambda m: None)
        with mock.patch.dict(sys.modules, {"appworld": type("A", (), {"AppWorld": BrokenWorld})}):
            traj = solver.solve({"task_id": "t1"}, "", "fix it")
        self.assertTrue(traj.meta["task_completed"]); self.assertFalse(traj.completed)


class TestCriticPrompt(unittest.TestCase):
    def _traj(self):
        return Trajectory(text='[step 1] AGENT:\nx = {"a": 1}\n[step 1] OUTPUT:\nok', completed=True,
                          meta={"instruction": "Pay {rent}", "supervisor": {"first_name": "A", "last_name": "B",
                                "email": "a@b.c", "phone_number": "1"}, "task_completed": True,
                                "last_output": "Marked the active task complete.", "steps": 7})

    def test_assembly(self):
        p = build_critic_prompt(self._traj(), "[aw-00001] helpful=2 :: L", "old critique", 2, 3, adaptive=True)
        self.assertIn("Pay {rent}", p); self.assertIn('x = {"a": 1}', p)          # values keep their braces
        self.assertIn("task submitted via complete_task = True", p); self.assertIn("steps used = 7", p)
        self.assertIn("attempt 2 of 3", p); self.assertIn("old critique", p)
        self.assertIn("CURRENT MEMORY", p); self.assertIn("[aw-00001]", p)
        self.assertIn(ADAREMO_FIELDS.format(), p)
        p2 = build_critic_prompt(self._traj(), "[aw-00001] helpful=2 :: L", None, 1, 3, adaptive=False)
        self.assertNotIn("CURRENT MEMORY", p2); self.assertNotIn("attempt 1 of", p2); self.assertIn(REMO_FIELDS.format(), p2)


class _FakeChat:
    """openai client look-alike: returns the scripted texts in order, or raises."""
    def __init__(self, texts=(), fail=False):
        self.texts, self.fail, self.calls = list(texts), fail, 0
        self.chat = type("C", (), {})(); self.chat.completions = type("X", (), {})(); self.chat.completions.create = self._create
    def _create(self, **kw):
        self.calls += 1
        if self.fail:
            raise ConnectionError("server down")
        txt = self.texts[min(self.calls, len(self.texts)) - 1]
        msg = type("M", (), {"content": txt})(); ch = type("Ch", (), {"message": msg})()
        return type("R", (), {"choices": [ch]})()


class TestCriticFallbacks(unittest.TestCase):
    TRAJ = Trajectory(text="t", completed=True, meta={"instruction": "I", "supervisor": {}, "task_completed": True,
                                                       "last_output": "ok", "steps": 3})

    def test_parsed_reply(self):
        c = AppWorldCritic(_FakeChat(['{"verdict":"no_errors","critique":"c","refine":false,"store":true,'
                                      '"novelty_reason":"","lesson":"L"}']), "m", adaptive=True, log=lambda m: None)
        r = c.reflect({}, self.TRAJ, "", None, 1, 3)
        self.assertTrue(r.parsed and r.correct and r.store); self.assertEqual(r.lesson, "L"); self.assertEqual(c.calls, 1)

    def test_unparseable_is_retried_then_kept_conservative(self):
        client = _FakeChat(["I think it is fine.", "Still prose, no JSON."])
        c = AppWorldCritic(client, "m", adaptive=True, retries=1, log=lambda m: None)
        r = c.reflect({}, self.TRAJ, "", None, 1, 3)
        self.assertEqual(client.calls, 2); self.assertEqual(c.parse_failures, 1)
        self.assertFalse(r.parsed); self.assertFalse(r.correct)                 # env signal does NOT decide the verdict
        self.assertTrue(r.refine); self.assertFalse(r.store); self.assertEqual(r.lesson, "")
        self.assertIn("Still prose", r.critique)

    def test_transport_failure_never_stores_and_does_not_burn_rounds(self):
        c = AppWorldCritic(_FakeChat(fail=True), "m", adaptive=True, log=lambda m: None)
        r = c.reflect({}, self.TRAJ, "", None, 1, 3)
        self.assertFalse(r.parsed); self.assertTrue(r.correct); self.assertFalse(r.refine); self.assertFalse(r.store)
        self.assertEqual(r.lesson, ""); self.assertIn("critic call failed", r.critique); self.assertEqual(c.call_failures, 1)
        r2 = c.reflect({}, Trajectory(text="t", completed=False, meta={}), "", None, 1, 3)
        self.assertFalse(r2.correct); self.assertFalse(r2.refine)
        # through the core: admitted (env clean) but nothing written; not completed -> critic_stop in AdaReMo
        from remo import ReMoAgent
        class S:
            def __init__(self, ok): self.ok = ok
            def solve(self, task, memory_text, critique): return Trajectory(text="t", completed=self.ok, meta={})
        rec = ReMoAgent(RemoConfig(mode="adaremo", K=3), S(True), c).run_task("t", 0)
        self.assertEqual((rec["gate"], rec["store_decision"], rec["rounds"][0]["parsed"]), ("round1_clean", "skipped", False))
        rec = ReMoAgent(RemoConfig(mode="adaremo", K=3), S(False), c).run_task("t", 1)
        self.assertEqual((rec["gate"], len(rec["rounds"])), ("critic_stop", 1))


class TestConfig(unittest.TestCase):
    def test_arms(self):
        c = make_config("react", None); self.assertEqual((c.mode, c.K, c.use_memory), ("remo", 1, False))
        c = make_config("refine", 3); self.assertEqual((c.mode, c.K, c.use_memory), ("remo", 3, False))
        c = make_config("memory", None); self.assertEqual((c.mode, c.K, c.use_memory), ("remo", 1, True))
        c = make_config("remo", None); self.assertEqual((c.mode, c.K, c.use_memory), ("remo", 3, True))
        c = make_config("adaremo", 2, redundant_mode="gate"); self.assertTrue(c.adaptive); self.assertEqual(c.K, 2)
        self.assertEqual(c.inject_cap_chars, 60000); self.assertEqual(c.redundant_mode, "gate")
        with self.assertRaises(SystemExit):
            make_config("react", 3)
        with self.assertRaises(SystemExit):
            make_config("refine", 1)

    def test_read_only_memory_wrapper(self):
        inner = RemoPolicy(RemoConfig(mode="adaremo", K=2))
        ro = ReadOnlyMemory(inner)
        from remo import EpisodeState
        st = EpisodeState()
        self.assertEqual(ro.after_round(st, True, Reflection("correct", lesson="L", store=True)), "accept")
        self.assertEqual(ro.gate(st), "round1_clean")
        self.assertEqual(ro.memory_decision(st, 0, ""), "memory_frozen")
        self.assertFalse(ro.frozen); self.assertEqual(ro.state()["store_window"], [])   # nothing recorded


class FakeSolver:
    """Mimics AppWorldSolver's contract: appends a meta per round, returns the transcript."""
    def __init__(self, misc_dir, clean_by_round):
        self.misc_dir, self.clean_by_round, self.round_metas, self.seen = misc_dir, clean_by_round, [], []
    def solve(self, task, memory_text, critique):
        r = len(self.round_metas) + 1
        self.seen.append((task["task_id"], len(memory_text), critique))
        clean = self.clean_by_round[min(r, len(self.clean_by_round)) - 1]
        meta = {"task_id": task["task_id"], "round": r, "instruction": "Do X", "supervisor": {}, "misc_dir": self.misc_dir,
                "output_dir": os.path.dirname(self.misc_dir), "steps": 3 + r, "task_completed": clean, "env_clean": clean,
                "error": "", "no_code_steps": 0, "elapsed_s": 1.5, "prompt_tokens": 100, "completion_tokens": 10,
                "memory_chars": len(memory_text), "critique_chars": len(critique or ""), "last_output": "done",
                "steps_full": [{"step": 1, "reply": "r", "code": "c", "output": "o"}]}
        self.round_metas.append(meta)
        return Trajectory(text="transcript", answer="apis.supervisor.complete_task()", completed=clean, meta=meta)


class ScriptedCritic:
    def __init__(self, *refls): self.refls = refls
    def reflect(self, task, traj, memory_text, prior, r, K):
        return self.refls[min(r, len(self.refls)) - 1]


class TestRecords(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.misc = os.path.join(self.d, "aw_out", "tasks", "t1", "misc")

    def test_records_and_resume(self):
        critic = ScriptedCritic(Reflection("incorrect", critique="fix paging", lesson="L1", refine=True, raw="{1}"),
                                Reflection("correct", lesson="paginate to the end", store=True, raw="{2}"))
        agent = AppWorldAgent(make_config("adaremo", 3), FakeSolver(self.misc, [False, True]), critic, cli_mode="adaremo",
                              playbook=Playbook(prefix="aw"), run_dir=self.d)
        rec = run_task(agent, "t1", 0, self.d, log=lambda m: None)
        # episodes.jsonl line: core record + adapter fields
        eps = read_episodes(self.d)
        self.assertEqual(len(eps), 1); e = eps[0]
        self.assertEqual((e["task_id"], e["task_index"], e["mode"], e["K"]), ("t1", 0, "adaremo", 3))
        self.assertEqual(e["gate"], "cross_round_validated"); self.assertEqual(len(e["rounds"]), 2)
        self.assertEqual(e["rounds"][0]["solver"]["steps"], 4); self.assertFalse(e["rounds"][0]["completed"])
        self.assertEqual(e["store_decision"], "stored"); self.assertEqual(e["entry_id"], "aw-00001")
        self.assertEqual(e["elapsed_s"], 3.0); self.assertFalse(e["memory_readonly"])
        self.assertEqual(agent.playbook.entries[0].text, "paginate to the end")
        # the retry carried the critique into the solver
        self.assertEqual(agent.solver.seen[1][2], "fix paging")
        # trajs/<task_id>.json and misc/remo_rounds.json
        tj = json.load(open(os.path.join(self.d, "trajs", "t1.json")))
        self.assertEqual([r["round"] for r in tj["rounds"]], [1, 2]); self.assertEqual(tj["rounds"][1]["critic_raw"], "{2}")
        self.assertEqual(tj["rounds"][0]["steps_full"][0]["code"], "c")
        rr = json.load(open(os.path.join(self.misc, "remo_rounds.json")))
        self.assertEqual(set(rr) >= {"clean", "curation", "rounds", "store_decision", "memory_frozen"}, True)
        self.assertTrue(rr["clean"]); self.assertEqual(rr["curation"], "cross_round_validated")
        self.assertEqual(rr["rounds"], [{"round": 1, "env_clean": False, "verdict_no_errors": False, "refine": True, "store": False},
                                        {"round": 2, "env_clean": True, "verdict_no_errors": True, "refine": True, "store": True}])
        self.assertEqual(rr["store_decision"], "stored"); self.assertFalse(rr["memory_frozen"])
        # resume: task 0 is skipped, the playbook is reloaded
        agent2 = AppWorldAgent(make_config("adaremo", 3), FakeSolver(self.misc, [True]), critic, cli_mode="adaremo",
                               playbook=Playbook(prefix="aw"), run_dir=self.d)
        self.assertEqual(agent2.done_indices(), {0}); self.assertEqual(len(agent2.playbook), 1)
        self.assertEqual(rec["task_id"], "t1")

    def test_freeze_after_reads_but_never_writes(self):
        critic = ScriptedCritic(Reflection("correct", lesson="L", store=True))
        agent = AppWorldAgent(make_config("remo", 1), FakeSolver(self.misc, [True]), critic, cli_mode="memory",
                              playbook=Playbook(prefix="aw"), run_dir=self.d)
        run_task(agent, "t0", 0, self.d, freeze_after=1, log=lambda m: None)      # index 0 < A: consolidates
        run_task(agent, "t1", 1, self.d, freeze_after=1, log=lambda m: None)      # index 1 >= A: read-only
        eps = read_episodes(self.d)
        self.assertEqual([e["store_decision"] for e in eps], ["stored", "memory_frozen"])
        self.assertEqual(len(agent.playbook), 1); self.assertTrue(agent.memory_readonly)
        self.assertGreater(agent.solver.seen[1][1], 0)                              # memory still injected
        self.assertTrue(json.load(open(os.path.join(self.misc, "remo_rounds.json")))["memory_frozen"])

    def test_no_memory_arm_and_summary(self):
        critic = ScriptedCritic(Reflection("correct", lesson="L", store=True))
        agent = AppWorldAgent(make_config("react", None), FakeSolver(self.misc, [True]), critic, cli_mode="react",
                              playbook=Playbook(prefix="aw"), run_dir=self.d)
        run_task(agent, "t0", 0, self.d, log=lambda m: None)
        self.assertEqual(agent.solver.seen[0][1], 0); self.assertEqual(len(agent.playbook), 0)
        eps = read_episodes(self.d)
        self.assertEqual(eps[0]["store_decision"], "no_memory")
        s = summarize(eps, agent.playbook, {"TGC": 100.0, "SGC": 100.0, "n_evaluated": 1, "per_task_success": {"t0": True},
                                            "errors": {}}, None)
        self.assertEqual((s["n_tasks"], s["TGC"], s["mean_rounds"]), (1, 100.0, 1.0))
        self.assertEqual(s["gate_distribution"], {"round1_clean": 1}); self.assertEqual(s["tgc_by_gate"]["round1_clean"]["rate"], 1.0)
        self.assertEqual(s["memory"]["entries"], 0); self.assertEqual(s["per_task"]["t0"]["success"], True)

    def test_rounds_record_shape(self):
        rec = {"gate": "never_clean", "rounds": [{"round": 1, "completed": False, "verdict": "incorrect", "refine": False, "store": False}],
               "store_decision": "skipped", "task_id": "x", "task_index": 3, "stop_reason": "critic_stop", "entry_id": ""}
        rr = rounds_record(rec, memory_frozen=True)
        self.assertFalse(rr["clean"]); self.assertEqual(rr["curation"], "never_clean"); self.assertTrue(rr["memory_frozen"])
        self.assertEqual(rr["rounds"][0], {"round": 1, "env_clean": False, "verdict_no_errors": False, "refine": False, "store": False})


if __name__ == "__main__":
    unittest.main()
