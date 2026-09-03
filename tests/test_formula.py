"""Formula adapter: parsing, answer extraction, scoring, manifest selection, critic prompt/parsing and the runner
(fake LLM, no model, no network)."""
import json
import os
import tempfile
import unittest

from benchmarks.formula import run_formula as rf
from benchmarks.formula.critic import FormulaCritic, LLMConsolidator, build_critic_prompt
from benchmarks.formula.data import (ANSWER_FORMAT, check_against_manifest, formula_name, load_rows, make_task,
                                     normalize_row, parse_question, row_hash, select_by_manifest, write_rows)
from benchmarks.formula.scoring import is_correct, score_run
from benchmarks.formula.solver import FormulaSolver, build_solver_prompt, extract_answer
from remo import Playbook, RemoConfig, Trajectory
from remo.critic import ADAREMO_FIELDS, REMO_FIELDS

CTX = ('Use formula Operating Margin to answer the question. Answer with a numerical answer with 2 decimal places. '
       'Formula: Operating Margin = (Operating Income / Revenue) × 100. Question:  "For a business with revenue of '
       '$300,000 and Operating Income of $45,000, find the Operating Margin.". Answer:')


def _rows(n):
    return [{"context": CTX.replace("$45,000", f"${i},000"), "target": f"{i}.0"} for i in range(n)]


class FakeLLM:
    """Routes by prompt: critic prompts get a JSON reflection, consolidator prompts one line, solver prompts a
    reply ending in Finish[...]. Optionally fails the first solver call (transport error)."""

    def __init__(self, answer="15.0", critic=None, fail_first_solver=False, consolidated="Use percent for margins."):
        self.answer, self.fail_first_solver, self.consolidated = answer, fail_first_solver, consolidated
        self.critic = critic or {"checks": "ok", "verdict": "no_errors", "critique": "fine", "refine": False, "store": True,
                                 "novelty_reason": "", "lesson": "Multiply margins by 100 when the formula says so."}
        self.prompts = []

    def served_models(self):
        return ["FAKE"]

    def __call__(self, prompt, max_tokens=8192, temperature=0.0):
        self.prompts.append(prompt)
        usage = {"prompt_tokens": 10, "completion_tokens": 5, "finish_reason": "stop"}
        if "reviewing another analyst" in prompt:
            c = self.critic(prompt) if callable(self.critic) else self.critic
            return json.dumps(c), usage
        if "Rewrite it as ONE entry" in prompt:
            return self.consolidated + "\n", usage
        if self.fail_first_solver:
            self.fail_first_solver = False
            raise ConnectionError("boom")
        return f"Margin = 45,000 / 300,000 = 0.15 -> 15%.\nFinish[{self.answer}]", usage


class TestParsing(unittest.TestCase):
    def test_question_and_suffix(self):
        q = parse_question(CTX)
        self.assertTrue(q.startswith("For a business with revenue"))
        self.assertTrue(q.endswith(ANSWER_FORMAT))
        self.assertNotIn('"', q.split(" Your answer")[0])
        self.assertNotIn("Formula:", q)                      # the instruction part is not shown

    def test_curly_quote_and_missing_markers(self):
        q = parse_question('x Question:  "What is 2+2?”. Answer:')
        self.assertTrue(q.startswith("What is 2+2?"))
        self.assertTrue(parse_question("no markers here").startswith("no markers here"))

    def test_formula_name_and_task_has_no_target(self):
        self.assertEqual(formula_name(CTX), "Operating Margin")
        self.assertEqual(formula_name("something else"), "")
        t = make_task(3, {"context": CTX, "target": "15.0"})
        self.assertEqual(set(t), {"task_index", "question", "formula"})

    def test_normalize_row(self):
        self.assertEqual(normalize_row({"input": "c", "output": 15.0}), {"context": "c", "target": "15.0"})
        self.assertEqual(normalize_row({"context": "c", "target": " 0.05 "}), {"context": "c", "target": "0.05"})
        self.assertIsNone(normalize_row({"foo": 1}))


class TestAnswer(unittest.TestCase):
    def test_extract(self):
        self.assertEqual(extract_answer("a\nFinish[15.0]"), "15.0")
        self.assertEqual(extract_answer("Finish[1] then Finish[ \"2,500.5\" ]."), "2,500.5")
        self.assertEqual(extract_answer("Finish[[15]]"), "15")
        self.assertEqual(extract_answer("45,000 / 300,000 = 0.15"), "0.15")     # fallback: last number
        self.assertEqual(extract_answer(""), "")

    def test_prompt_blocks(self):
        p = build_solver_prompt("Q?", "", None)
        self.assertNotIn("Lessons from earlier", p); self.assertNotIn("reviewer", p); self.assertIn("Finish[<number>]", p)
        p = build_solver_prompt("Q?", "[calc-00001] helpful=1 :: L", "wrong scale")
        self.assertIn("[calc-00001]", p); self.assertIn("wrong scale", p)

    def test_solver_failure_spends_round(self):
        s = FormulaSolver(FakeLLM(fail_first_solver=True))
        tr = s.solve({"question": "Q"}, "", None)
        self.assertFalse(tr.completed); self.assertEqual(tr.answer, ""); self.assertIn("boom", tr.meta["error"])
        tr2 = s.solve({"question": "Q"}, "", "fix")
        self.assertTrue(tr2.completed); self.assertEqual(tr2.answer, "15.0")
        self.assertEqual([h["answer"] for h in s.drain()], ["", "15.0"]); self.assertEqual(s.history, [])


class TestScoring(unittest.TestCase):
    def test_is_correct(self):
        self.assertTrue(is_correct("15.00", "15.0")); self.assertTrue(is_correct("1,500", "1500.0"))
        self.assertTrue(is_correct(" 0.05", "0.05")); self.assertFalse(is_correct("0.15", "15.0"))
        self.assertTrue(is_correct("n/a", "n/a")); self.assertFalse(is_correct("", "15.0")); self.assertFalse(is_correct("$15", "15.0"))

    def test_score_run(self):
        d = tempfile.mkdtemp()
        eps = [
            {"task_index": 0, "gate": "round1_clean", "stop_reason": "accepted", "store_decision": "stored", "final_answer": "0.0",
             "rounds": [{"round": 1, "answer": "0.0", "usage": {"solver_calls": 1, "critic_calls": 1, "prompt_tokens": 10, "completion_tokens": 5}}]},
            {"task_index": 1, "gate": "cross_round_validated", "stop_reason": "accepted", "store_decision": "reinforced", "final_answer": "1",
             "rounds": [{"round": 1, "answer": "0.01", "usage": {"solver_calls": 1, "critic_calls": 1, "prompt_tokens": 10, "completion_tokens": 5}},
                        {"round": 2, "answer": "1", "usage": {"solver_calls": 1, "critic_calls": 1, "prompt_tokens": 10, "completion_tokens": 5}}]},
            {"task_index": 2, "gate": "never_clean", "stop_reason": "max_rounds", "store_decision": "skipped", "final_answer": "",
             "memory_readonly": True, "rounds": [{"round": 1, "answer": "", "usage": {"solver_calls": 1, "critic_calls": 0}},
                                                 {"round": 2, "answer": "", "usage": {"solver_calls": 1, "critic_calls": 0}}]},
        ]
        with open(os.path.join(d, "episodes.jsonl"), "w") as f:
            for e in eps:
                f.write(json.dumps(e) + "\n")
        pb = Playbook(prefix="calc"); pb.add("one"); pb.add("two"); pb.save(os.path.join(d, "playbook.txt"))
        res = score_run(d, _rows(4), {"n_tasks": 4, "freeze_after": 2, "mode": "adaremo"})
        self.assertEqual((res["n_scored"], res["complete"]), (3, False))
        self.assertAlmostEqual(res["accuracy"], 2 / 3, places=4); self.assertEqual(res["correct"], 2); self.assertEqual(res["no_answer"], 1)
        self.assertAlmostEqual(res["round1_accuracy"], 1 / 3, places=4)
        self.assertEqual(res["mean_rounds"], round(5 / 3, 4))                 # failed rounds count
        self.assertEqual(res["gates"], {"cross_round_validated": 1, "never_clean": 1, "round1_clean": 1})
        self.assertEqual(res["store_decisions"], {"reinforced": 1, "skipped": 1, "stored": 1})
        self.assertEqual(res["memory"]["entries"], 2); self.assertEqual(res["memory"]["reinforced"], 1)
        self.assertIn("tokens", res["memory"])
        self.assertEqual(res["llm"], {"solver_calls": 5, "critic_calls": 3, "consolidator_calls": 0, "prompt_tokens": 30, "completion_tokens": 15})
        self.assertEqual(res["segments"]["learn"]["n"], 2); self.assertEqual(res["segments"]["frozen"]["accuracy"], 0.0)
        self.assertEqual(res["per_formula"]["Operating Margin"]["n"], 3)


class TestManifest(unittest.TestCase):
    def test_select_order_missing_and_roundtrip(self):
        rows = _rows(5)
        manifest = [row_hash(rows[3]), row_hash(rows[1]), "0" * 64, row_hash(rows[4])]
        sel, missing, unmatched = select_by_manifest(rows + [rows[1]], manifest)     # duplicate source row
        self.assertEqual([r["target"] for r in sel], ["3.0", "1.0", "4.0"])
        self.assertEqual(missing, ["0" * 64]); self.assertEqual(unmatched, 2)
        d = tempfile.mkdtemp(); p = os.path.join(d, "out.jsonl")
        sha = write_rows(sel, p)
        self.assertEqual(len(sha), 64); self.assertEqual(load_rows(p), sel)
        self.assertEqual(check_against_manifest(sel, [row_hash(r) for r in sel])["exact"], True)
        self.assertEqual(check_against_manifest(sel, manifest)["matched_in_order"], 2)
        self.assertEqual(len(row_hash(rows[0])), 64)


class TestCritic(unittest.TestCase):
    def _traj(self, text='reply with {"json": 1} braces\nFinish[15.0]', answer="15.0"):
        return Trajectory(text=text, answer=answer, completed=bool(answer))

    def test_prompt_assembly(self):
        task = {"question": "Why {braces}?"}
        p = build_critic_prompt(task, self._traj(), "[calc-00001] helpful=1 :: L", "old critique", 2, 3, adaptive=True)
        self.assertIn("Why {braces}?", p); self.assertIn('{"json": 1}', p); self.assertIn("attempt 2 of 3", p)
        self.assertIn("CURRENT MEMORY", p); self.assertIn("[calc-00001]", p); self.assertIn(ADAREMO_FIELDS.format(), p)
        p2 = build_critic_prompt(task, self._traj(), "[calc-00001] helpful=1 :: L", None, 1, 3, adaptive=False)
        self.assertNotIn("CURRENT MEMORY", p2); self.assertIn(REMO_FIELDS.format(), p2); self.assertNotIn("attempt 1 of", p2)

    def test_reflect_paths(self):
        llm = FakeLLM()
        c = FormulaCritic(llm, adaptive=True)
        r = c.reflect({"question": "Q"}, self._traj(), "", None, 1, 3)
        self.assertTrue(r.correct and r.parsed and r.store and not r.refine); self.assertTrue(r.lesson)
        # no answer: no model call, errors_found with a retry-worthy critique
        n = len(llm.prompts)
        r = c.reflect({"question": "Q"}, self._traj(text="cut off", answer=""), "", None, 1, 3)
        self.assertFalse(r.correct); self.assertTrue(r.refine); self.assertEqual(len(llm.prompts), n)
        self.assertEqual(c.skipped_no_answer, 1)
        # unparseable output: one retry, then conservative defaults
        c2 = FormulaCritic(lambda prompt, max_tokens, temperature: ("garbage", {"prompt_tokens": 1, "completion_tokens": 1}),
                           adaptive=True)
        r = c2.reflect({"question": "Q"}, self._traj(), "", None, 1, 3)
        self.assertFalse(r.parsed); self.assertFalse(r.correct); self.assertTrue(r.refine); self.assertFalse(r.store)
        self.assertEqual(c2.parse_failures, 1); self.assertEqual(c2.drain()[0]["calls"], 2)
        # transport failure: no_errors, unparsed, no lesson -> nothing is written
        def boom(prompt, max_tokens, temperature): raise TimeoutError("t")
        c3 = FormulaCritic(boom, adaptive=False)
        r = c3.reflect({"question": "Q"}, self._traj(), "", None, 1, 3)
        self.assertTrue(r.correct); self.assertFalse(r.parsed); self.assertEqual(r.lesson, ""); self.assertEqual(c3.call_failures, 1)

    def test_llm_consolidator(self):
        pb = Playbook(prefix="calc")
        cons = LLMConsolidator(FakeLLM(consolidated='"Use percent for margins."'))
        eid = cons.consolidate(pb, "raw lesson about 45,000", None, None)
        self.assertEqual(pb.get(eid).text, "Use percent for margins.")
        cons2 = LLMConsolidator(lambda prompt, max_tokens, temperature: (_ for _ in ()).throw(RuntimeError("x")))
        cons2.consolidate(pb, "raw lesson", None, None)
        self.assertEqual(pb.entries[-1].text, "raw lesson"); self.assertEqual(cons2.failures, 1)
        self.assertEqual([h["fallback"] for h in cons2.drain()], [True])


class TestRunner(unittest.TestCase):
    def test_config_for(self):
        self.assertEqual((rf.config_for("react", 3).K, rf.config_for("react", 3).use_memory), (1, False))
        self.assertEqual((rf.config_for("refine", 3).K, rf.config_for("refine", 3).use_memory), (3, False))
        self.assertEqual((rf.config_for("memory", 3).K, rf.config_for("memory", 3).mode), (1, "remo"))
        self.assertEqual(rf.config_for("remo", 2).mode, "remo"); self.assertTrue(rf.config_for("adaremo", 2).adaptive)
        with self.assertRaises(ValueError):
            rf.config_for("refine", 1)

    def test_agent_freeze_after_and_resume(self):
        d = tempfile.mkdtemp()
        llm = FakeLLM(fail_first_solver=True)
        cfg = RemoConfig(mode="adaremo", K=2)
        agent = rf.FormulaAgent(cfg, FormulaSolver(llm), FormulaCritic(llm, adaptive=True), Playbook(prefix="calc"),
                                LLMConsolidator(llm), run_dir=d, freeze_after=2)
        recs, learn_state = [], None
        for i, r in enumerate(_rows(4)):
            recs.append(agent.run_task(make_task(i, r), i))
            if i == 1:
                learn_state = agent.policy.state()                        # saturation state at the end of the learn phase
        self.assertEqual(len(recs[0]["rounds"]), 2)                      # failed generation spent a round
        self.assertEqual(recs[0]["rounds"][0]["answer"], ""); self.assertEqual(recs[0]["rounds"][1]["answer"], "15.0")
        self.assertEqual(recs[0]["gate"], "cross_round_validated"); self.assertEqual(recs[0]["store_decision"], "stored")
        self.assertEqual(recs[0]["consolidator_usage"]["calls"], 1)
        self.assertEqual(recs[1]["store_decision"], "stored"); self.assertEqual(len(agent.playbook), 2)
        self.assertEqual(agent.playbook.entries[0].text, "Use percent for margins.")
        for r in recs[2:]:
            self.assertTrue(r["memory_readonly"]); self.assertEqual(r["store_decision"], "skipped_readonly")
            self.assertEqual(r["entry_id"], ""); self.assertGreater(r["memory_chars_at_start"], 0)
        self.assertEqual(len(agent.playbook), 2); self.assertIsInstance(agent.playbook, Playbook)
        self.assertEqual(recs[2]["formula"], "Operating Margin")
        self.assertEqual(learn_state["store_window"], [True, True])
        self.assertEqual(agent.policy.state(), learn_state)             # read-only phase: no saturation bookkeeping
        with open(os.path.join(d, "policy_state.json")) as f:
            self.assertEqual(json.load(f), learn_state)
        with open(os.path.join(d, "trajs.jsonl")) as f:
            trajs = [json.loads(l) for l in f]
        self.assertEqual([len(t["rounds"]) for t in trajs], [2, 1, 1, 1]); self.assertIn("Finish[15.0]", trajs[1]["rounds"][0]["text"])
        # resume: everything done, playbook restored with the calc prefix
        agent2 = rf.FormulaAgent(cfg, FormulaSolver(llm), FormulaCritic(llm, adaptive=True), Playbook(prefix="calc"), run_dir=d)
        self.assertEqual(agent2.done_indices(), {0, 1, 2, 3}); self.assertEqual(agent2.playbook.add("x"), "calc-00003")
        fresh = rf.FormulaAgent(cfg, FormulaSolver(llm), FormulaCritic(llm, adaptive=True), Playbook(prefix="calc"))
        self.assertEqual(fresh.playbook.add("y"), "calc-00001")             # empty playbook is falsy in ReMoAgent.__init__

    def test_no_memory_arm_injects_nothing(self):
        d = tempfile.mkdtemp(); llm = FakeLLM()
        agent = rf.FormulaAgent(rf.config_for("refine", 2), FormulaSolver(llm), FormulaCritic(llm, adaptive=False), Playbook(prefix="calc"),
                                run_dir=d)
        rec = agent.run_task(make_task(0, _rows(1)[0]), 0)
        self.assertEqual(rec["store_decision"], "no_memory"); self.assertEqual(len(agent.playbook), 0)
        self.assertNotIn("Lessons from earlier", llm.prompts[0])

    def test_main_end_to_end_with_fake_llm(self):
        d = tempfile.mkdtemp(); data = os.path.join(d, "data.jsonl"); out = os.path.join(d, "run")
        write_rows(_rows(3), data)
        real = rf.ChatLLM
        rf.ChatLLM = lambda base_url, model, timeout: FakeLLM(answer="1.0")
        try:
            argv = ["--mode", "remo", "--K", "2", "--data", data, "--out", out, "--limit", "3", "--model", "FAKE"]
            self.assertEqual(rf.main(argv), 0)
            res = json.load(open(os.path.join(out, "final_results.json")))
            self.assertEqual((res["n_scored"], res["correct"], res["accuracy"]), (3, 1, round(1 / 3, 4)))
            self.assertEqual(res["mode"], "remo"); self.assertEqual(res["gates"], {"round1_clean": 3}); self.assertEqual(res["memory"]["entries"], 3)
            self.assertEqual(rf.main(argv), 0)                                       # resume: nothing to do
            self.assertEqual(sum(1 for _ in open(os.path.join(out, "episodes.jsonl"))), 3)
            with self.assertRaises(SystemExit):
                rf.main(argv[:3] + ["3"] + argv[4:])                                  # K changed -> refused
        finally:
            rf.ChatLLM = real


if __name__ == "__main__":
    unittest.main()
