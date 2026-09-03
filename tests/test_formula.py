"""Formula adapter: question parsing, prompt files, answer extraction, critic parsing, the curator consolidator,
lesson text, scoring, manifest selection and the runner (fake LLM, no model, no network)."""
import json
import os
import tempfile
import unittest

from benchmarks.formula import run_formula as rf
from benchmarks.formula.consolidator import AppendConsolidator, CuratorConsolidator, lesson_text
from benchmarks.formula.critic import EMPTY_PLAYBOOK, NO_PRIOR, FormulaCritic, build_critic_prompt
from benchmarks.formula.data import (ANSWER_FORMAT, check_against_manifest, formula_name, load_rows, make_task,
                                     normalize_row, parse_question, row_hash, select_by_manifest, write_rows)
from benchmarks.formula.scoring import is_correct, score_run
from benchmarks.formula.solver import (FIRST_REFLECTION, NO_ANSWER, RETRY_REFLECTION, SOLVER_PROMPT, FormulaSolver,
                                       build_solver_prompt, extract_answer)
from remo import Reflection, RemoConfig, SectionedPlaybook, Trajectory
from remo.policy import EpisodeState, RoundRecord

CTX = ('Use formula Operating Margin to answer the question. Answer with a numerical answer with 2 decimal places. '
       'Formula: Operating Margin = (Operating Income / Revenue) × 100. Question:  "For a business with revenue of '
       '$300,000 and Operating Income of $45,000, find the Operating Margin.". Answer:')
CURATOR_REPLY = json.dumps({"reasoning": "r", "operations": [
    {"type": "ADD", "section": "formulas_and_calculations", "content": "Operating margin = operating income / revenue x 100."},
    {"type": "UPDATE", "bullet_id": "calc-00001"}]})


def _rows(n):
    return [{"context": CTX.replace("$45,000", f"${i},000"), "target": f"{i}.0"} for i in range(n)]


def _task(i=0):
    return make_task(i, _rows(i + 1)[i])


class FakeLLM:
    """Routes by prompt: critic prompts get a JSON reflection, curator prompts a curator reply, solver prompts the
    JSON reply the solver prompt asks for. Optionally fails the first solver call (transport error)."""

    def __init__(self, answer="15.0", critic=None, fail_first_solver=False, curator=CURATOR_REPLY):
        self.answer, self.fail_first_solver, self.curator = answer, fail_first_solver, curator
        self.critic = critic or {"checks": "ok", "verdict": "no_errors", "critique": "fine", "refine": False, "store": True,
                                 "novelty_reason": "not covered", "key_insight": "Multiply margins by 100."}
        self.prompts = []

    def served_models(self):
        return ["FAKE"]

    def __call__(self, prompt, max_tokens):
        self.prompts.append(prompt)
        usage = {"prompt_tokens": 10, "completion_tokens": 5, "finish_reason": "stop"}
        if "financial-reasoning reviewer" in prompt:
            c = self.critic(prompt) if callable(self.critic) else self.critic
            return json.dumps(c), usage
        if "master curator of knowledge" in prompt:
            return self.curator, usage
        if self.fail_first_solver:
            self.fail_first_solver = False
            raise ConnectionError("boom")
        return json.dumps({"reasoning": "45,000 / 300,000 = 0.15", "bullet_ids": [], "final_answer": self.answer}), usage


def _traj(text='{"reasoning": "x", "final_answer": "15.0"}', answer="15.0"):
    return Trajectory(text=text, answer=answer)


class TestParsing(unittest.TestCase):
    def test_question_is_the_runs_question(self):
        q = parse_question(CTX)
        self.assertTrue(q.startswith("For a business with revenue"))
        self.assertTrue(q.endswith(ANSWER_FORMAT)); self.assertTrue(q.endswith("5000000.0. "))
        self.assertNotIn("Formula:", q); self.assertNotIn('"', q.split(" Your answer")[0])

    def test_curly_quotes_kept_and_missing_markers(self):
        self.assertTrue(parse_question('x Question:  “What is 2+2?”. Answer:').startswith("“What is 2+2?”"))
        self.assertEqual(parse_question("no markers here"), "no markers here")
        self.assertEqual(parse_question("Question: only one marker"), "Question: only one marker")

    def test_task_fields(self):
        self.assertEqual(formula_name(CTX), "Operating Margin"); self.assertEqual(formula_name("something else"), "")
        t = make_task(3, {"context": CTX, "target": "15.0"})
        self.assertEqual(set(t), {"task_index", "question", "context", "formula"}); self.assertEqual(t["context"], "")

    def test_normalize_row(self):
        self.assertEqual(normalize_row({"input": "c", "output": 15.0}), {"context": "c", "target": "15.0"})
        self.assertEqual(normalize_row({"context": "c", "target": " 0.05 "}), {"context": "c", "target": "0.05"})
        self.assertIsNone(normalize_row({"foo": 1}))


class TestPrompts(unittest.TestCase):
    def test_solver_prompt(self):
        p = build_solver_prompt(_task(), "PLAYBOOK", None)
        self.assertIn("**Playbook:**\nPLAYBOOK\n\n**Reflection:**\n(empty)\n\n**Question:**\nFor a business", p)
        self.assertTrue(p.endswith('"final_answer": "[Your concise final answer here]"\n}\n\n---\n'))
        self.assertIn("**Context:**\n\n\n**Answer in this exact JSON format:**", p)
        p2 = build_solver_prompt(_task(), "", 'use {braces} and "quotes"')
        self.assertIn("**Reflection:**\n" + RETRY_REFLECTION.format(critique='use {braces} and "quotes"'), p2)
        self.assertEqual(FIRST_REFLECTION, "(empty)"); self.assertEqual(SOLVER_PROMPT.count("{}"), 4)

    def test_critic_prompts(self):
        t = _task(); tr = _traj()
        p = build_critic_prompt(t, tr, "PB", None, 1, 3, adaptive=False)
        self.assertIn("**Question:**\n" + t["question"] + "\n\n**Model's Reasoning Trace:**\n" + tr.text, p)
        self.assertTrue(p.endswith("**Model's Final Answer:**\n15.0\n")); self.assertNotIn("Round", p)
        p = build_critic_prompt(t, tr, "PB", None, 1, 3, adaptive=True)
        self.assertIn(f"**Round:** 1 of at most 3\n**Previous reviewer critique (if this is a retry):**\n{NO_PRIOR}\n", p)
        self.assertIn("**Current playbook (memory the solver already has):**\nPB\n", p)
        p = build_critic_prompt(t, tr, "", "old {critique}", 2, 3, adaptive=True)
        self.assertIn("**Round:** 2 of at most 3\n**Previous reviewer critique (if this is a retry):**\nold {critique}\n", p)
        self.assertIn(f"already has):**\n{EMPTY_PLAYBOOK}\n", p)

    def test_curator_prompt(self):
        pb = SectionedPlaybook.from_skeleton("counts")
        cons = CuratorConsolidator(FakeLLM(), n_tasks=200, adaptive=True)
        st = EpisodeState(admitted=True, stop_reason="accepted")
        st.rounds.append(RoundRecord(1, True, Reflection("correct", critique="c", lesson="k", novelty_reason="n")))
        p = cons.build_prompt(pb, st, _task(4))
        self.assertIn("- Total token budget: 80000 tokens\n- Training progress: Sample 5 out of 200\n", p)
        self.assertIn("**Current Playbook Stats:**\n" + json.dumps(pb.stats(), indent=2) + "\n", p)
        self.assertIn("**Recent Reflection:**\n" + lesson_text(st, True) + "\n\n**Current Playbook:**\n" + pb.text + "\n", p)
        self.assertIn("**Question Context:**\n" + _task(4)["question"] + "\n", p)


class TestAnswer(unittest.TestCase):
    def test_extract_branches(self):
        self.assertEqual(extract_answer('{"reasoning": "r", "final_answer": "15.0"}'), "15.0")
        self.assertEqual(extract_answer('{"reasoning": "r", "final_answer": 15}'), "15")
        self.assertEqual(extract_answer('{"reasoning": "r"}'), NO_ANSWER)                  # JSON without the key
        self.assertEqual(extract_answer("42"), NO_ANSWER)                                    # JSON scalar -> fallbacks
        self.assertEqual(extract_answer("a Finish[1] b Finish[ 2,500.5 ]"), " 2,500.5 ")
        self.assertEqual(extract_answer('text {"final_answer": "0.15"} trailing'), "0.15")
        self.assertEqual(extract_answer("text {'final_answer': '7'}"), "7")
        self.assertEqual(extract_answer('text {"final_answer": 12.5}'), "12.5")
        self.assertEqual(extract_answer("The final answer is $\\boxed{3.5}$."), "3.5")
        self.assertEqual(extract_answer("The final answer is: $1,200"), "1,200")
        self.assertEqual(extract_answer("nothing here"), NO_ANSWER)
        self.assertFalse(is_correct(NO_ANSWER, "15.0"))

    def test_solver_failure_stops(self):
        s = FormulaSolver(FakeLLM(fail_first_solver=True))
        tr = s.solve(_task(), "", None)
        self.assertTrue(tr.failed); self.assertFalse(tr.completed); self.assertIn("boom", tr.meta["error"])
        tr2 = s.solve(_task(), "", "fix")
        self.assertTrue(tr2.completed); self.assertEqual(tr2.answer, "15.0"); self.assertIn('"final_answer": "15.0"', tr2.text)
        h = s.drain()
        self.assertEqual([x["answer"] for x in h], ["", "15.0"]); self.assertEqual(h[1]["reflection"], RETRY_REFLECTION.format(critique="fix"))


class TestCritic(unittest.TestCase):
    def test_parsing_and_fallbacks(self):
        c = FormulaCritic(FakeLLM(), adaptive=True)
        r = c.reflect(_task(), _traj(), "PB", None, 1, 3)
        self.assertTrue(r.correct and r.parsed and r.store and not r.refine)
        self.assertEqual((r.lesson, r.novelty_reason, r.cited_id), ("Multiply margins by 100.", "not covered", ""))
        replies = iter(['no json but "no_errors" quoted', "garbage", '{"verdict": "maybe", "critique": "x", "cited_id": "calc-00001"}',
                        'prose {"verdict": "errors_found", "critique": "bad", "refine": "yes"} tail'])
        c2 = FormulaCritic(lambda prompt, max_tokens: (next(replies), {}), adaptive=True)
        r = c2.reflect(_task(), _traj(), "", None, 1, 3)
        self.assertTrue(r.correct); self.assertFalse(r.parsed); self.assertFalse(r.store); self.assertTrue(r.refine)
        r = c2.reflect(_task(), _traj(), "", None, 1, 3)
        self.assertFalse(r.correct); self.assertEqual(r.critique, "garbage"); self.assertEqual(r.lesson, "")
        r = c2.reflect(_task(), _traj(), "", None, 1, 3)
        self.assertFalse(r.correct); self.assertEqual(r.cited_id, "")               # ids come from novelty_reason only
        r = c2.reflect(_task(), _traj(), "", None, 1, 3)
        self.assertFalse(r.correct); self.assertTrue(r.refine); self.assertTrue(r.parsed)
        self.assertEqual(c2.parse_failures, 2)

    def test_transport_failure(self):
        def boom(prompt, max_tokens): raise TimeoutError("t")
        c = FormulaCritic(boom, adaptive=False)
        r = c.reflect(_task(), _traj(), "", None, 1, 3)
        self.assertTrue(r.failed); self.assertEqual(r.verdict, "none"); self.assertIn("TimeoutError", r.critique)
        self.assertEqual(c.call_failures, 1); self.assertEqual(c.drain()[0]["calls"], 1)


class TestConsolidator(unittest.TestCase):
    def _episode(self, n, lesson="k{r}"):
        st = EpisodeState(admitted=True, stop_reason="accepted")
        for r in range(1, n + 1):
            st.rounds.append(RoundRecord(r, True, Reflection("incorrect" if r < n else "correct", critique=f"c{r}",
                                                             lesson=lesson.format(r=r), novelty_reason=f"n{r}")))
        return st

    def test_lesson_text(self):
        self.assertEqual(lesson_text(self._episode(1), False),
                         "[validated: round1_clean] The final answer passed independent review. "
                         "Reviewer assessment: c1\nKey reusable insight: k1")
        self.assertEqual(lesson_text(self._episode(3), False),
                         "[validated: cross_round_validated] The final answer passed independent review after 3 rounds "
                         "of refinement. Reviewer assessment: c3\nKey reusable insight: k3")
        self.assertEqual(lesson_text(self._episode(1), True),
                         "[validated: round1_clean] The final answer passed independent review. "
                         "Reviewer assessment: c1\nKey reusable insight: k1\nWhy this is new to the playbook: n1")
        self.assertTrue(lesson_text(self._episode(2), True).startswith("[validated: cross_round_validated] The final answer "
                                                                       "passed independent review after 2 rounds of refinement. "))

    def test_curator_adds_and_fails(self):
        pb = SectionedPlaybook.from_skeleton("counts")
        cons = CuratorConsolidator(FakeLLM(), n_tasks=3, adaptive=False)
        self.assertEqual(cons.consolidate(pb, self._episode(1), _task(), None), "calc-00001")
        self.assertIn("## FORMULAS & CALCULATIONS\n\n[calc-00001] helpful=0 harmful=0 :: Operating margin", pb.text)
        self.assertEqual(cons.drain()[0]["outcome"], "added")
        cons = CuratorConsolidator(FakeLLM(curator='{"reasoning": "r", "operations": []}'), n_tasks=3, adaptive=False)
        self.assertEqual(cons.consolidate(pb, self._episode(1), _task(), None), "")
        self.assertEqual(cons.drain()[0]["outcome"], "no_ops")
        before = pb.text
        cons = CuratorConsolidator(FakeLLM(curator="not json"), n_tasks=3, adaptive=False)
        self.assertEqual(cons.consolidate(pb, self._episode(1), _task(), None), "")
        self.assertEqual((cons.drain()[0]["outcome"], cons.failures, pb.text), ("parse_error", 1, before))
        def boom(prompt, max_tokens): raise ConnectionError("x")
        cons = CuratorConsolidator(boom, n_tasks=3, adaptive=False)
        self.assertEqual(cons.consolidate(pb, self._episode(1), _task(), None), "")
        self.assertEqual((cons.drain()[0]["outcome"], pb.text), ("llm_error", before))
        broken = SectionedPlaybook("stray line before any header\n## OTHERS", "counts")
        cons = CuratorConsolidator(FakeLLM(), n_tasks=3, adaptive=False)
        self.assertEqual(cons.consolidate(broken, self._episode(1), _task(), None), "")
        self.assertEqual(cons.drain()[0]["outcome"], "apply_error")

    def test_append(self):
        pb = SectionedPlaybook.from_skeleton("counts")
        self.assertEqual(AppendConsolidator().consolidate(pb, self._episode(1), None, None), "misc-00001")
        self.assertTrue(pb.text.endswith("## OTHERS\n[misc-00001] helpful=0 harmful=0 :: k1"))
        self.assertEqual(AppendConsolidator().consolidate(pb, self._episode(1, lesson=""), None, None), "")


class TestScoring(unittest.TestCase):
    def test_is_correct(self):
        self.assertTrue(is_correct("15.00", "15.0")); self.assertTrue(is_correct("1,500", "1500.0"))
        self.assertTrue(is_correct(" 0.05", "0.05")); self.assertFalse(is_correct("0.15", "15.0"))
        self.assertTrue(is_correct("n/a", "n/a")); self.assertFalse(is_correct("", "15.0")); self.assertFalse(is_correct("$15", "15.0"))

    def test_score_run(self):
        d = tempfile.mkdtemp()
        eps = [
            {"task_index": 0, "gate": "round1_clean", "stop_reason": "accepted", "store_decision": "stored", "final_answer": "0.0",
             "rounds": [{"round": 1, "answer": "0.0", "usage": {"solver_calls": 1, "critic_calls": 1, "prompt_tokens": 10, "completion_tokens": 5}}],
             "consolidator": {"calls": 1, "prompt_tokens": 7, "completion_tokens": 3, "outcome": "added"}},
            {"task_index": 1, "gate": "cross_round_validated", "stop_reason": "accepted", "store_decision": "reinforced", "final_answer": "1",
             "rounds": [{"round": 1, "answer": "0.01", "usage": {"solver_calls": 1, "critic_calls": 1, "prompt_tokens": 10, "completion_tokens": 5}},
                        {"round": 2, "answer": "1", "usage": {"solver_calls": 1, "critic_calls": 1, "prompt_tokens": 10, "completion_tokens": 5}}]},
            {"task_index": 2, "gate": "never_clean", "stop_reason": "solver_error", "store_decision": "skipped", "final_answer": NO_ANSWER,
             "memory_readonly": True, "rounds": [{"round": 1, "answer": NO_ANSWER, "usage": {"solver_calls": 1, "critic_calls": 1}},
                                                 {"round": 2, "answer": "", "usage": {"solver_calls": 1, "critic_calls": 0}}]},
        ]
        with open(os.path.join(d, "episodes.jsonl"), "w") as f:
            for e in eps:
                f.write(json.dumps(e) + "\n")
        pb = SectionedPlaybook.from_skeleton("counts")
        pb.apply_add_ops([{"type": "ADD", "section": "others", "content": "one"}, {"type": "ADD", "section": "others", "content": "two"}])
        pb.save(os.path.join(d, "playbook.txt"))
        res = score_run(d, _rows(4), {"n_tasks": 4, "freeze_after": 2, "mode": "adaremo"})
        self.assertEqual((res["n_scored"], res["complete"]), (3, False))
        self.assertAlmostEqual(res["accuracy"], 2 / 3, places=4); self.assertEqual(res["correct"], 2); self.assertEqual(res["no_answer"], 0)   # the sentinel is an answer (wrong), as in the runs
        self.assertAlmostEqual(res["round1_accuracy"], 1 / 3, places=4)
        self.assertEqual(res["mean_rounds"], round(5 / 3, 4))                 # failed rounds count
        self.assertEqual(res["gates"], {"cross_round_validated": 1, "never_clean": 1, "round1_clean": 1})
        self.assertEqual(res["store_decisions"], {"reinforced": 1, "skipped": 1, "stored": 1})
        self.assertEqual(res["memory"]["entries"], 2); self.assertEqual(res["memory"]["reinforced"], 1)
        self.assertEqual(res["memory"]["chars"], len(pb.text)); self.assertIn("tokens", res["memory"])
        self.assertEqual(res["llm"], {"solver_calls": 5, "critic_calls": 4, "consolidator_calls": 1, "prompt_tokens": 37, "completion_tokens": 18})
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


class TestRunner(unittest.TestCase):
    def _agent(self, llm, cfg, d, consolidator=None, freeze_after=None):
        return rf.FormulaAgent(cfg, FormulaSolver(llm), FormulaCritic(llm, adaptive=cfg.adaptive), SectionedPlaybook.from_skeleton("counts"),
                               consolidator, run_dir=d, freeze_after=freeze_after)

    def test_config_for(self):
        self.assertEqual((rf.config_for("react", 3).K, rf.config_for("react", 3).use_memory), (1, False))
        self.assertEqual((rf.config_for("refine", 3).K, rf.config_for("refine", 3).use_memory), (3, False))
        self.assertEqual((rf.config_for("memory", 3).K, rf.config_for("memory", 3).mode), (1, "remo"))
        self.assertEqual(rf.config_for("remo", 2).mode, "remo"); self.assertTrue(rf.config_for("adaremo", 2).adaptive)
        with self.assertRaises(ValueError):
            rf.config_for("refine", 1)

    def test_freeze_after_and_resume(self):
        d = tempfile.mkdtemp()
        llm = FakeLLM()
        cfg = RemoConfig(mode="adaremo", K=2)
        agent = self._agent(llm, cfg, d, CuratorConsolidator(llm, 4, True), freeze_after=2)
        recs, learn_state = [], None
        for i, r in enumerate(_rows(4)):
            recs.append(agent.run_task(make_task(i, r), i))
            if i == 1:
                learn_state = agent.policy.state()                        # saturation state at the end of the learn phase
        self.assertEqual((recs[0]["gate"], recs[0]["store_decision"], recs[0]["entry_id"]), ("round1_clean", "stored", "calc-00001"))
        self.assertEqual(recs[0]["consolidator"]["outcome"], "added"); self.assertEqual(recs[0]["final_answer"], "15.0")
        self.assertEqual(recs[1]["entry_id"], "calc-00002"); self.assertEqual(len(agent.playbook), 2)
        self.assertIn("[calc-00002] helpful=0 harmful=0 :: Operating margin", agent.playbook.text)
        for r in recs[2:]:
            self.assertTrue(r["memory_readonly"]); self.assertEqual(r["store_decision"], "skipped_readonly")
            self.assertEqual(r["entry_id"], ""); self.assertGreater(r["memory_chars_at_start"], 0)
        self.assertEqual(len(agent.playbook), 2); self.assertIsInstance(agent.playbook, SectionedPlaybook)
        self.assertEqual(len(llm.prompts), 4 * 2 + 2)                       # no curator call in the read-only phase
        self.assertEqual(recs[2]["formula"], "Operating Margin")
        self.assertEqual(learn_state["store_window"], [True, True])
        self.assertEqual(agent.policy.state(), learn_state)             # read-only phase: no saturation bookkeeping
        with open(os.path.join(d, "policy_state.json")) as f:
            self.assertEqual(json.load(f), learn_state)
        with open(os.path.join(d, "trajs.jsonl")) as f:
            trajs = [json.loads(l) for l in f]
        self.assertEqual([len(t["rounds"]) for t in trajs], [1, 1, 1, 1]); self.assertEqual(trajs[1]["rounds"][0]["reflection"], "(empty)")
        agent2 = self._agent(llm, cfg, d)
        self.assertEqual(agent2.done_indices(), {0, 1, 2, 3}); self.assertEqual(agent2.playbook.text, agent.playbook.text)
        self.assertEqual(agent2.playbook.next_id, 3)

    def test_reinforce_retry_and_curator_error(self):
        d = tempfile.mkdtemp()
        calls = {"n": 0}
        def critic(prompt):
            calls["n"] += 1
            if calls["n"] == 1:      # first attempt of task 0 is rejected with an actionable fix
                return {"verdict": "errors_found", "critique": "scale {wrong}", "refine": True, "store": False, "key_insight": "k"}
            if calls["n"] == 2:
                return {"verdict": "no_errors", "critique": "ok", "refine": False, "store": True, "novelty_reason": "new", "key_insight": "k"}
            return {"verdict": "no_errors", "critique": "ok", "refine": False, "store": False,
                    "novelty_reason": "covered by calc-00001 and calc-99999", "key_insight": "k"}
        llm = FakeLLM(critic=critic)
        agent = self._agent(llm, RemoConfig(mode="adaremo", K=3), d, CuratorConsolidator(llm, 3, True))
        rec = agent.run_task(_task(0), 0)
        self.assertEqual((rec["gate"], len(rec["rounds"]), rec["store_decision"], rec["entry_id"]), ("cross_round_validated", 2, "stored", "calc-00001"))
        self.assertIn(RETRY_REFLECTION.format(critique="scale {wrong}"), llm.prompts[2])          # solver retry
        self.assertIn("**Previous reviewer critique (if this is a retry):**\nscale {wrong}\n", llm.prompts[3])
        self.assertIn("after 2 rounds of refinement", llm.prompts[4])                                # curator lesson
        rec = agent.run_task(_task(1), 1)
        self.assertEqual((rec["store_decision"], rec["entry_id"]), ("reinforced", "calc-00001"))
        self.assertIn("[calc-00001] helpful=1 harmful=0", agent.playbook.text)
        llm.curator = "garbage"
        agent2 = self._agent(llm, RemoConfig(mode="remo", K=1), d, CuratorConsolidator(llm, 3, False))
        rec = agent2.run_task(_task(2), 2)
        self.assertEqual((rec["store_decision"], rec["entry_id"], rec["consolidator"]["outcome"]), ("curator_error", "", "parse_error"))
        self.assertEqual(len(agent2.playbook), 1)

    def test_error_rounds(self):
        d = tempfile.mkdtemp()
        llm = FakeLLM(fail_first_solver=True)
        agent = self._agent(llm, RemoConfig(mode="remo", K=2), d, CuratorConsolidator(llm, 1, False))
        rec = agent.run_task(_task(0), 0)
        self.assertEqual((rec["stop_reason"], rec["gate"], rec["store_decision"], rec["final_answer"]), ("solver_error", "never_clean", "skipped", ""))
        self.assertEqual(len(rec["rounds"]), 1); self.assertEqual(rec["rounds"][0]["answer"], ""); self.assertIn("boom", rec["rounds"][0]["solver"]["error"])
        n = 0
        def flaky(prompt, max_tokens):
            nonlocal n
            n += 1
            if "financial-reasoning reviewer" in prompt:
                if n == 2:
                    return json.dumps({"verdict": "errors_found", "critique": "no"}), {}
                raise TimeoutError("critic down")
            return json.dumps({"final_answer": str(n)}), {}
        agent = self._agent(flaky, RemoConfig(mode="remo", K=3), d, CuratorConsolidator(flaky, 1, False))
        rec = agent.run_task(_task(0), 0)
        self.assertEqual((rec["stop_reason"], len(rec["rounds"]), rec["final_answer"]), ("critic_error", 2, "1"))   # last round with a critic

    def test_no_memory_arm_injects_nothing(self):
        d = tempfile.mkdtemp(); llm = FakeLLM()
        agent = self._agent(llm, rf.config_for("refine", 2), d)
        rec = agent.run_task(_task(0), 0)
        self.assertEqual(rec["store_decision"], "no_memory"); self.assertEqual(len(agent.playbook), 0)
        self.assertIn("**Playbook:**\n\n\n**Reflection:**", llm.prompts[0])

    def test_main_end_to_end_with_fake_llm(self):
        d = tempfile.mkdtemp(); data = os.path.join(d, "data.jsonl"); out = os.path.join(d, "run")
        write_rows(_rows(3), data)
        real = rf.ChatLLM
        rf.ChatLLM = lambda base_url, model, timeout: FakeLLM(answer="1.0")
        try:
            argv = ["--mode", "remo", "--K", "2", "--data", data, "--out", out, "--limit", "3", "--model", "FAKE"]
            self.assertEqual(rf.main(argv), 0)
            with open(os.path.join(out, "final_results.json")) as f:
                res = json.load(f)
            self.assertEqual((res["n_scored"], res["correct"], res["accuracy"]), (3, 1, round(1 / 3, 4)))
            self.assertEqual(res["mode"], "remo"); self.assertEqual(res["gates"], {"round1_clean": 3})
            self.assertEqual((res["memory"]["entries"], res["store_decisions"], res["llm"]["consolidator_calls"]), (3, {"stored": 3}, 3))
            self.assertEqual(rf.main(argv), 0)                                       # resume: nothing to do
            with open(os.path.join(out, "episodes.jsonl")) as f:
                self.assertEqual(sum(1 for _ in f), 3)
            with self.assertRaises(SystemExit):
                rf.main(argv[:3] + ["3"] + argv[4:])                                  # K changed -> refused
        finally:
            rf.ChatLLM = real


if __name__ == "__main__":
    unittest.main()
