"""Unit tests of Algorithms 1/2 semantics with fake solver/critic (no LLM)."""
import json
import os
import tempfile
import unittest

from remo import Playbook, Reflection, ReMoAgent, RemoConfig, SectionedPlaybook, Trajectory
from remo.critic import extract_json_balanced, parse_reflection


class FakeSolver:
    def __init__(self, completed=True, fail=None):     # fail: "raise" | "flag" | None
        self.completed, self.fail, self.calls = completed, fail, 0
    def solve(self, task, memory_text, critique):
        self.calls += 1
        if self.fail == "raise":
            raise ConnectionError("boom")
        if self.fail == "flag":
            return Trajectory(text="", completed=False, failed=True, meta={"error": "timeout"})
        return Trajectory(text=f"attempt for {task}", answer="42", completed=self.completed)


class ScriptedCritic:
    """Returns the scripted Reflection for round r (1-based); repeats the last one after."""
    def __init__(self, *refls): self.refls, self.calls = refls, 0
    def reflect(self, task, traj, memory_text, prior, r, K):
        self.calls += 1
        return self.refls[min(r, len(self.refls)) - 1]


OK = lambda lesson="L", store=True, cited="": Reflection("correct", lesson=lesson, store=store, cited_id=cited, novelty_reason="")
BAD = lambda refine=True: Reflection("incorrect", critique="fix X", refine=refine)


class TestPolicy(unittest.TestCase):
    def test_remo_retries_until_budget_and_never_clean(self):
        a = ReMoAgent(RemoConfig(mode="remo", K=3), FakeSolver(), ScriptedCritic(BAD()))
        rec = a.run_task("t", 0)
        self.assertEqual(len(rec["rounds"]), 3)
        self.assertEqual(rec["gate"], "never_clean"); self.assertEqual(rec["store_decision"], "skipped")
        self.assertEqual(len(a.playbook), 0)

    def test_remo_ignores_refine_false(self):      # Alg. 1 has no refine gate
        a = ReMoAgent(RemoConfig(mode="remo", K=3), FakeSolver(), ScriptedCritic(BAD(refine=False)))
        self.assertEqual(len(a.run_task("t", 0)["rounds"]), 3)

    def test_adaremo_critic_stop(self):
        a = ReMoAgent(RemoConfig(mode="adaremo", K=3), FakeSolver(), ScriptedCritic(BAD(refine=False)))
        rec = a.run_task("t", 0)
        self.assertEqual(len(rec["rounds"]), 1); self.assertEqual(rec["gate"], "critic_stop")

    def test_cross_round_validated_stores_the_flipping_lesson(self):
        a = ReMoAgent(RemoConfig(mode="remo", K=3), FakeSolver(),
                      ScriptedCritic(Reflection("incorrect", critique="c", lesson="from failure"), OK(lesson="")))
        rec = a.run_task("t", 0)
        self.assertEqual(rec["gate"], "cross_round_validated")
        self.assertEqual(a.playbook.entries[0].text, "from failure")   # last non-empty lesson

    def test_completed_signal_blocks_admission(self):
        a = ReMoAgent(RemoConfig(mode="remo", K=2), FakeSolver(completed=False), ScriptedCritic(OK()))
        rec = a.run_task("t", 0)
        self.assertFalse(rec["rounds"][0]["completed"]); self.assertEqual(rec["gate"], "never_clean")

    def test_solver_failure_stops_without_critic(self):
        for fail in ("raise", "flag"):
            s, c = FakeSolver(fail=fail), ScriptedCritic(OK())
            a = ReMoAgent(RemoConfig(mode="remo", K=3), s, c)
            rec = a.run_task("t", 0)
            self.assertEqual((s.calls, c.calls, len(rec["rounds"])), (1, 0, 1))
            self.assertEqual((rec["stop_reason"], rec["gate"], rec["store_decision"]), ("solver_error", "never_clean", "skipped"))
            self.assertFalse(rec["rounds"][0]["completed"]); self.assertEqual(rec["rounds"][0]["verdict"], "none")
            self.assertFalse(rec["final_completed"]); self.assertEqual(len(a.playbook), 0)
        self.assertIn("boom", ReMoAgent(RemoConfig(mode="remo", K=1), FakeSolver(fail="raise"), ScriptedCritic(OK()))
                      .run_task("t", 0)["rounds"][0]["critique"])

    def test_critic_failure_stops_unadmitted(self):
        a = ReMoAgent(RemoConfig(mode="adaremo", K=3), FakeSolver(), ScriptedCritic(Reflection("correct", failed=True)))
        rec = a.run_task("t", 0)
        self.assertEqual((len(rec["rounds"]), rec["stop_reason"], rec["gate"], rec["store_decision"]), (1, "critic_error", "never_clean", "skipped"))
        self.assertEqual(a.policy.store_window, [])

    def test_admitted_episode_without_lesson_reaches_the_consolidator(self):
        class Counting:
            calls = 0
            def consolidate(self, playbook, episode, task, traj):
                self.calls += 1; self.episode = episode; return ""
        cons = Counting()
        a = ReMoAgent(RemoConfig(mode="remo", K=1), FakeSolver(), ScriptedCritic(OK(lesson="")), consolidator=cons)
        rec = a.run_task("t", 0)
        self.assertEqual((cons.calls, rec["store_decision"], rec["entry_id"]), (1, "stored", ""))
        self.assertTrue(cons.episode.admitted); self.assertEqual(cons.episode.lesson(), "")
        self.assertEqual(len(a.playbook), 0)                     # AppendConsolidator would add nothing either

    def test_adaremo_store_gate_and_reinforce(self):
        cfg = RemoConfig(mode="adaremo", K=1, redundant_mode="reinforce")
        a = ReMoAgent(cfg, FakeSolver(), ScriptedCritic(OK(lesson="use average equity")))
        a.run_task("t1", 0)
        eid = a.playbook.entries[0].id
        a.critic = ScriptedCritic(Reflection("correct", lesson="dup", store=False, novelty_reason=f"covered by [{eid}]"))
        rec = a.run_task("t2", 1)
        self.assertEqual(rec["store_decision"], "reinforced"); self.assertEqual(rec["entry_id"], eid)
        self.assertEqual(a.playbook.get(eid).helpful, 2); self.assertEqual(len(a.playbook), 1)
        self.assertEqual(a.policy.store_window, [True, True])

    def test_reinforce_every_cited_entry_that_exists(self):
        a = ReMoAgent(RemoConfig(mode="adaremo", K=1), FakeSolver(),
                      ScriptedCritic(Reflection("correct", lesson="dup", store=False, cited_id="[les-00002]",
                                                novelty_reason="see [les-00001], [les-00002] and [les-00009]")))
        a.playbook.add("one"); a.playbook.add("two")
        rec = a.run_task("t", 0)
        self.assertEqual((rec["store_decision"], rec["entry_id"]), ("reinforced", "les-00002,les-00001"))
        self.assertEqual([e.helpful for e in a.playbook.entries], [2, 2])

    def test_cited_entry_missing_is_discarded(self):
        a = ReMoAgent(RemoConfig(mode="adaremo", K=1), FakeSolver(),
                      ScriptedCritic(Reflection("correct", lesson="dup", store=False, novelty_reason="[les-00007]")))
        rec = a.run_task("t", 0)
        self.assertEqual((rec["store_decision"], rec["entry_id"]), ("discarded", "")); self.assertEqual(a.policy.store_window, [False])

    def test_adaremo_gate_mode_discards(self):
        a = ReMoAgent(RemoConfig(mode="adaremo", K=1, redundant_mode="gate"), FakeSolver(),
                      ScriptedCritic(Reflection("correct", lesson="dup", store=False, novelty_reason="[les-00001]")))
        a.playbook.add("existing")
        self.assertEqual(a.run_task("t", 0)["store_decision"], "discarded"); self.assertEqual(a.playbook.get("les-00001").helpful, 1)

    def test_freeze_and_probe(self):
        cfg = RemoConfig(mode="adaremo", K=1, freeze_w=4, freeze_rho=0.5, probe_p=3, redundant_mode="gate")
        a = ReMoAgent(cfg, FakeSolver(), ScriptedCritic(Reflection("correct", lesson="x", store=False)))
        decisions = [a.run_task(f"t{i}", i)["store_decision"] for i in range(4)]   # 4 discards -> freeze
        self.assertTrue(a.policy.frozen); self.assertEqual(a.policy.freeze_events[0]["event"], "freeze")
        a.critic = ScriptedCritic(OK(lesson="new"))
        self.assertEqual(a.run_task("t4", 4)["store_decision"], "skipped_frozen")  # index 4: not a probe
        self.assertEqual(a.run_task("t5", 5)["store_decision"], "stored")          # index 5: (5+1)%3==0 probe -> unfreeze
        self.assertFalse(a.policy.frozen)


class TestBaselines(unittest.TestCase):
    def test_no_memory_never_writes_and_injects_nothing(self):
        seen = []
        class Spy(FakeSolver):
            def solve(self, task, memory_text, critique):
                seen.append(memory_text); return super().solve(task, memory_text, critique)
        a = ReMoAgent(RemoConfig(mode="remo", K=2, use_memory=False), Spy(), ScriptedCritic(OK()))
        rec = a.run_task("t", 0)
        self.assertEqual(rec["store_decision"], "no_memory"); self.assertEqual(len(a.playbook), 0); self.assertEqual(seen, [""])


class TestPlaybookPrefixKept(unittest.TestCase):
    def test_empty_prefixed_playbook_is_not_replaced(self):
        a = ReMoAgent(RemoConfig(mode="remo", K=1), FakeSolver(), ScriptedCritic(OK()), playbook=Playbook(prefix="calc"))
        a.run_task("t", 0)
        self.assertTrue(a.playbook.entries[0].id.startswith("calc-"))


class TestParsing(unittest.TestCase):
    def test_parse_adaremo(self):
        r = parse_reflection('{"verdict":"no_errors","critique":"ok","refine":false,"store":false,'
                             '"novelty_reason":"covered by [les-00007]","lesson":"L"}', adaptive=True)
        self.assertTrue(r.correct); self.assertFalse(r.store); self.assertFalse(r.refine); self.assertEqual(r.lesson, "L")
        self.assertEqual(Playbook.find_cited_ids(r.novelty_reason), ["les-00007"])

    def test_unparseable_defaults_conservative(self):
        r = parse_reflection("garbage", adaptive=True)
        self.assertFalse(r.parsed); self.assertTrue(r.refine); self.assertFalse(r.store); self.assertFalse(r.correct)
        self.assertEqual(r.critique, "garbage"); self.assertEqual(r.raw, "garbage"); self.assertFalse(r.failed)

    def test_verdict_fallbacks(self):
        self.assertTrue(parse_reflection('not json but says "no_errors"', adaptive=False).correct)
        self.assertFalse(parse_reflection('{"verdict": "No Errors", "critique": "c"}', adaptive=False).correct)  # off-literal value
        self.assertTrue(parse_reflection('{"critique": "c"} and "no_errors" outside', adaptive=False).correct)     # absent -> literal
        r = parse_reflection('{"trajectory_verdict": "no_errors", "key_insight": "K"}', adaptive=True,
                             verdict_key="trajectory_verdict", lesson_keys=("key_insight",), extract=extract_json_balanced)
        self.assertTrue(r.correct); self.assertEqual(r.lesson, "K"); self.assertEqual(r.critique[:1], "{")
        self.assertFalse(parse_reflection('{"trajectory_verdict": "fine"}', adaptive=False).correct)           # wrong key, no literal

    def test_remo_mode_ignores_adaptive_fields(self):
        r = parse_reflection('{"verdict":"errors_found","refine":false,"critique":"c"}', adaptive=False)
        self.assertTrue(r.refine)

    def test_balanced_extractor(self):
        self.assertEqual(extract_json_balanced('x {"a": "}"} y {"b": 1} }'), {"a": "}"})
        self.assertEqual(extract_json_balanced("```json\n[1]\n```"), [1]); self.assertIsNone(extract_json_balanced("{ nope"))


class TestPlaybook(unittest.TestCase):
    def test_roundtrip_and_cap(self):
        pb = Playbook(prefix="fin"); a = pb.add("short  one\nsplit"); b = pb.add("second entry"); pb.reinforce(b)
        self.assertEqual(pb.render(), "[fin-00001] helpful=1 short  one split\n[fin-00002] helpful=2 second entry")   # inner spaces kept, line break folded (as the run wrote lesson.strip())
        p = os.path.join(tempfile.mkdtemp(), "pb.txt"); pb.save(p)
        with open(p) as f:
            self.assertEqual(f.read(), pb.render() + "\n")
        pb2 = Playbook.load(p, prefix="fin")
        self.assertEqual(pb2.get(b).helpful, 2); self.assertEqual(pb2.ids(), [a, b]); self.assertEqual(pb2.add("third"), "fin-00003")
        self.assertEqual(pb2.render(cap_chars=40), "[fin-00002] helpful=2 second entry")   # ranked by (-helpful, line), first overflow stops
        self.assertEqual(Playbook.load(os.path.join(tempfile.mkdtemp(), "none.txt"), prefix="fin").ids(), [])


class TestSectionedPlaybookInTheLoop(unittest.TestCase):
    """The loop with the sectioned memory: the consolidator applies ADD operations, reinforcement edits the
    cited bullet, the run dir round-trips the text."""

    class AddConsolidator:
        def consolidate(self, playbook, episode, task, traj):
            ops = playbook.parse_ops_response('{"reasoning": "r", "operations": [{"type": "ADD", "section": "OTHERS", "content": "%s"}]}'
                                                  % episode.lesson())
            return ",".join(playbook.apply_add_ops(ops))

    def test_store_reinforce_and_resume(self):
        d = tempfile.mkdtemp()
        cfg = RemoConfig(mode="adaremo", K=1)
        a = ReMoAgent(cfg, FakeSolver(), ScriptedCritic(OK(lesson="first")), SectionedPlaybook.from_skeleton("counts"),
                      self.AddConsolidator(), run_dir=d)
        rec = a.run_task("t1", 0)
        self.assertEqual((rec["store_decision"], rec["entry_id"], rec["memory_chars_at_start"]), ("stored", "misc-00001", len(a.playbook.text) - len("[misc-00001] helpful=0 harmful=0 :: first\n")))
        a.critic = ScriptedCritic(Reflection("correct", lesson="dup", store=False, novelty_reason="[misc-00001]"))
        rec = a.run_task("t2", 1)
        self.assertEqual((rec["store_decision"], rec["entry_id"]), ("reinforced", "misc-00001"))
        self.assertIn("[misc-00001] helpful=1 harmful=0 :: first", a.playbook.text)
        with open(os.path.join(d, "playbook.txt")) as f:
            self.assertEqual(f.read(), a.playbook.text)
        b = ReMoAgent(cfg, FakeSolver(), ScriptedCritic(OK()), SectionedPlaybook.from_skeleton("counts"), self.AddConsolidator(), run_dir=d)
        self.assertIsInstance(b.playbook, SectionedPlaybook); self.assertEqual(b.playbook.text, a.playbook.text)
        self.assertEqual((b.playbook.style, b.done_indices(), b.policy.store_window), ("counts", {0, 1}, [True, True]))
        with open(os.path.join(d, "episodes.jsonl")) as f:
            self.assertEqual([json.loads(l)["entry_id"] for l in f], ["misc-00001", "misc-00001"])


if __name__ == "__main__":
    unittest.main()


# -- redundant_mode "structural" ------------------------------------------------------------------------
from remo import LexicalRedundancyChecker, LexicalRetriever, LLMRedundancyChecker, RedundancyResult   # noqa: E402
from remo.redundancy import specific_tokens, weighted_jaccard                                            # noqa: E402


class _JudgeLLM:
    def __init__(self, replies):
        self.replies, self.requests = list(replies), []
    def chat(self, messages, max_tokens, temperature):
        self.requests.append(messages[0]["content"])
        return self.replies[min(len(self.requests), len(self.replies)) - 1]


def _plain_pb(*bullets):
    pb = SectionedPlaybook.from_skeleton("plain")
    pb.apply_add_ops([{"type": "ADD", "section": "strategies_and_hard_rules", "content": b} for b in bullets])
    return pb


class TestStructuralRedundancy(unittest.TestCase):
    def test_tokens_and_similarity(self):
        t = specific_tokens("Call `apis.amazon.show_payment_cards` and keep cards with expiry_year > 2023; the rating field")
        self.assertEqual(t["apis.amazon.show_payment_cards"], 3.0); self.assertEqual(t["expiry_year"], 3.0)
        self.assertEqual(t["2023"], 3.0); self.assertEqual(t["rating"], 1.0); self.assertNotIn("the", t)
        a = specific_tokens("apis.amazon.show_payment_cards returns expiry_year and expiry_month")
        self.assertEqual(weighted_jaccard(a, a), 1.0)
        self.assertGreater(weighted_jaccard(a, specific_tokens("filter payment cards by expiry_year from show_payment_cards")),
                           weighted_jaccard(a, specific_tokens("always look at the API docs before calling an API")))
        self.assertEqual(weighted_jaccard(a, {}), 0.0)

    def test_retriever_excludes_principles_and_ranks(self):
        entries = [("shr-00005", "Always look at API specifications before calling an API."),
                   ("vc-00010", "apis.amazon.show_payment_cards: pick a card whose expiry_year is later than the current year"),
                   ("cms-00011", "apis.gmail.search_emails needs page_index pagination")]
        r = LexicalRetriever(k=3, min_sim=0.15)
        top = r.top("use show_payment_cards and compare expiry_year with the current year", entries, exclude={"shr-00005"})
        self.assertEqual(top[0][0], "vc-00010"); self.assertNotIn("shr-00005", [x[0] for x in top])
        self.assertEqual(r.top("completely unrelated words here", entries), [])

    def test_policy_structural_decisions(self):
        cfg = RemoConfig(mode="adaremo", K=1, redundant_mode="structural")
        a = ReMoAgent(cfg, FakeSolver(), ScriptedCritic(OK(lesson="L", store=False)))   # the critic's store=False is ignored
        rec = a.run_task("t", 0)
        self.assertEqual(rec["store_decision"], "stored"); self.assertEqual(rec["redundancy"]["reason"], "no similar entry")
        self.assertEqual(len(a.playbook), 1)

    def test_llm_checker_makes_no_call_without_candidates(self):
        pb = _plain_pb("apis.venmo.show_transactions is paginated over page_index")
        llm = _JudgeLLM(['{"covered_by": null, "reason": "x"}'])
        c = LLMRedundancyChecker(llm, principle_ids=set())
        res = c.check("unrelated: the gmail draft subject must be non-empty", pb)
        self.assertEqual((res.covered_by, res.reason, llm.requests), ("", "no similar entry", []))
        self.assertEqual(c.check("none", pb).reason, "no lesson"); self.assertEqual(c.stats()["no_lesson"], 1)

    def test_llm_checker_covered_and_invalid_id(self):
        pb = _plain_pb("apis.venmo.show_transactions is paginated: loop over page_index until an empty page",
                       "apis.gmail.search_emails returns thread ids, not emails")
        ids = pb.ids()
        llm = _JudgeLLM([f'{{"covered_by": "{ids[0]}", "reason": "same page_index rule"}}'])
        c = LLMRedundancyChecker(llm, principle_ids=set())
        res = c.check("loop apis.venmo.show_transactions over page_index until an empty page is returned", pb)
        self.assertEqual((res.covered_by, res.reason), (ids[0], "same page_index rule")); self.assertIn(ids[0], res.candidates)
        self.assertIn(f"[{ids[0]}]", llm.requests[0]); self.assertNotIn(f"[{ids[1]}]", llm.requests[0])   # only retrieved candidates are shown
        llm2 = _JudgeLLM(['{"covered_by": "shr-99999", "reason": "made up"}'])
        c2 = LLMRedundancyChecker(llm2, principle_ids=set(), log=lambda m: None)
        self.assertEqual(c2.check("loop apis.venmo.show_transactions over page_index", pb).covered_by, "")   # not a candidate -> novel
        c3 = LLMRedundancyChecker(_JudgeLLM(["garbage"]), principle_ids=set(), log=lambda m: None)
        self.assertEqual(c3.check("loop apis.venmo.show_transactions over page_index", pb).covered_by, "")
        class Down:
            def chat(self, *a): raise ConnectionError("down")
        c4 = LLMRedundancyChecker(Down(), principle_ids=set(), log=lambda m: None)
        r4 = c4.check("loop apis.venmo.show_transactions over page_index", pb)
        self.assertEqual(r4.covered_by, ""); self.assertTrue(r4.reason.startswith("judge failed")); self.assertEqual(c4.failures, 1)

    def test_agent_structural_reinforces_and_seed_never_covers(self):
        seed = _plain_pb("Always look at API specifications before calling an API.")
        seed_id = seed.ids()[0]
        cfg = RemoConfig(mode="adaremo", K=1, redundant_mode="structural")
        critic = ScriptedCritic(OK(lesson="Always look at API specifications before calling an API.", store=False))
        a = ReMoAgent(cfg, FakeSolver(), critic, playbook=seed,
                      consolidator=type("C", (), {"consolidate": lambda self, pb, ep, task, traj: ",".join(pb.apply_add_ops(
                          [{"type": "ADD", "section": "verification_checklist", "content": ep.lesson()}]))})())
        rec = a.run_task("t", 0)                       # identical to the seed principle, yet the seed can never cover it
        self.assertEqual(rec["store_decision"], "stored"); self.assertEqual(rec["redundancy"]["candidates"], [])
        new_id = rec["entry_id"]; self.assertTrue(new_id.startswith("vc-"))
        rec2 = a.run_task("t2", 1)                     # the stored copy is not a principle: the lexical checker covers it
        self.assertEqual((rec2["store_decision"], rec2["entry_id"]), ("reinforced", new_id))
        self.assertIn(f"[{new_id}] Always look at API specifications before calling an API. [confirmed x2]", a.playbook.text)
        self.assertEqual(len(a.playbook), 2); self.assertEqual(a.playbook.ids(), [seed_id, new_id])
        self.assertEqual(rec2["redundancy"]["covered_by"], new_id)
