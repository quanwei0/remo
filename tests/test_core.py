"""Unit tests of Algorithms 1/2 semantics with fake solver/critic (no LLM)."""
import unittest

from remo import RemoConfig, ReMoAgent, Playbook, Reflection, Trajectory
from remo.critic import parse_reflection


class FakeSolver:
    def __init__(self, completed=True): self.completed = completed
    def solve(self, task, memory_text, critique):
        return Trajectory(text=f"attempt for {task}", answer="42", completed=self.completed)


class ScriptedCritic:
    """Returns the scripted Reflection for round r (1-based); repeats the last one after."""
    def __init__(self, *refls): self.refls = refls
    def reflect(self, task, traj, memory_text, prior, r, K):
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

    def test_adaremo_store_gate_and_reinforce(self):
        cfg = RemoConfig(mode="adaremo", K=1, redundant_mode="reinforce")
        a = ReMoAgent(cfg, FakeSolver(), ScriptedCritic(OK(lesson="use average equity")))
        a.run_task("t1", 0)
        eid = a.playbook.entries[0].id
        a.critic = ScriptedCritic(Reflection("correct", lesson="dup", store=False, novelty_reason=f"covered by [{eid}]"))
        rec = a.run_task("t2", 1)
        self.assertEqual(rec["store_decision"], "reinforced"); self.assertEqual(a.playbook.get(eid).helpful, 2)
        self.assertEqual(len(a.playbook), 1)

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
        self.assertTrue(r.correct); self.assertFalse(r.store)
        self.assertEqual(Playbook.find_cited_id(r.novelty_reason), "les-00007")

    def test_unparseable_defaults_conservative(self):
        r = parse_reflection("garbage", adaptive=True)
        self.assertFalse(r.parsed); self.assertTrue(r.refine); self.assertFalse(r.store); self.assertFalse(r.correct)

    def test_remo_mode_ignores_adaptive_fields(self):
        r = parse_reflection('{"verdict":"errors_found","refine":false,"critique":"c"}', adaptive=False)
        self.assertTrue(r.refine)


class TestPlaybook(unittest.TestCase):
    def test_roundtrip_and_cap(self):
        pb = Playbook(prefix="fin"); a = pb.add("short one"); b = pb.add("second entry"); pb.reinforce(b)
        import tempfile, os
        p = os.path.join(tempfile.mkdtemp(), "pb.txt"); pb.save(p)
        pb2 = Playbook.load(p, prefix="fin")
        self.assertEqual(pb2.get(b).helpful, 2); self.assertEqual(pb2.add("third"), "fin-00003")
        self.assertIn(b, pb2.render(cap_chars=40))   # most-reinforced survives the cap


if __name__ == "__main__":
    unittest.main()
