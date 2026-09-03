"""FinanceGym adapter: what the paper's runs sent and how they read the replies (prompt file formatted with
the original arguments, exact question strings, critic call parameters, parse fallbacks, playbook line
format and cap ranking), plus the async concurrent driver over RemoPolicy (fake solver/critic, no harness,
no LLM) — persistence/resume, the min-docs floor, AdaReMo reinforce, the baseline and ablation arms,
learn-then-freeze, the retry-round budget, the quality floor and the post-hoc summary."""
import asyncio
import json
import os
import tempfile
import unittest
from types import SimpleNamespace

from benchmarks.financegym import check_answers, make_answers
from benchmarks.financegym.common import (CRITIC_PROMPT, FinanceGymCritic, VerbatimConsolidator, build_critic_prompt,
                                          build_question, cited_entry, is_defective, make_config, make_trajectory,
                                          parse_critic_reply, summarize_run, trajectory_from_record)
from benchmarks.financegym.run_financegym import RunState, run_all, write_final_results
from remo import EpisodeState, Playbook, Reflection, RemoConfig, RoundRecord

PIT = "\n\n(Point-in-time constraint: use only information published on or before 2025-06-01. The search environment enforces this cutoff.)"


def _traj(report="R" * 2000, docs=10, queries=("q1",), citations=()):
    return make_trajectory(report, list(queries), docs, list(citations), 3, elapsed_s=1.0, steps=5, termination="answer",
                           empty_attempts=0, report_chars=len(report))


class FakeSolver:
    def __init__(self, per_task=None, per_round=None): self.per_task, self.per_round, self.calls = per_task or {}, per_round, []
    async def solve(self, task, memory_text, critique):
        self.calls.append((task["task_id"], len(memory_text), critique))
        await asyncio.sleep(0.001)
        if self.per_round:
            return self.per_round(task, critique)
        return self.per_task.get(task["task_id"], _traj())


class FakeCritic:
    def __init__(self, fn): self.fn, self.calls = fn, 0
    async def reflect(self, task, traj, memory_text, prior, r, K):
        self.calls += 1
        await asyncio.sleep(0.001)
        return self.fn(task, traj, memory_text, r)


class FakeChatClient:
    """openai.AsyncOpenAI look-alike returning a fixed completion (or raising); records every call's kwargs."""
    def __init__(self, text, fail=False):
        self.text, self.fail, self.kwargs = text, fail, []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
    async def _create(self, **kw):
        self.kwargs.append(kw)
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


def _reflect(client_text, fail=False):
    client = FakeChatClient(client_text, fail=fail)
    critic = FinanceGymCritic(client, "M")
    refl = asyncio.run(critic.reflect(_tasks(1)[0], _traj(), "", None, 1, 3))
    return refl, critic, client


class TestWhatIsSent(unittest.TestCase):
    def test_question_first_round_retry_and_empty_playbook(self):
        t = _tasks(1)[0]
        pb = "[fin-00001] helpful=2 Search the primary filing first.\n[fin-00002] helpful=1 Cite the page."
        self.assertEqual(build_question(t, pb, None),
                         "Analyst playbook — lessons from prior research tasks; apply when relevant:\n" + pb
                         + "\n\nResearch question: Q0?" + PIT)
        self.assertEqual(build_question(t, "", None), "Research question: Q0?" + PIT)
        crit = "  Figure X is unsourced.\n" + "c" * 4000
        self.assertEqual(build_question(t, pb, crit),
                         build_question(t, pb, None) + "\n\nA reviewer found these issues in a previous attempt — run a "
                         "fresh, better investigation that fixes them:\n" + crit[:3000])
        self.assertEqual(build_question(t, "", ""), "Research question: Q0?" + PIT + "\n\nA reviewer found these issues "
                         "in a previous attempt — run a fresh, better investigation that fixes them:\n")
        self.assertEqual(build_question(t, pb, "x", plain=True), "Q0?" + PIT)      # the official-harness baseline

    def test_critic_prompt_formats_the_file_with_the_original_arguments(self):
        self.assertTrue(CRITIC_PROMPT.startswith("You are a rigorous financial research reviewer."))
        self.assertTrue(CRITIC_PROMPT.endswith('"cited_id": "covering entry id or empty string"}}'))
        for k in ("{q}", "{cutoff}", "{nq}", "{queries}", "{ndocs}", "{ncit}", "{report}", "{playbook}", '{{"verdict"'):
            self.assertIn(k, CRITIC_PROMPT)
        task = {"task_id": "x", "question": "Why {braces}?", "cutoff": "2025-03-01"}
        queries = [f"q{i} é" for i in range(25)]
        report = 'Report with {"json": 1} braces ' + "r" * 20000
        tr = _traj(report=report, docs=7, queries=queries, citations=["a", "b"])
        memory = "[fin-00001] helpful=1 L\n" + "m" * 30000
        p = build_critic_prompt(task, tr, memory)
        self.assertEqual(p, CRITIC_PROMPT.format(q="Why {braces}?", cutoff="2025-03-01", nq=25,
                                                 queries=json.dumps(queries[:20]), ndocs=7, ncit=2,
                                                 report=report[:12000], playbook=memory[:20000]))
        self.assertIn('"q0 \\u00e9"', p)                                            # json.dumps ensure_ascii default
        self.assertIn("Prior playbook (for the novelty check only):\n(empty)\n", build_critic_prompt(task, tr, ""))
        self.assertIn("Documents actually fetched: 7 | Citations listed: 2", p)
        self.assertEqual(FinanceGymCritic(None, "M").build_prompt(task, tr, memory), p)
        # calibrate.py rebuilds the same Trajectory from a saved record
        rec = {"final_answer": report, "queries": queries, "docs_retrieved": 7, "citations": ["a", "b"]}
        self.assertEqual(build_critic_prompt(task, trajectory_from_record(rec), memory), p)

    def test_critic_call_parameters(self):
        refl, critic, client = _reflect('{"verdict": "no_errors", "confidence": 0.9, "critique": "fine", "refine": false, '
                                        '"store": true, "lesson": "Do X.", "novelty_reason": "new", "cited_id": ""}')
        kw = client.kwargs[0]
        self.assertEqual(set(kw), {"model", "max_tokens", "messages"})                 # no temperature: server default
        self.assertEqual((kw["model"], kw["max_tokens"]), ("M", 2048))
        self.assertEqual([m["role"] for m in kw["messages"]], ["user"])
        self.assertEqual(kw["messages"][0]["content"], build_critic_prompt(_tasks(1)[0], _traj(), ""))
        self.assertTrue(refl.correct and refl.parsed and refl.store and not refl.refine and not refl.failed)
        self.assertEqual((refl.lesson, refl.critique, refl.novelty_reason), ("Do X.", "fine", "new"))
        self.assertEqual((critic.calls, critic.parse_failures, critic.call_failures), (1, 0, 0))


class TestHowRepliesAreRead(unittest.TestCase):
    def test_original_parse_semantics(self):
        r = parse_critic_reply('text before {"verdict": "errors_found", "critique": "bad", "refine": "no", "store": 0, '
                               '"lesson": 5, "cited_id": "[fin-00003]"} text after')
        self.assertFalse(r.correct); self.assertTrue(r.parsed)
        self.assertTrue(r.refine)                                    # bool("no") is True, as in the runs
        self.assertFalse(r.store); self.assertEqual(r.lesson, "5"); self.assertEqual(r.cited_id, "[fin-00003]")
        self.assertEqual(cited_entry(r), "fin-00003")
        r = parse_critic_reply('{"critique": "x", "lesson": "L"}')     # verdict missing -> no_errors
        self.assertTrue(r.correct and r.parsed and not r.refine and not r.store); self.assertEqual(r.lesson, "L")
        self.assertFalse(parse_critic_reply('{"verdict": "unsure"}').correct)
        self.assertEqual(cited_entry(Reflection("correct", novelty_reason="covered by [fin-00012] and fin-00013")), "fin-00012")
        self.assertEqual(cited_entry(Reflection("correct", cited_id="  ", novelty_reason="new")), "")

    def test_unparseable_reply_is_accepted_without_a_lesson(self):
        for bad in ("no json here", '{"verdict": "errors_found", "lesson": "L"', '{"verdict": "errors_found", "confidence": "high"}',
                    '[1, 2]', ""):
            refl, critic, _ = _reflect(bad)
            self.assertTrue(refl.correct and not refl.parsed and not refl.failed, bad)
            self.assertEqual((refl.lesson, refl.refine, refl.store, refl.raw), ("", False, False, bad))
            self.assertEqual(critic.parse_failures, 1)

    def test_call_failure_is_a_failed_reflection(self):
        refl, critic, _ = _reflect("", fail=True)
        self.assertTrue(refl.failed and not refl.parsed); self.assertEqual(refl.verdict, "none")
        self.assertIn("model down", refl.critique)
        self.assertEqual((critic.calls, critic.parse_failures, critic.call_failures), (1, 0, 1))


def _store(pb, lesson):
    st = EpisodeState()
    st.rounds.append(RoundRecord(1, True, Reflection(verdict="no_errors", critique="", lesson=lesson)))
    return VerbatimConsolidator().consolidate(pb, st, {}, None)


class TestPlaybookFormat(unittest.TestCase):
    def test_lesson_stored_verbatim(self):
        # the runs wrote f"[{eid}] helpful=1 {lesson.strip()}\n": internal whitespace (double spaces, tabs,
        # U+202F as in "Form\u202f4") is kept byte for byte, only the ends are stripped
        pb = Playbook(prefix="fin")
        for lesson in ("  Two  spaces \n", "tab\there", "Form\u202f4 filings", ""):
            eid = _store(pb, lesson)
            self.assertEqual(eid, "" if not lesson.strip() else f"fin-{len(pb):05d}")
        self.assertEqual(pb.render(), "[fin-00001] helpful=1 Two  spaces\n[fin-00002] helpful=1 tab\there"
                         "\n[fin-00003] helpful=1 Form\u202f4 filings")
        d = tempfile.mkdtemp()
        pb.save(os.path.join(d, "playbook.txt"))
        pb2 = Playbook.load(os.path.join(d, "playbook.txt"), prefix="fin")
        self.assertEqual([e.text for e in pb2.entries], ["Two  spaces", "tab\there", "Form\u202f4 filings"])
        self.assertEqual(_store(pb2, "next"), "fin-00004")                     # the id counter resumes

    def test_line_format_and_cap_ranking(self):
        pb = Playbook(prefix="fin")
        _store(pb, "Search the primary filing first.")
        _store(pb, "Cite the page.")
        _store(pb, "Avoid post-cutoff sources.")
        self.assertEqual(pb.render(), "[fin-00001] helpful=1 Search the primary filing first.\n[fin-00002] helpful=1 Cite the page."
                         "\n[fin-00003] helpful=1 Avoid post-cutoff sources.")
        self.assertTrue(pb.reinforce("fin-00003")); self.assertFalse(pb.reinforce("fin-00009"))
        ranked = pb.render(30000).split("\n")
        self.assertEqual(ranked[0], "[fin-00003] helpful=2 Avoid post-cutoff sources.")   # (-helpful, line)
        self.assertEqual(ranked[1:], ["[fin-00001] helpful=1 Search the primary filing first.", "[fin-00002] helpful=1 Cite the page."])
        n = len(ranked[0]) + len(ranked[1])
        self.assertEqual(pb.render(n).split("\n"), ranked[:2])           # stops at the first line that would overflow
        self.assertEqual(pb.render(n - 1).split("\n"), ranked[:1])
        d = tempfile.mkdtemp()
        pb.save(os.path.join(d, "playbook.txt"))
        with open(os.path.join(d, "playbook.txt")) as f:
            self.assertEqual(f.read(), pb.render() + "\n")
        self.assertEqual(Playbook.load(os.path.join(d, "playbook.txt"), prefix="fin").render(), pb.render())


class TestDriver(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()

    def test_remo_concurrent_persist_and_resume(self):
        tasks = _tasks(6)
        critic = FakeCritic(lambda t, tr, m, r: Reflection("correct", lesson=f"lesson {t['task_id']}"))
        rs = RunState(RemoConfig(mode="remo", K=3), self.d)
        s = _run(tasks, rs, FakeSolver(), critic)
        self.assertEqual(s["saved"], 6); self.assertEqual(s["gates"], {"round1_clean": 6})
        self.assertEqual((s["playbook_entries"], s["extra_rounds_used"]), (6, 0))
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
        self.assertIn(("t000", 0, "fix X"), solver.calls)      # the retry carried the critique into the solver
        # t001: completed=False every round -> 3 rounds, never admitted, not saved, no memory write
        self.assertEqual(seen["t001"], 3); self.assertEqual(len(rs.playbook), 1)
        self.assertEqual(rs.extra_rounds_used, 3)                # 1 (t000) + 2 (t001, unsaved rounds count too)

    def test_adaremo_reinforce_one_cited_entry_and_critic_stop(self):
        tasks = _tasks(4)
        def fn(t, tr, m, r):
            if t["task_id"] == "t000":
                return Reflection("correct", lesson="use average equity", store=True)
            if t["task_id"] == "t001":
                return Reflection("incorrect", critique="no fix", refine=False)
            if t["task_id"] == "t002":       # id only in novelty_reason; the second id is not voted on
                return Reflection("correct", lesson="dup", store=False, novelty_reason="covered by [fin-00001], [fin-00002]")
            return Reflection("correct", lesson="dup", store=False, cited_id="[fin-00009]", novelty_reason="[fin-00001]")
        rs = RunState(RemoConfig(mode="adaremo", K=3), self.d)
        s = _run(tasks, rs, FakeSolver(), FakeCritic(fn), conc=1)      # conc=1: t000 stored before t002 cites it
        self.assertEqual(s["gates"], {"round1_clean": 3, "critic_stop": 1})
        self.assertEqual(s["store_decisions"], {"stored": 1, "skipped": 1, "reinforced": 1, "discarded": 1})
        self.assertEqual(rs.playbook.get("fin-00001").helpful, 2); self.assertEqual(len(rs.playbook), 1)
        eps = {e["task_id"]: e for e in map(json.loads, open(os.path.join(self.d, "episodes.jsonl")))}
        self.assertEqual(eps["t002"]["entry_id"], "fin-00001"); self.assertEqual(eps["t003"]["entry_id"], "")
        st = json.load(open(os.path.join(self.d, "policy_state.json")))
        self.assertEqual(st["store_window"], [True, True, False])

    def test_no_lesson_means_no_memory_bookkeeping(self):
        # admitted without a lesson (e.g. an unparseable reply): nothing stored, nothing reinforced, no window entry
        def fn(t, tr, m, r):
            return Reflection("correct", store=False, cited_id="fin-00001") if t["task_id"] == "t001" \
                else Reflection("correct", lesson="L", store=True)
        rs = RunState(RemoConfig(mode="adaremo", K=1), self.d)
        s = _run(_tasks(2), rs, FakeSolver(), FakeCritic(fn), conc=1)
        self.assertEqual(s["store_decisions"], {"stored": 1, "skipped": 1})
        self.assertEqual(rs.playbook.get("fin-00001").helpful, 1)
        self.assertEqual(json.load(open(os.path.join(self.d, "policy_state.json")))["store_window"], [True])

    def test_baseline_no_critic_no_memory(self):
        cfg, baseline = make_config("baseline", None)
        self.assertTrue(baseline); self.assertEqual((cfg.mode, cfg.K, cfg.use_memory), ("remo", 1, False))
        solver = FakeSolver({"t001": _traj(docs=1)})
        rs = RunState(cfg, self.d)
        s = _run(_tasks(2), rs, solver, None)                 # critic=None: no critic call at all
        self.assertEqual(s["saved"], 1); self.assertEqual(s["unsaved_min_docs"], 1)   # doc floor still applies
        self.assertEqual(s["gates"], {"no_critic": 1}); self.assertEqual(s["store_decisions"], {"no_memory": 1})
        e = json.loads(open(os.path.join(self.d, "episodes.jsonl")).readline())
        self.assertEqual(len(e["rounds"]), 1); self.assertEqual(e["rounds"][0]["verdict"], "none")
        self.assertEqual(e["stop_reason"], "no_critic"); self.assertEqual(len(rs.playbook), 0)
        self.assertEqual(solver.calls[0][1:], (0, None))       # empty memory, no critique

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
        full = rs.playbook.render(30000)
        for tid in ("t004", "t005"):                              # read-only tasks saw the complete learned memory
            self.assertTrue(eps[tid]["readonly_memory"]); self.assertEqual(eps[tid]["memory_chars_at_start"], len(full))
            self.assertIn((tid, len(full), None), solver.calls)
        self.assertEqual(len(eps["t000"]["rounds"]), 1)
        # resume: learning done -> barrier opens immediately, a new read-only task runs and does not write
        rs2 = RunState(RemoConfig(mode="remo", K=1), self.d, freeze_after=4)
        s2 = _run(tasks + _tasks(7)[6:], rs2, FakeSolver(), critic)
        self.assertEqual(s2["store_decisions"], {"readonly": 1}); self.assertEqual(len(rs2.playbook), 4)

    def test_extra_rounds_budget_is_run_wide_and_persisted(self):
        critic = FakeCritic(lambda t, tr, m, r: Reflection("incorrect", critique="c", lesson="L"))
        rs = RunState(RemoConfig(mode="remo", K=3), self.d, extra_rounds_budget=3)
        s = _run(_tasks(3), rs, FakeSolver(), critic, conc=1)
        self.assertEqual(s["extra_rounds_used"], 3)
        eps = [json.loads(l) for l in open(os.path.join(self.d, "episodes.jsonl"))]
        self.assertEqual([len(e["rounds"]) for e in eps], [3, 2, 1])
        self.assertEqual([e["stop_reason"] for e in eps], ["max_rounds", "round_budget", "round_budget"])
        self.assertEqual(json.load(open(os.path.join(self.d, "policy_state.json")))["extra_rounds_used"], 3)
        rs2 = RunState(RemoConfig(mode="remo", K=3), self.d, extra_rounds_budget=3)
        self.assertEqual(rs2.extra_rounds_used, 3)
        self.assertEqual(_run(_tasks(4)[3:], rs2, FakeSolver(), critic)["extra_rounds_used"], 3)   # still exhausted

    def test_critic_call_failure_stops_the_episode_unadmitted(self):
        fb = Reflection("none", critique="(critic call failed: x)", refine=False, store=False, parsed=False, failed=True)
        def fn(t, tr, m, r):
            return Reflection("incorrect", critique="fix X", lesson="L1") if r == 1 else fb
        rs = RunState(RemoConfig(mode="remo", K=3), self.d)
        s = _run(_tasks(1), rs, FakeSolver(), FakeCritic(fn))
        e = json.loads(open(os.path.join(self.d, "episodes.jsonl")).readline())
        self.assertEqual((len(e["rounds"]), e["gate"], e["stop_reason"], e["store_decision"]), (2, "never_clean", "critic_error", "skipped"))
        self.assertEqual(len(rs.playbook), 0); self.assertEqual(s["store_decisions"], {"skipped": 1})
        tf = _write_tasks(self.d, _tasks(1))
        self.assertEqual(summarize_run(self.d, tf)["critic_error_episodes"], 1)

    def test_empty_report_stops_and_submits_the_first_round(self):
        # an empty report is never retried (the critic still runs); the submitted report is then the first round's
        def solve(task, critique):
            return _traj(report="first " * 400, docs=5) if critique is None else _traj(report="", docs=5)
        critic = FakeCritic(lambda t, tr, m, r: Reflection("incorrect", critique="c"))
        rs = RunState(RemoConfig(mode="remo", K=3), self.d)
        s = _run(_tasks(1), rs, FakeSolver(per_round=solve), critic)
        e = json.loads(open(os.path.join(self.d, "episodes.jsonl")).readline())
        self.assertEqual((e["final_round"], e["final_answer"][:6], len(e["rounds"]), e["gate"], e["stop_reason"]),
                         (1, "first ", 2, "never_clean", "empty_report"))
        self.assertEqual((critic.calls, s["extra_rounds_used"]), (2, 1))

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
            if t["task_id"] == "t002":
                return Reflection("correct", parsed=False)                        # unparseable reply: accepted, no lesson
            return Reflection("correct", lesson=f"lesson {t['task_id']}", store=True)
        rs = RunState(RemoConfig(mode="adaremo", K=2), self.d)
        _run(tasks, rs, FakeSolver(), FakeCritic(fn))
        tf = _write_tasks(self.d, tasks)
        with open(os.path.join(self.d, "run_config.json"), "w") as f:
            json.dump({"mode": "adaremo", "K": 2, "freeze_after": None}, f)
        res = write_final_results(self.d, tf, {"this_invocation": {"saved": 4}})
        self.assertTrue(os.path.exists(os.path.join(self.d, "final_results.json")))
        self.assertIsNone(res["accuracy"]); self.assertEqual((res["n_tasks"], res["n_saved"]), (4, 4))
        self.assertEqual(res["gate_distribution"], {"round1_clean": 3, "never_clean": 1})
        self.assertEqual(res["store_decisions"], {"stored": 2, "skipped": 2})
        self.assertEqual((res["mean_rounds"], res["extra_rounds_used"], res["critic_unparsed_rounds"]), (1.25, 1, 1))
        self.assertEqual(res["round1_clean_rate"], 0.75); self.assertEqual(res["final_clean_rate"], 0.75)
        self.assertEqual(res["memory_entries"], 2); self.assertGreater(res["memory_tokens_cl100k"] or 0, 0)
        self.assertEqual(res["this_invocation"], {"saved": 4}); self.assertEqual(res["mode"], "adaremo")
        base = summarize_run(tempfile.mkdtemp(), tf)                                  # empty run dir is fine
        self.assertEqual((base["n_saved"], base["mean_rounds"], base["round1_clean_rate"]), (0, None, None))

    def test_is_defective(self):
        self.assertEqual(is_defective(""), "empty"); self.assertEqual(is_defective("x" * 100), "short")
        self.assertEqual(is_defective('{"a": 1}' + "x" * 3000), "json_fragment"); self.assertEqual(is_defective("x" * 1500), "")


if __name__ == "__main__":
    unittest.main()
