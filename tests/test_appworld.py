"""AppWorld adapter: prompt files, message construction (template turns, first-block extraction, output
wrapping, context trimming, retry injection), env_clean, critic input / parsing, reflection selection, the
curator consolidator and the runner's records — fake model and fake world, no appworld package."""
import importlib.util
import json
import os
import random
import sys
import tempfile
import unittest
from unittest import mock

from benchmarks.appworld import run_appworld as ra
from benchmarks.appworld.consolidator import (CONSOLIDATOR_PROMPT_PATH, VALIDATED_PREFIX, CuratorConsolidator,
                                              InsightConsolidator, build_curator_input, select_reflection)
from benchmarks.appworld.critic import CRITIC_PROMPT_PATHS, AppWorldCritic, EnvCritic, build_critic_input, parse_critic_reply
from benchmarks.appworld.solver import (EMPTY_PLAYBOOK, NOT_SHOWN, REACT_MAX_OUTPUT_LENGTH, REACT_PROMPT_PATH, RETRY_INTRO,
                                        RETRY_USE, RETRY_USE_WITH_PLAYBOOK, SOLVER_PROMPT_PATH, TRIMMED, AppWorldSolver, ChatLLM,
                                        conversation_history, env_clean, extract_code, messages_to_text, output_message,
                                        read_prompt, retry_messages, text_to_messages, trimmed_messages, truncate_output)
from remo import EpisodeState, Reflection, SectionedPlaybook, Trajectory
from remo.policy import RoundRecord

HAS_JINJA2 = importlib.util.find_spec("jinja2") is not None
SEED = ra.INITIAL_PLAYBOOK_PATH


class FakeLLM:
    """`chat(messages, max_tokens, temperature)`: scripted replies in order (the last one repeats), or raises."""
    def __init__(self, replies=(), fail=False):
        self.replies, self.fail, self.requests = list(replies), fail, []
    def chat(self, messages, max_tokens, temperature):
        self.requests.append(json.loads(json.dumps(messages)))
        if self.fail:
            raise ConnectionError("server down")
        return self.replies[min(len(self.requests), len(self.replies)) - 1]


class TestPromptFiles(unittest.TestCase):
    def test_files_load(self):
        for p in (SOLVER_PROMPT_PATH, CRITIC_PROMPT_PATHS[False], CRITIC_PROMPT_PATHS[True], CONSOLIDATOR_PROMPT_PATH):
            self.assertTrue(read_prompt(p).strip(), p)
        self.assertTrue(read_prompt(SOLVER_PROMPT_PATH).startswith("USER:\n"))
        self.assertIn("### PLAYBOOK BEGIN\n{{ playbook }}\n### PLAYBOOK END", read_prompt(SOLVER_PROMPT_PATH))
        react = read_prompt(REACT_PROMPT_PATH)                  # AppWorld's official ReAct prompt: no playbook anywhere
        self.assertTrue(react.startswith("USER:\n")); self.assertNotIn("playbook", react.lower())
        self.assertTrue(react.rstrip("\n").endswith("Task: {{ input_str }}"))
        for k in ("{{generated_code}}", "{{generated_rationale}}", "{{playbook}}", "{{previous_reflection}}"):
            self.assertIn(k, read_prompt(CRITIC_PROMPT_PATHS[False])); self.assertIn(k, read_prompt(CRITIC_PROMPT_PATHS[True]))
        self.assertIn('"confidence"', read_prompt(CRITIC_PROMPT_PATHS[True]))
        self.assertNotIn('"confidence"', read_prompt(CRITIC_PROMPT_PATHS[False]))
        pb = SectionedPlaybook.load(SEED, "plain")
        self.assertEqual(len(pb), 8); self.assertEqual(pb.next_id, 9); self.assertIn("shr-00001", pb.ids())


class TestMessages(unittest.TestCase):
    FIXTURE = "USER:\nhello {x}\n\nASSISTANT:\n```python\nprint(1)\n```\n\nUSER:\nnext\n\nUSER:\nTask: do it\n\n"

    def test_text_to_messages_round_trip(self):
        ms = text_to_messages(self.FIXTURE)
        self.assertEqual([m["role"] for m in ms], ["user", "assistant", "user", "user"])
        self.assertEqual(ms[0]["content"], "hello {x}\n\n"); self.assertEqual(ms[-1]["content"], "Task: do it\n\n")
        self.assertEqual(messages_to_text(ms), self.FIXTURE)
        with self.assertRaises(ValueError):
            text_to_messages("no role\nUSER:\nx")
        with self.assertRaises(ValueError):
            messages_to_text([{"role": "system", "content": "s"}])

    def test_first_block_only_and_content_cut(self):
        t = "Plan.\n```python\nprint(1)\n```\nthen\n```python\nprint(2)\n```\ntail"
        self.assertEqual(extract_code(t), ("print(1)", "Plan.\n```python\nprint(1)\n```"))
        self.assertEqual(extract_code("x\n```python\nfor p in pages:\n    print(p)"),
                         ("for p in pages:\n    print(p)", "x\n```python\nfor p in pages:\n    print(p)\n```"))
        self.assertEqual(extract_code("```python\n   \n```"), ("", "```python\n   \n```"))
        self.assertEqual(extract_code("I would call the API now."), ("", "I would call the API now."))
        self.assertEqual(extract_code("```\nx = 1\n```"), ("", "```\nx = 1\n```"))      # only ```python fences count
        self.assertEqual(extract_code(None), ("", "")); self.assertEqual(extract_code(""), ("", ""))

    def test_truncate_output_and_wrapper(self):
        self.assertEqual(truncate_output("a" * 20000), "a" * 20000)
        self.assertEqual(truncate_output("a" * 20001), "a" * 20000 + "\n[REST NOT SHOWN FOR BREVITY]")
        self.assertEqual(truncate_output("a" * 20001, None), "a" * 20001)             # ReAct scaffold: no cap
        self.assertEqual(output_message("a" * 30, 10)["content"], "Output:\n```\n" + "a" * 10 + "\n[REST NOT SHOWN FOR BREVITY]```\n\n")
        self.assertEqual(output_message("ok\n"), {"role": "user", "content": "Output:\n```\nok\n```\n\n"})
        self.assertEqual(output_message(""), {"role": "user", "content": "Output:\n```\n```\n\n"})

    def _history(self, n_steps, obs_len):
        ms = [{"role": "user", "content": "intro\n\n"}, {"role": "user", "content": "My name is X.\nTask: T\n\n"}]
        for i in range(n_steps):
            ms.append({"role": "assistant", "content": f"```python\nstep{i}\n```\n\n"})
            ms.append({"role": "user", "content": f"Output:\n```\n{'o' * obs_len}```\n\n"})
        return ms

    def test_trimmed_messages_blanks_oldest_observations_then_drops_messages(self):
        ms = self._history(8, 100)
        self.assertEqual(trimmed_messages(ms, 2, 10 ** 6), ms)
        out = trimmed_messages(ms, 2, 800)
        self.assertEqual(out[:2], ms[:2]); self.assertEqual(len(out), len(ms))
        blanked = [i for i, m in enumerate(out) if m["content"] == NOT_SHOWN]
        self.assertEqual(blanked, [3, 5, 7, 9, 11, 13])                 # oldest first, never among the last 5 messages
        self.assertEqual(out[-1], ms[-1]); self.assertEqual(out[15], ms[15])
        self.assertEqual(ms[3]["content"][:7], "Output:")               # input untouched
        out = trimmed_messages(ms, 2, 300)                               # blanking is not enough: drop after the task message
        self.assertTrue(out[1]["content"].endswith(TRIMMED)); self.assertLess(len(out), len(ms))
        self.assertEqual(out[-1], ms[-1]); self.assertEqual(out[0], ms[0])

    def test_retry_messages(self):
        ms = retry_messages("REFLECTION", True)
        self.assertEqual(ms, [{"role": "user", "content": RETRY_INTRO}, {"role": "assistant", "content": "REFLECTION\n\n"},
                              {"role": "user", "content": RETRY_USE_WITH_PLAYBOOK}])
        self.assertEqual(retry_messages("R", False)[2]["content"], RETRY_USE)
        self.assertIn("along with the playbook", RETRY_USE_WITH_PLAYBOOK); self.assertNotIn("playbook", RETRY_USE)

    def test_env_clean_and_history(self):
        self.assertTrue(env_clean(True, "Marked the active task complete."))
        self.assertFalse(env_clean(True, "Execution failed. Traceback:\n  KeyError"))
        self.assertFalse(env_clean(True, "...Traceback (most recent call last)..."))
        self.assertFalse(env_clean(False, ""))
        h = conversation_history([{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}])
        self.assertEqual(h, "\n\n=== FULL CONVERSATION HISTORY ===\n[0] USER: a\n\n[1] ASSISTANT: b\n\n")


class TestChatLLM(unittest.TestCase):
    """The client sends model / messages / max_tokens / temperature only, makes at most `attempts` calls
    10 s apart (0 here) and gives up at once on a context-length overflow."""
    def _llm(self, create, attempts=3):
        completions = type("Completions", (), {"create": staticmethod(create)})()
        client = type("Client", (), {"chat": type("Chat", (), {"completions": completions})()})()
        with mock.patch.dict(sys.modules, {"openai": type("openai", (), {"OpenAI": staticmethod(lambda **kw: client)})}):
            return ChatLLM("http://localhost:1/v1", "m", attempts=attempts, retry_after_s=0, log=lambda m: None)

    def test_retries_then_reply(self):
        seen = []
        def create(**kw):
            seen.append(kw)
            if len(seen) < 3:
                raise ConnectionError("down")
            return type("R", (), {"choices": [type("C", (), {"message": type("M", (), {"content": "hi"})()})()],
                                  "usage": type("U", (), {"prompt_tokens": 5, "completion_tokens": 2})()})()
        llm = self._llm(create)
        self.assertEqual(llm.chat([{"role": "user", "content": "q"}], 8192, 0.0), "hi")
        self.assertEqual((llm.calls, llm.failures, llm.prompt_tokens, llm.completion_tokens), (3, 2, 5, 2))
        self.assertEqual(seen[-1], {"model": "m", "messages": [{"role": "user", "content": "q"}], "max_tokens": 8192, "temperature": 0.0})

    def test_gives_up_after_the_attempts_or_at_once_on_overflow(self):
        def down(**kw):
            raise ConnectionError("down")
        llm = self._llm(down)
        with self.assertRaises(ConnectionError):
            llm.chat([], 10, 0.0)
        self.assertEqual(llm.calls, 3)
        def overflow(**kw):
            raise ValueError("This model's maximum context length is 131072 tokens")
        llm = self._llm(overflow)
        with self.assertRaises(ValueError):
            llm.chat([], 10, 0.0)
        self.assertEqual(llm.calls, 1)


class TestCritic(unittest.TestCase):
    def test_input_binding(self):
        p = read_prompt(CRITIC_PROMPT_PATHS[True])
        s = build_critic_input(p, "PB TEXT", None, "\n\nHIST")
        self.assertNotIn("{{", s.replace("{{\n", "").replace("}}", ""))   # only the JSON example braces remain
        self.assertIn("<<<PLAYBOOK_GUIDE>>>\nPB TEXT\n<<<PLAYBOOK_GUIDE>>>", s)
        self.assertIn("<<<PRIOR_REFLECTION>>>\nN/A\n<<<PRIOR_REFLECTION>>>", s)
        self.assertEqual(s.count("See full conversation history below"), 2); self.assertTrue(s.endswith("\n\nHIST"))
        self.assertIn("<<<PRIOR_REFLECTION>>>\nPREV\n<<<PRIOR_REFLECTION>>>", build_critic_input(p, "x", "PREV", ""))
        c = AppWorldCritic(FakeLLM(["{}"]), adaptive=False)
        c.reflect({}, Trajectory(text="H", completed=True), "", None, 1, 3)
        self.assertIn("<<<PLAYBOOK_GUIDE>>>\n" + EMPTY_PLAYBOOK + "\n", c.llm.requests[0][0]["content"])
        self.assertEqual([m["role"] for m in c.llm.requests[0]], ["user"])

    def test_parsing(self):
        raw = ('```json\n{"trajectory_verdict": "no_errors", "confidence": 0.9, "refine": "false", "store": true, '
               '"novelty_reason": "covered by [shr-00001]", "reasoning": "r", "key_insight": "L"}\n```')
        r, conf = parse_critic_reply(raw, True, env_clean=False)
        self.assertTrue(r.correct and r.parsed and r.store); self.assertFalse(r.refine)
        self.assertEqual((r.lesson, r.critique, r.raw, conf), ("L", raw, raw, 0.9))
        self.assertEqual(r.novelty_reason, "covered by [shr-00001]")
        r, conf = parse_critic_reply('{"trajectory_verdict": "Errors_Found", "key_insight": "L"}', False, True)
        self.assertFalse(r.correct); self.assertTrue(r.refine); self.assertFalse(r.store); self.assertEqual(conf, 0.0)
        self.assertTrue(parse_critic_reply('{"trajectory_verdict": " NO_ERRORS "}', False, False)[0].correct)
        # no key: the quoted literal in the text decides; neither: the env signal
        self.assertTrue(parse_critic_reply('verdict "no_errors" I think', True, False)[0].correct)
        self.assertFalse(parse_critic_reply('"errors_found" surely', True, True)[0].correct)
        for clean in (True, False):
            r, _ = parse_critic_reply("no json at all", True, clean)
            self.assertEqual((r.correct, r.parsed, r.lesson, r.critique), (clean, False, "", "no json at all"))
        self.assertEqual(parse_critic_reply('{"confidence": "high"}', True, True)[1], 0.0)
        self.assertEqual(parse_critic_reply('{"confidence": 7}', True, True)[1], 1.0)

    def test_store_confidence_gate(self):
        low = '{"trajectory_verdict": "no_errors", "confidence": 0.5, "store": true, "novelty_reason": "covered by [shr-00001]", "key_insight": "L"}'
        c = AppWorldCritic(FakeLLM([low]), adaptive=True)
        r = c.reflect({}, Trajectory(text="H", completed=True), "pb", None, 1, 3)
        self.assertTrue(r.correct); self.assertFalse(r.store); self.assertEqual((r.novelty_reason, r.cited_id, r.lesson), ("", "", "L"))
        self.assertEqual(r.critique, low)                                              # the retry / consolidator text is untouched
        self.assertEqual(c.drain()[0]["low_confidence"], {"store": True, "novelty_reason": "covered by [shr-00001]", "cited_id": ""})
        for conf, kept in ((0.7, True), (0.69, False), ("0.9", True)):
            c = AppWorldCritic(FakeLLM([low.replace("0.5", json.dumps(conf))]), adaptive=True)
            r = c.reflect({}, Trajectory(text="H", completed=True), "pb", None, 1, 3)
            self.assertEqual((r.store, "low_confidence" in c.drain()[0]), (kept, not kept), conf)
        c = AppWorldCritic(FakeLLM([low]), adaptive=True, store_conf=0.5)
        self.assertTrue(c.reflect({}, Trajectory(text="H", completed=True), "pb", None, 1, 3).store)
        c = AppWorldCritic(FakeLLM([low.replace('"store": true', '"store": false')]), adaptive=True)   # no store asked: no gate
        r = c.reflect({}, Trajectory(text="H", completed=True), "pb", None, 1, 3)
        self.assertEqual(r.novelty_reason, "covered by [shr-00001]"); self.assertNotIn("low_confidence", c.drain()[0])

    def test_env_critic_makes_no_call(self):
        c = EnvCritic()
        self.assertTrue(c.reflect({}, Trajectory(text="H", completed=True), "", None, 1, 1).correct)
        r = c.reflect({}, Trajectory(text="H", completed=False), "", None, 1, 1)
        self.assertFalse(r.correct); self.assertFalse(r.parsed); self.assertTrue(r.refine); self.assertFalse(r.store)
        self.assertEqual((c.drain(), c.calls, c.parse_failures, c.call_failures), ([], 0, 0, 0))

    def test_call_failure(self):
        c = AppWorldCritic(FakeLLM(fail=True), adaptive=True, log=lambda m: None)
        r = c.reflect({}, Trajectory(text="H", completed=True), "pb", None, 1, 3)
        self.assertTrue(r.failed); self.assertEqual(r.verdict, "none"); self.assertEqual(c.call_failures, 1)
        self.assertEqual(c.drain()[0]["confidence"], None)


class TestConsolidator(unittest.TestCase):
    def _episode(self, *raws):
        st = EpisodeState()
        for i, raw in enumerate(raws):
            st.rounds.append(RoundRecord(i + 1, i == len(raws) - 1, Reflection("correct", raw=raw, lesson=f"L{i + 1}")))
        st.admitted = True
        return st

    def test_reflection_selection(self):
        self.assertEqual(select_reflection(self._episode("R1")), "R1")
        self.assertEqual(select_reflection(self._episode("R1", "R2")), VALIDATED_PREFIX + "R1")
        self.assertEqual(select_reflection(self._episode("R1", "R2", "R3")), VALIDATED_PREFIX + "R2")
        self.assertTrue(VALIDATED_PREFIX.startswith("[VALIDATED BY RETRY: after applying this reflection in a retry, "))

    def test_curator_input(self):
        s = build_curator_input(read_prompt(CONSOLIDATOR_PROMPT_PATH), "REFL {x}", "PB {y}", "Pay {z}", "\n\nHIST")
        self.assertIn("`Pay {z}`", s); self.assertIn("`PB {y}`", s); self.assertIn("`REFL {x}`", s)
        self.assertIn('{\n  "reasoning": "[Your chain', s)                # the example braces are single after .format
        self.assertEqual(s.count("`See full conversation history below`"), 1); self.assertTrue(s.endswith("\n\nHIST"))

    def test_curator_adds_bullets(self):
        pb = SectionedPlaybook.load(SEED, "plain")
        llm = FakeLLM(['{"reasoning": "r", "operations": [{"type": "ADD", "section": "verification_checklist", "content": "Check pages."},'
                       ' {"type": "ADD", "section": "no such", "content": "dropped"}]}'])
        c = CuratorConsolidator(llm, log=lambda m: None)
        traj = Trajectory(text="\n\nHIST", completed=True, meta={"instruction": "Count playlists"})
        self.assertEqual(c.consolidate(pb, self._episode("R1", "R2"), {}, traj), "vc-00009")
        self.assertIn("## VERIFICATION CHECKLIST\n\n[vc-00009] Check pages.\n## TROUBLESHOOTING", pb.text)
        self.assertNotIn("dropped", pb.text); self.assertEqual(c.last_error, "")
        req = llm.requests[0][0]["content"]
        self.assertIn(VALIDATED_PREFIX + "R1", req); self.assertIn("`Count playlists`", req); self.assertTrue(req.endswith("HIST"))
        self.assertIn("[shr-00001]", req)
        # nothing to add is still a success; an empty reflection makes no call
        llm.replies = ['{"reasoning": "r", "operations": []}']
        self.assertEqual(c.consolidate(pb, self._episode("R1"), {}, traj), ""); self.assertEqual(c.last_error, "")
        self.assertEqual(c.consolidate(pb, self._episode(""), {}, traj), ""); self.assertEqual(c.calls, 2)

    def test_curator_failures_leave_the_playbook_unchanged(self):
        traj = Trajectory(text="H", completed=True, meta={"instruction": "I"})
        for llm in (FakeLLM(["not json"]), FakeLLM(['{"reasoning": "r", "operations": [{"type": "UPDATE"}]}']), FakeLLM(fail=True)):
            pb = SectionedPlaybook.load(SEED, "plain"); before = pb.text
            c = CuratorConsolidator(llm, log=lambda m: None)
            self.assertEqual(c.consolidate(pb, self._episode("R1"), {}, traj), "")
            self.assertTrue(c.last_error); self.assertEqual(pb.text, before); self.assertEqual(c.failures, 1)
        pb = SectionedPlaybook("stray line before the first header\n## OTHERS", "plain")   # apply raises KeyError
        c = CuratorConsolidator(FakeLLM(['{"reasoning": "r", "operations": [{"type": "ADD", "section": "others", "content": "x"}]}']),
                                log=lambda m: None)
        self.assertEqual(c.consolidate(pb, self._episode("R1"), {}, traj), ""); self.assertIn("KeyError", c.last_error)

    def test_insight_consolidator(self):
        pb = SectionedPlaybook.from_skeleton("plain")
        self.assertEqual(InsightConsolidator().consolidate(pb, self._episode("R1", "R2"), {}, None), "misc-00001")
        self.assertTrue(pb.text.endswith("## OTHERS\n[misc-00001] L2"))
        self.assertEqual(InsightConsolidator().consolidate(pb, EpisodeState(), {}, None), "")


class _FakeWorld:
    """Stand-in for appworld.AppWorld: scripted outputs; complete_task submits."""
    instances = []
    def __init__(self, task_id, experiment_name, **kw):
        self.kw, self.executed, self.done = kw, [], False
        self.task = type("T", (), {"instruction": "Count my playlists.", "app_descriptions": {"api_docs": "Docs.", "spotify": "Music."},
                                   "supervisor": {"first_name": "Ann", "last_name": "Lee", "email": "a@x.org", "phone_number": "1"}})()
        d = tempfile.mkdtemp(); self.output_directory = os.path.join(d, "outputs", experiment_name, "tasks", task_id)
        self.output_misc_directory = os.path.join(self.output_directory, "misc")
        _FakeWorld.instances.append(self)
    def __enter__(self): return self
    def __exit__(self, *a): pass
    def execute(self, code):
        self.executed.append(code)
        if "complete_task" in code:
            self.done = True
            return ""
        return "No code available to execute." if not code else f"out{len(self.executed)}"
    def task_completed(self): return self.done


@unittest.skipUnless(HAS_JINJA2, "jinja2 not installed")
class TestSolveLoop(unittest.TestCase):
    def _solve(self, replies, memory="PB", critique=None, world=_FakeWorld, **kw):
        llm = FakeLLM(replies)
        solver = AppWorldSolver(llm, "exp", max_steps=kw.pop("max_steps", 10), round1_experiment=None, log=lambda m: None, **kw)
        with mock.patch.dict(sys.modules, {"appworld": type("A", (), {"AppWorld": world})}):
            traj = solver.solve({"task_id": "t1"}, memory, critique)
        return traj, llm, solver

    def test_messages_and_steps(self):
        traj, llm, solver = self._solve(["Thought only, no block.", "x\n```python\nprint(1)\n```\nignored\n```python\nprint(2)\n```",
                                         "```python\napis.supervisor.complete_task(answer=3)\n```"])
        self.assertTrue(traj.completed); self.assertFalse(traj.failed); self.assertEqual(traj.meta["steps"], 3)
        r1 = llm.requests[0]
        self.assertTrue(all(m["role"] in ("user", "assistant") for m in r1)); self.assertEqual(r1[0]["role"], "user")
        self.assertTrue(r1[0]["content"].startswith("I am your supervisor")); self.assertIn("### PLAYBOOK BEGIN\nPB\n### PLAYBOOK END", r1[0]["content"])
        self.assertIn('[\n {\n  "name": "api_docs",\n  "description": "Docs."\n },', messages_to_text(r1))
        self.assertTrue(r1[-1]["content"].endswith("My name is: Ann Lee. My personal email is a@x.org and phone number is 1.\n"
                                                    "Task: Count my playlists.\n\n"))
        n = traj.meta["num_instruction_messages"]; self.assertEqual(n, len(r1))
        # step 2: the no-code reply stays in context, "" was executed and its output shown
        self.assertEqual(_FakeWorld.instances[-1].executed, ["", "print(1)", "apis.supervisor.complete_task(answer=3)"])
        r2 = llm.requests[1]
        self.assertEqual(r2[n:], [{"role": "assistant", "content": "Thought only, no block.\n\n"},
                                  {"role": "user", "content": "Output:\n```\nNo code available to execute.```\n\n"}])
        # step 3: the assistant content is cut after the first block
        self.assertEqual(llm.requests[2][n + 2]["content"], "x\n```python\nprint(1)\n```\n\n")
        self.assertEqual(llm.requests[2][n + 3]["content"], "Output:\n```\nout2```\n\n")
        # the critic's history ends with the last assistant turn: the final output is never appended
        self.assertTrue(traj.text.endswith("ASSISTANT: ```python\napis.supervisor.complete_task(answer=3)\n```\n\n\n\n"))
        self.assertEqual(traj.text[:len("\n\n=== FULL CONVERSATION HISTORY ===\n[0] USER: ")], "\n\n=== FULL CONVERSATION HISTORY ===\n[0] USER: ")
        self.assertEqual(traj.answer, "apis.supervisor.complete_task(answer=3)"); self.assertEqual(traj.meta["last_output"], "out2")
        self.assertEqual(_FakeWorld.instances[-1].kw, {"random_seed": 123, "timeout_seconds": 100, "load_ground_truth": False})
        after = random.random(); random.seed(123)                 # Python's RNG was re-seeded with the world seed at round start
        self.assertEqual(after, random.random())

    def test_retry_injection_and_no_memory(self):
        traj, llm, _ = self._solve(["```python\napis.supervisor.complete_task()\n```"], memory="PB", critique="REFL")
        r1 = llm.requests[0]; n = traj.meta["num_instruction_messages"]
        self.assertEqual(r1[n:], retry_messages("REFL", True)); self.assertEqual(len(r1), n + 3)
        traj, llm, _ = self._solve(["```python\napis.supervisor.complete_task()\n```"], memory="", critique="REFL")
        self.assertEqual(llm.requests[0][-1]["content"], RETRY_USE)
        self.assertIn("### PLAYBOOK BEGIN\n(empty)\n### PLAYBOOK END", llm.requests[0][0]["content"])

    def test_error_in_last_output_and_max_steps(self):
        class Broken(_FakeWorld):
            def execute(self, code):
                super().execute(code); return "Execution failed. Traceback:\nKeyError"
        traj, _, _ = self._solve(["```python\napis.supervisor.complete_task()\n```"], world=Broken)
        self.assertTrue(traj.meta["task_completed"]); self.assertFalse(traj.completed)
        traj, llm, _ = self._solve(["```python\nprint(1)\n```"], max_steps=4)
        self.assertEqual((len(llm.requests), traj.completed, traj.meta["steps"]), (4, False, 4))

    def test_react_scaffold(self):
        big = "o" * 25000
        class Verbose(_FakeWorld):
            def execute(self, code):
                super().execute(code); return big if len(self.executed) == 1 else ""
        settings = ra.solver_settings("react")
        self.assertEqual(settings, {"template_path": REACT_PROMPT_PATH, "max_output_length": 50000, "output_cap": None})
        traj, llm, _ = self._solve(["```python\nprint(1)\n```", "```python\napis.supervisor.complete_task()\n```"],
                                   memory="", world=Verbose, **settings)
        r1 = llm.requests[0]
        self.assertTrue(r1[0]["content"].startswith("I am your supervisor, and you are an AI Assistant whose job is to complete"))
        self.assertNotIn("PLAYBOOK", messages_to_text(r1)); self.assertNotIn(EMPTY_PLAYBOOK, messages_to_text(r1))
        self.assertTrue(r1[-1]["content"].endswith("Task: Count my playlists.\n\n"))
        self.assertEqual(llm.requests[1][-1]["content"], "Output:\n```\n" + big + "```\n\n")     # shown whole
        traj, llm, _ = self._solve(["```python\nprint(1)\n```"] * 8, memory="", world=Verbose, max_steps=8,
                                   **{**settings, "max_output_length": 25000})
        self.assertIn(NOT_SHOWN, [m["content"] for m in llm.requests[-1]])                    # trimmed at the scaffold's limit
        self.assertEqual(ra.solver_settings("refine"), {"template_path": REACT_PROMPT_PATH})
        for mode in ("memory", "remo", "adaremo"):
            self.assertEqual(ra.solver_settings(mode), {"template_path": SOLVER_PROMPT_PATH})
        self.assertEqual(REACT_MAX_OUTPUT_LENGTH, 50000)

    def test_model_failure_is_a_failed_attempt(self):
        solver = AppWorldSolver(FakeLLM(fail=True), "exp", round1_experiment=None, log=lambda m: None)
        with mock.patch.dict(sys.modules, {"appworld": type("A", (), {"AppWorld": _FakeWorld})}):
            traj = solver.solve({"task_id": "t1"}, "PB", None)
        self.assertTrue(traj.failed); self.assertFalse(traj.completed); self.assertIn("ConnectionError", traj.meta["error"])


class FakeSolver:
    """Mimics AppWorldSolver's contract: appends a meta per round, returns the history."""
    def __init__(self, misc_dir, clean_by_round):
        self.misc_dir, self.clean_by_round, self.round_metas, self.seen = misc_dir, clean_by_round, [], []
    def solve(self, task, memory_text, critique):
        r = len(self.round_metas) + 1
        self.seen.append((task["task_id"], memory_text, critique))
        clean = self.clean_by_round[min(r, len(self.clean_by_round)) - 1]
        meta = {"task_id": task["task_id"], "round": r, "instruction": "Do X", "supervisor": {}, "misc_dir": self.misc_dir,
                "output_dir": os.path.dirname(self.misc_dir), "steps": 3 + r, "task_completed": clean, "env_clean": clean,
                "error": "", "elapsed_s": 1.5, "memory_chars": len(memory_text), "critique_chars": len(critique or ""),
                "last_output": "done", "messages": [{"role": "user", "content": "u"}], "num_instruction_messages": 1,
                "steps_full": [{"step": 1, "reply": "r", "code": "c", "output": "o"}]}
        self.round_metas.append(meta)
        return Trajectory(text=f"\n\nHIST{r}", answer="apis.supervisor.complete_task()", completed=clean, meta=meta)


REFL_BAD = '{"trajectory_verdict": "errors_found", "refine": true, "store": false, "key_insight": "paginate"}'
REFL_OK = '{"trajectory_verdict": "no_errors", "confidence": 0.9, "refine": false, "store": true, "novelty_reason": "new", "key_insight": "L"}'
REFL_DUP = '{"trajectory_verdict": "no_errors", "confidence": 0.9, "store": false, "novelty_reason": "covered by [psw-00007]", "key_insight": "L"}'
REFL_LOW = '{"trajectory_verdict": "no_errors", "confidence": 0.5, "store": true, "novelty_reason": "covered by [psw-00007]", "key_insight": "L"}'
CURATOR_OK = '{"reasoning": "r", "operations": [{"type": "ADD", "section": "others", "content": "New bullet."}]}'


class TestVersions(unittest.TestCase):
    def test_check_versions(self):
        self.assertEqual((ra.APPWORLD_VERSION, ra.DATA_VERSION), ("0.1.4.dev0", "0.1.0"))
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(SystemExit):
                ra.check_versions(root, ra.APPWORLD_VERSION)                        # no data/version.txt
            os.makedirs(os.path.join(root, "data"))
            for data_version, appworld_version in (("0.2.0", ra.APPWORLD_VERSION), (ra.DATA_VERSION, "0.1.3.post1")):
                with open(os.path.join(root, "data", "version.txt"), "w") as f:
                    f.write(data_version + "\n")
                with self.assertRaises(SystemExit):
                    ra.check_versions(root, appworld_version)
            ra.check_versions(root, ra.APPWORLD_VERSION)


class TestRecords(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.misc = os.path.join(self.d, "aw_out", "tasks", "t1", "misc")

    def _agent(self, mode, K, solver, replies, cli_mode=None, **kw):
        llm = FakeLLM(replies)
        critic = EnvCritic() if mode == "react" else AppWorldCritic(llm, adaptive=(mode == "adaremo"))
        cons = CuratorConsolidator(llm, log=lambda m: None)
        agent = ra.AppWorldAgent(ra.make_config(mode, K), solver, critic, ra.load_playbook(SEED), cons,
                                 cli_mode=cli_mode or mode, run_dir=self.d, **kw)
        return agent, llm

    def test_cross_round_records_and_resume(self):
        agent, llm = self._agent("adaremo", 3, FakeSolver(self.misc, [False, True]), [REFL_BAD, REFL_OK, CURATOR_OK])
        rec = ra.run_task(agent, "t1", 0, self.d, log=lambda m: None)
        self.assertEqual((rec["gate"], rec["store_decision"], rec["entry_id"], len(rec["rounds"])), ("cross_round_validated", "stored", "misc-00009", 2))
        self.assertIn("[misc-00009] New bullet.", agent.playbook.text)
        # the retry carried the WHOLE round-1 reflection; the round-2 critic saw it as the prior reflection
        self.assertEqual(agent.solver.seen[1][2], REFL_BAD); self.assertEqual(agent.solver.seen[1][1], SectionedPlaybook.load(SEED, "plain").text)
        self.assertIn("<<<PRIOR_REFLECTION>>>\n" + REFL_BAD + "\n<<<PRIOR_REFLECTION>>>", llm.requests[1][0]["content"])
        self.assertIn("<<<PRIOR_REFLECTION>>>\nN/A\n", llm.requests[0][0]["content"])
        self.assertTrue(llm.requests[0][0]["content"].endswith("\n\nHIST1")); self.assertTrue(llm.requests[1][0]["content"].endswith("\n\nHIST2"))
        # the consolidator got the flipping reflection with the prefix and the accepted attempt's history
        self.assertIn(VALIDATED_PREFIX + REFL_BAD, llm.requests[2][0]["content"]); self.assertTrue(llm.requests[2][0]["content"].endswith("HIST2"))
        eps = ra.read_episodes(self.d); e = eps[0]
        self.assertEqual((e["task_id"], e["mode"], e["K"], e["playbook_entries"]), ("t1", "adaremo", 3, 9))
        self.assertEqual(e["rounds"][0]["solver"]["steps"], 4); self.assertEqual(e["rounds"][1]["critic"]["confidence"], 0.9)
        self.assertEqual(e["rounds"][0]["raw"], REFL_BAD); self.assertEqual(e["elapsed_s"], 3.0)
        tj = json.load(open(os.path.join(self.d, "trajs", "t1.json")))
        self.assertEqual([r["round"] for r in tj["rounds"]], [1, 2]); self.assertEqual(tj["rounds"][1]["critic_raw"], REFL_OK)
        rr = json.load(open(os.path.join(self.misc, "remo_rounds.json")))
        self.assertTrue(rr["clean"]); self.assertEqual(rr["curation"], "cross_round_validated")
        self.assertEqual(rr["rounds"][0]["reflection"], REFL_BAD); self.assertEqual(rr["rounds"][1]["confidence"], 0.9)
        self.assertEqual([(r["env_clean"], r["verdict_no_errors"], r["store"]) for r in rr["rounds"]], [(False, False, False), (True, True, True)])
        agent2, _ = self._agent("adaremo", 3, FakeSolver(self.misc, [True]), [])
        self.assertEqual(agent2.done_indices(), {0}); self.assertEqual(len(agent2.playbook), 9)

    def test_round1_clean_stores_the_whole_reflection_and_reinforce(self):
        agent, llm = self._agent("remo", 3, FakeSolver(self.misc, [True]), [REFL_OK, CURATOR_OK])
        rec = ra.run_task(agent, "t1", 0, self.d, log=lambda m: None)
        self.assertEqual((rec["gate"], rec["store_decision"]), ("round1_clean", "stored"))
        self.assertIn("`" + REFL_OK + "`", llm.requests[1][0]["content"])
        agent, llm = self._agent("adaremo", 3, FakeSolver(self.misc, [True]), [REFL_DUP])
        rec = ra.run_task(agent, "t2", 1, self.d, log=lambda m: None)
        self.assertEqual((rec["store_decision"], rec["entry_id"], len(llm.requests)), ("reinforced", "psw-00007", 1))
        self.assertIn("[psw-00007] Many APIs return items in \"pages\". Make sure to run through all the pages by looping over `page_index`. [confirmed x2]",
                      agent.playbook.text)

    def test_curator_error_and_never_clean(self):
        agent, llm = self._agent("remo", 2, FakeSolver(self.misc, [True]), [REFL_OK, "garbage"])
        rec = ra.run_task(agent, "t1", 0, self.d, log=lambda m: None)
        self.assertEqual(rec["store_decision"], "curator_error"); self.assertEqual(len(agent.playbook), 8)
        self.assertEqual(ra.read_episodes(self.d)[0]["store_decision"], "curator_error")
        agent, llm = self._agent("remo", 2, FakeSolver(self.misc, [False]), [REFL_BAD])
        rec = ra.run_task(agent, "t2", 1, self.d, log=lambda m: None)
        self.assertEqual((rec["gate"], rec["store_decision"], len(rec["rounds"]), len(llm.requests)), ("never_clean", "skipped", 2, 2))

    def test_low_confidence_store_is_skipped(self):
        agent, llm = self._agent("adaremo", 3, FakeSolver(self.misc, [True]), [REFL_LOW])
        rec = ra.run_task(agent, "t1", 0, self.d, log=lambda m: None)
        self.assertEqual((rec["gate"], rec["store_decision"], rec["entry_id"], len(llm.requests)), ("round1_clean", "skipped_lowconf", "", 1))
        self.assertEqual(agent.playbook.text, SectionedPlaybook.load(SEED, "plain").text)     # neither stored nor reinforced
        self.assertEqual((agent.policy.store_window, agent.policy.frozen), ([False], False))
        e = ra.read_episodes(self.d)[0]; rd = e["rounds"][0]
        self.assertEqual((rd["store"], rd["novelty_reason"], rd["critic"]["confidence"]), (True, "covered by [psw-00007]", 0.5))
        self.assertEqual(rd["critic"]["low_confidence"]["store"], True); self.assertEqual(e["store_decision"], "skipped_lowconf")
        rr = json.load(open(os.path.join(self.misc, "remo_rounds.json")))
        self.assertEqual((rr["rounds"][0]["store"], rr["store_decision"]), (True, "skipped_lowconf"))
        # frozen memory at a probe index: a low-confidence store neither unfreezes nor writes
        agent, llm = self._agent("adaremo", 3, FakeSolver(self.misc, [True]), [REFL_LOW])
        agent.policy.frozen, agent.policy.store_window = True, [False] * 20
        rec = ra.run_task(agent, "t2", 19, self.d, log=lambda m: None)
        self.assertEqual((rec["store_decision"], agent.policy.frozen, agent.policy.store_window[-1], len(llm.requests)), ("skipped_lowconf", True, False, 1))
        agent, llm = self._agent("adaremo", 3, FakeSolver(self.misc, [True]), [REFL_OK, CURATOR_OK])
        agent.policy.frozen, agent.policy.store_window = True, [False] * 17 + [True] * 3
        rec = ra.run_task(agent, "t3", 19, self.d, log=lambda m: None)                        # confident store: probe unfreezes
        self.assertEqual((rec["store_decision"], agent.policy.frozen, len(llm.requests)), ("stored", False, 2))

    def test_unparseable_verdict_follows_env_clean(self):
        agent, _ = self._agent("remo", 2, FakeSolver(self.misc, [True]), ["no json", CURATOR_OK])
        rec = ra.run_task(agent, "t1", 0, self.d, log=lambda m: None)
        self.assertEqual((rec["gate"], rec["rounds"][0]["verdict"], rec["rounds"][0]["parsed"]), ("round1_clean", "correct", False))
        agent, _ = self._agent("remo", 2, FakeSolver(self.misc, [False]), ["no json"])
        self.assertEqual(ra.run_task(agent, "t2", 1, self.d, log=lambda m: None)["gate"], "never_clean")

    def test_freeze_after_reads_but_never_writes(self):
        agent, llm = self._agent("adaremo", 1, FakeSolver(self.misc, [True]), [REFL_OK, CURATOR_OK, REFL_OK], freeze_after=1)
        ra.run_task(agent, "t0", 0, self.d, log=lambda m: None)      # index 0 < A: consolidates
        ra.run_task(agent, "t1", 1, self.d, log=lambda m: None)      # index 1 >= A: read-only, no curator call
        eps = ra.read_episodes(self.d)
        self.assertEqual([e["store_decision"] for e in eps], ["stored", "skipped_readonly"]); self.assertEqual(len(llm.requests), 3)
        self.assertEqual(len(agent.playbook), 9); self.assertTrue(agent.memory_readonly); self.assertFalse(eps[0]["memory_readonly"])
        self.assertEqual(agent.solver.seen[1][1], agent.playbook.text)                      # memory still injected
        self.assertEqual(agent.policy.store_window, [True]); self.assertEqual(json.load(open(os.path.join(self.d, "policy_state.json")))["store_window"], [True])
        self.assertTrue(json.load(open(os.path.join(self.misc, "remo_rounds.json")))["memory_frozen"])

    def test_no_memory_arm_and_summary(self):
        agent, llm = self._agent("react", None, FakeSolver(self.misc, [True]), [], cli_mode="react")
        ra.run_task(agent, "t0", 0, self.d, log=lambda m: None)
        self.assertEqual(agent.solver.seen[0][1], ""); self.assertEqual(len(agent.playbook), 8); self.assertEqual(len(llm.requests), 0)
        eps = ra.read_episodes(self.d)
        self.assertEqual((eps[0]["store_decision"], eps[0]["gate"], eps[0]["rounds"][0]["verdict"]), ("no_memory", "round1_clean", "correct"))
        self.assertNotIn("critic", eps[0]["rounds"][0])
        agent, llm = self._agent("refine", 2, FakeSolver(self.misc, [False, True]), [REFL_BAD, REFL_OK])
        rec = ra.run_task(agent, "t1", 1, self.d, log=lambda m: None)
        self.assertEqual((rec["gate"], rec["store_decision"], len(llm.requests)), ("cross_round_validated", "no_memory", 2))
        self.assertEqual(agent.solver.seen[1], ("t1", "", REFL_BAD))
        self.assertIn("<<<PLAYBOOK_GUIDE>>>\n" + EMPTY_PLAYBOOK + "\n", llm.requests[0][0]["content"])
        eps = ra.read_episodes(self.d)[:1]
        s = ra.summarize(eps, agent.playbook, {"TGC": 100.0, "SGC": 100.0, "n_evaluated": 1, "per_task_success": {"t0": True}, "errors": {}}, None)
        self.assertEqual((s["n_tasks"], s["TGC"], s["mean_rounds"], s["memory"]["entries"]), (1, 100.0, 1.0, 8))
        self.assertEqual(s["gate_distribution"], {"round1_clean": 1}); self.assertEqual(s["tgc_by_gate"]["round1_clean"]["rate"], 1.0)

    def test_config_and_records(self):
        c = ra.make_config("react", None); self.assertEqual((c.mode, c.K, c.use_memory, c.inject_cap_chars), ("remo", 1, False, None))
        c = ra.make_config("refine", 3); self.assertEqual((c.mode, c.K, c.use_memory), ("remo", 3, False))
        c = ra.make_config("memory", None); self.assertEqual((c.mode, c.K, c.use_memory), ("remo", 1, True))
        c = ra.make_config("remo", None); self.assertEqual((c.mode, c.K, c.use_memory), ("remo", 3, True))
        c = ra.make_config("adaremo", 2, redundant_mode="gate"); self.assertTrue(c.adaptive); self.assertEqual((c.K, c.redundant_mode), (2, "gate"))
        for bad in (("react", 3), ("refine", 1)):
            with self.assertRaises(SystemExit):
                ra.make_config(*bad)
        self.assertEqual(len(ra.load_playbook(None)), 0); self.assertEqual(ra.load_playbook(None).style, "plain")
        rec = {"gate": "never_clean", "rounds": [{"round": 1, "completed": False, "verdict": "incorrect", "raw": "R", "refine": False,
                                                   "store": False, "novelty_reason": "", "parsed": True, "critic": {"confidence": 0.2}}],
               "store_decision": "skipped", "task_id": "x", "task_index": 3, "stop_reason": "critic_stop", "entry_id": ""}
        rr = ra.rounds_record(rec, memory_frozen=True)
        self.assertFalse(rr["clean"]); self.assertTrue(rr["memory_frozen"])
        self.assertEqual(rr["rounds"][0], {"round": 1, "env_clean": False, "verdict_no_errors": False, "reflection": "R", "confidence": 0.2,
                                           "refine": False, "store": False, "novelty_reason": "", "parsed": True})


if __name__ == "__main__":
    unittest.main()


class TestExperimentName(unittest.TestCase):
    def test_memory_is_rewritten(self):
        notes = []
        self.assertEqual(ra.experiment_name_for("runs/appworld/x_react_r1", None, log=notes.append), "x_react_r1")
        self.assertEqual(notes, [])
        self.assertEqual(ra.experiment_name_for("runs/appworld/x_memory_r1", None, log=notes.append), "x_mem_r1")
        self.assertEqual(ra.experiment_name_for("runs/appworld/x", "memory_arm", log=notes.append), "mem_arm")
        self.assertEqual(len(notes), 2)
