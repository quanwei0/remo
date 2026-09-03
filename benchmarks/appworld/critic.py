"""AppWorld critic: reviews one round's transcript with NO ground truth and no code execution.

The domain part of the prompt (what counts as an error for an API-calling agent) is ours; the
decision fields are `remo.critic.fields_spec(adaptive)` and the reply is parsed by
`remo.critic.parse_reflection`, so the policy receives the same Reflection as on every benchmark.
The critic also sees the objective environment signals of the attempt (task submitted, last output
clean) but never any test result.
"""
from remo.critic import fields_spec, parse_reflection
from remo.interfaces import Reflection, Trajectory

CRITIC_MEMORY_CAP_CHARS = 20000     # playbook chars shown to the critic for the novelty check (adaptive only)

CRITIC_TEMPLATE = """You are reviewing one attempt of an AI assistant that solves a supervisor's everyday task by calling app APIs \
from a Python REPL (one code block per step; the environment prints the result). You see the task and the transcript \
of the attempt. There is NO ground truth and you cannot run code: judge the attempt from the transcript alone, and \
be skeptical — a submitted task is not a solved task.

Task given to the assistant: {instruction}
Supervisor: {supervisor}
Environment signals of this attempt: task submitted via complete_task = {submitted}; last execution output free of \
errors = {last_clean}; steps used = {steps}.
{prior_block}
=== TRANSCRIPT ===
{transcript}
=== END TRANSCRIPT ===

Check, from what the transcript shows:
1. Every requirement of the task was carried out: all items (not just the first page), the right time window, the \
right people, the right direction (sent vs received, from vs to), the right units and answer format.
2. The right source of truth was used: relationships and people from the phone app's contacts, credentials and \
personal data from the supervisor app — never guessed from free text or made up.
3. Nothing was invented: every value used or answered appears in an API output earlier in the transcript.
4. complete_task was called exactly once, at the end, with an answer if and only if the task asks for information, \
and that answer has the right type and follows from the evidence.
5. The final steps ran without errors and no side effect outside the task's scope happened (no extra payment, \
message, deletion or change).
The verdict is "no_errors" ONLY if you are confident the task was solved correctly and completely; if a check fails \
or cannot be confirmed from the transcript, it is "errors_found". When errors_found, the critique must say exactly \
what is wrong and the concrete correction for a fresh attempt (which API, filter, value or step). The lesson is ONE \
imperative, generalizable sentence about working in this environment (an API's actual behaviour or return format, a \
source-of-truth rule, a workflow) that would help on OTHER tasks — never a fact specific to this task.

"""

PRIOR_TEMPLATE = """
This is attempt {round_idx} of {K}. The reviewer of the previous attempt (the environment was reset before this one) \
reported:
{prior_critique}
Judge THIS attempt on its own transcript and say whether those problems were fixed.
"""

MEMORY_TEMPLATE = """

CURRENT MEMORY (playbook entries the assistant was shown; used for the novelty check only):
{memory}

Reply with ONLY the JSON object."""


def _cap_lines(text: str, cap: int) -> str:
    """Whole lines up to `cap` chars (the playbook render is already ranked most-helpful first)."""
    out, total = [], 0
    for line in (text or "").splitlines():
        if total + len(line) + 1 > cap:
            break
        out.append(line); total += len(line) + 1
    return "\n".join(out)


def build_critic_prompt(traj: Trajectory, memory_text: str, prior_critique: str | None, round_idx: int, K: int,
                        adaptive: bool, memory_cap: int = CRITIC_MEMORY_CAP_CHARS) -> str:
    """The template is formatted ONCE; transcript / memory / critique are values, so their braces are safe."""
    template = CRITIC_TEMPLATE + fields_spec(adaptive)
    if adaptive:
        template += MEMORY_TEMPLATE
    prior = ""
    if prior_critique and prior_critique.strip():
        prior = PRIOR_TEMPLATE.format(round_idx=round_idx, K=K, prior_critique=prior_critique.strip()[:3000])
    m = traj.meta or {}
    sup = m.get("supervisor") or {}
    supervisor = (f"{sup.get('first_name', '')} {sup.get('last_name', '')} <{sup.get('email', '')}>, "
                  f"phone {sup.get('phone_number', '')}").strip()
    return template.format(instruction=m.get("instruction", "(unknown)"), supervisor=supervisor,
                           submitted=bool(m.get("task_completed", traj.completed)),
                           last_clean=not any(k in (m.get("last_output") or "") for k in ("Execution failed", "Traceback")),
                           steps=m.get("steps", "?"), prior_block=prior, transcript=traj.text,
                           memory=_cap_lines(memory_text, memory_cap) or "(empty)")


class AppWorldCritic:
    """One chat call per round (synchronous; the AppWorld loop is sequential). Failures are handled
    like on the other benchmarks:

    * unparseable reply: retried `retries` times, then the conservative parse of
      remo.critic.parse_reflection is returned as it is (verdict from the text, refine=True,
      store=False, no lesson, parsed=False) — the policy behaves like ReMo and nothing is written;
    * transport failure: the verdict follows the objective environment signal (`traj.completed`) with
      refine=False, store=False, no lesson and parsed=False — a critic outage neither burns solver
      rounds nor writes memory (a completed round is admitted, flagged parsed=False in the record).
    """

    def __init__(self, client, model: str, adaptive: bool, max_tokens: int = 4096, temperature: float = 0.0,
                 memory_cap: int = CRITIC_MEMORY_CAP_CHARS, retries: int = 1, log=print):
        self.client, self.model, self.adaptive = client, model, adaptive
        self.max_tokens, self.temperature, self.memory_cap, self.retries, self.log = (
            max_tokens, temperature, memory_cap, retries, log)
        self.calls = self.parse_failures = self.call_failures = 0

    def build_prompt(self, traj, memory_text, prior_critique, round_idx, K) -> str:
        return build_critic_prompt(traj, memory_text, prior_critique, round_idx, K, self.adaptive, self.memory_cap)

    def reflect(self, task, traj: Trajectory, memory_text: str, prior_critique: str | None,
                round_idx: int, K: int) -> Reflection:
        prompt = self.build_prompt(traj, memory_text, prior_critique, round_idx, K)
        refl = None
        for _ in range(self.retries + 1):
            self.calls += 1
            try:
                r = self.client.chat.completions.create(model=self.model, max_tokens=self.max_tokens,
                                                        temperature=self.temperature,
                                                        messages=[{"role": "user", "content": prompt}])
                txt = r.choices[0].message.content or ""
            except Exception as e:                       # noqa: BLE001 — transport / server failure
                self.call_failures += 1
                self.log(f"[critic] call failed: {type(e).__name__}: {e}")
                return Reflection(verdict="correct" if traj.completed else "incorrect",
                                  critique=f"(critic call failed: {type(e).__name__}: {e})"[:500],
                                  refine=False, store=False, parsed=False, raw="")
            refl = parse_reflection(txt, self.adaptive)
            if refl.parsed:
                return refl
        self.parse_failures += 1
        return refl
