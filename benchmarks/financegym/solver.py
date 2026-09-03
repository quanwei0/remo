"""FinanceGym solver = the official FinanceHarness `run_research` over the point-in-time
FinanceGymBackend, harness untouched. Importing this module applies two runtime patches (needed
before any harness call) — import it only in the env `remo-financegym`, where the harness is available.

Runtime patches (the same two the paper's runs used):
* httpx.AsyncClient -> subclass with verify=False. The harness builds a fresh AsyncClient per
  search/fetch; each one loaded an SSL context and, with 8 concurrent agents + parse-pool threads,
  that crashed the process in ssl.SSLContext.__new__ (SIGSEGV). All endpoints here are plain http.
* assembly.EQUITY_DATA_SPECS / MARKET_DATA_SPECS = (): the official registry also exposes deferred
  live market-data tools (yfinance) that return CURRENT prices/fundamentals — information from after
  the task cutoff. Dropping them keeps the run PIT-compliant (calc/plan/pure-compute tools stay).
"""
import asyncio
import faulthandler
import sys
import time

from .common import EMBED_MODEL, FH_ROOT, MIN_DOCS, build_question, make_trajectory
from remo.interfaces import Trajectory

faulthandler.enable()
if FH_ROOT not in sys.path:                                   # the vendored harness wins over any other install
    sys.path.insert(0, FH_ROOT)

import httpx as _httpx                                        # noqa: E402


class _NoSSLAsyncClient(_httpx.AsyncClient):                  # SIGSEGV workaround, see module docstring
    def __init__(self, *a, **kw):
        kw.setdefault("verify", False); super().__init__(*a, **kw)


_httpx.AsyncClient = _NoSSLAsyncClient
import financeharness.tools.research.assembly as _asm         # noqa: E402  PIT: no live market-data tools
_asm.EQUITY_DATA_SPECS = (); _asm.MARKET_DATA_SPECS = ()
from financeharness.providers import get_profile               # noqa: E402
from financeharness.research import run_research              # noqa: E402
from financeharness.tools.research.financegym_backend import FinanceGymBackend   # noqa: E402


def harness_profiles(model: str | None, base_url: str | None, reader_base_url: str | None = None):
    """The harness backbone + page-reader profiles (configs/providers.json, env FH_*_BASE_URL) with
    `--model` / `--base-url` applied to both, so one flag pair selects the model everywhere."""
    profile = get_profile()
    reader = get_profile(profile.reader_profile or "vllm-reader")
    upd = {k: v for k, v in (("model", model), ("base_url", base_url)) if v}
    rupd = {k: v for k, v in (("model", model), ("base_url", reader_base_url or base_url)) if v}
    return profile.model_copy(update=upd), reader.model_copy(update=rupd)


class FinanceGymSolver:
    """One FinanceGymBackend per attempt (pinned to the task cutoff). `solve` is a coroutine
    (the harness is async); it retries an attempt whose final report is EMPTY up to
    `max_empty_retries` times (gpt-oss sometimes ends the turn with an empty final message).
    A harness exception or timeout is an attempt with an empty report (termination = the error),
    as in the paper's runs. `plain=True` (--mode baseline) sends the bare question + PIT sentence
    exactly as the leaderboard baseline did."""

    def __init__(self, pit_url: str, embed_url: str, embed_model: str = EMBED_MODEL, timeout_s: float = 120.0,
                 task_timeout_s: float = 3660.0, max_empty_retries: int = 3, min_docs: int = MIN_DOCS,
                 model: str | None = None, base_url: str | None = None, reader_base_url: str | None = None,
                 plain: bool = False, log=print):
        self.pit_url, self.embed_url, self.embed_model, self.timeout_s = pit_url, embed_url, embed_model, timeout_s
        self.task_timeout_s, self.max_empty_retries, self.min_docs, self.log = task_timeout_s, max_empty_retries, min_docs, log
        self.plain = plain
        self.profile, self.reader_profile = harness_profiles(model, base_url, reader_base_url)

    def backend(self, cutoff: str) -> FinanceGymBackend:
        return FinanceGymBackend(cutoff=cutoff, search_url=self.pit_url, embed_url=self.embed_url,
                                 embed_model=self.embed_model, timeout_s=self.timeout_s)

    async def health(self) -> None:
        await self.backend("2025-01-01").health()

    async def _attempt(self, cutoff: str, question: str) -> dict:
        t0 = time.time()
        backend = self.backend(cutoff)
        traj, err = None, None
        try:
            traj = await asyncio.wait_for(
                run_research(question, profile=self.profile, reader_profile=self.reader_profile,
                             backend=backend, fetcher=backend.fetch, mode="research"),
                timeout=self.task_timeout_s)
        except asyncio.TimeoutError:
            err = "driver_timeout"
        except Exception as e:                             # noqa: BLE001 — recorded, the round counts as failed
            err = f"{type(e).__name__}: {e}"
        if traj is None:
            traj = {"prediction": "", "rounds": 0, "termination": err}
        return {"report": traj.get("prediction") or "", "elapsed_s": round(time.time() - t0, 1),
                "docs_retrieved": len(backend.doc_ids_fetched), "steps": traj.get("rounds", 0),
                "termination": traj.get("termination"), "citations": traj.get("citations", []),
                "queries": list(backend.queries)}

    async def solve(self, task: dict, memory_text: str, critique: str | None) -> Trajectory:
        question = build_question(task, memory_text, critique, plain=self.plain)
        attempts = []
        for attempt in range(1, self.max_empty_retries + 1):
            rec = await self._attempt(task["cutoff"], question)
            rec["attempt"] = attempt
            attempts.append(rec)
            if rec["report"].strip():
                break
            if attempt < self.max_empty_retries:
                self.log(f"[retry-empty] {task['task_id']} attempt {attempt} term={rec['termination']} "
                         f"steps={rec['steps']} docs={rec['docs_retrieved']}")
        rec = attempts[-1]
        return make_trajectory(rec["report"], rec["queries"], rec["docs_retrieved"], rec["citations"], self.min_docs,
                               task_id=task["task_id"], cutoff=task["cutoff"],
                               elapsed_s=round(sum(a["elapsed_s"] for a in attempts), 1),
                               steps=rec["steps"], termination=rec["termination"], empty_attempts=len(attempts) - 1,
                               question_chars=len(question), memory_chars=len(memory_text), report_chars=len(rec["report"]))
