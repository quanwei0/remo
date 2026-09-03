# benchmarks/financegym — ReMo / AdaReMo on FinanceGym (point-in-time finance research)

[FinanceGym](https://financegym.github.io): 400 open-ended finance research questions
(`third_party/finance_harness/FinanceGym/data/benchmark_400_public.jsonl`: `task_id`, `question`, `cutoff`), each
answered with a cited research report written from a **point-in-time (PIT) corpus** — nothing published after the
task's cutoff may be used. **There is no local ground truth**: the rubric is withheld and the organizers grade a
submitted `answers.jsonl`. The solver is the official **FinanceHarness** deep-research agent
(`financeharness.research.run_research`) over the official **FinanceGymBackend** (PIT search + fetch); the
harness is imported from `third_party/finance_harness`, not modified (provenance: `third_party/NOTICE.md`).

| paper | here | reused component (imported, not copied) |
|---|---|---|
| `Solve(task, M, ρ)` | `solver.FinanceGymSolver` | `run_research(question, profile, reader_profile, backend=FinanceGymBackend(cutoff…), fetcher=backend.fetch, mode="research")`; `M` is prepended to the question as an *Analyst playbook*, `ρ` appended as reviewer issues (`common.build_question`) |
| `Reflect` | `common.FinanceGymCritic` | one chat call; prompt = five severe-only checks (our words) + `remo.critic.fields_spec(adaptive)`; parsed by `remo.critic.parse_reflection` |
| memory `M` | `remo.Playbook(prefix="fin")` | `[fin-00012] helpful=N :: lesson`; inject cap 30 000 chars; `remo.agent.AppendConsolidator` or `common.LLMConsolidator` (`--consolidator llm`) |
| Alg. 1 / 2 | `remo.RemoPolicy` driven by `run_financegym.py` | the harness is async and tasks run concurrently, so the driver replays `ReMoAgent.run_task` step by step under one `asyncio.Lock` |

## Files

```
common.py          benchmark file, question builder, record-level view, critic prompt + FinanceGymCritic, LLMConsolidator,
                   arm -> RemoConfig (make_config), quality floor, post-hoc summary (no harness import)
solver.py          runtime patches + FinanceGymSolver (imports the harness; env remo-financegym only)
run_financegym.py  CLI: concurrent driver for every arm, resumable run dir, answers.jsonl + final_results.json at the end
calibrate.py       zero-rollout critic calibration over finished reports (flag / refine / store / parse rates)
check_answers.py   pre-submission quality floor: list / delete defective episodes so a rerun redoes them
make_answers.py    submission file answers.jsonl in benchmark order, question verbatim from the benchmark file
corpus_pipeline/   rebuild of the point-in-time corpus (WARC list + scripts; see its README)
```

## Setup

* conda env **`remo-financegym`** (top-level README → Install): `pip install -e ".[financegym]"` and
  `pip install -e third_party/finance_harness`. `solver.py` puts `third_party/finance_harness` first on `sys.path`
  (override the location with `FINHARNESS_ROOT`), so the vendored harness is the one that runs.
* The harness reads `configs/providers.json` / `configs/runtime.json` next to its package. Model names come from
  `--model` (default `GPT-OSS-120B`, applied to the backbone, the page reader and the critic); base URLs from
  `--base-url` (default `$FH_VLLM_BASE_URL`), the reader alone from `FH_VLLM_READER_BASE_URL` when set.
* Three live services:
  * a vLLM serving the backbone with **tool-call parsing on** (`--enable-auto-tool-choice --tool-call-parser openai`;
    without it the harness gets empty content and ends after one round) — `scripts/servers/vllm_120b.sh`;
  * the FinanceGym PIT search service (`scripts/servers/pit_server.sh`, ~450 GB RAM for the 145 M-document corpus) —
    `--pit-url` / `FH_PIT_URL`;
  * a query-embedding endpoint serving `Qwen/Qwen3-Embedding-4B` (`scripts/servers/embed_server.sh`) —
    `--embed-url` / `FH_EMBED_URL` (`.../v1/embeddings`).
* `SSL_CERT_FILE`: the runner sets it to certifi's bundle when unset (the harness's SSL context creation under
  NFS jitter froze the process).

Runtime patches applied by `solver.py` (the paper's runs used the same two): `httpx.AsyncClient` → `verify=False`
subclass (SIGSEGV in `ssl.SSLContext.__new__` under 8-way concurrency; all endpoints are plain http), and
`financeharness.tools.research.assembly.EQUITY_DATA_SPECS = MARKET_DATA_SPECS = ()` — the registry's deferred
yfinance tools return *current* prices/fundamentals, i.e. post-cutoff information; dropping them keeps the run
PIT-compliant (state this in any write-up).

## Run

```bash
cd third_party/finance_harness                    # the harness looks for project skills in ./skills of the cwd
URL=http://HOST:8125/v1                           # vLLM, tool-call parsing on
export FH_VLLM_BASE_URL=$URL FH_VLLM_READER_BASE_URL=$URL
export FH_PIT_URL=http://PIT_HOST:8889 FH_EMBED_URL=http://EMB_HOST:8888/v1/embeddings
RUN=../../benchmarks/financegym/run_financegym.py

python $RUN --mode baseline        --out ../../runs/financegym/baseline    # official harness alone (leaderboard 31.9)
python $RUN --mode react           --out ../../runs/financegym/react       # K=1, critic verdict recorded, no memory
python $RUN --mode refine  --K 3   --out ../../runs/financegym/refine_k3   # refinement only
python $RUN --mode memory          --out ../../runs/financegym/memory      # memory only (K=1)
python $RUN --mode remo    --K 3   --out ../../runs/financegym/remo_k3     # Algorithm 1
python $RUN --mode adaremo --K 3   --out ../../runs/financegym/adaremo_k3  # Algorithm 2
python $RUN --mode adaremo --K 3 --freeze-after 200 --out ../../runs/financegym/adaremo_freeze200   # learn-then-freeze
python $RUN --mode adaremo --K 3 --limit 1 --conc 1 --out ../../runs/financegym/smoke                 # smoke (3-25 min: one task, one or two rounds)
```

Flags shared by every runner: `--mode`, `--K` (default 1 for baseline/react/memory, else 3), `--out`, `--limit N`
(first N benchmark tasks), `--base-url`, `--model`, `--redundant-mode {reinforce,gate,off}` (AdaReMo: a lesson the
critic judges covered gives `helpful+1` to the cited entry, no model call; `gate` discards it), `--freeze-after A`
(consolidate on the first A tasks, then read-only memory; the read-only tasks wait until the A learning tasks have
finished so they all see the same memory), `--consolidator {append,llm}` (`llm`: one model call rewrites the lesson as
one general line, run outside the lock).

FinanceGym-specific: `--conc` (default `$FH_CONC` or 8 concurrent tasks), `--min-docs` (default 3: a final report
with fewer fetched documents is **not saved** and is redone by the next run — essential: when the embedding service
dies mid-run every task returns 0 docs), `--max-empty-retries` (3: gpt-oss sometimes ends the turn with an empty final
message; the attempt is rerun), `--task-timeout-s` (3660 per attempt), `--inject-cap` (30000 playbook chars prepended
to the question), `--freeze-w 20 --freeze-rho 0.1 --probe-p 20` (saturation freeze), critic
`--critic-max-tokens 2048 --critic-temperature 0 --critic-memory-cap 20000`, `--skip-health` (skip the PIT/embed/LLM
probe), `--loglevel` (harness logging; httpx is quieted), `--tasks` (another question file).

**`--mode baseline`** is the plain official harness: one attempt, no critic call, no memory, and the bare question +
PIT sentence (no "Research question:" header) — the configuration of the leaderboard entry
`gptoss120b-financeharness` (31.9). It still applies the two runtime patches and the `--min-docs` floor.

**Resumable**: rerunning with the same `--out` skips every `task_id` already in `episodes.jsonl` and restores
`playbook.txt` and `policy_state.json`. Kill and restart freely (the driver refuses a run dir started with a
different `--mode` / `--K` / `--freeze-after`). No ground truth exists, so nothing in the loop can leak it.

**Concurrency semantics**: each task reads the playbook when it starts (after acquiring the semaphore), runs its ≤K
solve/critic rounds concurrently with other tasks, and then — under the single lock — applies `gate` /
`memory_decision` / playbook write and persists. `task_index` (benchmark position) drives the AdaReMo probe
schedule and the `--freeze-after` split, so both are independent of completion order.

## Outputs (RUN_DIR)

* `episodes.jsonl` — one record per saved task: `task_index`, `task_id`, `question`, `cutoff`, `mode`, `K`,
  `use_memory`, `readonly_memory`, `gate` (`round1_clean` / `cross_round_validated` / `critic_stop` / `never_clean`;
  `no_critic` for the baseline), `stop_reason`, `rounds[]` (per round: `completed`, critic `verdict` / `critique` /
  `lesson` / `refine` / `store` / `novelty_reason` / `cited_id` / `parsed` / `raw`, plus `solver` = `elapsed_s`,
  `docs_retrieved`, `steps`, `termination`, `empty_attempts`, `report_chars`), `store_decision` (`stored` /
  `reinforced` / `discarded` / `skipped` / `skipped_frozen` / `readonly` / `no_memory`), `entry_id`, `frozen`,
  `critic_failed` (accepted on a failed critic call, see below), `memory_chars_at_start`, `final_answer` (the report),
  `final_completed`, `final_round`, `elapsed_s` (all rounds),
  `docs_retrieved`, `steps`, `termination`, `citations`, `queries`.
* `trajs/<task_id>.json` — every round's full report, queries, citations and the critic's raw output.
* `playbook.txt`, `policy_state.json`, `run_config.json`, `answers.jsonl` (rebuilt at the end of every run).
* `final_results.json` (post hoc, from the files above): `n_saved`, `gate_distribution`, `store_decisions`,
  `mean_rounds` (`len(rounds)`, failed rounds included), `critic_failed_episodes`, `round1_clean_rate` / `final_clean_rate` (the **critic's**
  verdicts — FinanceGym has no accuracy; `accuracy` is `null` and the score comes from the organizers),
  `memory_entries` / `memory_chars` / `memory_tokens_cl100k` (tiktoken), freeze events, mean docs / report length /
  time, `defective_reports`, and `this_invocation` (what the last run did, critic and consolidator call counts).

`Trajectory.completed` = report non-empty **and** `docs_retrieved >= min_docs` (the harness leaves `prediction`
empty on `max_rounds` / `timeout` / `error`, so this is exactly "the harness produced an answer"). A critic transport
failure falls back to `no_errors` with `parsed=false` and no lesson; the episode is recorded with `critic_failed:
true` and `store_decision: skipped` — no memory write, not even of an earlier round's lesson, and no saturation
bookkeeping — so a critic outage neither burns solver rounds nor writes memory. Unparseable critic output is retried
once, then parsed conservatively (`errors_found`, `refine=true`, `store=false`).

## Score / submit

There is no local scoring. The pipeline before submission (from the repository root):

```bash
# 1. quality floor: a report is defective if empty, < 1500 chars, or starts with a JSON brace
python benchmarks/financegym/check_answers.py RUN_DIR            # list
python benchmarks/financegym/check_answers.py RUN_DIR --delete   # drop them (stop the driver first), then
python benchmarks/financegym/run_financegym.py ... --out RUN_DIR # rerun redoes exactly those tasks
# 2. submission file (benchmark order; question/cutoff verbatim from benchmark_400_public.jsonl)
python benchmarks/financegym/make_answers.py RUN_DIR             # -> RUN_DIR/answers.jsonl
```

`answers.jsonl` rows: `question`, `cutoff`, `report` (required by `FinanceGym/docs/participate.md`), `searches`,
`docs_retrieved`, `steps`, `elapsed_s` (optional). Submit via the FinanceGym GitHub issue template or e-mail with the
agent metadata (agent, org, base model, date, contact, PIT-compliance note); the maintainers run the judge.

Deleting defective episodes does not roll back the playbook / policy state (memory is append-only) — the redo simply
happens with the memory as it stands.

## Critic calibration (no rollouts)

```bash
python benchmarks/financegym/calibrate.py --mode adaremo --run-dir RUN_DIR --out cal.jsonl --base-url $URL
python benchmarks/financegym/calibrate.py --mode remo --run-dir runs/financegym/baseline --out cal.jsonl --base-url $URL --limit 50
```

Runs the exact production critic prompt over finished reports (a run dir's `episodes.jsonl`, or `--trajs-dir` with
one `{"record": {...}}` JSON per task) and prints the `errors_found` rate, the `refine` rate among flagged, the `store`
rate and parse failures. Use it to tune the domain instructions before spending retry GPU-time (a wide-net prompt
flagged 99 % of baseline reports; the five severe-only checks brought it to ~56 %, ~63 % in the full ReMo run).
