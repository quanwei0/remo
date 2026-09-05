# ReMo / AdaReMo

Reference implementation of **From More to Enough: Rethinking Refinement and Memory in Self-Improving LLM Agents**, with
runners for AppWorld, Formula and FinanceGym.

- **ReMo** (Algorithm 1): a solver–critic refinement loop inside each task (round budget `K`) plus an append-only memory across
  tasks. Memory is written only for episodes the outcome gate admits (`round1_clean`, `cross_round_validated`).
- **AdaReMo** (Algorithm 2): the critic also decides `refine` (retry only with an actionable fix, else `critic_stop`) and
  `store` (evidence-backed, generalizable, novel — a covered lesson must cite the entry it duplicates); a saturation freeze
  stops consolidation once the last `freeze_w` admitted episodes stop demanding writes. Redundant lessons are *reinforced*
  (`helpful+1` on the cited entry, no model call) or, with `--redundant-mode gate`, discarded.
- Every baseline is a configuration of the same loop: ReAct = `K=1`, no memory; refinement only = `K>1`, no memory;
  memory only = `K=1`, memory on.

## Layout

```
remo/                          core: RemoConfig, RemoPolicy (decisions), Playbook / SectionedPlaybook (memory), ReMoAgent loop
prompts/{solver,critic,consolidator}/   the prompts used for the paper's runs, byte for byte: one .txt per role and benchmark
benchmarks/formula/            numeric financial QA, 200 questions, exact-match accuracy            run_formula.py
benchmarks/appworld/           ReAct agent in the AppWorld REPL, test_normal 168 tasks, TGC / SGC    run_appworld.py
benchmarks/financegym/         FinanceHarness research reports on a point-in-time corpus              run_financegym.py
benchmarks/financegym/corpus_pipeline/   scripts + WARC list that rebuild the point-in-time news corpus
scripts/servers/               vLLM launch scripts (gpt-oss-120b / 20b), embedding server, search server
third_party/finance_harness/   the official FinanceGym agent (Apache-2.0; CC BY-NC 4.0 notice; see third_party/NOTICE.md)
tests/                         unit tests: Algorithm 1/2 semantics, the three adapters, repository hygiene (no model needed)
```

- `remo/` has no benchmark code and never sees ground truth; each `benchmarks/<name>/` plugs in a solver, a critic, a
  consolidator, a data loader and a post-hoc scorer, and reads its prompts from `prompts/` at import time: solver prompts
  for Formula and AppWorld (`appworld_react.txt` = AppWorld's official ReAct prompt, used by the no-memory arms), critic
  prompts per benchmark and arm (`*_remo.txt` / `*_adaremo.txt`; FinanceGym has one for both), curator prompts for Formula
  and AppWorld. FinanceGym's solver prompt is the harness's own, and it has no consolidator prompt (lessons are appended verbatim).
- Paper ↔ code: `Solve` = `Solver.solve(task, M, ρ)` · `Reflect` = `Critic.reflect` → `Reflection(v, ρ, g_ref, g_sto)` ·
  `completed(τ)` = `Trajectory.completed` · gate / `admitted` = `RemoPolicy` · `Consolidate` = append-only `Consolidator` ·
  `Saturated` = the freeze window in `RemoPolicy.memory_decision`.

## Install

Two environments (the benchmarks' dependencies conflict):

```bash
git clone https://github.com/quanwei0/remo.git && cd remo
# Formula + AppWorld
conda create -y -n remo-agents python=3.12 && conda activate remo-agents
git lfs install && pip install -e ".[agents]" && appworld install   # appworld at the paper's revision (git-lfs bundles)
# FinanceGym
conda create -y -n remo-financegym python=3.12 && conda activate remo-financegym
pip install -e ".[financegym]" && pip install -e third_party/finance_harness
# sanity check
python -m unittest discover -s tests
```

## Data

| benchmark | what | in this repo | how to get it |
|---|---|---|---|
| Formula | 200 numeric questions over financial formulas (FinLoRA; no license declared) | only `benchmarks/formula/data/formula_test.sha256` (one hash per row) | `benchmarks/formula/prepare_data.py` downloads the public source and keeps exactly the 200 matching rows |
| AppWorld | 9 apps, ~450 APIs; `test_normal` 168 tasks / 56 scenarios (Apache-2.0) | nothing | `appworld install` + `appworld download data` (data `0.1.0`; the package revision is pinned in `pyproject.toml`) |
| FinanceGym questions | 400 research questions with a point-in-time cutoff (Apache-2.0 + CC BY-NC 4.0 notice) | `third_party/finance_harness/FinanceGym/data/benchmark_400_public.jsonl` | already there |
| FinanceGym corpus | CC-NEWS Oct-2024 … Nov-2025, 6,758 WARCs → 145.3M articles, Qwen3-Embedding-4B, IVF-SQ8 (nlist 12,053 / nprobe 32) | `corpus_pipeline/warc_paths.txt` + build scripts | rebuild with `corpus_pipeline/` (Common Crawl is public; ~3 days on 8 H100s) |
| FinanceGym rubrics | withheld by the organizers | — | scores come back by e-mail |

## Models

Every runner talks to an OpenAI-compatible chat endpoint: `--base-url`, `--model`, key in `REMO_API_KEY` (default `EMPTY`).

**Local vLLM (paper setting: gpt-oss-120b / 20b)**

| script | serves |
|---|---|
| `scripts/servers/vllm_120b.sh` | GPT-OSS-120B, TP=4, port 8125. `TOOLS=1` adds the tool-call parser FinanceGym needs; Formula and AppWorld run **without** it (paper setting — the parser changes what gpt-oss returns as `content`) |
| `scripts/servers/vllm_20b.sh` | GPT-OSS-20B, one GPU, port 8126 |
| `scripts/servers/embed_server.sh` | Qwen3-Embedding-4B, port 8888 (FinanceGym queries; same model as the corpus) |
| `scripts/servers/pit_server.sh` | FinanceGym point-in-time search, port 8889 (CPU node, ~450 GB RAM) |

**Any OpenAI-compatible API**

```bash
export REMO_API_KEY=sk-...
python benchmarks/formula/run_formula.py --mode adaremo --K 3 --data data/formula_test.jsonl \
       --base-url https://api.openai.com/v1 --model gpt-4o --out runs/formula/gpt4o_adaremo_k3
```

- Works for OpenAI and any compatible endpoint (Anthropic `https://api.anthropic.com/v1/`, DeepSeek, Together, OpenRouter …);
  same flags for `run_appworld.py`.
- FinanceGym: the model must support function calling; the harness reads the backbone / reader key from its own profile —
  add `"api_key_env": "OPENAI_API_KEY", "api_key_literal": null` to the `vllm` profile in
  `third_party/finance_harness/configs/providers.json` and export that variable.
- Cost: one AppWorld task ≈ 200k prompt tokens per round (a `test_normal` replicate ≈ 40–50M input tokens); FinanceGym is
  similar; Formula is small.

## Arms

| paper arm | `--mode` | `--K` |
|---|---|---|
| ReAct | `react` (FinanceGym: `baseline`, the official harness alone) | 1 |
| refinement only | `refine` | 2 … 5 |
| memory only | `memory` | 1 |
| ReMo | `remo` | 1 … 5 (default 3) |
| AdaReMo | `adaremo` (`--redundant-mode gate` for the ablation) | 1 … 5 |

- Replicates: five independent runs per cell (`r1` … `r5`), launched concurrently against one server — never copied run directories.
- `--freeze-after A`: consolidate on the first *A* tasks, then run with the memory read-only (learn-then-freeze, RQ3).
- Consolidation: Formula and AppWorld default to `--consolidator curator` — the curator model call of the paper's runs
  (`prompts/consolidator/<benchmark>.txt`), which may add 0 … n playbook bullets; `append` stores the accepting round's
  key insight as one bullet, no call. FinanceGym has no such flag: the lesson is appended verbatim.
- Memory: Formula / AppWorld a sectioned markdown playbook (bullets `[calc-00012] helpful=0 harmful=0 :: text` /
  `[shr-00012] text` under `## SECTION` headers) injected whole, AppWorld's seeded from
  `benchmarks/appworld/initial_playbook.txt`; FinanceGym flat `[fin-00001] helpful=N text` lines, injected under a
  30 000-char cap ranked by (`-helpful`, line).
- Failures: a failed solver or critic call ends the episode (not admitted; the round counts); an unparseable critic reply
  is read as in the original run (Formula: `no_errors` iff that literal occurs; AppWorld: the environment signal;
  FinanceGym: `no_errors`, no lesson); a failed or unusable curator call leaves the playbook unchanged (`curator_error`).
- Mean rounds = `len(rounds)`: a failed generation still spends a round.
- Runs resume from `episodes.jsonl`; a run directory refuses a different `--mode` or `--K` (plus, per runner, its other
  locked settings: Formula the model and data file, AppWorld the split and seed playbook, FinanceGym `--freeze-after`).

**What a run directory contains**

- `episodes.jsonl` — one record per task: rounds, verdicts, gate, store decision
- `playbook.txt`, `policy_state.json` — the memory and the freeze window (FinanceGym: also the retry-round budget)
- full trajectories: Formula `trajs.jsonl` (every reply per round); AppWorld `trajs/<task_id>.json` (every step's
  reply / code / output and the critic's raw reply) plus AppWorld's own `experiments/outputs/<name>/tasks/<task_id>/`
  (databases, logs, `misc/remo_rounds.json`); FinanceGym `trajs/<task_id>.json` (every round's report, queries, critic reply)
- `final_results.json` — accuracy or TGC/SGC (not for FinanceGym), round-1 accuracy, gate distribution, store decisions,
  mean rounds, memory entries (= bullet / line count) / tokens

## Formula (env `remo-agents`)

```bash
python benchmarks/formula/prepare_data.py --out data/formula_test.jsonl
for r in 1 2 3 4 5; do
  python benchmarks/formula/run_formula.py --mode adaremo --K 3 --data data/formula_test.jsonl \
         --base-url $URL --model GPT-OSS-120B --out runs/formula/adaremo_k3_r$r &
done; wait
```

- Correct iff the prediction equals the reference as a float (commas stripped).
- Solver, critic and curator: one user message, temperature 0, `max_tokens` 8192 each (`--max-tokens`, `--critic-max-tokens`,
  `--consolidator-max-tokens`); `--token-budget 80000` quoted to the curator; `--consolidator curator|append`.
- 20B rows: `--model GPT-OSS-20B` against the 20B server. Learn-then-freeze: `--freeze-after 100`.
- `prepare_data.py --from-file PATH` converts a local copy; all 200 manifest hashes must match.
- `scoring.py RUN_DIR --data …` rescores a finished run. Details: `benchmarks/formula/README.md`.

## AppWorld (env `remo-agents`)

```bash
export APPWORLD_ROOT=$PWD/data/appworld && mkdir -p $APPWORLD_ROOT && (cd $APPWORLD_ROOT && appworld download data)
python benchmarks/appworld/run_appworld.py --mode adaremo --K 3 --split test_normal \
       --base-url $URL --model GPT-OSS-120B --out runs/appworld/adaremo_k3_r1
```

- ReAct loop in the AppWorld REPL: one code block per step, at most `--max-steps` (40).
- A round is `completed` when the task is submitted and the last execution output shows no error; a retry opens a fresh
  world with the critique injected.
- Prompts: `memory` / `remo` / `adaremo` render the paper's generator prompt (`prompts/solver/appworld.txt`, shows the
  playbook); `refine` and `react` render AppWorld's official ReAct prompt (`appworld_react.txt`), `react` in the plain
  scaffold without a critic call.
- Solver, critic and curator: `--temperature 0`, `max_tokens` 8192 each (`--max-tokens`, `--critic-max-tokens`,
  `--consolidator-max-tokens`), world seed `--random-seed 123`; `--consolidator curator|append`; seed playbook
  `--initial-playbook PATH` (default `benchmarks/appworld/initial_playbook.txt`) / `--no-initial-playbook`; AdaReMo
  stores only with `confidence` ≥ `--store-conf 0.7`.
- Scoring is post hoc with AppWorld's unit tests (`appworld.evaluator.evaluate_task`, = `appworld evaluate <name> test_normal --root $APPWORLD_ROOT`):
  TGC = % tasks passing all assertions, SGC = % scenarios whose tasks all pass; `--eval-only` recomputes them.
- Paper: `test_normal`, five replicates per cell; learn-then-freeze `--freeze-after 90`.
- Package and data are pinned to the paper's versions (upstream commit `5725335`, `0.1.4.dev0`; data `0.1.0`) — later
  versions return other API responses and score with other tests, so the runner refuses them.
  Details: `benchmarks/appworld/README.md`.

## FinanceGym (env `remo-financegym`)

No local ground truth: `answers.jsonl` is e-mailed to the organizers (https://financegym.github.io/). Our entries:
`gptoss120b-financeharness` (baseline, 31.9), `gptoss120b-financeharness-remo`, `gptoss120b-financeharness-adaremo`.

1. Corpus (once, heavy): `benchmarks/financegym/corpus_pipeline/` — download → extract → embed → merge → `fg_finalize.sh`.
2. Servers: `pit_server.sh` (`DATA=<corpus dir>`), `embed_server.sh`, `TOOLS=1 vllm_120b.sh` (tool-call parsing on).
3. Runs, from `third_party/finance_harness` (the harness reads its `skills/` from the working directory):
   ```bash
   cd third_party/finance_harness
   export FH_VLLM_BASE_URL=$URL FH_VLLM_READER_BASE_URL=$URL FH_PIT_URL=http://PIT:8889 FH_EMBED_URL=http://EMBED:8888/v1/embeddings
   python ../../benchmarks/financegym/run_financegym.py --mode baseline --out ../../runs/financegym/baseline
   python ../../benchmarks/financegym/run_financegym.py --mode remo    --K 3 --out ../../runs/financegym/remo
   python ../../benchmarks/financegym/run_financegym.py --mode adaremo --K 3 --out ../../runs/financegym/adaremo
   ```
4. Submission: gzip `answers.jsonl` and e-mail it with the metadata block described on the benchmark page.

- `--mode baseline` = the official harness alone (one attempt, no critic, no memory) — the leaderboard entry.
- Solver = the harness with its own prompts and `vllm` profile (temperature 0.6, `max_tokens` 16384; page reader 0.7 / 8192);
  the playbook (≤ `--inject-cap` 30000 chars) is prepended to the question, the previous critique appended on a retry.
- Critic: `prompts/critic/financegym.txt` for both arms, `--critic-max-tokens 2048`, no temperature sent (server default);
  no `--consolidator`, no `--critic-temperature`; retry rounds draw on one run-wide `--extra-rounds-budget` (550).
- `--conc` tasks run concurrently (8); a task is saved only if ≥ `--min-docs` (3) documents were retrieved.
- `check_answers.py` deletes defective reports (< 1500 chars or a tool-call fragment) so a rerun redoes them;
  `make_answers.py` writes `answers.jsonl` with each question verbatim in benchmark order;
  `calibrate.py` measures the critic's flag rate on a finished run before spending retries.

## Citation

TBD.
