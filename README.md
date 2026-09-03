# ReMo / AdaReMo

Reference implementation of **From More to Enough: Rethinking Refinement and Memory in Self-Improving LLM Agents**,
with runners for the three benchmarks used in the paper: AppWorld, Formula and FinanceGym.

* **ReMo** (Algorithm 1) couples a solver–critic refinement loop within each task (round budget `K`) with an append-only
  memory across tasks. Memory is written only for episodes the outcome gate admits (`round1_clean`, `cross_round_validated`);
  episodes that never became clean leave no trace.
* **AdaReMo** (Algorithm 2) hands two budget decisions to the critic — `refine` (retry only with an actionable fix, otherwise
  `critic_stop`) and `store` (evidence-backed, generalizable, novel; a covered lesson must cite the entry it duplicates) — and
  adds a saturation freeze over the last `freeze_w` admitted episodes. Redundant lessons are *reinforced* by default
  (`helpful+1` on the cited entry, no model call); `--redundant-mode gate` discards them instead.

Every baseline in the paper is a configuration of the same loop: ReAct (`K=1`, no memory), refinement only (`K>1`, no
memory), memory only (`K=1`, memory on).

## Layout

```
remo/                          core: RemoConfig, RemoPolicy (decisions), Playbook (memory), critic contract, ReMoAgent loop
benchmarks/formula/            numeric financial QA, 200 questions, exact-match accuracy                  run_formula.py
benchmarks/appworld/           ReAct agent in the AppWorld REPL, test_normal 168 tasks, TGC / SGC          run_appworld.py
benchmarks/financegym/         FinanceHarness research reports on a point-in-time corpus, e-mailed grading run_financegym.py
benchmarks/financegym/corpus_pipeline/   scripts and the WARC list that rebuild the point-in-time news corpus
scripts/servers/               vLLM launch scripts (gpt-oss-120b / 20b), embedding server, search server
third_party/finance_harness/   the official FinanceGym agent (Apache-2.0 + CC BY-NC 4.0 notice; provenance in third_party/NOTICE.md)
tests/                         unit tests of the Algorithm 1/2 semantics (fake solver and critic, no model needed)
```

`remo/` contains no benchmark code and never sees ground truth; each `benchmarks/<name>/` folder plugs a solver, a critic
prompt, a data loader and a post-hoc scorer into it. Mapping to the paper's pseudocode: `Solve` = `Solver.solve(task, M, ρ)`;
`Reflect` = `Critic.reflect` → `Reflection(v, ρ, g_ref, g_sto)`; `completed(τ)` = `Trajectory.completed`; the gate and
`admitted` = `RemoPolicy`; `Consolidate` = an append-only `Consolidator`; `Saturated` = the freeze window in
`RemoPolicy.memory_decision`.

## Reproducing the paper

All experiments use gpt-oss-120b and gpt-oss-20b served locally with vLLM 0.23.0 on NVIDIA H100 80GB GPUs; no API keys are
needed. The benchmarks have incompatible Python dependencies, so two environments are used.

### Install

```bash
git clone https://github.com/quanwei0/remo.git && cd remo
# Formula + AppWorld
conda create -y -n remo-agents python=3.12 && conda activate remo-agents
pip install -e ".[agents]" && appworld install
# FinanceGym
conda create -y -n remo-financegym python=3.12 && conda activate remo-financegym
pip install -e ".[financegym]" && pip install -e third_party/finance_harness
# sanity check (no model needed)
python -m unittest discover -s tests
```

### Data

| benchmark | what | in this repo | how you get it | license |
|---|---|---|---|---|
| Formula | 200 numeric questions over financial formulas (`context` = instruction + question, `target` = number) | **not the questions** (the source declares no license); `benchmarks/formula/data/formula_test.sha256` — one SHA-256 per row, so the exact 200 items can be reconstructed | `benchmarks/formula/prepare_data.py` downloads the public FinLoRA source and keeps the rows whose hashes match | source: FinLoRA (no license declared) |
| AppWorld | 9 simulated apps, ~450 APIs, populated databases; `test_normal` 168 tasks / 56 scenarios (`test_challenge` 417 / 139) | nothing | `appworld install` + `appworld download data` (official package) | Apache-2.0 |
| FinanceGym questions | 400 research questions, each with a point-in-time cutoff | `third_party/finance_harness/FinanceGym/data/benchmark_400_public.jsonl` | already there | Apache-2.0 file headers; the harness README adds a CC BY-NC 4.0 non-commercial notice (`third_party/NOTICE.md`) |
| FinanceGym corpus | point-in-time news corpus: CC-NEWS Oct-2024 … Nov-2025, 6,758 WARC files → 145.3M articles, Qwen3-Embedding-4B, IVF-SQ8 index (nlist 12,053, nprobe 32) | **not the corpus** (terabytes); `corpus_pipeline/warc_paths.txt` is the exact WARC list, next to the build scripts | rebuild with `corpus_pipeline/` (Common Crawl is public; ~3 days of embedding on 8 H100s) | Common Crawl terms |
| FinanceGym rubrics | withheld by the organizers | — | scores come back by e-mail | — |

Run outputs (`runs/`) are not versioned.

### Model servers

| script | serves | notes |
|---|---|---|
| `scripts/servers/vllm_120b.sh` | GPT-OSS-120B, tensor-parallel 4, port 8125 | `--enable-auto-tool-choice --tool-call-parser openai` is required by FinanceGym (tool calling); harmless elsewhere |
| `scripts/servers/vllm_20b.sh` | GPT-OSS-20B, one GPU, port 8126 | |
| `scripts/servers/embed_server.sh` | Qwen3-Embedding-4B, port 8888 | FinanceGym query embeddings — must be the model the corpus was embedded with |
| `scripts/servers/pit_server.sh` | FinanceGym point-in-time search, port 8889 | CPU node; ~450 GB RAM for the full corpus |

Every runner takes `--base-url http://HOST:PORT/v1 --model GPT-OSS-120B` (or `GPT-OSS-20B`). AppWorld and Formula decode
at temperature 0; FinanceGym uses the harness defaults. Runs are resumable: each run directory holds `episodes.jsonl`
(one record per task: rounds, verdicts, gate, store decision), `playbook.txt`, `policy_state.json` and, when finished,
`final_results.json`.

### Arms

The same flags select every arm on every benchmark:

| paper arm | `--mode` | `--K` |
|---|---|---|
| ReAct | `react` | 1 |
| refinement only | `refine` | 2 … 5 |
| memory only | `memory` | 1 |
| ReMo | `remo` | 1 … 5 (paper default 3) |
| AdaReMo (reinforce; `--redundant-mode gate` for the ablation) | `adaremo` | 1 … 5 |

`--K` defaults to 3 and is fixed at 1 for the K=1 arms. FinanceGym adds `--mode baseline` (the official harness alone, see
below). Replicates are independent runs (`r1`, `r2`, …) launched concurrently against one server — never copies of a run
directory. Learn-then-freeze (RQ3): `--freeze-after A` consolidates on the first *A* tasks and runs the rest with the memory
read-only. `--consolidator llm` (default `append`) condenses each stored lesson into one general line with one model call.
Mean rounds are counted as `len(rounds)`, so a failed generation still spends a round.

### Formula (env `remo-agents`)

A prediction is correct iff it equals the reference as a float (commas stripped).

```bash
python benchmarks/formula/prepare_data.py --out data/formula_test.jsonl
URL=http://HOST:8125/v1
for r in 1 2 3; do
  python benchmarks/formula/run_formula.py --mode adaremo --K 3 --data data/formula_test.jsonl \
         --base-url $URL --model GPT-OSS-120B --out runs/formula/adaremo_k3_r$r &
done; wait
```

The 20B rows use `--model GPT-OSS-20B` against the 20B server. `final_results.json` reports accuracy, round-1 accuracy, the
gate distribution, store decisions, mean rounds and the memory size (entries / tokens); `scoring.py RUN_DIR --data …`
rescores a finished run. `prepare_data.py --from-file PATH`
converts a local copy instead of downloading (all 200 manifest hashes must match; the rebuilt file is byte-identical to the
one evaluated). `--consolidator llm` rewrites each admitted lesson into one general entry with one extra call (default
`append` stores the critic's lesson verbatim); the learn-then-freeze split is `--freeze-after 100`. Details and the
output format: `benchmarks/formula/README.md`.

### AppWorld (env `remo-agents`)

```bash
export APPWORLD_ROOT=$PWD/data/appworld && mkdir -p $APPWORLD_ROOT && (cd $APPWORLD_ROOT && appworld download data)
URL=http://HOST:8125/v1
python benchmarks/appworld/run_appworld.py --mode adaremo --K 3 --split test_normal \
       --base-url $URL --model GPT-OSS-120B --out runs/appworld/adaremo_k3_r1
```

The agent is a ReAct loop in the AppWorld Python REPL (one code block per step, at most `--max-steps` steps, default 40). A round is
`completed` when the task is submitted and the last execution output shows no error; a retry starts a fresh world with the
critic's critique injected. AppWorld writes each world's end state to `$APPWORLD_ROOT/experiments/outputs/<name>/` (`--root` overrides the root)
(`<name>` = basename of `--out`, or `--experiment-name`). Evaluation is post hoc with AppWorld's unit tests: after the
last task the runner scores every task it ran (`appworld.evaluator.evaluate_task`; for a whole split this equals
`appworld evaluate <name> test_normal --root $APPWORLD_ROOT`) and writes TGC = % of tasks passing all assertions and
SGC = % of scenarios whose tasks all pass, plus the same numbers for the round-1 snapshot `<name>__round1`, into
`final_results.json` (`--eval-only` recomputes it). The paper reports `test_normal` (168 tasks, 56 scenarios), five
replicates at K=3 and three otherwise; learn-then-freeze uses `--freeze-after 90`. Details: `benchmarks/appworld/README.md`.

### FinanceGym (env `remo-financegym`)

FinanceGym has no local ground truth: `answers.jsonl` is e-mailed to the organizers, who score it against a withheld rubric
(https://financegym.github.io/). Our entries: `gptoss120b-financeharness` (baseline, 31.9), `gptoss120b-financeharness-remo`,
`gptoss120b-financeharness-adaremo`.

1. **Corpus** (once; heavy). Rebuild the point-in-time corpus with `benchmarks/financegym/corpus_pipeline/`
   (download → extract → embed → merge → `fg_finalize.sh` builds the index); set the paths at the top of each script.
2. **Servers**: `pit_server.sh` (`DATA=<corpus dir>`), `embed_server.sh`, `vllm_120b.sh` (tool-call flags!).
3. **Runs**, from `third_party/finance_harness` (the harness discovers project skills in `./skills` of the working
   directory; its `configs/*.json` sit next to its package, and `--model` / `--base-url` override the model name and URL
   for the backbone, the page reader and the critic):
   ```bash
   cd third_party/finance_harness
   export FH_VLLM_BASE_URL=$URL FH_VLLM_READER_BASE_URL=$URL FH_PIT_URL=http://PIT:8889 FH_EMBED_URL=http://EMBED:8888/v1/embeddings
   python ../../benchmarks/financegym/run_financegym.py --mode baseline --out ../../runs/financegym/baseline   # official harness alone
   python ../../benchmarks/financegym/run_financegym.py --mode remo    --K 3 --out ../../runs/financegym/remo
   python ../../benchmarks/financegym/run_financegym.py --mode adaremo --K 3 --out ../../runs/financegym/adaremo
   ```
   `--mode baseline` is the official harness with one attempt, no critic and no memory (the leaderboard entry); the
   other arms are as in the table above, plus `--consolidator llm` to condense each stored lesson with one model call.
   Eight tasks run concurrently (`--conc`). A task is saved only if at least three documents were retrieved
   (`--min-docs`, a guard against a dead embedding service). `check_answers.py` deletes defective reports (< 1500
   characters or a tool-call fragment) so a rerun redoes them; `make_answers.py` writes `answers.jsonl` with each
   question verbatim in benchmark order; `calibrate.py` runs the critic over a finished run to measure its flag rate
   before spending retries. `final_results.json` has no accuracy (the score comes from the organizers) but the same
   gate / store / rounds / memory statistics as the other benchmarks.
4. **Submission**: gzip `answers.jsonl` and e-mail it with the metadata block as described on the benchmark page.

## Citation

TBD.
