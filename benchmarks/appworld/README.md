# benchmarks/appworld — ReMo / AdaReMo on AppWorld

[AppWorld](https://appworld.dev) (Apache-2.0; the `appworld` package at the exact upstream revision the paper's runs
used, see Setup): 9 simulated apps, ~450 APIs, tasks solved by writing Python in a REPL; `test_normal` = 168 tasks /
56 scenarios (`test_challenge` = 417 / 139). Scoring is AppWorld's own unit tests over the world's end state:
**TGC** = % of tasks passing every test, **SGC** = % of scenarios whose tasks all pass.

The adapter reproduces the paper's runs: same prompts (byte-identical files under `prompts/`), same message
construction, parameters and decisions, and — because the package and the data are pinned to the paper's versions —
the same API responses in the observations and the same unit tests in the scores.

| paper | here |
|---|---|
| `Solve(task, M, ρ)` | `solver.AppWorldSolver`: a ReAct loop in a **fresh** `AppWorld(task_id, random_seed=123)` per round. Instruction messages = the arm's template rendered with jinja2 (task, supervisor, app descriptions, the whole playbook), split into USER / ASSISTANT turns, no system message: `prompts/solver/appworld.txt` (the paper's generator prompt, shows the playbook) for `memory` / `remo` / `adaremo`, `prompts/solver/appworld_react.txt` (AppWorld's official ReAct prompt, no playbook) for `react` / `refine`. Per step the first ```python block is executed (the reply is cut after it; a reply without a block executes `""`), the output comes back as `Output:` (20 000 chars, then `[REST NOT SHOWN FOR BREVITY]`), the context is trimmed at 400 000 chars (observations blanked first); `react` keeps AppWorld's plain scaffold limits instead (outputs uncapped, context trimmed at 50 000 chars). `--max-steps 40`, temperature 0, `max_tokens` 8192; execution timeout = AppWorld's 100 s + an outer 300 s guard. A retry injects ρ (the previous round's whole reflection) as three messages after the instructions. `completed` = task submitted **and** the last non-empty output has no `Execution failed` / `Traceback`. |
| `Reflect` | `critic.AppWorldCritic`: one call with `prompts/critic/appworld_remo.txt` (ReMo) or `appworld_adaremo.txt` (AdaReMo) + the whole playbook (`(empty)` without memory) + the previous reflection (`N/A` in round 1) + the `=== FULL CONVERSATION HISTORY ===` block of the attempt. `trajectory_verdict` decides the verdict (unparseable → the env signal), `key_insight` is the lesson, `refine` / `store` / `novelty_reason` drive AdaReMo; a `store` with `confidence` < `--store-conf 0.7` is not honoured (nothing written or reinforced, no memory demand in the saturation window; `store_decision: skipped_lowconf`). No ground truth, no test results. `react` makes no critic call (`critic.EnvCritic`: the verdict is the env signal). |
| memory `M` | `remo.SectionedPlaybook(style="plain")` — `## SECTION` headers, `[shr-00012] text` bullets — seeded from `initial_playbook.txt` (`--initial-playbook PATH`, `--no-initial-playbook` = empty skeleton), injected whole; reinforcement = the `[confirmed xN]` tag |
| `Consolidate` | `consolidator.LLMConsolidator` (default): `prompts/consolidator/appworld.txt` + the admitted reflection (round 1's, or the flipping round's behind `[VALIDATED BY RETRY: …]`) + playbook + task + history → ADD operations appended as bullets; a failed call / unusable reply leaves the playbook unchanged (`store_decision: consolidator_error`). `--consolidator append` = the `key_insight` as one OTHERS bullet, no call. |
| Alg. 1 / 2 | `remo.ReMoAgent` (subclass `run_appworld.AppWorldAgent`: record fields, `consolidator_error`, the read-only phase of `--freeze-after`) |

## Setup (env `remo-agents`)

```bash
git lfs install                                         # the pinned appworld ships its app/test sources as git-lfs bundles
pip install -e ".[agents]" && appworld install          # appworld at the paper's revision (+ click<8.2, jinja2); unpack its app/test sources
export APPWORLD_ROOT=$PWD/data/appworld                 # data/ lives inside it; outputs go to experiments/outputs/
mkdir -p $APPWORLD_ROOT && (cd $APPWORLD_ROOT && appworld download data)
cat $APPWORLD_ROOT/data/version.txt                     # must print 0.1.0
python -m unittest tests.test_appworld                  # no model, no world needed
```

**Versions matter.** The paper's runs used AppWorld at upstream commit
[`5725335`](https://github.com/stonybrooknlp/appworld/tree/57253350edf00922f370a8c7dbe94f1a4d3ee456) (package version `0.1.4.dev0`, May 2025;
apps bundle sha256 `ed68e817…`), which `pyproject.toml` pins exactly, with data version `0.1.0` (`appworld download data`
at that revision fetches `data-0.1.0.bundle`; the PyPI release fetches the same file). The PyPI release `0.1.3.post1`
and the later `0.2.0` data are *not* equivalent: the apps' responses differ (e.g. `spotify.search_songs` returns an
`album_title` field at the pinned revision and not in the release), so from step 2 on every generator prompt — and the
history the critic and the consolidator read — would diverge from the paper's; and `0.2.0` ships other base DBs and unit
tests, so scores would not be comparable. `run_appworld.py` therefore refuses to start (`check_versions`) with another
`appworld.__version__` or `data/version.txt`, and records both in `run_config.json`.

## Run

```bash
URL=http://HOST:8125/v1
R=benchmarks/appworld/run_appworld.py
python $R --mode remo    --K 3 --out runs/appworld/remo_k3_r1    --base-url $URL --model GPT-OSS-120B   # paper ReMo
python $R --mode adaremo --K 3 --out runs/appworld/adaremo_k3_r1 --base-url $URL --model GPT-OSS-120B   # paper AdaReMo
python $R --mode adaremo --K 3 --redundant-mode gate --out ...                                           # ablation
python $R --mode remo    --K 3 --freeze-after 90 --out ...                                               # learn-then-freeze
python $R --mode memory  --out runs/appworld/memory_r1 --base-url $URL --model GPT-OSS-120B              # K=1, memory
python $R --mode refine  --K 3 --out runs/appworld/refine_k3_r1 --base-url $URL --model GPT-OSS-120B     # paper ReMo-mem (k=3): official prompt, no memory
python $R --mode react   --out runs/appworld/react_r1 --base-url $URL --model GPT-OSS-120B              # paper ReAct: official prompt, plain scaffold
python $R --mode adaremo --K 2 --limit 1 --out runs/appworld/smoke --base-url $URL --model GPT-OSS-120B  # smoke
```

- Flags: `--split` (default `test_normal`), `--limit N` (first N tasks in file order), `--K` (default 3;
  `react` / `memory` fix K=1), `--max-steps 40`, `--freeze-after A`, `--redundant-mode {reinforce,gate,off}`,
  `--freeze-w 20 --freeze-rho 0.1 --probe-p 20`, `--store-conf 0.7`, `--consolidator {llm,append}`, `--initial-playbook PATH` /
  `--no-initial-playbook`, `--max-tokens 8192` / `--critic-max-tokens 8192` / `--consolidator-max-tokens 8192`,
  `--temperature 0`, `--random-seed 123`, `--exec-timeout 100` / `--guard-timeout 300`, `--llm-timeout-s 600`,
  `--llm-attempts 50`, `--experiment-name` (default: basename of `--out`), `--root` (default `$APPWORLD_ROOT`),
  `--no-round1-snapshot`, `--no-eval`, `--eval-only`, `--skip-health`.
- `--mode refine` is the paper's refinement-only baseline ("ReMo-mem" in `results/appworld/`): the same
  reflect loop and critic, AppWorld's official ReAct prompt, no playbook anywhere. `--mode react` is the
  paper's ReAct row: that prompt in AppWorld's plain ReAct scaffold (one round, no critic call, 50 000-char
  context, uncapped outputs; its `gate` is the env signal). The paper's runs of both used the official
  scaffold's own executor, without the 300 s outer guard.
- Resumable: the same `--out` skips every `task_index` already in `episodes.jsonl` and restores `playbook.txt` /
  `policy_state.json` (a run dir refuses another `--mode`, `--K`, `--split`, `--consolidator`, seed playbook).
- Tasks run sequentially (AppWorld worlds are process-global); replicates are separate `--out` dirs.

## Outputs

- `RUN_DIR/episodes.jsonl` — per task: `task_index`, `task_id`, `gate`, `stop_reason`, `rounds[]` (critic fields,
  `raw` reply, `solver` = steps / task_completed / env_clean / error / elapsed, `critic` = confidence / parsed),
  `store_decision` (`stored`, `reinforced`, `discarded`, `skipped`, `skipped_frozen`, `skipped_lowconf`,
  `consolidator_error`, `skipped_readonly`, `no_memory`), `entry_id`, `memory_readonly`, `playbook_entries`,
  `playbook_chars`. A round whose `store` the confidence gate removed keeps the critic's own `store` /
  `novelty_reason` (`critic.low_confidence`).
- `RUN_DIR/trajs/<task_id>.json` — every round's trimmed messages, steps (reply / code / output), critic raw reply.
- `RUN_DIR/playbook.txt`, `policy_state.json`, `run_config.json`, `final_results.json`.
- `$APPWORLD_ROOT/experiments/outputs/<experiment_name>/tasks/<task_id>/`: AppWorld's `dbs/` (end state of the
  **last** round — what gets scored), `logs/`, `misc/remo_rounds.json` (`clean`, `curation`, `rounds[{round,
  env_clean, verdict_no_errors, reflection, confidence, refine, store, novelty_reason, parsed}]`, `store_decision`,
  `memory_frozen`) and, after scoring, `evaluation/report.md`. The end state of every task's **first** round is
  copied to `<experiment_name>__round1/tasks/<task_id>/dbs`.

## Scoring (post hoc, never inside the loop)

- After the last task the runner calls `appworld.evaluator.evaluate_task(task_id, experiment_name)` for every
  task it ran (the same tests as `appworld evaluate <experiment_name> <split> --root $APPWORLD_ROOT`) and writes
  `final_results.json`: `TGC`, `SGC`, `round1_TGC`, `round1_SGC`, `per_task`, `tgc_by_gate`, `gate_distribution`,
  `stop_reasons`, `store_decisions`, `mean_rounds` (= mean `len(rounds)`, failed rounds included), `memory`
  (bullets, chars, cl100k_base tokens, per section).
- `python $R --mode ... --out RUN_DIR --eval-only` recomputes it; `--no-eval` skips scoring. With `--limit`, SGC
  is over the scenarios of the evaluated tasks only.
