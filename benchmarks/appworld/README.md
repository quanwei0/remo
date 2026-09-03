# benchmarks/appworld — ReMo / AdaReMo on AppWorld

[AppWorld](https://appworld.dev) (official package `appworld==0.1.3.post1`, Apache-2.0): 9 simulated apps,
~450 APIs, tasks solved by writing Python in a REPL; `test_normal` = 168 tasks / 56 scenarios
(`test_challenge` = 417 / 139). Scoring is AppWorld's own unit tests over the world's end state: **TGC** =
% of tasks passing every test, **SGC** = % of scenarios whose tasks all pass.

| paper | here |
|---|---|
| `Solve(task, M, ρ)` | `solver.AppWorldSolver`: a ReAct loop in a **fresh** `AppWorld(task_id, experiment_name)` per round — one ```python block per step → `world.execute` → output, until `world.task_completed()` or `--max-steps` (40). `M` is prepended as a playbook, `ρ` (the previous critique) is injected at the start of the retry. `completed` = task submitted **and** the last execution output has no `Execution failed` / `Traceback`. |
| `Reflect` | `critic.AppWorldCritic`: one chat call over the trimmed transcript; prompt = our checks (requirements, source of truth, invented values, `complete_task` use, side effects) + `remo.critic.fields_spec(adaptive)`; parsed by `remo.critic.parse_reflection` (an unparseable reply is retried once, then kept with its conservative defaults: refine, do not store; a failed call is `parsed=false`, never stored). No ground truth, no test results. |
| memory `M` | `remo.Playbook(prefix="aw")`, 60 000 chars injected (most-reinforced first); `AppendConsolidator`, or `--consolidator llm` (one call rewriting the lesson into one general line) |
| Alg. 1 / 2 | `remo.ReMoAgent` (subclass `AppWorldAgent` only enriches the saved record) |

## Setup (env `remo-agents`)

```bash
pip install -e ".[agents]" && appworld install          # package + its app/test sources
export APPWORLD_ROOT=$PWD/data/appworld                 # data/ lives inside it; outputs go to experiments/outputs/
mkdir -p $APPWORLD_ROOT && (cd $APPWORLD_ROOT && appworld download data)
python -m unittest tests.test_appworld                  # no model needed
```

## Run

```bash
URL=http://HOST:8125/v1
R=benchmarks/appworld/run_appworld.py
python $R --mode react   --out runs/appworld/react_r1        --base-url $URL --model GPT-OSS-120B   # K=1, no memory
python $R --mode refine  --K 3 --out runs/appworld/refine_k3_r1 --base-url $URL --model GPT-OSS-120B
python $R --mode memory  --out runs/appworld/memory_r1       --base-url $URL --model GPT-OSS-120B   # K=1, memory
python $R --mode remo    --K 3 --out runs/appworld/remo_k3_r1   --base-url $URL --model GPT-OSS-120B
python $R --mode adaremo --K 3 --out runs/appworld/adaremo_k3_r1 --base-url $URL --model GPT-OSS-120B
python $R --mode adaremo --K 3 --redundant-mode gate --out ...                                       # ablation
python $R --mode remo    --K 3 --freeze-after 90 --out ...                                           # learn-then-freeze
python $R --mode adaremo --K 2 --limit 1 --out runs/appworld/smoke --base-url $URL --model GPT-OSS-120B  # smoke
```

Flags: `--split` (default `test_normal`), `--limit N` (first N tasks in file order), `--K` (default 3;
`react`/`memory` fix K=1), `--max-steps 40`, `--freeze-after A`, `--redundant-mode {reinforce,gate,off}`,
`--freeze-w 20 --freeze-rho 0.1 --probe-p 20`, `--consolidator {append,llm}`, `--inject-cap 60000`,
`--max-tokens 8192` / `--critic-max-tokens 4096` / `--critic-memory-cap 20000` / `--temperature 0` /
`--llm-timeout-s 600`, `--random-seed 100` and `--exec-timeout 100` (AppWorld world seed and per-execution
timeout), `--experiment-name` (default: basename of `--out`), `--root` (default `$APPWORLD_ROOT`),
`--no-round1-snapshot`, `--no-eval`, `--eval-only`, `--skip-health`.
Replicates are separate runs with distinct `--out` (hence distinct experiment names) against one server.

**Resumable**: rerunning with the same `--out` skips every `task_index` already in `episodes.jsonl` and
restores `playbook.txt` / `policy_state.json` (a run dir refuses a different `--mode`, `--K`, `--split`).
Tasks run sequentially (AppWorld worlds are process-global). Wall clock at K=3 with GPT-OSS-120B: a few
hours per replicate.

## Outputs

`RUN_DIR/`: `episodes.jsonl` (per task: `task_index`, `task_id`, `gate`, `stop_reason`, `rounds[]` with the
critic fields and `solver` = steps / task_completed / env_clean / error / tokens, `store_decision`, `entry_id`,
`memory_readonly`), `trajs/<task_id>.json` (every round's steps: reply / code / output, critic raw reply),
`playbook.txt`, `policy_state.json`, `run_config.json`, `final_results.json`.

`$APPWORLD_ROOT/experiments/outputs/<experiment_name>/tasks/<task_id>/`: AppWorld's own `dbs/` (end state of
the **last** round — a fresh world wipes the task directory, so this is what gets scored), `logs/`,
`misc/remo_rounds.json` (`clean`, `curation`, `rounds[{round, env_clean, verdict_no_errors, refine, store}]`,
`store_decision`, `memory_frozen`) and, after scoring, `evaluation/report.md`. The end state of every task's
**first** round is copied to `<experiment_name>__round1/tasks/<task_id>/dbs`.

## Scoring (post hoc, never inside the loop)

The worlds are opened with `load_ground_truth=False`; the critic sees no test. After the last task the runner
calls `appworld.evaluator.evaluate_task(task_id, experiment_name)` for every task it ran (the same tests as
`appworld evaluate <experiment_name> <split> --root $APPWORLD_ROOT`, which needs the whole split) and folds the
results with AppWorld's `Metric` into `final_results.json`: `TGC`, `SGC`, `round1_TGC`, `round1_SGC` (the first
attempt of every task), `per_task`, `tgc_by_gate`, `gate_distribution`, `stop_reasons`, `store_decisions`,
`mean_rounds` (= mean `len(rounds)`, failed rounds included), `memory` (entries, chars, cl100k_base tokens).
`python $R --mode ... --out RUN_DIR --eval-only` recomputes it; `--no-eval` skips scoring and leaves an existing
`final_results.json` untouched. With `--limit`, SGC is over the scenarios of the evaluated tasks only.
