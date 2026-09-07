# benchmarks/formula — ReMo / AdaReMo on Formula (numeric financial QA)

200 questions from the FinLoRA Formula test set: each row is `{"context": "<instruction incl. the formula>
Question: \"...\". Answer:", "target": "15.0"}`. The loop sees the question text plus the fixed answer-format
sentence (`data.parse_question`, built exactly as in the paper's runs); the target is read post hoc by `scoring.py`.
A prediction is correct iff it equals the target as a float (commas stripped). Main usage is in the top-level README
(section *Formula*).

| paper | here |
|---|---|
| `Solve(task, M, ρ)` | `solver.FormulaSolver`: one chat call with `prompts/solver/formula.txt` (playbook text, reflection, question, empty context); `extract_answer` reads the JSON `final_answer` with the runs' fallbacks (`Finish[...]`, `"final_answer":` regexes, `\boxed{}`, "the final answer is") and the `No final answer found` sentinel |
| `Reflect` | `critic.FormulaCritic`: `prompts/critic/formula_remo.txt` (ReMo) / `formula_adaremo.txt` (AdaReMo: round, previous critique, whole playbook); parsed by `remo.critic.parse_reflection` (greedy JSON locator, no re-ask) |
| memory `M` | `remo.SectionedPlaybook` style `counts` (7-section skeleton, `[calc-00012] helpful=0 harmful=0 :: <text>`); the whole text is injected into the solver and, in AdaReMo, into the critic |
| `Consolidate` | `consolidator.LLMConsolidator` (default): `prompts/consolidator/formula.txt` with token budget 80000, step / total, playbook stats, the lesson built from the accepting round, the playbook and the question; ADD operations applied through `SectionedPlaybook`. `--consolidator append`: the key insight becomes one bullet in OTHERS, no call |
| Alg. 1 / 2 | `remo.ReMoAgent`, sequential over the 200 questions; `run_formula.FormulaAgent` adds per-round bookkeeping, `consolidator_error`, and the read-only phase of `--freeze-after` |

- Prompts: the ones used for the paper's runs, byte for byte, in `prompts/{solver,critic,consolidator}/formula*.txt`;
  the adapters read them at import time.
- Parameters of the runs: temperature 0, one user message per call, `max_tokens` 8192 for solver, critic and
  consolidator, K = 3, freeze window 20 / rate 0.1 / probe 20, redundant lessons reinforced (`helpful+1` on every cited id).
- Reflection sent to the solver: `(empty)` on the first attempt, then
  `A reviewer found problems with your previous attempt. Reviewer critique:\n<critique>`.
- Lesson sent to the consolidator: `[validated: <gate>] The final answer passed independent review[ after N rounds of
  refinement]. Reviewer assessment: <critique>\nKey reusable insight: <key_insight>`; AdaReMo appends
  `\nWhy this is new to the playbook: <novelty_reason>`.
- A failed solver or critic call ends the episode (not admitted; the round is counted); a failed consolidator call or an
  unusable consolidator reply leaves the playbook unchanged and records `store_decision: consolidator_error`.

```
data.py           row schema, question parsing, row hashes / manifest selection
prepare_data.py   download the public source (or --from-file) -> keep + order the 200 manifest rows -> data/formula_test.jsonl
solver.py         ChatLLM (OpenAI-compatible), solver prompt, extract_answer, FormulaSolver
critic.py         critic prompts + FormulaCritic
consolidator.py   lesson text, LLMConsolidator, AppendConsolidator
run_formula.py    CLI: arms, resumable run dir, final_results.json
scoring.py        exact-match scorer, post-hoc final_results.json (also a CLI to rescore a run dir)
data/             formula_test.sha256 (manifest) + README; the questions themselves are NOT here
```

## Run (env `remo-agents`)

```bash
python benchmarks/formula/prepare_data.py --out data/formula_test.jsonl        # 200 of 200 rows must match
URL=http://HOST:8125/v1
python benchmarks/formula/run_formula.py --mode adaremo --K 3 --data data/formula_test.jsonl \
       --base-url $URL --model GPT-OSS-120B --out runs/formula/adaremo_k3_r1
```

`--mode react | refine | memory | remo | adaremo` (ReAct = remo, K=1, no memory; refinement only = remo, K>1, no
memory; memory only = remo, K=1), `--K` (default 3), `--limit N` (first N questions), `--redundant-mode
{reinforce,gate,off}`, `--freeze-after A` (consolidate on the first A questions, then read-only memory),
`--consolidator {llm,append}`, `--freeze-w 20 --freeze-rho 0.1 --probe-p 20`, `--token-budget 80000`,
`--max-tokens 8192 --critic-max-tokens 8192 --consolidator-max-tokens 8192`, `--timeout-s 600`, `--skip-health`.

Resumable: rerunning with the same `--out` skips every `task_index` already in `episodes.jsonl` and restores
`playbook.txt` / `policy_state.json` (a run dir started with another mode / K / model / data file is refused). Nothing
in the loop reads the targets: the per-task log shows the answer, never its correctness. Replicates are independent
runs launched concurrently against one server. In the read-only phase of `--freeze-after A` (questions `A, A+1, …`)
the playbook is injected but nothing changes: no consolidator call, no bullet added or reinforced, and the saturation
window / freeze state stay as they were after question `A-1`; such records carry `memory_readonly: true` and
`store_decision: skipped_readonly` (`skipped` when the episode was not admitted).

## Outputs (RUN_DIR)

* `episodes.jsonl` — one record per question: `task_index`, `formula`, `gate`, `stop_reason`, `rounds[]` (critic
  fields + `answer`, `solver` timing, `usage` tokens per round), `store_decision`, `entry_id`, `frozen`,
  `memory_readonly`, `final_answer` (last round whose solver and critic both ran), `consolidator` (calls, tokens,
  outcome).
* `trajs.jsonl` — every round's full reply and the reflection it received.
* `playbook.txt`, `policy_state.json`, `run_config.json`, and post hoc `final_results.json`: accuracy, round-1
  accuracy, gate distribution, store decisions, stop reasons, mean rounds (`len(rounds)`, failed rounds included),
  memory bullets / chars / tokens (tiktoken `cl100k_base`; set `TIKTOKEN_CACHE_DIR` on offline nodes), LLM calls and
  tokens, per-formula and per-task correctness, learn / frozen segments when `--freeze-after` is set.

Rescore an existing run: `python benchmarks/formula/scoring.py RUN_DIR --data data/formula_test.jsonl`.
