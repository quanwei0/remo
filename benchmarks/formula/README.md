# benchmarks/formula — ReMo / AdaReMo on Formula (numeric financial QA)

200 questions from the FinLoRA Formula test set: each row is `{"context": "<instruction incl. the formula>
Question: \"...\". Answer:", "target": "15.0"}`. The loop sees only the question text plus a fixed formatting
instruction (`data.parse_question`); the target is read post hoc by `scoring.py`. A prediction is correct iff it
equals the target as a float (commas stripped). Main usage is in the top-level README (section *Formula*).

| paper | here |
|---|---|
| `Solve(task, M, ρ)` | `solver.FormulaSolver`: one chat call; prompt = playbook lines (if the arm uses memory) + question + reviewer critique (retries); answer = last `Finish[<number>]`, fallback last number; `completed` = an answer was produced |
| `Reflect` | `critic.FormulaCritic`: re-derives the computation (inputs, scale/units, rounding, stated requirements), no answer key; fields from `remo.critic.fields_spec(adaptive)`, parsed by `remo.critic.parse_reflection` |
| memory `M` | `remo.Playbook(prefix="calc")` — `[calc-00012] helpful=N :: lesson`; whole playbook injected (`--inject-cap` to cap) |
| `Consolidate` | `remo.agent.AppendConsolidator` (default) or `critic.LLMConsolidator` (`--consolidator llm`: one call rewrites the lesson into one general line) |
| Alg. 1 / 2 | `remo.ReMoAgent`, sequential over the 200 questions; `run_formula.FormulaAgent` only adds per-round bookkeeping and the read-only phase of `--freeze-after` |

```
data.py           row schema, question parsing, row hashes / manifest selection
prepare_data.py   download the public source (or --from-file) -> keep + order the 200 manifest rows -> data/formula_test.jsonl
solver.py         ChatLLM (OpenAI-compatible), prompt, Finish[...] extraction, FormulaSolver
critic.py         critic prompt + FormulaCritic, LLMConsolidator
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
`--consolidator {append,llm}`, `--freeze-w 20 --freeze-rho 0.1 --probe-p 20`, `--inject-cap`, `--max-tokens 8192`,
`--critic-max-tokens 8192`, `--temperature 0`, `--critic-memory-cap 40000`, `--skip-health`.

Resumable: rerunning with the same `--out` skips every `task_index` already in `episodes.jsonl` and restores
`playbook.txt` / `policy_state.json` (a run dir started with another mode / K / model / data file is refused). Nothing
in the loop reads the targets: the per-task log shows the answer, never its correctness. Replicates are independent
runs launched concurrently against one server. In the read-only phase of `--freeze-after A` (questions `A, A+1, …`)
the playbook is injected but nothing changes: no entry is added or reinforced and the saturation window / freeze
state stay as they were after question `A-1`; such records carry `memory_readonly: true` and
`store_decision: skipped_readonly` (`skipped` when the episode was not admitted).

## Outputs (RUN_DIR)

* `episodes.jsonl` — one record per question: `task_index`, `formula`, `gate`, `stop_reason`, `rounds[]` (critic
  fields + `answer`, `solver` timing, `usage` tokens per round), `store_decision`, `entry_id`, `frozen`,
  `memory_readonly`, `final_answer`, `consolidator_usage`.
* `trajs.jsonl` — every round's full reply and the critique it received.
* `playbook.txt`, `policy_state.json`, `run_config.json`, and post hoc `final_results.json`: accuracy, round-1
  accuracy, gate distribution, store decisions, stop reasons, mean rounds (`len(rounds)`, failed rounds included),
  memory entries / tokens (tiktoken `cl100k_base`; set `TIKTOKEN_CACHE_DIR` on offline nodes), LLM calls and tokens,
  per-formula and per-task correctness, learn / frozen segments when `--freeze-after` is set.

Rescore an existing run: `python benchmarks/formula/scoring.py RUN_DIR --data data/formula_test.jsonl`.
