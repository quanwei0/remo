# Formula test set (200 questions)

Not redistributed here (the source, FinLoRA, declares no license). `../prepare_data.py` downloads the public source and
selects exactly the 200 items used in the paper by matching `formula_test.sha256` (one SHA-256 per row, evaluation order).
Whole-file SHA-256 of the exact `formula_test.jsonl` we evaluated: `079919e9631e50ba0489d2be1221ecf59952a9230ff91c5325c27b37ed99e342`.
Row schema: `{"context": "<instruction with the formula> Question: \"...\". Answer:", "target": "<number>"}`.
Public source: the FinLoRA repository (https://github.com/Open-Finance-Lab/FinLoRA), file `data/test/formula_test.jsonl` — the same 200 rows in a different order; `prepare_data.py` restores the evaluation order from the manifest (`--from-file` for a local copy).
