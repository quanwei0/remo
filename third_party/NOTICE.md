# Third-party code

The only third-party code in this repository is the official FinanceGym agent, **FinanceHarness**, vendored as a
plain copy (no submodule) under `third_party/finance_harness/`.

| path | source | pinned commit | license | our changes |
|---|---|---|---|---|
| `third_party/finance_harness/` | https://github.com/google-research/google-research/tree/master/finance_harness (directory `finance_harness/` of the google-research monorepo; it also contains the `FinanceGym/` benchmark package and the public question file `FinanceGym/data/benchmark_400_public.jsonl`) | `5f07ba9` (2026-08-13) | Apache-2.0 (the google-research repository license; every source file carries the Apache-2.0 header). The harness's own `README.md` / `pyproject.toml` additionally state a CC BY-NC 4.0 non-commercial notice for the tool and the benchmark data — academic, non-commercial use only. | three edits, recorded in `third_party/patches/gr-financegym.patch` and already applied to the vendored copy (see below) |

## Our three edits (`third_party/patches/gr-financegym.patch`)

1. `FinanceGym/financegym/corpus/extract_embed.py` — `embed_batch` forwards `truncate_prompt_tokens` to the vLLM
   embedding server when the environment variable `EMBED_TRUNCATE_TOKENS` is set (one over-length article otherwise
   fails the whole batch while embedding 145 M articles). Unset, the official behaviour is unchanged.
2. `FinanceGym/financegym/env/server.py` — the FastAPI handlers `_search` / `_fetch` declare their request models
   (`req: SearchRequest`, `req: FetchRequest`) so the body is parsed and validated (without the annotation FastAPI
   treats `req` as a query parameter and the endpoints reject every request).
3. `configs/providers.json` — the two built-in vLLM profiles (`vllm` backbone, `vllm-reader` page reader) are given
   the served model name `GPT-OSS-120B`; their base URLs come from `FH_VLLM_BASE_URL` / `FH_VLLM_READER_BASE_URL`
   (or the runner's `--base-url`), and the runner's `--model` overrides the name.

Nothing else in `third_party/finance_harness/` is modified. The runner in `benchmarks/financegym/` imports the harness
(`financeharness.research.run_research`, `financeharness.tools.research.financegym_backend.FinanceGymBackend`) and
applies two *runtime* patches in `benchmarks/financegym/solver.py` (an `httpx.AsyncClient` with `verify=False`, and an
empty set of live market-data tools for point-in-time compliance); those live in our code, not in the vendored copy.

To refresh the vendored copy: check out google-research at the pinned commit, copy its `finance_harness/` directory
here, and re-apply the patch (`git apply -p1 --directory=third_party third_party/patches/gr-financegym.patch` from the
repository root, or apply the three hunks by hand).

## Not third-party code

* The Formula questions (FinLoRA) are **not** redistributed; `benchmarks/formula/data/formula_test.sha256` identifies the
  200 rows and `benchmarks/formula/prepare_data.py` rebuilds the file from the public source.
* AppWorld data is downloaded by the `appworld` package (Apache-2.0); nothing of it is stored here.
* The FinanceGym point-in-time corpus is rebuilt from Common Crawl with `benchmarks/financegym/corpus_pipeline/`.
