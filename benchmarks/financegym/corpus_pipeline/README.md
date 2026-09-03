# Rebuilding the FinanceGym point-in-time corpus

The corpus the paper used: CC-NEWS Oct-2024 … Nov-2025, the 6,758 WARC files listed in `warc_paths.txt`, 145.3M extracted
articles, embedded with Qwen3-Embedding-4B, indexed with FAISS IVF-SQ8 (nlist 12,053, nprobe 32). Common Crawl is public;
the build is heavy (embedding ≈ 3 days on 8 H100s; the finished index needs ~450 GB RAM to serve).

All scripts read their locations from `env.sh` / `_env.py` (override with environment variables):

| variable | default | meaning |
|---|---|---|
| `FINHARNESS_ROOT` | `third_party/finance_harness` | the official FinanceGym package (its `financegym.corpus` code does extraction/embedding) |
| `FG_CORPUS_ROOT` | `<repo>/data/financegym_corpus` | working root: `warc_all/`, `shards/`, `search_all/`, `logs/` |
| `FG_SEARCH_DIR` | `$FG_CORPUS_ROOT/search_all` | the final index served by `scripts/servers/pit_server.sh` |
| `PYTHON` | `python` | interpreter of the `remo-financegym` env |

Stages (run in order; every stage is restartable):

1. `fg_download_all.sh` — fetch the WARC files in `warc_paths.txt` from Common Crawl into `warc_all/`.
2. `fg_extract_shard.py` — extract news articles from WARCs into JSONL shards (CPU; `--workers`).
3. `fg_emb_servers.sh` + `fg_embed_shards.py` / `fg_embed_worker.py` — start embedding servers (one per GPU) and embed the shards;
   `fg_build_all.sh` / `fg_build_node.sh` are the single-node orchestration wrappers we used (Slurm-free inside).
4. `fg_merge_parts.py` + `fg_finalize.sh` — merge per-node parts, verify completeness against the WARC list, build the FAISS index and `corpus.db`.
5. Serve: `DATA=$FG_SEARCH_DIR scripts/servers/pit_server.sh`.

`fg_month_download.sh` / `fg_month_process.sh` are the same pipeline for a single month tag (useful for a small pilot corpus).
