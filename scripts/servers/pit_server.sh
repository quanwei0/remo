#!/bin/bash
# FinanceGym point-in-time search server over the built corpus (CPU node; ~450 GB RAM for the full 145M-article index).
DATA=${DATA:?path to the built corpus dir (embeddings.bin, faiss_index.bin, corpus.db, metadata.jsonl)}
PORT=${PORT:-8889}
cd "$(dirname "$0")/../../third_party/finance_harness/FinanceGym" && exec python -m financegym.env.server --data-dir "$DATA" --host 0.0.0.0 --port "$PORT"
