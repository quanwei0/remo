#!/bin/bash
# Query-embedding service for FinanceGym search (must be the SAME model the corpus was embedded with).
# A 4B embedder needs ~10 GB; 0.25 of an 80 GB GPU is plenty (the vLLM default 0.9 wastes the card).
PORT=${PORT:-8888}; UTIL=${UTIL:-0.25}
exec vllm serve Qwen/Qwen3-Embedding-4B --port "$PORT" --max-model-len 8192 --gpu-memory-utilization "$UTIL"
