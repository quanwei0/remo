#!/bin/bash
# On a 4xH100 node: Qwen3-Embedding dp4 server + official build pipeline over warc_all.
set -uo pipefail
source "$(dirname "$0")/env.sh"
# vllm must be on PATH
pkill -9 -f -i vllm 2>/dev/null || true; sleep 5
nohup vllm serve Qwen/Qwen3-Embedding-4B --port 8888 --max-model-len 8192 \
  --gpu-memory-utilization 0.9 --data-parallel-size 4 > $LOGDIR/vllm_emb_build.log 2>&1 &
for i in $(seq 1 150); do curl -sf http://localhost:8888/v1/models >/dev/null 2>&1 && { echo "embed server ready"; break; }; sleep 10; done
bash "$SCRIPTS/fg_build_all.sh"
