#!/bin/bash
# Start N independent single-GPU embed servers for THIS allocation.
# Independent servers (not --data-parallel-size N) avoid vLLM's DP lockstep,
# which caps a dp4 server at ~132 art/s and wedges under concurrent requests.
#
# Multi-tenant safety: several allocations can land on one node, so this must
# never touch processes it does not own. It therefore does NOT pkill node-wide,
# and derives its port range from the Slurm job id so two allocations on one
# host cannot collide.
set -uo pipefail
source "$(dirname "$0")/env.sh"
# vllm must be on PATH
NGPU=${NGPU:-4}
JOB=${SLURM_JOB_ID:-0}
# 8890 + (job % 40)*8 keeps ranges 8 apart and inside an unprivileged band
BASEPORT=${BASEPORT:-$((8890 + (JOB % 40) * 8))}
TAG=$(hostname)-j$JOB
echo "[emb] job=$JOB ngpu=$NGPU ports=$BASEPORT..$((BASEPORT+NGPU-1))"

for i in $(seq 0 $((NGPU-1))); do
  p=$((BASEPORT+i))
  if curl -sf -m 3 http://localhost:$p/v1/models >/dev/null 2>&1; then
    echo "[emb] port $p already serving, reusing"; continue
  fi
  CUDA_VISIBLE_DEVICES=$i nohup vllm serve Qwen/Qwen3-Embedding-4B \
    --port $p --max-model-len 8192 --gpu-memory-utilization 0.85 \
    >> $LOGDIR/vllm_emb_${TAG}_$i.log 2>&1 &
done
ok=0
for i in $(seq 0 $((NGPU-1))); do
  p=$((BASEPORT+i))
  for t in $(seq 1 180); do
    curl -sf -m 5 http://localhost:$p/v1/models >/dev/null 2>&1 && { echo "[emb] port $p ready $(date '+%T')"; ok=$((ok+1)); break; }
    sleep 10
  done
done
echo "[emb] $ok/$NGPU SERVERS UP $(date '+%T')"
[ "$ok" -gt 0 ] || exit 1
echo "$BASEPORT" > $LOGDIR/.baseport.$JOB
