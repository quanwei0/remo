#!/bin/bash
# Rolling processor for one CC-NEWS month on the local embed server.
# Usage: TAG=2025-06 PORT=8888 bash fg_month_process.sh
set -uo pipefail
TAG=${TAG:?} PORT=${PORT:?}
source "$(dirname "$0")/env.sh"; IN=$CORPUS_ROOT/warc_$TAG
OUT=$CORPUS_ROOT/search_$TAG

source "$(dirname "$0")/env.sh"
mkdir -p "$IN" "$OUT"
cd $FG
export EMBED_URL=http://localhost:$PORT/v1/embeddings
export EMBED_TRUNCATE_TOKENS=8192
while :; do
  # free disk: drop WARCs the checkpoint says are done
  if [ -f "$OUT/checkpoint.txt" ]; then
    while read -r w; do rm -f "$IN/$w"; done < "$OUT/checkpoint.txt"
  fi
  pending=$(ls "$IN"/*.warc.gz 2>/dev/null | wc -l)
  if [ "$pending" -gt 0 ]; then
    $PY -m financegym.corpus.extract_embed "$IN" --output "$OUT" --workers 10 --embed-batch 32 --embed-workers 4 \
      >> $LOGDIR/proc_$TAG.log 2>&1 || sleep 30
  elif [ -f "$IN/DOWNLOAD_DONE" ]; then
    echo "[proc $TAG] month complete $(date)"; break
  else
    sleep 60
  fi
done
