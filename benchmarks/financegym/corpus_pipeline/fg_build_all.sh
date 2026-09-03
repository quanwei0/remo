#!/bin/bash
# Official build over the full static corpus (runs on gpu4b):
# wait for download completion + local embed server, then the official stages.
set -uo pipefail
source "$(dirname "$0")/env.sh"



cd $FG
export EMBED_URL=http://localhost:8888/v1/embeddings
export EMBED_TRUNCATE_TOKENS=8192
echo "[build-all] waiting for embed server..."
until curl -sf -m 3 http://localhost:8888/v1/models >/dev/null 2>&1; do sleep 30; done
# incremental passes: start on whatever is downloaded, checkpoint skips done WARCs;
# stop when the downloader is finished and a pass found nothing new to do.
while :; do
  echo "[build-all] extract+embed pass start $(date)"
  time $PY -m financegym.corpus.extract_embed "$WARC" --output "$OUT" --workers 48 --embed-batch 16 --embed-workers 4
  done_ct=$(wc -l < "$OUT/checkpoint.txt" 2>/dev/null || echo 0)
  on_disk=$(ls "$WARC"/*.warc.gz 2>/dev/null | wc -l)
  echo "[build-all] pass done: $done_ct processed / $on_disk on disk"
  if [ -f "$WARC/DOWNLOAD_ALL_DONE" ] && [ "$done_ct" -ge "$on_disk" ]; then break; fi
  sleep 60
done
echo "[build-all] build_db $(date)"
time $PY -m financegym.corpus.build_db --input "$OUT"
echo "[build-all] index.build $(date)"
time $PY -m financegym.index.build --input "$OUT" --index-type ivf_sq8
ls -la "$OUT"; wc -l "$OUT/metadata.jsonl" 2>/dev/null
echo "[build-all] ALL DONE $(date)"
