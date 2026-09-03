#!/bin/bash
# Rolling downloader for one CC-NEWS month (login node), throttled by disk buffer.
# Usage: MONTH=2025/06 TAG=2025-06 bash fg_month_download.sh
set -uo pipefail
MONTH=${MONTH:?} TAG=${TAG:?}
source "$(dirname "$0")/env.sh"; IN=$CORPUS_ROOT/warc_$TAG
CKPT=$CORPUS_ROOT/search_$TAG/checkpoint.txt
BUF=${BUF:-25}
mkdir -p "$IN"
LIST=$(mktemp)
for try in 1 2 3 4 5; do
  curl -s -m 120 "https://data.commoncrawl.org/crawl-data/CC-NEWS/$MONTH/warc.paths.gz" | zcat > "$LIST" 2>/dev/null
  [ -s "$LIST" ] && break; sleep 10
done
total=$(wc -l < "$LIST")
echo "[dl $TAG] $total files to fetch"
n=0
while read -r path; do
  name=$(basename "$path")
  n=$((n+1))
  [ -f "$IN/$name" ] && continue
  grep -q "^$name$" "$CKPT" 2>/dev/null && continue
  while [ "$(ls "$IN"/*.warc.gz 2>/dev/null | wc -l)" -ge "$BUF" ]; do sleep 60; done
  for try in 1 2 3; do
    curl -s -m 1800 -o "$IN/.$name.part" "https://data.commoncrawl.org/$path" && break
    sleep 20
  done
  if [ -s "$IN/.$name.part" ]; then
    mv "$IN/.$name.part" "$IN/$name"
  else
    rm -f "$IN/.$name.part"; echo "[dl $TAG] FAILED $name" >> $LOGDIR/dl_$TAG.failed
  fi
  [ $((n % 25)) -eq 0 ] && echo "[dl $TAG] $n/$total $(date '+%H:%M')"
done < "$LIST"
rm -f "$LIST"
touch "$IN/DOWNLOAD_DONE"
echo "[dl $TAG] DOWNLOAD DONE ($total files) $(date)"
