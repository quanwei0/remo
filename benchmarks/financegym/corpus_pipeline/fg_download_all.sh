#!/bin/bash
# Download ALL CC-NEWS WARCs for the target months into one dir (official-style
# static corpus). 4 parallel fetch lanes, skip existing, retry, no deletion.
set -uo pipefail
MONTHS="${MONTHS:-2025/04 2025/05 2025/06 2025/07}"
source "$(dirname "$0")/env.sh"; DEST=$WARC
LOG=$LOGDIR/dl_all.log
mkdir -p "$DEST"
LIST=$WARC_LIST
: > "$LIST"
for M in $MONTHS; do
  for try in 1 2 3 4 5; do
    n=$(curl -s -m 120 "https://data.commoncrawl.org/crawl-data/CC-NEWS/$M/warc.paths.gz" | zcat 2>/dev/null | tee -a "$LIST" | wc -l)
    [ "$n" -gt 0 ] && break; sleep 15
  done
done
total=$(wc -l < "$LIST")
echo "[dl-all] $total files total $(date)" | tee -a "$LOG"

fetch_lane() {
  local lane=$1
  local i=0
  while read -r path; do
    i=$((i+1))
    [ $((i % 4)) -ne "$lane" ] && continue
    name=$(basename "$path")
    [ -s "$DEST/$name" ] && continue
    for try in 1 2 3; do
      curl -sf -m 1800 -o "$DEST/.$name.part" "https://data.commoncrawl.org/$path" && break
      sleep 20
    done
    if [ -s "$DEST/.$name.part" ]; then mv "$DEST/.$name.part" "$DEST/$name"
    else rm -f "$DEST/.$name.part"; echo "FAILED $name" >> "$LOG"; fi
  done < "$LIST"
}
for lane in 0 1 2 3; do fetch_lane $lane & done
wait
got=$(ls "$DEST"/*.warc.gz | wc -l)
echo "[dl-all] COMPLETE: $got/$total on disk $(date)" | tee -a "$LOG"
touch "$DEST/DOWNLOAD_ALL_DONE"
