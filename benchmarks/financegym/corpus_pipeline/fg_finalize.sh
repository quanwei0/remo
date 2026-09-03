#!/bin/bash
# Post-embedding pipeline: merge parts -> build_db -> FAISS index -> verify.
#
# Runs unattended after days of GPU work, so every stage is guarded and every
# "skip" check verifies CONTENT, not mere existence: a build killed at hour 3
# leaves a non-empty corpus.db / short faiss_index.bin that a `[ -s ]` test
# would happily skip, shipping a truncated corpus with no error.
#
# This job needs CPUs and RAM, NOT GPUs: the env has faiss-cpu
# (faiss.get_num_gpus()==0), so index.build's GPU k-means branch never fires and
# index.add is CPU-only regardless. Measured budget at 155M docs:
#   merge ~2h (I/O) | build_db 3-5h | index ~6h (2.8h of that is the random
#   training gather at ~314 vec/s over NFS) | peak RAM ~410GB | disk +5.2TB
set -uo pipefail
source "$(dirname "$0")/env.sh"


cd $FG || { echo "cd $FG failed"; exit 1; }   # `python -m` resolves financegym from CWD

hdr_n() { $PY -c "import struct,sys;print(struct.unpack('<ii',open('$OUT/embeddings.bin','rb').read(8))[0])" 2>/dev/null || echo 0; }

echo "===== 1/4 merge parts $(date) ====="
# all three legs must be present: build_db zips metadata with texts using
# strict=False, so a short texts.jsonl silently truncates the whole DB.
if [ -s "$OUT/embeddings.bin" ] && [ -s "$OUT/metadata.jsonl" ] && [ -s "$OUT/texts.jsonl" ]; then
  echo "[skip] merged triplet already present"
else
  # NOTE: the prefix must be a LITERAL assignment — bash parses assignment
  # prefixes before expansion, so ${VAR:+VAR=$VAR} becomes a command name.
  ALLOW_PARTIAL=${ALLOW_PARTIAL:-0} FG_CORPUS_ROOT=$CORPUS_ROOT \
    $PY $SCRIPTS/fg_merge_parts.py \
    || { echo "MERGE FAILED — 停止(不在未校验的语料上继续)"; exit 2; }
fi
N=$(hdr_n)
[ "$N" -gt 0 ] 2>/dev/null || { echo "无法读取 embeddings.bin 表头"; exit 2; }
echo "语料规模: $N 向量"

echo "===== 2/4 build_db $(date) ====="
db_rows() { $PY -c "
import sqlite3,sys
try: print(sqlite3.connect('file:$OUT/corpus.db?mode=ro',uri=True).execute('select count(*) from docs').fetchone()[0])
except Exception: print(0)" 2>/dev/null || echo 0; }
if [ "$(db_rows)" = "$N" ]; then
  echo "[skip] corpus.db complete ($N rows)"
else
  echo "(现有 $(db_rows) 行 != $N,重建;预计 3-5 小时,产物约 $((N/1000000*4375/1000)) GB)"
  time $PY -m financegym.corpus.build_db --input "$OUT" || { echo "BUILD_DB FAILED"; exit 3; }
  [ "$(db_rows)" = "$N" ] || { echo "BUILD_DB 行数不符: $(db_rows) != $N"; exit 3; }
fi

echo "===== 3/4 FAISS index $(date) ====="
IDX="$OUT/faiss_index.bin"          # financegym/index/build.py: INDEX_FILENAME
want=$(( N * 2568 ))                # 2560B sq8 code + 8B int64 id per vector
have=$( [ -f "$IDX" ] && stat -c%s "$IDX" || echo 0 )
if [ "$have" -gt $(( want * 98 / 100 )) ]; then
  echo "[skip] faiss_index.bin complete (${have}B)"
else
  echo "(现有 ${have}B,目标约 ${want}B;预计 6 小时,纯 CPU)"
  time $PY -m financegym.index.build --input "$OUT" --index-type ivf_sq8 \
    || { echo "INDEX FAILED"; exit 4; }
fi

echo "===== 4/4 verify $(date) ====="
$PY - <<PYEOF || { echo "校验失败 — 语料不可用"; exit 5; }
import json, os, sqlite3, struct
out = "$OUT"
n, d = struct.unpack("<ii", open(os.path.join(out,"embeddings.bin"),"rb").read(8))
sz = os.path.getsize(os.path.join(out,"embeddings.bin"))
ml = sum(1 for _ in open(os.path.join(out,"metadata.jsonl")))
tl = sum(1 for _ in open(os.path.join(out,"texts.jsonl")))
rows = sqlite3.connect(os.path.join(out,"corpus.db")).execute("select count(*) from docs").fetchone()[0]
import faiss
ntot = faiss.read_index(os.path.join(out,"faiss_index.bin"), faiss.IO_FLAG_READ_ONLY).ntotal
print(f"embeddings.bin {n:,} x {d}  ({sz:,}B, 应为 {8+n*d*4:,})")
print(f"metadata.jsonl {ml:,} 行 | texts.jsonl {tl:,} 行 | corpus.db {rows:,} 行 | faiss ntotal {ntot:,}")
bad = []
if sz != 8+n*d*4: bad.append("向量文件大小与表头不符")
if ml != n: bad.append(f"metadata 行数 {ml} != {n}")
if tl != n: bad.append(f"texts 行数 {tl} != {n}")
if rows != n: bad.append(f"corpus.db 行数 {rows} != {n}")
if ntot != n: bad.append(f"faiss ntotal {ntot} != {n}")
if bad: raise SystemExit("不一致: " + "; ".join(bad))
print("ALL CHECKS PASSED")
PYEOF
echo "===== ALL DONE $(date) ====="
echo "下一步:"
echo "  cd $FG && $PY -m financegym.env.server --data-dir $OUT   # 127.0.0.1:8889, 需 ~505GB 内存"
echo "  then run benchmarks/financegym/run_financegym.py against it"
echo "注意: 合并完成前不要删除 warc_all/ —— fg_merge_parts.py 用它数 WARC 判完整性,"
echo "      目录没了 nwarc=0,完整性检查会静默通过。"
