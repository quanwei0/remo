#!/bin/bash
# Shared locations for the corpus pipeline. Override with environment variables; defaults live inside the repo.
#   FINHARNESS_ROOT  the official FinanceGym agent (default: third_party/finance_harness)
#   FG_CORPUS_ROOT   working root for WARCs, shards, index (default: <repo>/data/financegym_corpus)
#   FG_SEARCH_DIR    final search index dir (default: $FG_CORPUS_ROOT/search_all)
#   PYTHON           interpreter with financeharness installed (default: python)
SCRIPTS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPTS/../../.." && pwd)
FG=${FINHARNESS_ROOT:-$REPO/third_party/finance_harness}/FinanceGym
CORPUS_ROOT=${FG_CORPUS_ROOT:-$REPO/data/financegym_corpus}
WARC=$CORPUS_ROOT/warc_all
SHARDS=$CORPUS_ROOT/shards
OUT=${FG_SEARCH_DIR:-$CORPUS_ROOT/search_all}
LOGDIR=${FG_LOGDIR:-$CORPUS_ROOT/logs}
PY=${PYTHON:-python}
WARC_LIST=${FG_WARC_LIST:-$SCRIPTS/warc_paths.txt}
mkdir -p "$WARC" "$SHARDS" "$OUT" "$LOGDIR"
