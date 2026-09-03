"""Shared locations for the corpus pipeline (Python side); mirrors env.sh. Override with environment variables."""
import os
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
REPO = SCRIPTS.parents[2]
FG_DIR = Path(os.environ.get("FINHARNESS_ROOT", REPO / "third_party" / "finance_harness")) / "FinanceGym"
CORPUS_ROOT = Path(os.environ.get("FG_CORPUS_ROOT", REPO / "data" / "financegym_corpus"))
WARC_DIR = CORPUS_ROOT / "warc_all"
SHARD_DIR = CORPUS_ROOT / "shards"
OUT_DIR = Path(os.environ.get("FG_SEARCH_DIR", CORPUS_ROOT / "search_all"))
LOG_DIR = Path(os.environ.get("FG_LOGDIR", CORPUS_ROOT / "logs"))
if str(FG_DIR) not in sys.path:
    sys.path.insert(0, str(FG_DIR))
