#!/usr/bin/env python
"""Phase 1: extraction only, one shard file per WARC (atomic, resumable).

Reuses the official extract_article() from financegym.corpus.extract_embed so
filtering/normalization semantics are identical; only orchestration differs
(per-WARC shard files instead of one fused pass, so kills lose at most one WARC).
"""
import argparse
import json
import logging
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

from _env import FG_DIR, CORPUS_ROOT, WARC_DIR, SHARD_DIR, OUT_DIR, LOG_DIR  # noqa: E402,F401
from warcio.archiveiterator import ArchiveIterator  # noqa: E402
from financegym.corpus.extract_embed import extract_article  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("extract_shard")

# SHARD_DIR from _env
# WARC_DIR from _env


def process_warc(warc_name):
    wf = WARC_DIR / warc_name
    out = SHARD_DIR / (warc_name + ".jsonl")
    if out.exists():
        return warc_name, -1  # already done
    tmp = SHARD_DIR / (warc_name + f".tmp.{os.getpid()}")
    n = 0
    try:
        with open(wf, "rb") as f, open(tmp, "w") as w:
            try:
                for record in ArchiveIterator(f):
                    if record.rec_type != "response":
                        continue
                    ct = (record.http_headers.get_header("Content-Type")
                          if record.http_headers else "")
                    if not ct or "html" not in ct.lower():
                        continue
                    url = record.rec_headers.get_header("WARC-Target-URI") or ""
                    crawl_date = record.rec_headers.get_header("WARC-Date") or ""
                    try:
                        html = record.content_stream().read().decode("utf-8", errors="ignore")
                    except Exception:
                        continue
                    a = extract_article(html, url, crawl_date, warc_name=warc_name)
                    if a is not None:
                        w.write(json.dumps(a) + "\n")
                        n += 1
            except Exception as e:  # mirror official reader: keep partial WARC output
                log.warning("reader failed mid-%s: %s (keeping %d articles)", warc_name, e, n)
        os.replace(tmp, out)
        return warc_name, n
    except Exception as e:
        log.error("shard failed %s: %s", warc_name, e)
        try:
            os.unlink(tmp)
        except OSError:
            pass
        return warc_name, -2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True, help="file with one WARC filename per line")
    ap.add_argument("--workers", type=int, required=True)
    args = ap.parse_args()

    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    names = [l.strip() for l in open(args.list) if l.strip()]
    todo = [n for n in names if not (SHARD_DIR / (n + ".jsonl")).exists()]
    log.info("%d WARCs assigned, %d to do, %d workers", len(names), len(todo), args.workers)

    t0 = time.time()
    done = 0
    arts = 0
    with Pool(args.workers) as pool:
        for name, n in pool.imap_unordered(process_warc, todo, chunksize=1):
            done += 1
            if n > 0:
                arts += n
            if done % 20 == 0:
                rate = done / (time.time() - t0) * 3600
                eta_h = (len(todo) - done) / max(rate, 1e-9)
                log.info("progress %d/%d shards, %d articles, %.0f shards/h, ETA %.1f h",
                         done, len(todo), arts, rate, eta_h)
    log.info("ALL DONE: %d shards, %d articles, %.1f h",
             done, arts, (time.time() - t0) / 3600)
    # Global completion marker: only when every WARC has a shard. Embed workers
    # use it to tell "extraction still running" from "this WARC will never come".
    nwarc = len(list(WARC_DIR.glob("*.warc.gz")))
    nshard = len(list(SHARD_DIR.glob("*.jsonl")))
    if nshard >= nwarc:
        (SHARD_DIR/"EXTRACT_ALL_DONE").touch()
        log.info("EXTRACT_ALL_DONE marker written (%d/%d)", nshard, nwarc)
    else:
        log.info("this slice done; %d/%d shards globally, marker not written", nshard, nwarc)


if __name__ == "__main__":
    main()
