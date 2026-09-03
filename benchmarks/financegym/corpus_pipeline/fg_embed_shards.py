#!/usr/bin/env python
"""Phase 2: embed shard files into official-format outputs, shard-checkpointed.

Consumes shards/<warc>.jsonl (from fg_extract_shard.py) and appends to
embeddings.bin / metadata.jsonl / texts.jsonl in the exact official layout:
8-byte (n, dim) header + float32 rows; row i <-> metadata line i <-> doc_i.

Shards are processed in whatever order they become available (sorted among the
available ones); the checkpoint records the realized order and resume replays
it exactly, so downstream row<->line alignment always holds. This avoids
head-of-line blocking on slow-to-extract WARCs.

Restart-safe at shard granularity: after each shard the three files are
flushed+fsynced, the bin header updated, and a checkpoint line
"name<TAB>count<TAB>emb_off<TAB>meta_off<TAB>texts_off" appended. On resume the
files are truncated back to the last checkpointed offsets, so a kill mid-shard
loses only that shard's partial work. Embed failures retry forever (server
restarts are transparent); persistent 4xx batches degrade per-article rather
than being discarded.
"""
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from _env import FG_DIR, CORPUS_ROOT, WARC_DIR, SHARD_DIR, OUT_DIR, LOG_DIR  # noqa: E402,F401
import numpy as np  # noqa: E402
import requests  # noqa: E402
from financegym.corpus.extract_embed import (  # noqa: E402
    DEFAULT_EMBED_DIM, embed_batch, metadata_line, pack_header, text_line,
)

REQ_TIMEOUT = float(os.environ.get("EMBED_REQ_TIMEOUT", "120"))
# One independent single-GPU server per port. vLLM data-parallel puts the DP
# ranks in lockstep (measured: dp4 caps at ~132 art/s and wedges on concurrent
# requests); N independent servers, one in-flight request each, scale linearly.
# EMBED_URLS: comma-separated full endpoints, so servers on other nodes can join.
_env_urls = os.environ.get("EMBED_URLS", "").strip()
if _env_urls:
    _urls = [u.strip() for u in _env_urls.split(",") if u.strip()]
else:
    EMBED_PORTS = [int(x) for x in os.environ.get("EMBED_PORTS", "8890,8891,8892,8893").split(",")]
    EMBED_HOST = os.environ.get("EMBED_HOST", "localhost")
    _urls = [f"http://{EMBED_HOST}:{p}/v1/embeddings" for p in EMBED_PORTS]
_slot = __import__("itertools").count()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("embed_shards")

# WARC_DIR from _env
# SHARD_DIR from _env
# OUT_DIR from _env
CKPT = OUT_DIR / "embed_checkpoint.txt"
BATCH = int(os.environ.get("EMBED_BATCH", "256"))
INFLIGHT = int(os.environ.get("EMBED_INFLIGHT", str(len(_urls))))
DIM = DEFAULT_EMBED_DIM


def embed_retry(texts, url=None):
    """Retry forever on server trouble; degrade per-article on persistent 4xx."""
    texts = [t if t.strip() else " " for t in texts]
    if url is None:
        url = _urls[next(_slot) % len(_urls)]
    delay = 5
    tries = 0
    while True:
        try:
            return embed_batch(texts, url=url, timeout=REQ_TIMEOUT)
        except requests.HTTPError as e:
            code = e.response.status_code if e.response is not None else 0
            if 400 <= code < 500:
                if len(texts) == 1:
                    short = texts[0][:2000]
                    if short != texts[0]:
                        texts = [short]
                        continue
                    log.error("article rejected by server (%s), zero-vector fallback", code)
                    return [[0.0] * DIM]
                mid = len(texts) // 2
                return embed_retry(texts[:mid], url) + embed_retry(texts[mid:], url)
            log.warning("embed 5xx (%s), retry in %ds", code, delay)
        except Exception as e:
            log.warning("embed error (%s), retry in %ds", type(e).__name__, delay)
        tries += 1
        if tries % 3 == 0:  # sticky failure: try a different server
            url = _urls[next(_slot) % len(_urls)]
        time.sleep(delay)
        delay = min(delay * 2, 60)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    canon = sorted(p.name for p in WARC_DIR.glob("*.warc.gz"))
    log.info("%d WARCs total", len(canon))

    done = {}
    emb_off, meta_off, texts_off, n_total = 8, 0, 0, 0
    if CKPT.exists():
        for line in open(CKPT):
            name, cnt, eo, mo, to = line.rstrip("\n").split("\t")
            done[name] = int(cnt)
            emb_off, meta_off, texts_off = int(eo), int(mo), int(to)
            n_total += int(cnt)
        log.info("resuming: %d shards, %d articles", len(done), n_total)

    emb_path = OUT_DIR / "embeddings.bin"
    meta_path = OUT_DIR / "metadata.jsonl"
    texts_path = OUT_DIR / "texts.jsonl"
    for p in (emb_path, meta_path, texts_path):
        if not p.exists():
            p.touch()
    emb_f = open(emb_path, "r+b")
    emb_f.truncate(emb_off)
    if emb_off == 8:
        emb_f.seek(0)
        emb_f.write(pack_header(0, DIM))
    meta_f = open(meta_path, "r+")
    meta_f.truncate(meta_off)
    texts_f = open(texts_path, "r+")
    texts_f.truncate(texts_off)
    emb_f.seek(0, 2)
    meta_f.seek(0, 2)
    texts_f.seek(0, 2)

    ckpt_f = open(CKPT, "a")
    pool = ThreadPoolExecutor(max_workers=INFLIGHT)
    t0 = time.time()
    arts0 = n_total
    sess_done = 0

    def process_shard(name):
        nonlocal n_total, sess_done
        arts = [json.loads(l) for l in open(SHARD_DIR / (name + ".jsonl"))]
        if arts:
            batches = [arts[i:i + BATCH] for i in range(0, len(arts), BATCH)]
            vec_lists = list(pool.map(
                lambda b: embed_retry([a["text"] for a in b]), batches))
            vecs = [v for vl in vec_lists for v in vl]
            arr = np.asarray(vecs, dtype=np.float32)
            assert arr.shape == (len(arts), DIM), f"{name}: {arr.shape}"
            emb_f.write(arr.tobytes())
            for i, a in enumerate(arts):
                doc_id = f"doc_{n_total + i}"
                meta_f.write(metadata_line(a, doc_id) + "\n")
                texts_f.write(text_line(a, doc_id) + "\n")
            n_total += len(arts)
        for f in (emb_f, meta_f, texts_f):
            f.flush()
            os.fsync(f.fileno())
        pos = emb_f.tell()
        emb_f.seek(0)
        emb_f.write(pack_header(n_total, DIM))
        emb_f.flush()
        os.fsync(emb_f.fileno())
        emb_f.seek(pos)
        ckpt_f.write(f"{name}\t{len(arts)}\t{emb_f.tell()}\t{meta_f.tell()}\t{texts_f.tell()}\n")
        ckpt_f.flush()
        os.fsync(ckpt_f.fileno())
        done[name] = len(arts)
        sess_done += 1
        if sess_done % 10 == 0:
            dt = time.time() - t0
            rate = (n_total - arts0) / max(dt, 1e-9)
            left = len(canon) - len(done)
            log.info("shard %d/%d, %d articles, %.1f art/s, ~%.1f h left",
                     len(done), len(canon), n_total, rate,
                     left * (dt / max(sess_done, 1)) / 3600)

    remaining = [n for n in canon if n not in done]
    while remaining:
        avail = [n for n in remaining if (SHARD_DIR / (n + ".jsonl")).exists()]
        if not avail:
            log.info("waiting for extraction (%d shards left)", len(remaining))
            time.sleep(60)
            remaining = [n for n in remaining if n not in done]
            continue
        for name in avail:
            process_shard(name)
        remaining = [n for n in remaining if n not in done]

    log.info("EMBED ALL DONE: %d shards, %d articles", len(done), n_total)


if __name__ == "__main__":
    main()
