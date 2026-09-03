#!/usr/bin/env python
"""Multi-node embedding worker: atomic shard claiming + per-worker output parts.

Several of these run at once (one per Slurm allocation, nodes are preemptible),
so they must never write the same bytes. Each worker:
  * claims a shard with O_CREAT|O_EXCL on claims/<shard>.claim (first wins),
  * writes into its OWN parts/<WORKER_ID>/ triplet, checkpointed per shard.
fg_merge_parts.py later concatenates the parts into the official single
embeddings.bin / metadata.jsonl / texts.jsonl with contiguous doc ids.

WORKER_ID must be unique per *allocation* (host+jobid), never just the host:
two allocations landing on one node would otherwise share a part directory and
destroy each other. A startup lock enforces this.

Preemption safety: a killed worker loses at most the in-flight shard, and its
claim is reaped by any worker after CLAIM_STALE_S once the shard is confirmed
absent from every part checkpoint.
"""
import errno, json, logging, os, socket, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from _env import FG_DIR, CORPUS_ROOT, WARC_DIR, SHARD_DIR, OUT_DIR, LOG_DIR  # noqa: E402,F401
import numpy as np
import requests
from financegym.corpus.extract_embed import (
    DEFAULT_EMBED_DIM, embed_batch, metadata_line, pack_header, text_line,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("embed_worker")

# WARC_DIR from _env
# SHARD_DIR from _env
OUT_ROOT = OUT_DIR
CLAIM_DIR = OUT_ROOT / "claims"
PARTS_DIR = OUT_ROOT / "parts"
EXTRACT_DONE = SHARD_DIR / "EXTRACT_ALL_DONE"

_host = socket.gethostname()
_job = os.environ.get("SLURM_JOB_ID", str(os.getpid()))
WORKER = os.environ.get("WORKER_ID") or f"{_host}-j{_job}"
BATCH = int(os.environ.get("EMBED_BATCH", "256"))
REQ_TIMEOUT = float(os.environ.get("EMBED_REQ_TIMEOUT", "60"))
CLAIM_STALE_S = float(os.environ.get("CLAIM_STALE_S", "1800"))
DIM = DEFAULT_EMBED_DIM
# Length control is done CLIENT-side by characters, deliberately not with vLLM's
# `truncate_prompt_tokens`: with that field set, the server intermittently drops
# embedding requests entirely (measured on this cluster: a hung request every
# ~131s, i.e. one full REQ_TIMEOUT, with the server otherwise idle and healthy).
# 24k chars sits under the 8192-token window for this tokenizer and leaves 99.8%
# of articles untouched (p99 = 19.3k chars); the alternative is a 400 that would
# otherwise cost the whole 256-item batch.
os.environ.pop("EMBED_TRUNCATE_TOKENS", None)
EMBED_MAX_CHARS = int(os.environ.get("EMBED_MAX_CHARS", "24000"))
MIN_EMBED_CHARS = int(os.environ.get("MIN_EMBED_CHARS", "512"))
_trunc = [0]   # articles shortened to fit the model window
_zero = [0]    # articles that failed even at MIN_EMBED_CHARS
_urls = [u.strip() for u in os.environ.get("EMBED_URLS", "").split(",") if u.strip()]
if not _urls:
    _urls = [f"http://localhost:{8890+i}/v1/embeddings" for i in range(int(os.environ.get("NGPU", "4")))]
INFLIGHT = int(os.environ.get("EMBED_INFLIGHT", str(len(_urls))))
_slot = iter(range(10**12))


def embed_retry(texts, url=None):
    texts = [(t[:EMBED_MAX_CHARS] if t.strip() else " ") for t in texts]
    if url is None:
        url = _urls[next(_slot) % len(_urls)]
    delay, tries = 5, 0
    while True:
        try:
            return embed_batch(texts, url=url, timeout=REQ_TIMEOUT)
        except requests.HTTPError as e:
            code = e.response.status_code if e.response is not None else 0
            if 400 <= code < 500:
                if len(texts) == 1:
                    # A single text can still exceed the 8192-token window even
                    # under EMBED_MAX_CHARS: CJK/Cyrillic run ~1 char/token, so a
                    # 19k-char Japanese article is ~19k tokens. Halve adaptively
                    # rather than dropping to a fixed prefix or a zero vector —
                    # this keeps as much of the article as the window allows and
                    # is script-agnostic.
                    t = texts[0]
                    if len(t) > MIN_EMBED_CHARS:
                        new = max(MIN_EMBED_CHARS, len(t) // 2)
                        _trunc[0] += 1
                        log.info("over-window article len=%d -> %d (adaptive)", len(t), new)
                        texts = [t[:new]]
                        continue
                    log.error("article rejected by server (%s) even at %d chars; ZERO VECTOR",
                              code, len(t))
                    _zero[0] += 1
                    return [[0.0] * DIM]
                mid = len(texts) // 2
                return embed_retry(texts[:mid], url) + embed_retry(texts[mid:], url)
            log.warning("embed 5xx %s, retry %ds", code, delay)
        except Exception as e:
            log.warning("embed %s, retry %ds", type(e).__name__, delay)
        tries += 1
        # move off the failing server immediately: a dropped request costs a full
        # REQ_TIMEOUT, so there is no reason to give the same endpoint 3 chances
        if len(_urls) > 1:
            url = _urls[next(_slot) % len(_urls)]
        time.sleep(delay); delay = min(delay * 2, 60)


def completed_shards():
    """Names present in ANY part checkpoint — the ground truth for 'embedded'."""
    out = set()
    if PARTS_DIR.exists():
        for ck in PARTS_DIR.glob("*/checkpoint.txt"):
            try:
                for line in open(ck):
                    if line.strip():
                        out.add(line.split("\t", 1)[0])
            except OSError:
                pass
    return out


def reap_stale_claims(done_all):
    """Unlink claims older than CLAIM_STALE_S whose shard no part ever finished.

    Without this a preempted worker's claim blocks that shard forever and its
    articles are silently dropped from the corpus.
    """
    now, n = time.time(), 0
    for c in CLAIM_DIR.glob("*.claim"):
        name = c.name[:-6]
        if name in done_all:
            continue
        try:
            if now - c.stat().st_mtime < CLAIM_STALE_S:
                continue
            c.unlink()
            n += 1
        except OSError:
            pass
    if n:
        log.warning("reaped %d stale claims", n)
    return n


def claim(name):
    p = CLAIM_DIR / (name + ".claim")
    try:
        fd = os.open(str(p), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w") as f:
        f.write(f"{WORKER}\t{int(time.time())}\n")
    return True


def unclaim(name):
    try:
        (CLAIM_DIR / (name + ".claim")).unlink()
    except OSError:
        pass


def open_at(path, off, binary, resuming):
    """Open for append at a checkpointed offset, refusing to invent data.

    ftruncate EXTENDS a short file with NULs, which would manufacture zero
    vectors / NUL-padded jsonl that still satisfy every size check downstream.
    The guard applies only when RESUMING: a brand-new part legitimately has a
    0-byte embeddings.bin while off==8, because the 8-byte header has not been
    written yet.
    """
    if not path.exists():
        path.touch()
    size = path.stat().st_size
    if resuming and size < off:
        raise SystemExit(
            f"ABORT: {path} is {size}B but checkpoint says {off}B — part is "
            f"damaged (truncated/partially copied). Refusing to pad with NULs.")
    f = open(path, "r+b" if binary else "r+")
    f.truncate(off)
    f.seek(0, 2)
    return f


def main():
    for d in (CLAIM_DIR, PARTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    mine = PARTS_DIR / WORKER
    mine.mkdir(parents=True, exist_ok=True)

    # one writer per part directory, ever
    lock = mine / ".lock"
    try:
        fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        os.write(fd, f"{WORKER}\t{os.getpid()}\t{int(time.time())}\n".encode())
        os.close(fd)
    except FileExistsError:
        if os.environ.get("FORCE_PART_LOCK") != "1":
            raise SystemExit(
                f"ABORT: {lock} exists — another worker owns this part. "
                f"Set a unique WORKER_ID, or FORCE_PART_LOCK=1 if you are certain "
                f"the previous owner is dead.")
        log.warning("taking over part %s (FORCE_PART_LOCK=1)", WORKER)

    ckpt_p = mine / "checkpoint.txt"
    done, emb_off, meta_off, texts_off, n_total = {}, 8, 0, 0, 0
    if ckpt_p.exists():
        for line in open(ckpt_p):
            if not line.strip():
                continue
            nm, cnt, eo, mo, to = line.rstrip("\n").split("\t")
            done[nm] = int(cnt); emb_off, meta_off, texts_off = int(eo), int(mo), int(to)
            n_total += int(cnt)
        log.info("resuming part %s: %d shards, %d articles", WORKER, len(done), n_total)

    resuming = bool(done)
    ef = open_at(mine/"embeddings.bin", emb_off, True, resuming)
    if not resuming:
        ef.seek(0); ef.truncate(0); ef.write(pack_header(0, DIM)); ef.seek(0, 2)
    mf = open_at(mine/"metadata.jsonl", meta_off, False, resuming)
    tf = open_at(mine/"texts.jsonl", texts_off, False, resuming)
    cf = open(ckpt_p, "a")
    pool = ThreadPoolExecutor(max_workers=INFLIGHT)
    t0, a0, sess = time.time(), n_total, 0

    def process(name):
        nonlocal n_total, sess
        arts = [json.loads(l) for l in open(SHARD_DIR / (name + ".jsonl"))]
        if arts:
            batches = [arts[i:i+BATCH] for i in range(0, len(arts), BATCH)]
            vl = list(pool.map(lambda b: embed_retry([a["text"] for a in b]), batches))
            vecs = [v for x in vl for v in x]
            if len(vecs) != len(arts):
                raise RuntimeError(f"{name}: got {len(vecs)} vectors for {len(arts)} articles")
            arr = np.asarray(vecs, dtype=np.float32)
            if arr.shape != (len(arts), DIM):
                raise RuntimeError(f"{name}: array shape {arr.shape} != {(len(arts), DIM)}")
            ef.write(arr.tobytes())
            for i, a in enumerate(arts):
                did = f"doc_{n_total+i}"          # local; merge renumbers globally
                mf.write(metadata_line(a, did) + "\n")
                tf.write(text_line(a, did) + "\n")
            n_total += len(arts)
        for f in (ef, mf, tf):
            f.flush(); os.fsync(f.fileno())
        pos = ef.tell(); ef.seek(0); ef.write(pack_header(n_total, DIM))
        ef.flush(); os.fsync(ef.fileno()); ef.seek(pos)
        cf.write(f"{name}\t{len(arts)}\t{ef.tell()}\t{mf.tell()}\t{tf.tell()}\n")
        cf.flush(); os.fsync(cf.fileno())
        done[name] = len(arts); sess += 1
        if sess % 10 == 0:
            dt = time.time() - t0
            log.info("part %s: +%d shards, %d articles, %.0f art/s, "
                     "adaptive-truncated=%d zero-vectors=%d",
                     WORKER, sess, n_total, (n_total - a0) / max(dt, 1e-9),
                     _trunc[0], _zero[0])

    canon = sorted(p.name for p in WARC_DIR.glob("*.warc.gz"))
    idle = 0
    while True:
        done_all = completed_shards()
        avail = [n for n in canon
                 if n not in done_all
                 and not (CLAIM_DIR / (n + ".claim")).exists()
                 and (SHARD_DIR / (n + ".jsonl")).exists()]
        if avail:
            idle = 0
            for name in avail:
                if not claim(name):
                    continue
                try:
                    process(name)
                except Exception as e:            # never die holding a claim
                    log.exception("shard %s FAILED (%s); releasing claim", name, type(e).__name__)
                    unclaim(name)
                    time.sleep(5)
            continue

        # nothing available: exit only when every shard is truly EMBEDDED,
        # never merely claimed (a dead worker's claim must not look like done).
        remaining = [n for n in canon if n not in done_all]
        if not remaining:
            log.info("all %d shards embedded; worker %s done", len(canon), WORKER); break
        reap_stale_claims(done_all)
        missing_shard = [n for n in remaining if not (SHARD_DIR / (n + ".jsonl")).exists()]
        if EXTRACT_DONE.exists() and missing_shard:
            log.error("extraction finished but %d WARCs never produced a shard; "
                      "cannot embed them: %s", len(missing_shard), missing_shard[:5])
            if len(missing_shard) == len(remaining):
                break
        idle += 1
        if idle % 10 == 1:
            log.info("waiting: %d shards left (%d not yet extracted)",
                     len(remaining), len(missing_shard))
        time.sleep(60)
    log.info("WORKER DONE %s: %d shards, %d articles", WORKER, len(done), n_total)


if __name__ == "__main__":
    main()
