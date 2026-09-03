#!/usr/bin/env python
"""Merge per-worker parts into the official single corpus triplet.

Reads parts/<worker>/{embeddings.bin,metadata.jsonl,texts.jsonl,checkpoint.txt}
and concatenates them into search_all/{embeddings.bin,metadata.jsonl,texts.jsonl}
with contiguous global doc ids and a correct (n, dim) header.

Everything downstream is POSITIONAL: financegym/env/server.py does
`meta = state.meta[idx]` on the FAISS row index, and build_db.py zips metadata
with texts line by line. A single missing line therefore mis-attributes every
document after it, so all validation happens BEFORE any output byte is written,
and output lands in a temp dir that is renamed into place only after the final
count check passes.

Only bytes up to each part's LAST checkpoint are used, so a part left mid-shard
by preemption contributes exactly its complete shards.
"""
import json, os, struct, sys
from pathlib import Path

# FG_CORPUS_ROOT lets a scaled-down copy of the tree be validated end to end
# before committing days of GPU time to the real one.
from _env import CORPUS_ROOT as ROOT  # noqa: E402
OUT = ROOT / "search_all"
PARTS = OUT / "parts"
WARC_DIR = ROOT / "warc_all"
DIM = int(os.environ.get("EMBED_DIM", "2560"))
DRY = os.environ.get("DRY_RUN", "0") == "1"
ALLOW_PARTIAL = os.environ.get("ALLOW_PARTIAL", "0") == "1"


def count_lines(p, limit=None):
    n = 0
    with open(p, "rb") as f:
        for _ in f:
            n += 1
            if limit is not None and n >= limit:
                break
    return n


def main():
    parts = sorted(p for p in PARTS.iterdir() if p.is_dir()) if PARTS.exists() else []
    if not parts:
        print("no parts found"); return 1
    plan, seen, total = [], {}, 0
    for p in parts:
        ck = p / "checkpoint.txt"
        # A part with embedded data but no/empty checkpoint is unusable, and
        # skipping it would silently drop days of GPU work. Abort instead.
        if not ck.exists() or not ck.stat().st_size:
            if (p/"embeddings.bin").exists() and (p/"embeddings.bin").stat().st_size > 8:
                print(f"ABORT {p.name}: has embeddings but no usable checkpoint"); return 2
            print(f"  {p.name}: empty part, ignored"); continue
        rows = [l.rstrip("\n").split("\t") for l in open(ck) if l.strip()]
        n = sum(int(r[1]) for r in rows)
        eo, mo, to = int(rows[-1][2]), int(rows[-1][3]), int(rows[-1][4])
        want = 8 + n * DIM * 4
        if eo != want:
            print(f"ABORT {p.name}: checkpoint emb_off {eo} != 8+n*dim*4 {want}"); return 2
        for f, off in ((p/"embeddings.bin", eo), (p/"metadata.jsonl", mo), (p/"texts.jsonl", to)):
            if not f.exists() or f.stat().st_size < off:
                print(f"ABORT {p.name}: {f.name} is {f.stat().st_size if f.exists() else 0}B "
                      f"< checkpointed {off}B"); return 2
        # the docstring invariant, actually enforced: enough lines to cover n
        for f in (p/"metadata.jsonl", p/"texts.jsonl"):
            got = count_lines(f, limit=n)
            if got < n:
                print(f"ABORT {p.name}: {f.name} has {got} lines, need {n}"); return 2
        for nm, *_ in rows:
            if nm in seen:
                print(f"ABORT: shard {nm} in both {seen[nm]} and {p.name}"); return 2
            seen[nm] = p.name
        plan.append((p, n, eo, mo, to)); total += n
        print(f"  {p.name}: {len(rows)} shards, {n:,} articles")

    nwarc = len(list(WARC_DIR.glob("*.warc.gz")))
    print(f"total: {len(plan)} parts, {len(seen):,}/{nwarc:,} shards, {total:,} articles")
    if len(seen) < nwarc and not ALLOW_PARTIAL:
        print(f"ABORT: only {len(seen)}/{nwarc} WARCs embedded. "
              f"Set ALLOW_PARTIAL=1 to merge an intentionally partial corpus.")
        return 2
    if DRY:
        print("DRY_RUN=1, not writing"); return 0

    tmp = OUT / ".merge_tmp"
    tmp.mkdir(exist_ok=True)
    with open(tmp/"embeddings.bin", "wb") as ef, \
         open(tmp/"metadata.jsonl", "w") as mf, \
         open(tmp/"texts.jsonl", "w") as tf:
        ef.write(struct.pack("<ii", total, DIM))
        gid = 0
        for p, n, eo, mo, to in plan:
            with open(p/"embeddings.bin", "rb") as src:
                src.seek(8); left = eo - 8
                while left > 0:
                    buf = src.read(min(1 << 24, left))
                    if not buf:
                        print(f"ABORT: short read in {p.name}"); return 2
                    ef.write(buf); left -= len(buf)
            with open(p/"metadata.jsonl") as ms, open(p/"texts.jsonl") as ts:
                for i in range(n):
                    md = json.loads(next(ms)); td = json.loads(next(ts))
                    if md["doc_id"] != td["doc_id"]:
                        print(f"ABORT {p.name}: row {i} metadata {md['doc_id']} "
                              f"!= texts {td['doc_id']} — part is misaligned"); return 2
                    md["doc_id"] = td["doc_id"] = f"doc_{gid+i}"
                    mf.write(json.dumps(md) + "\n")
                    tf.write(json.dumps(td) + "\n")
            gid += n
            print(f"  merged {p.name} -> doc_{gid-n}..doc_{gid-1}")

    sz = (tmp/"embeddings.bin").stat().st_size
    ml = count_lines(tmp/"metadata.jsonl")
    tl = count_lines(tmp/"texts.jsonl")
    want = 8 + total*DIM*4
    if not (sz == want and ml == total and tl == total):
        print(f"ABORT after write: emb {sz:,} (want {want:,}) meta {ml:,} texts {tl:,} "
              f"(want {total:,}). Leaving output in {tmp}, canonical files untouched.")
        return 3
    for f in ("embeddings.bin", "metadata.jsonl", "texts.jsonl"):
        os.replace(tmp/f, OUT/f)
    tmp.rmdir()
    print(f"OK: {total:,} docs -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
