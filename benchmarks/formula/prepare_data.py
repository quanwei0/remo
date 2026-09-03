"""Rebuild the Formula test file (200 rows) from the public FinLoRA source.

The questions are not redistributed (FinLoRA declares no license for them); data/formula_test.sha256 holds one
SHA-256 per row in evaluation order. This script downloads FinLoRA's `data/test/formula_test.jsonl` (or converts a
local file with --from-file), normalizes rows to {"context", "target"}, keeps exactly the rows whose hash is in
the manifest, orders them like the manifest and reports how many of the 200 were matched.

    python benchmarks/formula/prepare_data.py --out data/formula_test.jsonl
    python benchmarks/formula/prepare_data.py --from-file some_copy.jsonl --out data/formula_test.jsonl
"""
import argparse
import os
import sys
import urllib.request

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from benchmarks.formula.data import (DEFAULT_DATA_PATH, MANIFEST_PATH, load_rows, read_manifest,   # noqa: E402
                                     select_by_manifest, write_rows)

DEFAULT_SOURCE_URL = "https://raw.githubusercontent.com/Open-Finance-Lab/FinLoRA/main/data/test/formula_test.jsonl"
# whole-file SHA-256 of the exact file evaluated in the paper (also recorded in data/README.md)
EXPECTED_FILE_SHA256 = "079919e9631e50ba0489d2be1221ecf59952a9230ff91c5325c27b37ed99e342"


def download(url: str, dest: str, timeout: float = 120.0) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "remo-formula-prepare/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r, open(dest, "wb") as f:
        f.write(r.read())
    return dest


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=DEFAULT_DATA_PATH, help="output jsonl (default: data/formula_test.jsonl)")
    p.add_argument("--from-file", default=None, help="convert this local jsonl instead of downloading")
    p.add_argument("--source-url", default=DEFAULT_SOURCE_URL, help="public source (FinLoRA raw file)")
    p.add_argument("--manifest", default=MANIFEST_PATH)
    p.add_argument("--no-filter", action="store_true", help="keep every converted row in source order (NOT the paper's set)")
    p.add_argument("--keep-source", action="store_true", help="keep the downloaded file next to --out")
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    src = a.from_file
    downloaded = None
    if not src:
        downloaded = src = a.out + ".source.jsonl"
        print(f"downloading {a.source_url}")
        download(a.source_url, src)
    rows = load_rows(src)
    print(f"source: {src} ({len(rows)} rows)")
    manifest = read_manifest(a.manifest)
    if a.no_filter:
        selected, missing, unmatched = rows, [], 0
    else:
        selected, missing, unmatched = select_by_manifest(rows, manifest)
    sha = write_rows(selected, a.out)
    print(f"matched {len(selected)} of {len(manifest)} manifest rows; {unmatched} source rows are not in the manifest")
    print(f"wrote {a.out}: {len(selected)} rows, sha256 {sha}"
          + (" (identical to the file evaluated in the paper)" if sha == EXPECTED_FILE_SHA256 else ""))
    if downloaded and not a.keep_source:
        os.remove(downloaded)
    if missing:
        print(f"WARNING: {len(missing)} manifest rows not found in the source, e.g. {missing[:3]}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
