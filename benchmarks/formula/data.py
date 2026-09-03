"""Formula data: row schema, question parsing, row hashing against the manifest.

A row is {"context": "<instruction incl. the formula> Question: \"...\". Answer:", "target": "15.0"} (FinLoRA
Formula test set, 200 rows). The loop sees the question text (between "Question: " and ". Answer:") followed by the
fixed answer-format sentence, exactly as the paper's runs built it; the target stays outside the loop and is used
post hoc by scoring.py. The rows are not redistributed: data/formula_test.sha256 lists one SHA-256 per row in
evaluation order and prepare_data.py rebuilds the file from the public source.
"""
import hashlib
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
MANIFEST_PATH = os.path.join(HERE, "data", "formula_test.sha256")
DEFAULT_DATA_PATH = os.environ.get("REMO_FORMULA_DATA", os.path.join(REPO, "data", "formula_test.jsonl"))

ANSWER_FORMAT = (" Your answer should be a plain floating point number, round to the nearest hundredth if "
                 "necessary. Do the necessary conversions, for example 5 million should be 5000000.0. ")
_FORMULA_NAME = re.compile(r"^\s*Use formula (.+?) to answer the question", re.S)


# -- parsing --------------------------------------------------------------------------------------
def parse_question(context: str) -> str:
    """The question text of the paper's runs: between "Question: " and the first ". Answer:", stripped,
    one pair of straight double quotes removed, followed by the fixed answer-format sentence (which ends
    with a space). Without both markers the whole context is the question."""
    if "Question: " in context and ". Answer:" in context:
        q = context.split("Question: ", 1)[1].split(". Answer:")[0].strip()
        if q.startswith('"') and q.endswith('"'):
            q = q[1:-1]
        return q + ANSWER_FORMAT
    return context


def formula_name(context: str) -> str:
    """The formula named by the instruction ("Use formula X to answer the question"), "" if absent.
    Used only for the per-formula breakdown in final_results.json."""
    m = _FORMULA_NAME.match(context or "")
    return m.group(1).strip() if m else ""


def make_task(index: int, row: dict) -> dict:
    """What the loop sees: the question and an (always empty) context slot for the solver prompt; no target."""
    return {"task_index": index, "question": parse_question(row["context"]), "context": "",
            "formula": formula_name(row["context"])}


# -- rows -----------------------------------------------------------------------------------------
def normalize_row(obj: dict) -> dict | None:
    """Coerce a source record to {"context", "target"} (target as a string); None if it has no usable fields."""
    if not isinstance(obj, dict):
        return None
    context = obj.get("context")
    if context is None:
        context = obj.get("input", obj.get("prompt", obj.get("question")))
    target = obj.get("target")
    if target is None:
        target = obj.get("output", obj.get("answer", obj.get("label")))
    if context is None or target is None:
        return None
    if isinstance(target, float) and target.is_integer():
        target = f"{target:.1f}"
    return {"context": str(context), "target": str(target).strip()}


def load_rows(path: str, limit: int = 0) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = normalize_row(json.loads(line))
            if row is None:
                raise ValueError(f"{path}:{n}: row has no context/target")
            rows.append(row)
    return rows[:limit] if limit else rows


def write_rows(rows: list[dict], path: str) -> str:
    """Write rows as jsonl (key order context, target; ASCII-escaped like the source) and return the
    SHA-256 of the written file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"context": r["context"], "target": r["target"]}) + "\n")
    return file_sha256(path)


# -- hashing --------------------------------------------------------------------------------------
def row_hash(row: dict) -> str:
    payload = json.dumps({"context": row["context"], "target": row["target"]}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_manifest(path: str = MANIFEST_PATH) -> list[str]:
    """Row hashes in evaluation order; '#' lines are comments."""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                out.append(s.lower())
    return out


def select_by_manifest(rows: list[dict], manifest: list[str]) -> tuple[list[dict], list[str], int]:
    """Keep the rows whose hash is in the manifest, in manifest order (first occurrence wins on duplicates).
    Returns (selected rows, manifest hashes not found, number of source rows not in the manifest)."""
    by_hash: dict[str, dict] = {}
    unmatched = 0
    for r in rows:
        h = row_hash(r)
        if h in manifest:
            by_hash.setdefault(h, r)
        else:
            unmatched += 1
    selected = [by_hash[h] for h in manifest if h in by_hash]
    missing = [h for h in manifest if h not in by_hash]
    return selected, missing, unmatched


def check_against_manifest(rows: list[dict], manifest: list[str]) -> dict:
    """Row-by-row comparison of a data file with the manifest (used by the runner to label a run)."""
    hashes = [row_hash(r) for r in rows]
    in_order = sum(1 for a, b in zip(hashes, manifest) if a == b)
    return {"rows": len(rows), "manifest_rows": len(manifest), "matched_in_order": in_order,
            "matched_any_order": len(set(hashes) & set(manifest)),
            "exact": len(rows) == len(manifest) and in_order == len(manifest)}
