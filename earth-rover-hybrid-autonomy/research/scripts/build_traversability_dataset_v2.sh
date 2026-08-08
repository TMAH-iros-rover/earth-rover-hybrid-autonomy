#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APPROVED_V1="${APPROVED_V1:-$HOME/datasets/generated/traversability_dataset_v1/approved_120_v1}"
MANUAL_V2="${MANUAL_V2:-$HOME/datasets/review_bundles/traversability_manual_v2_33_imported}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/datasets/generated/traversability_dataset_v2/approved_153_v2}"
SEED="${SEED:-20260723}"

if command -v python3 >/dev/null 2>&1; then
    PYTHON=python3
elif command -v python >/dev/null 2>&1; then
    PYTHON=python
else
    echo "ERROR: Python 3 was not found." >&2
    exit 1
fi

for path in \
    "$APPROVED_V1/manifest.csv" \
    "$APPROVED_V1/merge_report.json" \
    "$APPROVED_V1/label_contract.yaml" \
    "$MANUAL_V2/metadata.csv" \
    "$MANUAL_V2/validation_report.json" \
    "$MANUAL_V2/label_contract.yaml"; do
    if [[ ! -f "$path" ]]; then
        echo "ERROR: Required approved input is missing: $path" >&2
        exit 1
    fi
done
if [[ -e "$OUTPUT_DIR" ]]; then
    echo "ERROR: Output path already exists; it will not be overwritten: $OUTPUT_DIR" >&2
    exit 1
fi
case "$(realpath -m "$OUTPUT_DIR")" in
    "$ROOT_DIR"/*)
        echo "ERROR: Generated v2 dataset must remain outside Git: $OUTPUT_DIR" >&2
        exit 1
        ;;
esac

tree_fingerprint() {
    "$PYTHON" - "$1" <<'PY'
import hashlib
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
digest = hashlib.sha256()
for path in sorted(item for item in root.rglob("*") if item.is_file()):
    stat = path.stat()
    digest.update(f"{path.relative_to(root).as_posix()}|{stat.st_size}|{stat.st_mtime_ns}|".encode())
    digest.update(hashlib.sha256(path.read_bytes()).digest())
print(digest.hexdigest())
PY
}

cd "$ROOT_DIR"
before_git_status="$(git status --porcelain)"
before_v1="$(tree_fingerprint "$APPROVED_V1")"
before_manual="$(tree_fingerprint "$MANUAL_V2")"

echo "[1/4] Running focused v2 dataset and loader tests"
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" -m pytest -p no:cacheprovider -q \
    tests/test_traversability_dataset_v2.py \
    tests/test_traversability_segmentation.py \
    tests/test_import_manual_traversability_v2.py

echo "[2/4] Building immutable approved_153_v2"
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" training/build_traversability_dataset_v2.py \
    --approved-v1 "$APPROVED_V1" \
    --manual-v2 "$MANUAL_V2" \
    --output-dir "$OUTPUT_DIR" \
    --expected-v1-count 120 \
    --expected-new-count 33 \
    --new-holdout-ratio 0.20 \
    --seed "$SEED"

echo "[3/4] Verifying fixed v1 splits and new group isolation"
"$PYTHON" - "$APPROVED_V1" "$OUTPUT_DIR" <<'PY'
import csv
import hashlib
import json
import sys
from pathlib import Path

v1 = Path(sys.argv[1]).resolve()
v2 = Path(sys.argv[2]).resolve()
v1_rows = list(csv.DictReader((v1 / "manifest.csv").open(newline="", encoding="utf-8")))
v2_rows = list(csv.DictReader((v2 / "manifest.csv").open(newline="", encoding="utf-8")))
report = json.loads((v2 / "merge_report.json").read_text(encoding="utf-8"))
split = json.loads((v2 / "split_report.json").read_text(encoding="utf-8"))
assert report["valid"] is True
assert len(v1_rows) == 120
assert len(v2_rows) == 153
assert len({row["sample_id"] for row in v2_rows}) == 153
v1_assignments = {row["sample_id"]: row["split"] for row in v1_rows}
v2_assignments = {row["sample_id"]: row["split"] for row in v2_rows}
assert all(v2_assignments[sample_id] == split_name for sample_id, split_name in v1_assignments.items())
new_rows = [row for row in v2_rows if row["source_bundle"] == "manual_v2_33"]
assert len(new_rows) == 33
assert {row["split"] for row in new_rows} == {"train", "new_holdout"}
groups = {}
for row in new_rows:
    key = row["ride_id"]
    groups.setdefault(key, set()).add(row["split"])
assert all(len(values) == 1 for values in groups.values())
assert report["label_contract_exact_match"] is True
assert report["exact_image_duplicate_count"] == 0
assert report["model_training_performed"] is False
assert split["existing_v1_split_preserved"] is True
assert split["new_group_leakage"] == []
for name in ("validation", "test"):
    original = (v1 / "splits" / f"{name}.csv").read_bytes()
    frozen = (v2 / "fixed_v1_splits" / f"{name}.csv").read_bytes()
    assert hashlib.sha256(original).digest() == hashlib.sha256(frozen).digest()
print(f"Samples: v1={len(v1_rows)} manual=33 total={len(v2_rows)}")
print(f"Split statistics: {split['statistics']}")
print(f"New groups: {split['new_groups']}")
print("Fixed-v1 split and new-group leakage gate: PASS")
PY

echo "[4/4] Verifying source immutability and Git exclusion"
after_v1="$(tree_fingerprint "$APPROVED_V1")"
after_manual="$(tree_fingerprint "$MANUAL_V2")"
after_git_status="$(git status --porcelain)"
if [[ "$before_v1" != "$after_v1" || "$before_manual" != "$after_manual" ]]; then
    echo "ERROR: An approved source dataset changed." >&2
    exit 1
fi
if [[ "$before_git_status" != "$after_git_status" ]]; then
    echo "ERROR: Git worktree changed during dataset build." >&2
    git status --short >&2
    exit 1
fi
if [[ -n "$(git ls-files -- ':(glob)**/approved_153_v2/**')" ]]; then
    echo "ERROR: Generated v2 dataset appears in git ls-files." >&2
    exit 1
fi
du -sh "$OUTPUT_DIR"
echo "Approved v1 and manual-v2 inputs unchanged: PASS"
echo "Git exclusion: PASS"
echo "Training/live rover operations: NOT PERFORMED"
echo "Traversability dataset v2: $OUTPUT_DIR"
