#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_BUNDLE="${SOURCE_BUNDLE:-$HOME/datasets/review_bundles/manual_candidates_v2}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/datasets/review_bundles/manual_candidates_v2_selected}"
SELECTION=(1 3 8 9 14 15 18 19 22 27 30 32 34 35 41 46 60 63 68 75 77 88 90 97 100 113 124 154 160 166 169 177 180)

if command -v python3 >/dev/null 2>&1; then
    PYTHON=python3
elif command -v python >/dev/null 2>&1; then
    PYTHON=python
else
    echo "ERROR: Python 3 was not found." >&2
    exit 1
fi

if [[ ! -f "$SOURCE_BUNDLE/candidates.csv" ]]; then
    echo "ERROR: Source candidates.csv is missing: $SOURCE_BUNDLE/candidates.csv" >&2
    exit 1
fi
if [[ -e "$OUTPUT_DIR" ]]; then
    echo "ERROR: Output path already exists; it will not be overwritten: $OUTPUT_DIR" >&2
    exit 1
fi
case "$(realpath -m "$OUTPUT_DIR")" in
    "$ROOT_DIR"/*)
        echo "ERROR: Selected dataset artifacts must remain outside Git: $OUTPUT_DIR" >&2
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
before_source="$(tree_fingerprint "$SOURCE_BUNDLE")"

echo "[1/4] Running focused manual-selection tests"
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" -m pytest -p no:cacheprovider -q \
    tests/test_select_manual_candidates_v2.py \
    tests/test_manual_candidate_sampling.py

echo "[2/4] Selecting and copying the reviewed 33 candidates"
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" training/select_manual_candidates_v2.py \
    --source-bundle "$SOURCE_BUNDLE" \
    --output-dir "$OUTPUT_DIR" \
    --selection "${SELECTION[@]}"

echo "[3/4] Verifying CSV, unchanged JPG bytes, and CVAT ZIP layout"
"$PYTHON" - "$SOURCE_BUNDLE" "$OUTPUT_DIR" <<'PY'
import csv
import hashlib
import sys
import zipfile
from pathlib import Path

source = Path(sys.argv[1]).resolve()
output = Path(sys.argv[2]).resolve()
rows = list(csv.DictReader((output / "selected_candidates.csv").open(newline="", encoding="utf-8")))
expected_ids = [
    f"manual_v2_{number:04d}"
    for number in (1, 3, 8, 9, 14, 15, 18, 19, 22, 27, 30, 32, 34, 35, 41, 46,
                   60, 63, 68, 75, 77, 88, 90, 97, 100, 113, 124, 154, 160, 166, 169, 177, 180)
]
assert len(rows) == 33
assert [row["candidate_id"] for row in rows] == expected_ids
assert (output / "selection.txt").read_text(encoding="utf-8").splitlines() == expected_ids
for row in rows:
    source_image = source / row["image_path"]
    selected_image = output / row["image_path"]
    assert source_image.name == selected_image.name == f"{row['candidate_id']}.jpg"
    assert hashlib.sha256(source_image.read_bytes()).digest() == hashlib.sha256(selected_image.read_bytes()).digest()
archive_path = output / "manual_candidates_v2_selected_33.zip"
with zipfile.ZipFile(archive_path) as archive:
    assert archive.testzip() is None
    assert archive.namelist() == [f"{candidate_id}.jpg" for candidate_id in expected_ids]
    assert all("/" not in name for name in archive.namelist())
    for candidate_id in expected_ids:
        selected = output / "images" / f"{candidate_id}.jpg"
        assert archive.read(selected.name) == selected.read_bytes()
print(f"Selected rows: {len(rows)}")
print(f"ZIP entries: {len(expected_ids)} top-level JPG files")
print("Byte-preservation and archive-layout gate: PASS")
PY

echo "[4/4] Verifying source immutability and Git exclusion"
after_source="$(tree_fingerprint "$SOURCE_BUNDLE")"
after_git_status="$(git status --porcelain)"
if [[ "$before_source" != "$after_source" ]]; then
    echo "ERROR: The source manual-candidate bundle changed." >&2
    exit 1
fi
if [[ "$before_git_status" != "$after_git_status" ]]; then
    echo "ERROR: Git working state changed during Dell execution." >&2
    git status --short >&2
    exit 1
fi
if [[ -n "$(git ls-files -- ':(glob)**/manual_candidates_v2_selected/**')" ]]; then
    echo "ERROR: Generated selected candidates appear in git ls-files." >&2
    exit 1
fi
du -sh "$OUTPUT_DIR"
echo "Source bundle unchanged: PASS"
echo "Git exclusion: PASS"
echo "Selected CVAT bundle: $OUTPUT_DIR/manual_candidates_v2_selected_33.zip"
