#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINT="${CHECKPOINT:-$HOME/datasets/experiments/traversability_segformer_b0_v2/full_training/segformer_b0_best.pt}"
CONFIG="${CONFIG:-$ROOT_DIR/configs/traversability_segformer_b0_v2.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/datasets/review_bundles/traversability_video_review_v2}"
DATASETS="${DATASETS:-all}"
LOW_CONFIDENCE_THRESHOLD="${LOW_CONFIDENCE_THRESHOLD:-}"
OVERWRITE="${OVERWRITE:-false}"

if command -v python3 >/dev/null 2>&1; then
    PYTHON=python3
elif command -v python >/dev/null 2>&1; then
    PYTHON=python
else
    echo "ERROR: Python 3 was not found." >&2
    exit 1
fi
for command_name in ffmpeg ffprobe; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "ERROR: $command_name is required for QuickTime-compatible H.264 output." >&2
        exit 1
    fi
done
ffmpeg_encoders="$(ffmpeg -hide_banner -encoders 2>/dev/null)"
if [[ "$ffmpeg_encoders" != *libx264* ]]; then
    echo "ERROR: ffmpeg does not provide the libx264 encoder." >&2
    exit 1
fi

cd "$ROOT_DIR"
echo "[1/3] Running focused synthetic video-review tests"
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" -m pytest -p no:cacheprovider -q \
    tests/test_traversability_video_review_v2.py \
    tests/test_traversability_checkpoint_schema.py

echo "[2/3] Running offline v2 inference for dataset selection: $DATASETS"
arguments=(
    --checkpoint "$CHECKPOINT"
    --config "$CONFIG"
    --output-dir "$OUTPUT_DIR"
    --datasets $DATASETS
    --require-cuda
)
if [[ -n "$LOW_CONFIDENCE_THRESHOLD" ]]; then
    arguments+=(--low-confidence-threshold "$LOW_CONFIDENCE_THRESHOLD")
fi
if [[ "$OVERWRITE" == "true" ]]; then
    arguments+=(--overwrite)
fi
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" training/run_traversability_video_review_v2.py "${arguments[@]}"

echo "[3/3] Verifying H.264 artifacts and Git exclusion"
"$PYTHON" - "$OUTPUT_DIR" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
report = json.loads((root / "review_manifest.json").read_text(encoding="utf-8"))
assert report["success"]
for dataset in report["datasets"].values():
    assert dataset["processed_frame_count"] > 0
    assert dataset["output_video"]["codec"] == "h264"
    assert dataset["output_video"]["pixel_format"] == "yuv420p"
    assert dataset["temporal_smoothing_applied"] is False
    assert dataset["sdk_or_live_rover_commands_sent"] is False
    print(
        f"{dataset['dataset_name']}: rides={dataset['selected_ride_count']} "
        f"processed={dataset['processed_frame_count']} skipped={dataset['skipped_frame_count']} "
        f"fps={dataset['effective_fps']:.2f}"
    )
print(f"Review manifest: {root / 'review_manifest.json'}")
PY
if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
    echo "ERROR: Repository changed while generating review artifacts." >&2
    git status --short
    exit 1
fi
echo "Traversability v2 video review: PASS"
