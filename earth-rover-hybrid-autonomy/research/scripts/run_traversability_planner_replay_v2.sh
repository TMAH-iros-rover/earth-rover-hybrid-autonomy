#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINT="${CHECKPOINT:-$HOME/datasets/experiments/traversability_segformer_b0_v2/full_training/segformer_b0_best.pt}"
TRAINING_CONFIG="${TRAINING_CONFIG:-$ROOT_DIR/configs/traversability_segformer_b0_v2.yaml}"
AUTONOMY_CONFIG="${AUTONOMY_CONFIG:-$ROOT_DIR/configs/urban_replay_v2.yaml}"
LATENCY_PROFILE="${LATENCY_PROFILE:-$ROOT_DIR/configs/urban_latency_2s.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$HOME/datasets/review_bundles/traversability_planner_replay_v2}"
DATASET_ROOT_0="${DATASET_ROOT_0:-$HOME/datasets/output_rides_0}"
DATASET_ROOT_1="${DATASET_ROOT_1:-$HOME/datasets/output_rides_1}"
DATASET_ROOT_2="${DATASET_ROOT_2:-$HOME/datasets/output_rides_2}"
DATASETS="${DATASETS:-0}"
RIDE_ID="${RIDE_ID:-}"
RIDE_COUNT="${RIDE_COUNT:-1}"
DURATION_SECONDS="${DURATION_SECONDS:-30}"
LATENCY_SEC="${LATENCY_SEC:-0}"
GOAL_HEADING_ERROR_DEG="${GOAL_HEADING_ERROR_DEG:-0}"
LOW_CONFIDENCE_THRESHOLD="${LOW_CONFIDENCE_THRESHOLD:-}"
OVERWRITE="${OVERWRITE:-false}"
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/latency_${LATENCY_SEC}s}"

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
if [[ "$(ffmpeg -hide_banner -encoders 2>/dev/null)" != *libx264* ]]; then
    echo "ERROR: ffmpeg does not provide the libx264 encoder." >&2
    exit 1
fi

cd "$ROOT_DIR"
git_state_before="$(git status --porcelain --untracked-files=all)"
echo "[1/3] Running focused offline planner replay tests"
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" -m pytest -p no:cacheprovider -q \
    tests/test_traversability_adapter.py \
    tests/test_goal_aware_local_planner.py \
    tests/test_traversability_replay.py \
    tests/test_traversability_planner_replay_v2.py \
    tests/test_command_filter.py \
    tests/test_hybrid_controller.py

echo "[2/3] Running log-only planner replay (latency=${LATENCY_SEC}s, datasets=${DATASETS})"
arguments=(
    --checkpoint "$CHECKPOINT"
    --training-config "$TRAINING_CONFIG"
    --autonomy-config "$AUTONOMY_CONFIG"
    --latency-profile "$LATENCY_PROFILE"
    --output-dir "$OUTPUT_DIR"
    --datasets $DATASETS
    --dataset-root-0 "$DATASET_ROOT_0"
    --dataset-root-1 "$DATASET_ROOT_1"
    --dataset-root-2 "$DATASET_ROOT_2"
    --ride-count "$RIDE_COUNT"
    --duration-seconds "$DURATION_SECONDS"
    --latency-sec "$LATENCY_SEC"
    --goal-heading-error-deg "$GOAL_HEADING_ERROR_DEG"
    --require-cuda
)
if [[ -n "$RIDE_ID" ]]; then
    arguments+=(--ride-id "$RIDE_ID")
fi
if [[ -n "$LOW_CONFIDENCE_THRESHOLD" ]]; then
    arguments+=(--low-confidence-threshold "$LOW_CONFIDENCE_THRESHOLD")
fi
if [[ "$OVERWRITE" == "true" ]]; then
    arguments+=(--overwrite)
fi
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" \
    training/run_traversability_planner_replay_v2.py "${arguments[@]}"

echo "[3/3] Verifying log-only, H.264, and Git invariants"
"$PYTHON" - "$OUTPUT_DIR" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
report = json.loads((root / "review_manifest.json").read_text(encoding="utf-8"))
assert report["success"]
assert report["command_transmitted"] is False
assert report["sdk_or_live_rover_used"] is False
for dataset in report["datasets"].values():
    rows = list(csv.DictReader(Path(dataset["csv_log_path"]).open(encoding="utf-8")))
    assert rows and all(row["command_transmitted"] == "False" for row in rows)
    assert dataset["video"]["codec"] == "h264"
    assert dataset["video"]["pixel_format"] == "yuv420p"
    print(
        f"{dataset['dataset_name']}: processed={dataset['processed_frame_count']} "
        f"failed={dataset['failed_frame_count']} fps={dataset['effective_fps']:.2f}"
    )
print(f"Review manifest: {root / 'review_manifest.json'}")
PY
git_state_after="$(git status --porcelain --untracked-files=all)"
if [[ "$git_state_before" != "$git_state_after" ]]; then
    echo "ERROR: Repository state changed while replay artifacts were generated." >&2
    git status --short
    exit 1
fi
echo "Traversability planner replay v2: PASS (log-only; no SDK commands)"
