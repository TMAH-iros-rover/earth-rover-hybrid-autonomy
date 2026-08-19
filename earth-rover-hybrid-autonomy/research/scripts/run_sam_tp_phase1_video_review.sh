#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RESEARCH_ROOT="$PROJECT_ROOT/research"
WORKSPACE_ROOT="$(cd "$PROJECT_ROOT/.." && pwd)"
UPSTREAM_ROOT="${UPSTREAM_ROOT:-$WORKSPACE_ROOT/external/GENIE-SAMTP}"
VENV_PATH="${VENV_PATH:-$WORKSPACE_ROOT/external/venvs/sam_tp_repro}"
PYTHON="${PYTHON:-$VENV_PATH/bin/python}"
CONFIG="${CONFIG:-$RESEARCH_ROOT/configs/sam_tp_reproduction.yaml}"
CHECKPOINT="${CHECKPOINT:-$UPSTREAM_ROOT/sam2_logs/configs/sam2.1_training_tiny/sam2_training_custom2_freezeNoneNone_f57.yaml/checkpoints/checkpoint_2.pt}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/datasets/review_bundles/sam_tp_phase1/$RUN_ID}"
REPORT_DIR="${REPORT_DIR:-$HOME/datasets/experiments/sam_tp_phase1/$RUN_ID}"
DATASETS="${DATASETS:-0}"
RIDES_PER_DATASET="${RIDES_PER_DATASET:-1}"
SECONDS_PER_RIDE="${SECONDS_PER_RIDE:-20}"
MINIMUM_PATH_SCORE="${MINIMUM_PATH_SCORE:-0.55}"
PATH_CORRIDOR_HALF_WIDTH_RATIO="${PATH_CORRIDOR_HALF_WIDTH_RATIO:-0.018}"

for path in "$PYTHON" "$UPSTREAM_ROOT" "$CONFIG" "$CHECKPOINT"; do
  if [[ ! -e "$path" ]]; then
    echo "ERROR: Required input is missing: $path" >&2
    exit 2
  fi
done
if [[ -e "$OUTPUT_DIR" || -e "$REPORT_DIR" ]]; then
  echo "ERROR: Output already exists; choose a new RUN_ID." >&2
  exit 2
fi
for command_name in ffmpeg ffprobe; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "ERROR: $command_name is required." >&2
    exit 2
  fi
done
read -r -a dataset_arguments <<< "$DATASETS"

cd "$PROJECT_ROOT"
mkdir -p "$REPORT_DIR"

echo "[1/3] Running Phase 1 and SDK-frame focused tests"
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" -m pytest -p no:cacheprovider -q \
  tests/test_sam_tp_adapter.py \
  tests/test_trajectory_sampler.py \
  tests/test_sam_tp_phase1_review.py \
  tests/test_sam_tp_sdk_shadow.py

echo "[2/3] Strictly validating the official checkpoint"
"$PYTHON" research/training/inspect_sam_tp_compatibility.py \
  --reproduction-config "$CONFIG" \
  --upstream-root "$UPSTREAM_ROOT" \
  --checkpoint "$CHECKPOINT" \
  --output "$REPORT_DIR/compatibility_report.json"

echo "[3/3] Creating the deterministic QuickTime-compatible Phase 1 video"
"$PYTHON" research/training/run_sam_tp_video_review.py \
  --reproduction-config "$CONFIG" \
  --upstream-root "$UPSTREAM_ROOT" \
  --checkpoint "$CHECKPOINT" \
  --compatibility-report "$REPORT_DIR/compatibility_report.json" \
  --output-dir "$OUTPUT_DIR" \
  --datasets "${dataset_arguments[@]}" \
  --rides-per-dataset "$RIDES_PER_DATASET" \
  --seconds-per-ride "$SECONDS_PER_RIDE" \
  --phase1-trajectories \
  --minimum-path-score "$MINIMUM_PATH_SCORE" \
  --path-corridor-half-width-ratio "$PATH_CORRIDOR_HALF_WIDTH_RATIO"

echo "Phase 1 review: $OUTPUT_DIR"
echo "The image-space path is visualization-only; metric camera projection was not performed."
echo "No SDK command or live rover motion was performed."
