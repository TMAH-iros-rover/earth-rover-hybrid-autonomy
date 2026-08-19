#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="$(cd "$PROJECT_ROOT/.." && pwd)"
UPSTREAM_ROOT="${UPSTREAM_ROOT:-$WORKSPACE_ROOT/external/GENIE-SAMTP}"
ENV_NAME="${ENV_NAME:-sam_tp_repro}"
ENV_BACKEND="${ENV_BACKEND:-auto}"
VENV_PATH="${VENV_PATH:-$WORKSPACE_ROOT/external/venvs/$ENV_NAME}"
CONFIG="${CONFIG:-$PROJECT_ROOT/configs/default.yaml}"
# The SDK shadow feeds Mission1 live control, so its default must match the
# controller's live profile. Set MISSION_CONFIG='' explicitly only for an
# offline/legacy image-heuristic experiment.
MISSION_CONFIG="${MISSION_CONFIG-$PROJECT_ROOT/configs/mission1_live.yaml}"
# Model selection defaults live in CONFIG under sam_tp. These optional
# overrides keep one-command rollback and experiment launches available.
MODEL_CONFIG="${MODEL_CONFIG:-}"
CHECKPOINT="${CHECKPOINT:-}"
EXPECTED_CHECKPOINT_SHA256="${EXPECTED_CHECKPOINT_SHA256:-}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/datasets/review_bundles/sam_tp_sdk_shadow/$RUN_ID}"
TARGET_FPS="${TARGET_FPS:-4}"
TELEMETRY_HZ="${TELEMETRY_HZ:-2}"
MAXIMUM_FRAME_AGE_SEC="${MAXIMUM_FRAME_AGE_SEC:-2.0}"
MAXIMUM_TELEMETRY_AGE_SEC="${MAXIMUM_TELEMETRY_AGE_SEC:-3.0}"
REQUEST_TIMEOUT_SEC="${REQUEST_TIMEOUT_SEC:-3.0}"
MAX_FRAMES="${MAX_FRAMES:-}"
MAXIMUM_CONSECUTIVE_FAILURES="${MAXIMUM_CONSECUTIVE_FAILURES:-60}"
SHOW_WINDOW="${SHOW_WINDOW:-false}"
DASHBOARD_HOST="${DASHBOARD_HOST:-127.0.0.1}"
DASHBOARD_PORT="${DASHBOARD_PORT:-8001}"
MISSION_ROUTE_LATEST_OVERRIDE="${MISSION_ROUTE_LATEST_OVERRIDE:-}"
PLANNER_MODE="${PLANNER_MODE:-}"
EVENT_CAPTURE="${EVENT_CAPTURE:-true}"
CAPTURE_PRE_EVENT_FRAMES="${CAPTURE_PRE_EVENT_FRAMES:-8}"
CAPTURE_POST_EVENT_FRAMES="${CAPTURE_POST_EVENT_FRAMES:-8}"
CAPTURE_BASELINE_INTERVAL_FRAMES="${CAPTURE_BASELINE_INTERVAL_FRAMES:-40}"
CAPTURE_QUEUE_SIZE="${CAPTURE_QUEUE_SIZE:-32}"

if [[ "$ENV_BACKEND" == "auto" ]]; then
  if command -v conda >/dev/null 2>&1 \
    && conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
    ENV_BACKEND="conda"
  elif [[ -x "$VENV_PATH/bin/python" ]]; then
    ENV_BACKEND="venv"
  else
    echo "ERROR: No SAM-TP environment found; run setup_sam_tp_reproduction.sh first." >&2
    exit 2
  fi
fi
if [[ "$ENV_BACKEND" == "conda" ]]; then
  env_python() {
    conda run --no-capture-output -n "$ENV_NAME" python "$@"
  }
elif [[ "$ENV_BACKEND" == "venv" ]]; then
  env_python() {
    "$VENV_PATH/bin/python" "$@"
  }
else
  echo "ERROR: ENV_BACKEND must be auto, conda, or venv." >&2
  exit 2
fi

arguments=(
  --config "$CONFIG"
  --upstream-root "$UPSTREAM_ROOT"
  --output-dir "$OUTPUT_DIR"
  --target-fps "$TARGET_FPS"
  --telemetry-hz "$TELEMETRY_HZ"
  --maximum-frame-age-sec "$MAXIMUM_FRAME_AGE_SEC"
  --maximum-telemetry-age-sec "$MAXIMUM_TELEMETRY_AGE_SEC"
  --request-timeout-sec "$REQUEST_TIMEOUT_SEC"
  --maximum-consecutive-failures "$MAXIMUM_CONSECUTIVE_FAILURES"
  --dashboard-host "$DASHBOARD_HOST"
  --dashboard-port "$DASHBOARD_PORT"
  --capture-pre-event-frames "$CAPTURE_PRE_EVENT_FRAMES"
  --capture-post-event-frames "$CAPTURE_POST_EVENT_FRAMES"
  --capture-baseline-interval-frames "$CAPTURE_BASELINE_INTERVAL_FRAMES"
  --capture-queue-size "$CAPTURE_QUEUE_SIZE"
)
if [[ "$EVENT_CAPTURE" != "true" ]]; then
  arguments+=(--no-event-capture)
fi
if [[ -n "$MISSION_CONFIG" ]]; then
  arguments+=(--mission-config "$MISSION_CONFIG")
fi
if [[ -n "$MODEL_CONFIG" ]]; then
  arguments+=(--model-config "$MODEL_CONFIG")
fi
if [[ -n "$CHECKPOINT" ]]; then
  arguments+=(--checkpoint "$CHECKPOINT")
fi
if [[ -n "$EXPECTED_CHECKPOINT_SHA256" ]]; then
  arguments+=(--expected-checkpoint-sha256 "$EXPECTED_CHECKPOINT_SHA256")
fi
if [[ -n "$MAX_FRAMES" ]]; then
  arguments+=(--max-frames "$MAX_FRAMES")
fi
if [[ -n "$MISSION_ROUTE_LATEST_OVERRIDE" ]]; then
  arguments+=(--mission-route-latest-override "$MISSION_ROUTE_LATEST_OVERRIDE")
fi
if [[ -n "$PLANNER_MODE" ]]; then
  arguments+=(--planner-mode "$PLANNER_MODE")
fi
if [[ "$SHOW_WINDOW" == "true" ]]; then
  arguments+=(--show-window)
fi

cd "$PROJECT_ROOT"
echo "Starting GET-only SAM-TP SDK shadow dashboard"
echo "Browser-only mode enabled; set SHOW_WINDOW=true for the legacy OpenCV window"
if [[ -n "$PLANNER_MODE" ]]; then
  echo "Planner mode override: $PLANNER_MODE"
fi
if [[ -n "$MISSION_CONFIG" ]]; then
  echo "Mission config overlay: $MISSION_CONFIG"
fi
if [[ -n "$MISSION_ROUTE_LATEST_OVERRIDE" ]]; then
  echo "Read-only route preview override: latest_scanned_checkpoint=$MISSION_ROUTE_LATEST_OVERRIDE"
fi
echo "No /control or mission endpoint will be called"
env_python training/run_sam_tp_sdk_shadow.py "${arguments[@]}" "$@"
