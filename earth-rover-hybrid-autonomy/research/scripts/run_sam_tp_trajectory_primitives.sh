#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROJECT_PYTHON="${PROJECT_PYTHON:-python3}"

cd "$PROJECT_ROOT"

if ! "$PROJECT_PYTHON" -c "import numpy, pytest" >/dev/null 2>&1; then
  echo "ERROR: PROJECT_PYTHON must provide numpy and pytest: $PROJECT_PYTHON" >&2
  exit 2
fi

echo "[1/3] Running SAM-TP adapter and trajectory sampler tests"
PYTHONDONTWRITEBYTECODE=1 "$PROJECT_PYTHON" -m pytest -p no:cacheprovider -q \
  tests/test_sam_tp_adapter.py \
  tests/test_trajectory_sampler.py

echo "[2/3] Running existing checkpoint-free SAM-TP regression tests"
PYTHONDONTWRITEBYTECODE=1 "$PROJECT_PYTHON" -m pytest -p no:cacheprovider -q \
  tests/test_sam_tp_reproduction.py \
  tests/test_sam_tp_sdk_shadow.py

echo "[3/3] Generating a deterministic seven-trajectory geometry dry run"
PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
  "$PROJECT_PYTHON" - <<'PY'
from earth_rover.planning.trajectory_sampler import (
    DEFAULT_CURVATURES,
    ConstantCurvatureTrajectorySampler,
)

# Geometry-only provisional values. They are not approved live-rover calibration.
sampler = ConstantCurvatureTrajectorySampler(
    curvatures=DEFAULT_CURVATURES,
    horizon_m=2.0,
    sample_interval_m=0.1,
    rover_width_m=0.4,
    safety_margin_m=0.1,
)
trajectories = sampler.sample()
assert len(trajectories) == 7
for trajectory in trajectories:
    end_x, end_y = trajectory.points_xy[-1]
    print(
        f"curvature={trajectory.curvature:+.2f} "
        f"points={len(trajectory.points_xy)} "
        f"end_xy=({end_x:+.3f},{end_y:+.3f}) "
        f"heading_rad={trajectory.headings_rad[-1]:+.3f}"
    )
print("SAM-TP trajectory primitives: PASS (geometry only; no CUDA or rover command)")
PY
