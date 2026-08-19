#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ "${1:-}" != "--enable-live-control" ]]; then
  echo "Starting Mission1 autonomy in DRY-RUN mode (no rover commands)"
  echo "For attended rover driving, rerun: $0 --enable-live-control"
  exec python3 scripts/run_mission1_autonomy.py "$@"
fi

shift
echo "LIVE CONTROL ARMED: motion begins only after Start Mission succeeds in the dashboard"
echo "Keep the dashboard visible and use End Mission or Ctrl+C to stop"
exec python3 scripts/run_mission1_autonomy.py --enable-live-control "$@"
