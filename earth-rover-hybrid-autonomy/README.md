# Earth Rover Hybrid Autonomy

## Safe keyboard teleoperation

Run the local SDK server first from the sibling `earth-rovers-sdk` directory,
then launch the OpenCV dashboard from this project. The teleop starts disarmed
and does not send control commands until `E` is pressed.

```bash
cd ~/IROS2026/earth-rovers-sdk
hypercorn main:app --bind 127.0.0.1:8000
```

In a second terminal, verify the read-only endpoints before enabling control:

```bash
curl http://127.0.0.1:8000/mission-status
curl --max-time 45 http://127.0.0.1:8000/data
curl http://127.0.0.1:8000/connection-diagnostics
```

In direct-bot mode, leave `MISSION_SLUG` unset and do not call
`/start-mission`. A ready rover reports RTM telemetry, at least one RTC remote
user, and a non-empty front frame in `connection-diagnostics`. If RTM is
`LOGGED_IN`/`JOINED` but `remoteUserCount` is zero and telemetry is absent, the
local SDK is connected but the rover is not publishing to the assigned channel.

Then start the dashboard:

```bash
cd ~/IROS2026/earth-rover-hybrid-autonomy
python3 scripts/teleop_dashboard.py \
  --control-timeout 0.4 \
  --deadman-timeout 0.65
```

This Dell workstation uses the GUI-enabled `opencv-python` package for both
projects. Do not install `opencv-python-headless` into the same user Python
environment because it can replace the GUI build and make `cv2.namedWindow()`
fail.

Controls:

- `E`: arm control; motion remains zero until a direction key is pressed;
- hold `W`/`S`: forward/reverse at the displayed low-speed setting;
- hold `A`/`D`: left/right using the project convention (`+angular` left);
- `Space`: immediately disarm and request three stop commands;
- `L`: toggle the lamp;
- `+`/`-`: adjust speed within the teleop caps (`linear <= 0.25`,
  `angular <= 0.40`);
- `Q` or `Esc`: stop and exit.

Keyboard input is treated as a heartbeat because OpenCV does not expose key-up
events. If repeats stop for the configured dead-man timeout, that axis returns
to zero. The SDK server independently sends a stop if `/control` heartbeats
cease. Confirm the rover's angular sign with its wheels clear or in an open,
controlled area before ground driving. Keep `Space` ready and do not use the
browser examples in `earth-rovers-sdk/examples/web` for live testing.

## Goal
Urban GPS MVP for Earth Rover Challenge using latency-aware hybrid reactive controller.

## Architecture
SDK -> Perception -> Candidate Planner -> Mode FSM -> Controller -> Command Filter -> SDK

## Setup
```bash
pip install -r requirements.txt
```

## Config
`configs/default.yaml`

## Run SDK smoke test
```bash
python scripts/run_sdk_smoke_test.py --config configs/default.yaml --no-motion
```

## Run Urban MVP (archived, see `research/`)
```bash
python research/scripts/run_urban_mvp.py --config configs/default.yaml
```

## Safety
Default motion limits are conservative. The system stops on stale frame, stale data, SDK failure, or emergency condition.

## SegFormer v2 Offline Planner Replay (archived, see `research/`)

`research/scripts/run_traversability_planner_replay_v2.sh` connects the approved SegFormer-B0 v2 checkpoint to the traversability adapter, goal-aware local planner, existing safety monitor, controller, and command filter. It reads recorded front-camera HLS data and writes expected commands to CSV/JSONL plus an H.264 review video.

This is a log-only integration gate. FrodoBots recordings do not provide the mission waypoint used by the Urban MVP, so the replay requires an explicit fixed heading error and records `gps_valid=false`, `goal_input_mode=fixed_heading_error`, and `command_transmitted=false`. It does not call the SDK or validate GPS navigation, recovery, or rover motion.

Run a five-second Dell smoke replay:

```bash
DURATION_SECONDS=5 RIDE_COUNT=1 LATENCY_SEC=0 GOAL_HEADING_ERROR_DEG=0 \
  ./research/scripts/run_traversability_planner_replay_v2.sh
```

Run the two-second delayed profile in a separate output directory:

```bash
LATENCY_SEC=2 GOAL_HEADING_ERROR_DEG=20 \
  ./research/scripts/run_traversability_planner_replay_v2.sh
```

The default output is
`$HOME/datasets/review_bundles/traversability_planner_replay_v2/latency_<N>s/`.
Each dataset directory contains `planner_replay.mp4`,
`logs/replay_steps.csv`, `logs/replay_steps.jsonl`, and
`review_manifest.json`. Generated data, checkpoints, and review videos remain
outside Git.

## Official SAM-TP Reproduction

SegFormer-B0 v2 remains the frozen lightweight semantic baseline. The separate
SAM-TP workflow reproduces only the official GeNIE perception model in an
independent Dell Conda or venv environment, applies a strict config/checkpoint gate, and
produces single-image logits, benchmark data, and deterministic FrodoBots
review videos. It does not train SAM-TP or connect it to the planner, SDK, or
live rover. See `docs/experiments/sam_tp_reproduction.md`.

### Read-only SDK shadow dashboard

The SDK shadow stage fetches the live front frame, telemetry, and the SDK
server's already-loaded checkpoint route. It runs SAM-TP once per frame and
publishes the overlay and navigation metrics to the browser dashboard. It has
no control, mission-start, checkpoint-report, or mission-end call:

```bash
./scripts/run_sam_tp_sdk_shadow.sh
```

The process also serves the latest browser overlay and metrics at
`http://127.0.0.1:8001`. Open the SDK mission dashboard at
`http://127.0.0.1:8000/dashboard`; after the first inference, its `SAM-TP`
camera button becomes available. The overlay uses blue for lower and red for
higher SAM-TP traversability evidence. A cyan line shows the GPS shortest-path
heading to the next checkpoint. By default, the local overlay is now the
selected motion primitive corridor rather than a newly connected pixel path
every frame. The map shows the current-position-to-next-checkpoint GPS segment
separately. Until camera calibration is completed, the local path is an
image-space visualization and does not represent metric obstacle clearance.

Planner mode is selected by `planner.mode` in config or with a launch override:

```bash
./scripts/run_sam_tp_sdk_shadow.sh --planner-mode motion_primitives
./scripts/run_sam_tp_sdk_shadow.sh --planner-mode connected_path
./scripts/run_sam_tp_sdk_shadow.sh --planner-mode gps_only
```

Supported modes:

- `motion_primitives` (default): evaluates fixed image-space candidate
  trajectories `[-45, -30, -15, 0, 15, 30, 45]` degrees. GPS target bearing is
  the primary guide; SAM-TP scores each candidate corridor for local safety and
  cost. Candidate score EMA, minimum commit time, switch confirmation, and
  time-based held plans reduce frame-to-frame left/right switching.
- `connected_path`: keeps the previous connected high-score image-space path
  behavior for rollback and A/B comparison.
- `gps_only`: steers from filtered GPS heading at very low speed and only uses
  SAM-TP near-field score as an emergency safety check. Use this to separate
  GPS/controller/network issues from SAM-TP local planning issues.

The `/status` payload remains backward-compatible:
`state`, `path_valid`, `path_reason`, `path_mean_score`,
`local_path_selected_heading_deg`, and `navigation` are still present. New
debug fields include `planner.mode`, selected candidate index/heading/score,
candidate scores, `near_field_safe`, `near_field_score`, `trajectory_valid`,
`trajectory_quality`, `planner_confidence`, `using_held_plan`, and
`plan_age_sec`.

The main planner parameters live in `configs/default.yaml` and can be
overridden by `configs/mission1_live.yaml`: candidate headings,
traversability weight, GPS heading penalty, continuity penalty, curvature
penalty, near-field risk penalty, `path_score_threshold`,
`near_field_stop_threshold`, candidate score EMA alpha, minimum commit time,
transient invalid grace time, max plan age, switch score margin, and switch
confirmation count.

The separate OpenCV window is disabled by default. Set `SHOW_WINDOW=true` only
when the legacy local window is explicitly needed. Outputs are written under
`$HOME/datasets/review_bundles/sam_tp_sdk_shadow/<RUN_ID>/`. Every record sets
`command_transmitted=false`. This is perception shadow mode, not planner
integration or autonomous driving.

### Mission1 live local-path control

Mission1 control is a separate process from SAM-TP. Start the SDK server and
the SAM-TP shadow process first, then arm the conservative live controller in a
third terminal:

```bash
./scripts/run_sam_tp_sdk_shadow.sh
./scripts/run_mission1_live.sh
```

Both launchers default to the calibrated `configs/mission1_live.yaml`; no
`MISSION_CONFIG` or `--mission-config` argument is required for the current
attended rover test. `run_mission1_live.sh` is the explicit live-control entry
point; `run_mission1_autonomy.sh` without flags remains a no-write dry-run.

It serves controller state at `http://127.0.0.1:8002/status` and waits without
moving until `Start Mission` succeeds in the browser dashboard. During an
active mission it sends a 5 Hz bounded command from the latest accepted local
trajectory, stops before reporting each reached checkpoint, and resets local
planner/controller history after checkpoint transitions. Large GPS heading
errors enter `ROTATING_TO_GOAL` with zero linear speed instead of requiring a
forward camera path while the target lies outside the front view. Transient
local planner uncertainty enters `PATH_HOLD`/cautious hold for a short time if
the near field remains safe; near-field danger, stale SAM-TP, invalid
GPS/heading, control bridge failure, request failures, or process shutdown
still result in a zero command. The SDK also has a 0.75 s command-heartbeat
watchdog.

Use `EMERGENCY STOP` in the dashboard (or press `Space`) for an immediate
latched stop and `Resume Auto` to release it. `End Mission` triggers the same
stop before ending the cloud
mission. Running the launcher without `--enable-live-control` is a no-write
dry-run. Because image heading is not yet camera-calibrated metric curvature,
the live limits in `configs/mission1_live.yaml` are deliberately low
and the first drive must be attended.

Recommended Mission1 live order:

1. `./scripts/run_sam_tp_sdk_shadow.sh`; verify selected candidate,
   confidence, near-field, GPS, and heading fields in the dashboard.
2. `./scripts/run_mission1_live.sh` only after the
   dashboard shows fresh SAM-TP, valid GPS, and stable candidate selection.
3. Compare the same route with `--planner-mode connected_path` only for
   rollback/A-B diagnostics.

### Metric camera projection

`configs/mission1_live.yaml` sets `planner.geometry_mode: metric_projected`
and loads the measured 1024x576 front-camera calibration from
`configs/calibration/mission1_camera.yaml`. The shadow process must report
`camera_projection_applied=true` and `image_path_metric_calibrated=true`
before live control is started. A resolution mismatch or invalid calibration
still hard-rejects every candidate and keeps Mission1 stopped.

For offline planner stability checks:

```bash
PYTHONPATH=src:. scripts/compare_mission1_planners.py
```

The script prints selected candidate switch count, switches per second,
angular sign flip count, valid ratio, held-plan duration, full-stop count,
near-field stop count, and mean planner confidence for synthetic sequences.

## Development order
1. SDK client
2. Logger
3. GPS utils
4. Candidate planner
5. Hybrid controller
6. Command filter
7. Safety/recovery
8. Urban main loop
