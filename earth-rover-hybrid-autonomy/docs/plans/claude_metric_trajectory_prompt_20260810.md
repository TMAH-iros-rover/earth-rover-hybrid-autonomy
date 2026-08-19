# Claude Prompt: Metric Camera Projection and Footprint Planner

Work in `/home/asl/IROS2026/earth-rover-hybrid-autonomy`.

## Context

The working tree is intentionally dirty. Do not revert, overwrite, or reformat
unrelated changes. Codex has already implemented and tested:

- asynchronous event/replay capture with source PNG, raw logits, score map,
  telemetry/navigation/planner JSON;
- localization fail-closed heading rejection and stable recovery;
- checkpoint reached latch and three-frame target-sequence confirmation in the
  live profile;
- planner stop-before-switch, three-frame unsafe switch confirmation, maximum
  10-degree adjacent switch, and disabled live `SEARCH_ROTATE`;
- same-frame baseline `checkpoint_2.pt` versus candidate `best_sam_tp.pt` replay
  A/B tooling.

Current tests pass: `241 passed, 3 skipped` under `tests`, and `171 passed, 1
skipped` under `research/tests`.

The unresolved safety defect is structural. The live planner currently draws
heuristic image-space curves and explicitly reports:

- `trajectory_geometry_only=true`
- `camera_projection_applied=false`
- `image_path_metric_calibrated=false`

There is no validated camera intrinsic/extrinsic calibration in this repo. The
existing `ConstantCurvatureTrajectorySampler` produces valid rover-frame metric
centerlines and footprint boundaries, but those are not projected into the
camera and are not used for metric collision scoring. The current image-space
planner therefore cannot establish that a candidate corridor corresponds to
the rover footprint, which is a plausible cause of wall-side path selection.

## Task

Implement a production-quality, fail-closed metric camera projection and
footprint-aware local planner path. Do not invent, estimate, or hard-code any
physical calibration value. No SDK write endpoint or rover motion is allowed.

### 1. Calibration contract

Add a typed calibration model loaded from YAML/JSON that includes at minimum:

- calibration schema version and stable calibration ID;
- calibrated image width and height;
- camera intrinsic matrix and distortion coefficients/model;
- a clearly named and documented rigid transform between rover frame
  (`+x` forward, `+y` left, `+z` up) and OpenCV camera frame;
- provenance fields (capture date/source and optional reprojection error);
- a deterministic content SHA-256 surfaced in status/output.

Validate finite values, matrix shapes, a nonsingular intrinsic matrix, a proper
rotation matrix, homogeneous transform last row, positive focal lengths, and
exact input image-size compatibility. Reject missing, malformed, placeholder,
or image-size-mismatched calibration. Do not silently rescale intrinsics.

Add an example/template clearly marked invalid/not-live-ready, but do not add a
fake live calibration.

### 2. Calibration acquisition tools

Add read-only offline tools and documentation for obtaining the required real
values from saved rover frames:

- intrinsic calibration from a configurable checkerboard or ChArUco board;
- ground-plane camera extrinsics from explicit measured rover-frame target
  coordinates matched to image pixels (or an equally rigorous target-based
  method);
- numerical reports for number of accepted images/points, reprojection RMSE,
  per-frame error, and rejected observations;
- output only after configurable quality thresholds pass.

The tool must never call `/control`, `/start-mission`, `/checkpoint-reached`, or
any SDK mutation endpoint. It must not claim calibration success from a single
unvalidated image or guessed camera height/pitch.

### 3. Projection

Project `CandidateTrajectory` centerlines and both footprint boundaries from
rover coordinates into undistorted image coordinates using the validated
calibration. Correctly reject points behind the camera, above/outside the
visible ground region, nonfinite projections, and candidates with insufficient
visible horizon/coverage. Keep frame conventions explicit and unit tested.

Rasterize the projected left/right footprint boundaries into a corridor mask.
Score SAM-TP traversability over the full footprint, with distinct near-field
and full-horizon statistics. A thin centerline score is not sufficient.

### 4. Planner integration

Integrate metric candidates into the existing `MotionPrimitivePlanner` without
removing the recently added temporal safety behavior:

- current candidate loss still requires a stop first;
- unsafe replacement still needs consecutive confirmation;
- a driving switch remains bounded to an adjacent candidate;
- no nonadjacent left/right jump;
- no automatic search rotation in `configs/mission1_live.yaml`;
- localization and target-sequence fail-closed behavior remains unchanged.

Use the existing metric sampler configuration for horizon, interval, rover
width, and safety margin, but require these values to be explicit in the live
profile and validate them. Map curvature to a physically meaningful candidate
terminal heading/yaw, and preserve the project-wide controller convention:
positive local heading means clockwise/right in image/controller status. The
rover-frame sampler currently uses positive curvature for left, so conversion
must be explicit and tested.

Do not retain heuristic `primitive_curve_points` as a silent fallback in live
metric mode. It may remain available only in an explicitly named replay/legacy
mode.

### 5. Fail-closed live gate and observability

The live-control process must refuse to arm or transmit positive linear/angular
motion when metric mode is selected and calibration is absent, invalid,
placeholder, wrong resolution, or projection coverage is insufficient. A zero
stop command is permitted. Make the reason visible in controller status and
logs.

Surface at least:

- `camera_projection_applied`
- `image_path_metric_calibrated`
- calibration ID and SHA prefix
- projected candidate coverage
- projected footprint pixel count
- rejection reason per candidate
- selected curvature, terminal heading, near/full footprint statistics

Remove or update stale unconditional flags only when the runtime has actually
performed validated metric projection. Never report calibrated=true merely
because a config path exists.

### 6. Tests

Add deterministic synthetic tests with known camera matrices/transforms for:

- straight/left/right projection direction and frame conventions;
- behind-camera and out-of-image rejection;
- malformed rotation/intrinsics/transform rejection;
- image resolution mismatch fail-closed behavior;
- footprint width and safety margin changing the rasterized corridor;
- obstacle pixels inside a footprint hard-rejecting that candidate;
- obstacle pixels outside the footprint not rejecting it;
- insufficient projected coverage causing stop;
- no-calibration live controller sends zero and cannot path-hold/search-rotate;
- existing stop-before-switch and maximum-switch-angle behavior remains intact.

Run:

```bash
.venv/bin/python -m pytest tests -q
.venv/bin/python -m pytest research/tests -q
```

Also run any focused tests in the independent SAM-TP environment that do not
require a live rover. Do not start live control.

## Deliverable

Implement the code, tests, config template, and calibration instructions. At
the end, report changed files, exact test results, and remaining blockers.
Explicitly state that live metric driving remains blocked until a real
calibration file is produced and validated. Do not weaken that gate to make a
demo pass.
