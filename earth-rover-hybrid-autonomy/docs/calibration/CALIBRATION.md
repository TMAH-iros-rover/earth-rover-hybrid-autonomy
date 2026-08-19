# Camera calibration procedure

This is the only supported path to producing a calibration file that
`configs/mission1_live.yaml` can point at. Nothing in this repository
fabricates, estimates, or hard-codes calibration values -- every number in
the final file must trace back to a real measurement or a passing offline
calibration run.

All tools here are read-only over local files. None of them import
`earth_rover.sdk_client` or call any network/SDK endpoint (`/control`,
`/start-mission`, `/checkpoint-reached`, or any other mutation endpoint).

## 1. Capture images

Using the live rover's front camera at its normal operating resolution
(matching what `training/run_sam_tp_sdk_shadow.py` actually receives),
save:

- 15-25 images of a checkerboard or ChArUco board, filling different parts
  of the frame and at varied distances/angles, for intrinsic calibration.
- One additional frame of the ground plane ahead of the rover, with at
  least 6-8 physical target points visible, for extrinsic calibration.

## 2. Intrinsic calibration

```bash
.venv/bin/python scripts/calibration/calibrate_intrinsics.py \
  --images-dir /path/to/checkerboard_images \
  --board chessboard \
  --columns <inner corners per row> \
  --rows <inner corners per column> \
  --square-size-m <measured square size> \
  --min-accepted-images 12 \
  --max-reprojection-rmse-px 0.75 \
  --output /path/to/intrinsics_fragment.yaml \
  --report /path/to/intrinsics_report.json
```

Use `--board charuco --marker-size-m <m>` for a ChArUco board instead (more
robust to partial occlusion/blur than a plain checkerboard).

The script prints a JSON report: `images_considered`, `accepted_count`,
`rejected_count`, per-image `reprojection_error_px`/`reason` (e.g.
`CHESSBOARD_NOT_FOUND`, `IMAGE_SIZE_MISMATCH`), and the overall
`reprojection_rmse_px`. **The intrinsics fragment at `--output` is only
written if `accepted_count >= --min-accepted-images` and
`reprojection_rmse_px <= --max-reprojection-rmse-px`.** A failing run exits
non-zero and writes nothing -- rerun with more/better images rather than
lowering the thresholds to force output.

## 3. Extrinsic calibration

With the rover stationary, measure the rover-frame 3D coordinates (`+x`
forward, `+y` left, `+z` up from the rover's reference origin; ground-plane
targets have `z = 0`) of at least 6 physical points visible in one saved
camera frame, and record the corresponding pixel coordinates by inspecting
that frame. Put them in a JSON/YAML list:

```json
[
  {"rover_xyz_m": [1.0, 0.3, 0.0], "pixel_uv": [412.5, 301.0]},
  {"rover_xyz_m": [1.0, -0.3, 0.0], "pixel_uv": [612.0, 298.5]},
  ...
]
```

Then run:

```bash
.venv/bin/python scripts/calibration/calibrate_extrinsics.py \
  --correspondences /path/to/correspondences.json \
  --intrinsics /path/to/intrinsics_fragment.yaml \
  --min-points 6 \
  --max-reprojection-rmse-px 1.5 \
  --output /path/to/extrinsics_fragment.yaml \
  --report /path/to/extrinsics_report.json
```

Same gating rule: the `camera_from_rover_transform` fragment is only
written if point count and RMSE both pass. No camera height or pitch is
ever guessed -- every entry in the transform comes from `cv2.solvePnP` over
the measured correspondences.

## 4. Assemble the final calibration file

Combine the two fragments plus the remaining required fields into one file
shaped like `configs/calibration/mission1_camera.EXAMPLE_TEMPLATE.yaml`
(read that file's comments for the full field list and frame-convention
documentation):

```yaml
schema_version: 1
calibration_id: "<descriptive, unique id, e.g. mission1_camera_2026_08_10>"
placeholder: false          # false is required for live use
image_width: <from intrinsics fragment>
image_height: <from intrinsics fragment>
camera_matrix: <from intrinsics fragment>
distortion_model: "opencv_pinhole"
distortion_coefficients: <from intrinsics fragment>
camera_from_rover_transform: <from extrinsics fragment>
provenance:
  capture_date: "<YYYY-MM-DD>"
  capture_source: "<how/where captured>"
  reprojection_error_px: <either fragment's error, or a combined figure>
```

## 5. Validate before pointing anything at it

```bash
.venv/bin/python -c "
from earth_rover.perception.camera_calibration import load_calibration, validate_for_live_use
cal = load_calibration('/path/to/mission1_camera.yaml')
validate_for_live_use(cal, (cal.image_height, cal.image_width))
print('OK', cal.calibration_id, cal.content_sha256[:16])
"
```

`load_calibration` runs full structural validation (finite values, matrix
shapes, a nonsingular intrinsic matrix, a proper rotation matrix, the
homogeneous transform's last row, positive focal lengths). If it raises
`CalibrationError`, fix the file and re-validate -- do not hand-edit around
a validation failure.

## 6. Point the live profile at it

Set `camera_calibration.path` in `configs/mission1_live.yaml` to the final
file's path, then run the SAM-TP shadow process with
`MISSION_CONFIG=configs/mission1_live.yaml` (see the README's "Metric
camera projection" section) and confirm the dashboard reports
`camera_projection_applied=true` and `image_path_metric_calibrated=true`
before ever running `run_mission1_autonomy.sh --enable-live-control`.
