from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from earth_rover.core.types import CandidateTrajectory
from earth_rover.perception.camera_calibration import CameraCalibration

# Frame conventions (see also CameraCalibration's docstring):
#
# - Rover frame: +x forward, +y left, +z up. All trajectory/footprint points
#   supplied to this module lie on the ground plane (z = 0).
# - Camera frame (OpenCV convention): +x right, +y down, +z forward.
# - `camera_from_rover_transform` maps rover-frame homogeneous points into
#   the camera frame: p_camera = camera_from_rover_transform @ p_rover.
# - "Behind the camera" means the transformed point's camera-frame Z <= 0.
# - Projected pixel coordinates (u, v) are in the same raw (as-calibrated,
#   distorted) pixel grid that SAM-TP scores -- this module applies the full
#   calibrated distortion model via cv2.projectPoints rather than targeting
#   a separately undistorted image buffer, since nothing else in this
#   pipeline undistorts frames before inference.

REASON_OK = "OK"
REASON_BEHIND_CAMERA = "BEHIND_CAMERA"
REASON_OUT_OF_IMAGE = "OUT_OF_IMAGE"
REASON_NONFINITE_PROJECTION = "NONFINITE_PROJECTION"
REASON_INSUFFICIENT_PROJECTED_COVERAGE = "INSUFFICIENT_PROJECTED_COVERAGE"


@dataclass(frozen=True)
class ProjectedPoints:
    """Per-point projection result, one entry per input rover-frame point."""

    image_uv: np.ndarray  # (N, 2) float64, meaningful only where valid
    valid: np.ndarray  # (N,) bool
    reasons: tuple[str, ...]  # length N, REASON_OK where valid


@dataclass(frozen=True)
class ProjectedFootprint:
    """Result of projecting one CandidateTrajectory's footprint into an image."""

    valid: bool
    reason: str
    centerline_uv: np.ndarray  # (K, 2) int32, valid arc-length prefix only
    left_uv: np.ndarray  # (K, 2) int32
    right_uv: np.ndarray  # (K, 2) int32
    covered_arc_length_m: float
    coverage_ratio: float
    footprint_mask: np.ndarray  # (H, W) bool
    footprint_pixel_count: int
    near_field_mask: np.ndarray  # (H, W) bool, subset of footprint_mask


def project_rover_points(
    points_xy: np.ndarray,
    calibration: CameraCalibration,
) -> ProjectedPoints:
    """Project ground-plane rover-frame points (x, y) into calibrated image pixels."""

    points_xy = np.asarray(points_xy, dtype=np.float64)
    if points_xy.ndim != 2 or points_xy.shape[1] != 2:
        raise ValueError("points_xy must have shape (N, 2)")
    count = points_xy.shape[0]
    homogeneous_rover = np.concatenate(
        [
            points_xy,
            np.zeros((count, 1), dtype=np.float64),
            np.ones((count, 1), dtype=np.float64),
        ],
        axis=1,
    )
    camera_points = (calibration.camera_from_rover_transform @ homogeneous_rover.T).T[:, :3]
    behind_camera = camera_points[:, 2] <= 0.0

    image_uv = np.full((count, 2), np.nan, dtype=np.float64)
    nonfinite = np.zeros(count, dtype=bool)
    in_front = ~behind_camera
    if np.any(in_front):
        projected, _ = cv2.projectPoints(
            camera_points[in_front].reshape(-1, 1, 3),
            np.zeros(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            calibration.camera_matrix,
            calibration.distortion_coefficients,
        )
        projected = projected.reshape(-1, 2)
        finite = np.isfinite(projected).all(axis=1)
        front_indices = np.flatnonzero(in_front)
        image_uv[front_indices] = projected
        nonfinite[front_indices[~finite]] = True

    out_of_image = (
        ~nonfinite
        & in_front
        & (
            (image_uv[:, 0] < 0.0)
            | (image_uv[:, 0] > calibration.image_width - 1)
            | (image_uv[:, 1] < 0.0)
            | (image_uv[:, 1] > calibration.image_height - 1)
        )
    )
    valid = in_front & ~nonfinite & ~out_of_image
    reasons = []
    for index in range(count):
        if behind_camera[index]:
            reasons.append(REASON_BEHIND_CAMERA)
        elif nonfinite[index]:
            reasons.append(REASON_NONFINITE_PROJECTION)
        elif out_of_image[index]:
            reasons.append(REASON_OUT_OF_IMAGE)
        else:
            reasons.append(REASON_OK)
    return ProjectedPoints(image_uv=image_uv, valid=valid, reasons=tuple(reasons))


def project_trajectory(
    trajectory: CandidateTrajectory,
    calibration: CameraCalibration,
    *,
    min_projected_coverage_ratio: float,
    near_field_horizon_fraction: float,
) -> ProjectedFootprint:
    """Project one candidate's centerline and footprint boundaries into image space.

    Points are truncated to the longest contiguous arc-length run that
    projects validly for the centerline *and* both boundaries. The run is
    not required to start at arc-length zero: the sample at distance 0 is
    the rover's own origin, which sits directly under/behind a
    forward-mounted, downward-pitched camera on essentially every real
    extrinsic calibration -- that leading blind spot is normal and must not
    itself reject every candidate. Coverage below
    ``min_projected_coverage_ratio`` of ``horizon_m`` (measured as the
    visible run's own arc-length span) rejects the candidate outright
    rather than scoring a near-useless sliver.

    The near-field sub-mask used for safety scoring is the closest
    ``near_field_horizon_fraction`` fraction of the *visible run*, not an
    absolute distance from the rover -- an absolute cutoff could fall
    entirely inside the blind spot and never be scoreable at all.
    """

    shape = (calibration.image_height, calibration.image_width)
    centerline = project_rover_points(trajectory.points_xy, calibration)
    left = project_rover_points(trajectory.left_boundary_xy, calibration)
    right = project_rover_points(trajectory.right_boundary_xy, calibration)
    combined_valid = centerline.valid & left.valid & right.valid

    run_start, run_end = _longest_valid_run(combined_valid)

    empty_mask = np.zeros(shape, dtype=bool)
    if run_start is None:
        first_reason = next(
            (
                r
                for r in (centerline.reasons[0], left.reasons[0], right.reasons[0])
                if r != REASON_OK
            ),
            REASON_INSUFFICIENT_PROJECTED_COVERAGE,
        )
        return ProjectedFootprint(
            valid=False,
            reason=first_reason,
            centerline_uv=np.zeros((0, 2), dtype=np.int32),
            left_uv=np.zeros((0, 2), dtype=np.int32),
            right_uv=np.zeros((0, 2), dtype=np.int32),
            covered_arc_length_m=0.0,
            coverage_ratio=0.0,
            footprint_mask=empty_mask,
            footprint_pixel_count=0,
            near_field_mask=empty_mask,
        )

    covered_arc_length_m = float(
        trajectory.sample_distances_m[run_end - 1] - trajectory.sample_distances_m[run_start]
    )
    coverage_ratio = covered_arc_length_m / trajectory.horizon_m
    run_length = run_end - run_start
    if coverage_ratio < min_projected_coverage_ratio or run_length < 2:
        return ProjectedFootprint(
            valid=False,
            reason=REASON_INSUFFICIENT_PROJECTED_COVERAGE,
            centerline_uv=np.zeros((0, 2), dtype=np.int32),
            left_uv=np.zeros((0, 2), dtype=np.int32),
            right_uv=np.zeros((0, 2), dtype=np.int32),
            covered_arc_length_m=covered_arc_length_m,
            coverage_ratio=coverage_ratio,
            footprint_mask=empty_mask,
            footprint_pixel_count=0,
            near_field_mask=empty_mask,
        )

    centerline_uv = np.rint(centerline.image_uv[run_start:run_end]).astype(np.int32)
    left_uv = np.rint(left.image_uv[run_start:run_end]).astype(np.int32)
    right_uv = np.rint(right.image_uv[run_start:run_end]).astype(np.int32)
    footprint_mask, footprint_pixel_count = rasterize_footprint(shape, left_uv, right_uv)

    near_run_length = max(2, min(run_length, int(round(run_length * near_field_horizon_fraction))))
    near_field_mask, _ = rasterize_footprint(
        shape, left_uv[:near_run_length], right_uv[:near_run_length]
    )

    return ProjectedFootprint(
        valid=True,
        reason=REASON_OK,
        centerline_uv=centerline_uv,
        left_uv=left_uv,
        right_uv=right_uv,
        covered_arc_length_m=covered_arc_length_m,
        coverage_ratio=coverage_ratio,
        footprint_mask=footprint_mask,
        footprint_pixel_count=footprint_pixel_count,
        near_field_mask=near_field_mask,
    )


def _longest_valid_run(valid: np.ndarray) -> tuple[int | None, int]:
    """Return the [start, end) index span of the longest contiguous True run."""

    best_start: int | None = None
    best_length = 0
    run_start: int | None = None
    for index, is_valid in enumerate(valid):
        if is_valid:
            if run_start is None:
                run_start = index
            run_length = index + 1 - run_start
            if run_length > best_length:
                best_length = run_length
                best_start = run_start
        else:
            run_start = None
    if best_start is None:
        return None, 0
    return best_start, best_start + best_length


def rasterize_footprint(
    shape: tuple[int, int],
    left_uv: np.ndarray,
    right_uv: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Rasterize the polygon bounded by left/right boundary polylines."""

    height, width = int(shape[0]), int(shape[1])
    mask = np.zeros((height, width), dtype=np.uint8)
    if len(left_uv) < 2 or len(right_uv) < 2:
        return mask.astype(bool), 0
    polygon = np.concatenate([left_uv, right_uv[::-1]], axis=0).astype(np.int32)
    cv2.fillPoly(mask, [polygon], 1)
    bool_mask = mask.astype(bool)
    return bool_mask, int(bool_mask.sum())
