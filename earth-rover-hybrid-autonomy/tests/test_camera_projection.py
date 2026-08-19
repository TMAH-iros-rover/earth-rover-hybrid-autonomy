from __future__ import annotations

import math

import numpy as np
import pytest

from earth_rover.perception.camera_calibration import CameraCalibration
from earth_rover.perception.camera_projection import (
    REASON_BEHIND_CAMERA,
    REASON_INSUFFICIENT_PROJECTED_COVERAGE,
    REASON_OUT_OF_IMAGE,
    project_rover_points,
    project_trajectory,
    rasterize_footprint,
)
from earth_rover.planning.trajectory_sampler import ConstantCurvatureTrajectorySampler


def _make_calibration(
    *,
    image_width: int = 640,
    image_height: int = 480,
    pitch_deg: float = 20.0,
    camera_height_m: float = 0.3,
    fx: float = 500.0,
    fy: float = 500.0,
) -> CameraCalibration:
    camera_matrix = np.array(
        [[fx, 0.0, image_width / 2.0], [0.0, fy, image_height / 2.0], [0.0, 0.0, 1.0]]
    )
    pitch = math.radians(pitch_deg)
    # Base mapping with zero pitch: camera_x = -rover_y, camera_y = -rover_z,
    # camera_z = rover_x (camera boresight along rover +x, camera "up" along
    # rover +z). Then an additional pitch-down rotation about the camera's
    # own x-axis tilts the boresight toward the ground.
    base = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
    pitch_rotation = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(pitch), -math.sin(pitch)],
            [0.0, math.sin(pitch), math.cos(pitch)],
        ]
    )
    rotation = pitch_rotation @ base
    camera_position_in_rover = np.array([0.0, 0.0, camera_height_m])
    translation = -rotation @ camera_position_in_rover
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return CameraCalibration(
        schema_version=1,
        calibration_id="synthetic_test",
        image_width=image_width,
        image_height=image_height,
        camera_matrix=camera_matrix,
        distortion_model="opencv_pinhole",
        distortion_coefficients=np.zeros(5),
        camera_from_rover_transform=transform,
        capture_date="2026-01-01",
        capture_source="synthetic",
        reprojection_error_px=0.0,
        placeholder=False,
        content_sha256="0" * 64,
    )


def _straight_trajectory(**sampler_overrides):
    defaults = dict(
        horizon_m=2.0, sample_interval_m=0.1, rover_width_m=0.4, safety_margin_m=0.1
    )
    defaults.update(sampler_overrides)
    (trajectory,) = ConstantCurvatureTrajectorySampler((0.0,), **defaults).sample()
    return trajectory


def _curved_trajectory(curvature: float, **sampler_overrides):
    defaults = dict(
        horizon_m=2.0, sample_interval_m=0.1, rover_width_m=0.4, safety_margin_m=0.1
    )
    defaults.update(sampler_overrides)
    (trajectory,) = ConstantCurvatureTrajectorySampler((curvature,), **defaults).sample()
    return trajectory


class TestProjectRoverPoints:
    def test_straight_ahead_point_projects_near_principal_point_column(self):
        calibration = _make_calibration()
        result = project_rover_points(np.array([[1.0, 0.0]]), calibration)
        assert result.valid[0]
        assert result.image_uv[0, 0] == pytest.approx(calibration.image_width / 2.0, abs=1e-6)

    def test_left_point_projects_to_smaller_u_than_straight(self):
        calibration = _make_calibration()
        points = np.array([[1.0, 0.0], [1.0, 0.3]])  # straight, then left (+y)
        result = project_rover_points(points, calibration)
        assert result.valid.all()
        assert result.image_uv[1, 0] < result.image_uv[0, 0]

    def test_right_point_projects_to_larger_u_than_straight(self):
        calibration = _make_calibration()
        points = np.array([[1.0, 0.0], [1.0, -0.3]])  # straight, then right (-y)
        result = project_rover_points(points, calibration)
        assert result.valid.all()
        assert result.image_uv[1, 0] > result.image_uv[0, 0]

    def test_point_behind_camera_is_rejected(self):
        calibration = _make_calibration()
        # Far behind the rover: definitely behind the pitched-forward camera.
        result = project_rover_points(np.array([[-5.0, 0.0]]), calibration)
        assert not result.valid[0]
        assert result.reasons[0] == REASON_BEHIND_CAMERA

    def test_point_far_off_axis_is_out_of_image(self):
        calibration = _make_calibration()
        result = project_rover_points(np.array([[1.0, 50.0]]), calibration)
        assert not result.valid[0]
        assert result.reasons[0] == REASON_OUT_OF_IMAGE

    def test_rover_origin_is_typically_not_visible(self):
        # Arc-length-zero sample sits directly under/behind a forward,
        # downward-pitched camera -- this is the expected leading blind spot
        # that project_trajectory must tolerate rather than treat as a
        # global rejection.
        calibration = _make_calibration()
        result = project_rover_points(np.array([[0.0, 0.0]]), calibration)
        assert not result.valid[0]


class TestProjectTrajectory:
    def test_straight_trajectory_is_valid_with_reasonable_coverage(self):
        calibration = _make_calibration()
        trajectory = _straight_trajectory()
        footprint = project_trajectory(
            trajectory,
            calibration,
            min_projected_coverage_ratio=0.3,
            near_field_horizon_fraction=0.35,
        )
        assert footprint.valid
        assert footprint.coverage_ratio > 0.3
        assert footprint.footprint_pixel_count > 0
        assert int(footprint.near_field_mask.sum()) > 0
        assert int(footprint.near_field_mask.sum()) <= footprint.footprint_pixel_count

    def test_left_and_right_curvature_bend_the_centerline_apart(self):
        calibration = _make_calibration()
        left = project_trajectory(
            _curved_trajectory(0.3),
            calibration,
            min_projected_coverage_ratio=0.3,
            near_field_horizon_fraction=0.35,
        )
        right = project_trajectory(
            _curved_trajectory(-0.3),
            calibration,
            min_projected_coverage_ratio=0.3,
            near_field_horizon_fraction=0.35,
        )
        assert left.valid and right.valid
        # Positive (rover-frame left) curvature bends toward smaller u;
        # negative (right) curvature bends toward larger u -- consistent
        # with TestProjectRoverPoints above.
        assert left.centerline_uv[-1, 0] < right.centerline_uv[-1, 0]

    def test_insufficient_coverage_is_rejected(self):
        calibration = _make_calibration()
        trajectory = _straight_trajectory()
        footprint = project_trajectory(
            trajectory,
            calibration,
            min_projected_coverage_ratio=0.999,
            near_field_horizon_fraction=0.35,
        )
        assert not footprint.valid
        assert footprint.reason == REASON_INSUFFICIENT_PROJECTED_COVERAGE
        assert footprint.footprint_pixel_count == 0

    def test_sharp_curvature_off_frame_is_rejected(self):
        calibration = _make_calibration(image_width=640)
        # A very tight turn's boundary swings off the side of a narrow FOV
        # image well before the horizon.
        trajectory = _curved_trajectory(1.2, horizon_m=2.0, sample_interval_m=0.1,
                                         rover_width_m=0.4, safety_margin_m=0.1)
        footprint = project_trajectory(
            trajectory,
            calibration,
            min_projected_coverage_ratio=0.9,
            near_field_horizon_fraction=0.35,
        )
        assert not footprint.valid

    def test_wider_footprint_yields_more_pixels(self):
        # A wide-FOV calibration so the wider footprint's boundary still
        # stays fully in-image (narrow-FOV cameras can make a wider
        # footprint clip *sooner*, which is exercised separately by
        # test_sharp_curvature_off_frame_is_rejected).
        calibration = _make_calibration(image_width=1280, image_height=720, fx=300.0, fy=300.0)
        narrow = _curved_trajectory(
            0.0, horizon_m=0.8, sample_interval_m=0.05, rover_width_m=0.2, safety_margin_m=0.0
        )
        wide = _curved_trajectory(
            0.0, horizon_m=0.8, sample_interval_m=0.05, rover_width_m=1.5, safety_margin_m=0.0
        )
        narrow_footprint = project_trajectory(
            narrow, calibration, min_projected_coverage_ratio=0.1, near_field_horizon_fraction=0.35
        )
        wide_footprint = project_trajectory(
            wide, calibration, min_projected_coverage_ratio=0.1, near_field_horizon_fraction=0.35
        )
        assert narrow_footprint.valid and wide_footprint.valid
        assert wide_footprint.footprint_pixel_count > narrow_footprint.footprint_pixel_count

    def test_larger_safety_margin_yields_more_pixels(self):
        calibration = _make_calibration(image_width=1280, image_height=720, fx=300.0, fy=300.0)
        tight = _curved_trajectory(
            0.0, horizon_m=0.8, sample_interval_m=0.05, rover_width_m=0.4, safety_margin_m=0.0
        )
        padded = _curved_trajectory(
            0.0, horizon_m=0.8, sample_interval_m=0.05, rover_width_m=0.4, safety_margin_m=0.5
        )
        tight_footprint = project_trajectory(
            tight, calibration, min_projected_coverage_ratio=0.1, near_field_horizon_fraction=0.35
        )
        padded_footprint = project_trajectory(
            padded, calibration, min_projected_coverage_ratio=0.1, near_field_horizon_fraction=0.35
        )
        assert tight_footprint.valid and padded_footprint.valid
        assert padded_footprint.footprint_pixel_count > tight_footprint.footprint_pixel_count


class TestObstacleScoring:
    def _footprint(self, calibration):
        trajectory = _straight_trajectory()
        return project_trajectory(
            trajectory,
            calibration,
            min_projected_coverage_ratio=0.3,
            near_field_horizon_fraction=0.35,
        )

    def test_obstacle_inside_footprint_is_detected_in_mask(self):
        calibration = _make_calibration()
        footprint = self._footprint(calibration)
        assert footprint.valid
        score = np.ones((calibration.image_height, calibration.image_width), dtype=np.float32)
        rows, cols = np.nonzero(footprint.footprint_mask)
        obstacle_row, obstacle_col = rows[len(rows) // 2], cols[len(rows) // 2]
        score[obstacle_row, obstacle_col] = 0.0
        assert score[footprint.footprint_mask].min() == 0.0

    def test_obstacle_outside_footprint_does_not_affect_mask_scores(self):
        calibration = _make_calibration()
        footprint = self._footprint(calibration)
        assert footprint.valid
        score = np.ones((calibration.image_height, calibration.image_width), dtype=np.float32)
        outside = ~footprint.footprint_mask
        assert outside.any()
        rows, cols = np.nonzero(outside)
        score[rows[0], cols[0]] = 0.0
        assert score[footprint.footprint_mask].min() == 1.0


class TestRasterizeFootprint:
    def test_too_few_points_returns_empty_mask(self):
        mask, count = rasterize_footprint((100, 100), np.zeros((1, 2), dtype=np.int32), np.zeros((1, 2), dtype=np.int32))
        assert count == 0
        assert not mask.any()

    def test_simple_rectangle_pixel_count(self):
        left = np.array([[10, 0], [10, 50]], dtype=np.int32)
        right = np.array([[30, 0], [30, 50]], dtype=np.int32)
        mask, count = rasterize_footprint((60, 60), left, right)
        assert count > 0
        assert mask[25, 20]
        assert not mask[25, 5]
