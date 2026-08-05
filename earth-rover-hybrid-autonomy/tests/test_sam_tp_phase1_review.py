from __future__ import annotations

import math

import cv2
import numpy as np

from earth_rover.planning.trajectory_sampler import (
    DEFAULT_CURVATURES,
    ConstantCurvatureTrajectorySampler,
)
from training.sam_tp_phase1_review import (
    SamTpPhase1FrameProcessor,
    draw_image_path_rgb,
    propose_image_space_path,
    render_trajectory_geometry_rgb,
    smooth_image_path_spline,
    smooth_image_path_temporally,
)
from training.sam_tp_reproduction import SamTpPrediction


class Predictor:
    def __init__(self) -> None:
        self.inputs: list[np.ndarray] = []

    def predict(self, image_rgb: np.ndarray) -> SamTpPrediction:
        self.inputs.append(image_rgb.copy())
        score = np.full(image_rgb.shape[:2], 0.75, dtype=np.float32)
        return SamTpPrediction(
            raw_logits=np.ones_like(score),
            traversability_score=score,
            heatmap=np.zeros_like(image_rgb),
            input_shape=image_rgb.shape,
            output_shape=score.shape,
            inference_time_ms=4.0,
            device="test",
        )


class SequencePredictor(Predictor):
    def __init__(self, scores: list[np.ndarray]) -> None:
        super().__init__()
        self.scores = list(scores)

    def predict(self, image_rgb: np.ndarray) -> SamTpPrediction:
        self.inputs.append(image_rgb.copy())
        score = self.scores.pop(0)
        return SamTpPrediction(
            raw_logits=np.ones_like(score),
            traversability_score=score,
            heatmap=np.zeros_like(image_rgb),
            input_shape=image_rgb.shape,
            output_shape=score.shape,
            inference_time_ms=4.0,
            device="test",
        )


def trajectories():
    return ConstantCurvatureTrajectorySampler(
        DEFAULT_CURVATURES,
        horizon_m=2.0,
        sample_interval_m=0.1,
        rover_width_m=0.4,
        safety_margin_m=0.1,
    ).sample()


def test_phase1_processor_accepts_replay_or_sdk_rgb_frame() -> None:
    predictor = Predictor()
    processor = SamTpPhase1FrameProcessor(
        predictor,
        trajectories(),
        "checkpoint:test",
    )
    image = np.zeros((36, 64, 3), dtype=np.uint8)

    result = processor.process(image, 100.0)

    assert len(result.trajectories) == 7
    assert result.traversability.score_map.shape == image.shape[:2]
    assert result.traversability.model_version == "checkpoint:test"
    assert np.array_equal(predictor.inputs[0], image)
    assert result.image_path.valid
    assert np.all(result.traversability.score_map[
        result.image_path.points_uv[:, 1],
        result.image_path.points_uv[:, 0],
    ] >= 0.55)


def test_phase1_processor_holds_previous_safe_path_between_replans() -> None:
    predictor = Predictor()
    processor = SamTpPhase1FrameProcessor(
        predictor,
        trajectories(),
        "checkpoint:test",
        path_replan_interval_frames=3,
    )
    image = np.zeros((120, 200, 3), dtype=np.uint8)

    first = processor.process(image, 100.0, target_heading_error_rad=math.radians(20.0))
    second = processor.process(image, 100.2, target_heading_error_rad=math.radians(22.0))
    third = processor.process(image, 100.4, target_heading_error_rad=math.radians(24.0))
    fourth = processor.process(image, 100.6, target_heading_error_rad=math.radians(26.0))

    assert first.image_path.valid
    assert second.image_path.reason.startswith("TEMPORAL_HOLD_")
    assert third.image_path.reason.startswith("TEMPORAL_HOLD_")
    assert np.array_equal(second.image_path.points_uv, first.image_path.points_uv)
    assert third.image_path.reason == second.image_path.reason
    assert not fourth.image_path.reason.startswith("TEMPORAL_HOLD_")


def test_phase1_processor_replans_immediately_when_goal_heading_jumps() -> None:
    predictor = Predictor()
    processor = SamTpPhase1FrameProcessor(
        predictor,
        trajectories(),
        "checkpoint:test",
        path_replan_interval_frames=3,
        heading_replan_threshold_deg=12.0,
    )
    image = np.zeros((120, 200, 3), dtype=np.uint8)

    first = processor.process(image, 100.0, target_heading_error_rad=math.radians(5.0))
    second = processor.process(image, 100.2, target_heading_error_rad=math.radians(45.0))

    assert first.image_path.valid
    assert second.image_path.valid
    assert not second.image_path.reason.startswith("TEMPORAL_HOLD_")
    assert not np.array_equal(second.image_path.points_uv, first.image_path.points_uv)


def test_geometry_panel_is_deterministic_and_not_blank() -> None:
    first = render_trajectory_geometry_rgb(trajectories(), 320, 180)
    second = render_trajectory_geometry_rgb(trajectories(), 320, 180)

    assert first.shape == (180, 320, 3)
    assert np.array_equal(first, second)
    assert np.unique(first.reshape(-1, 3), axis=0).shape[0] > 3


def test_image_path_stays_inside_connected_high_score_region() -> None:
    score = np.full((120, 200), 0.1, dtype=np.float32)
    for y in range(35, 115):
        center = 100 + (80 - y) // 3
        score[y, center - 24 : center + 25] = 0.9

    proposal = propose_image_space_path(
        score,
        np.ones_like(score, dtype=bool),
        minimum_score=0.55,
        corridor_half_width_ratio=0.02,
    )

    assert proposal.valid
    assert proposal.reason == "CONNECTED_HIGH_TRAVERSABILITY_IMAGE_PATH"
    assert np.all(score[proposal.points_uv[:, 1], proposal.points_uv[:, 0]] >= 0.55)
    rendered = draw_image_path_rgb(
        np.zeros((120, 200, 3), dtype=np.uint8),
        proposal,
        0.02,
    )
    assert rendered.any()


def test_image_path_is_rejected_when_safe_region_is_disconnected() -> None:
    score = np.full((120, 200), 0.9, dtype=np.float32)
    score[65:72] = 0.0

    proposal = propose_image_space_path(
        score,
        np.ones_like(score, dtype=bool),
        minimum_score=0.55,
        corridor_half_width_ratio=0.02,
    )

    assert proposal.valid
    assert proposal.reason == "PARTIAL_CONNECTED_TRAVERSABILITY_IMAGE_PATH"
    assert proposal.points_uv.shape[0] > 3
    assert proposal.points_uv[-1, 1] > round(score.shape[0] * 0.52)


def test_image_path_rejects_when_no_near_field_safe_start_exists() -> None:
    score = np.full((120, 200), 0.1, dtype=np.float32)

    proposal = propose_image_space_path(
        score,
        np.ones_like(score, dtype=bool),
        minimum_score=0.55,
        corridor_half_width_ratio=0.02,
    )

    assert not proposal.valid
    assert proposal.reason == "NO_CONNECTED_TRAVERSABLE_PATH"


def test_rejected_path_preserves_global_heading_for_visualization() -> None:
    score = np.full((120, 200), 0.1, dtype=np.float32)

    proposal = propose_image_space_path(
        score,
        np.ones_like(score, dtype=bool),
        minimum_score=0.55,
        corridor_half_width_ratio=0.02,
        target_heading_error_rad=math.radians(30.0),
    )

    assert proposal.valid is False
    assert proposal.target_heading_error_rad == math.radians(30.0)
    assert proposal.target_uv is not None
    assert proposal.goal_alignment_weight == 1.35


def test_goal_behind_uses_turn_cue_instead_of_forward_heading_line() -> None:
    score = np.full((120, 200), 0.9, dtype=np.float32)
    # target_heading_error_rad follows positive_clockwise_right, so +130 deg
    # means the goal is behind and to the right.
    proposal = propose_image_space_path(
        score,
        np.ones_like(score, dtype=bool),
        minimum_score=0.55,
        corridor_half_width_ratio=0.02,
        target_heading_error_rad=math.radians(130.0),
    )

    rendered = draw_image_path_rgb(
        np.zeros((120, 200, 3), dtype=np.uint8),
        proposal,
        0.02,
    )

    # Cyan turn cue occupies the lower-right edge for a behind-right goal.
    assert rendered[103, 100:184].sum() > 0


def test_goal_behind_left_uses_left_edge_turn_cue() -> None:
    score = np.full((120, 200), 0.9, dtype=np.float32)
    proposal = propose_image_space_path(
        score,
        np.ones_like(score, dtype=bool),
        minimum_score=0.55,
        corridor_half_width_ratio=0.02,
        target_heading_error_rad=math.radians(-130.0),
    )

    rendered = draw_image_path_rgb(
        np.zeros((120, 200, 3), dtype=np.uint8),
        proposal,
        0.02,
    )

    # Cyan turn cue occupies the lower-left edge for a behind-left goal.
    assert rendered[103, 16:100].sum() > 0


def test_gps_heading_biases_local_path_without_crossing_unsafe_pixels() -> None:
    score = np.full((120, 200), 0.9, dtype=np.float32)
    valid = np.ones_like(score, dtype=bool)

    left = propose_image_space_path(
        score,
        valid,
        minimum_score=0.55,
        corridor_half_width_ratio=0.02,
        target_heading_error_rad=math.radians(35.0),
    )
    right = propose_image_space_path(
        score,
        valid,
        minimum_score=0.55,
        corridor_half_width_ratio=0.02,
        target_heading_error_rad=math.radians(-35.0),
    )

    assert left.valid and right.valid
    assert left.reason == "GPS_HEADING_ALIGNED_TRAVERSABLE_PATH"
    assert left.points_uv[-1, 0] < score.shape[1] // 2
    assert right.points_uv[-1, 0] > score.shape[1] // 2
    assert left.selected_heading_rad > 0.0
    assert right.selected_heading_rad < 0.0
    assert left.target_uv is not None
    assert left.path_length_px > 0.0


def test_gps_preference_never_crosses_low_traversability_barrier() -> None:
    score = np.full((120, 200), 0.1, dtype=np.float32)
    score[35:116, 82:119] = 0.9

    proposal = propose_image_space_path(
        score,
        np.ones_like(score, dtype=bool),
        minimum_score=0.55,
        corridor_half_width_ratio=0.02,
        target_heading_error_rad=math.radians(50.0),
    )

    assert proposal.valid
    assert np.all(score[proposal.points_uv[:, 1], proposal.points_uv[:, 0]] >= 0.55)
    assert np.all((proposal.points_uv[:, 0] >= 82) & (proposal.points_uv[:, 0] < 119))


def test_relaxed_path_search_bridges_moderate_score_gaps_but_reports_reason() -> None:
    score = np.full((120, 200), 0.9, dtype=np.float32)
    score[65:72] = 0.48

    proposal = propose_image_space_path(
        score,
        np.ones_like(score, dtype=bool),
        minimum_score=0.55,
        corridor_half_width_ratio=0.02,
        target_heading_error_rad=math.radians(10.0),
    )

    assert proposal.valid
    assert proposal.reason == "RELAXED_GPS_HEADING_ALIGNED_TRAVERSABLE_PATH"
    assert proposal.mean_score >= 0.55


def test_global_heading_has_high_weight_among_safe_paths() -> None:
    score = np.full((120, 200), 0.90, dtype=np.float32)
    score[:, :85] = 0.62

    proposal = propose_image_space_path(
        score,
        np.ones_like(score, dtype=bool),
        minimum_score=0.55,
        corridor_half_width_ratio=0.02,
        target_heading_error_rad=math.radians(45.0),
    )

    assert proposal.valid
    assert proposal.goal_alignment_weight == 1.35
    assert proposal.points_uv[-1, 0] < score.shape[1] // 2
    assert proposal.selected_heading_rad > 0.0


def test_blocked_global_heading_uses_nearest_connected_safe_alternative() -> None:
    score = np.full((120, 200), 0.10, dtype=np.float32)
    score[35:116, 108:176] = 0.90

    proposal = propose_image_space_path(
        score,
        np.ones_like(score, dtype=bool),
        minimum_score=0.55,
        corridor_half_width_ratio=0.02,
        target_heading_error_rad=math.radians(45.0),
    )

    assert proposal.valid
    assert proposal.target_heading_error_rad > 0.0
    assert proposal.selected_heading_rad < 0.0
    assert np.all(score[proposal.points_uv[:, 1], proposal.points_uv[:, 0]] >= 0.55)


def test_cubic_spline_filter_reduces_path_cornering_in_linear_time() -> None:
    points = np.asarray(
        [
            [50, 90],
            [42, 80],
            [54, 70],
            [43, 60],
            [56, 50],
            [60, 40],
        ],
        dtype=np.int32,
    )
    safe = np.ones((100, 100), dtype=bool)

    smoothed, applied = smooth_image_path_spline(points, safe, iterations=3)

    original_cornering = np.abs(np.diff(points[:, 0], n=2)).sum()
    smoothed_cornering = np.abs(np.diff(smoothed[:, 0], n=2)).sum()
    assert applied is True
    assert smoothed_cornering < original_cornering
    assert np.array_equal(smoothed[[0, -1]], points[[0, -1]])
    assert smoothed.flags.writeable is False


def test_spline_falls_back_when_smoothing_would_cut_unsafe_corner() -> None:
    points = np.asarray(
        [[20, 90], [20, 80], [40, 70], [40, 60], [60, 50], [60, 40]],
        dtype=np.int32,
    )
    safe = np.zeros((100, 100), dtype=np.uint8)
    cv2.polylines(safe, [points], False, 1, 1, cv2.LINE_8)

    result, applied = smooth_image_path_spline(points, safe.astype(bool), iterations=3)

    assert applied is False
    assert np.array_equal(result, points)


def test_temporal_smoothing_reduces_frame_to_frame_jitter_and_stays_safe() -> None:
    previous = np.asarray(
        [[50, 90], [48, 80], [46, 70], [44, 60], [42, 50], [40, 40]],
        dtype=np.int32,
    )
    current = np.asarray(
        [[50, 90], [57, 80], [52, 70], [55, 60], [48, 50], [40, 40]],
        dtype=np.int32,
    )
    safe = np.ones((100, 100), dtype=bool)

    smoothed, applied = smooth_image_path_temporally(
        current,
        previous,
        safe,
        current_weight=0.45,
    )

    assert applied is True
    assert np.abs(smoothed[1:-1, 0] - previous[1:-1, 0]).sum() < np.abs(
        current[1:-1, 0] - previous[1:-1, 0]
    ).sum()
    assert np.array_equal(smoothed[[0, -1]], current[[0, -1]])
    assert smoothed.flags.writeable is False


def test_processor_keeps_temporal_reference_across_short_invalid_gap() -> None:
    open_score = np.full((120, 200), 0.90, dtype=np.float32)
    blocked_score = np.full((120, 200), 0.10, dtype=np.float32)
    predictor = SequencePredictor([open_score, blocked_score, open_score])
    processor = SamTpPhase1FrameProcessor(
        predictor,
        trajectories(),
        "checkpoint:test",
    )
    image = np.zeros((120, 200, 3), dtype=np.uint8)

    first = processor.process(image, 100.0, math.radians(40.0))
    gap = processor.process(image, 100.1, math.radians(0.0))
    third = processor.process(image, 100.2, math.radians(-40.0))

    assert first.image_path.valid
    assert not gap.image_path.valid
    assert third.image_path.valid
    assert third.image_path.smoothing_method == (
        "CONSTRAINED_CUBIC_B_SPLINE_TEMPORAL_EMA"
    )
    assert third.image_path.smoothing_applied is True
