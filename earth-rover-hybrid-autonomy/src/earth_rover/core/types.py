from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class RoverData:
    timestamp: float
    latitude: Optional[float]
    longitude: Optional[float]
    orientation: Optional[float]
    speed: Optional[float]
    rpms: Optional[list[float]]
    battery: Optional[float]
    signal_level: Optional[float]
    gps_signal: Optional[float]
    raw: dict
    sdk_timestamp: Optional[float] = None


@dataclass
class FrameData:
    timestamp: float
    image: np.ndarray
    source: str
    sdk_timestamp: Optional[float] = None
    source_frame_id: Optional[str] = None
    source_media_time_sec: Optional[float] = None
    source_frame_new: bool = True


@dataclass
class ControlCommand:
    linear: float
    angular: float
    lamp: int = 0
    mode: str = "NORMAL_DRIVE"


@dataclass
class PerceptionResult:
    left_free_score: float
    center_free_score: float
    right_free_score: float
    obstacle_confidence: float
    traversability_confidence: float
    debug: dict = field(default_factory=dict)


@dataclass
class CandidateDirection:
    name: str
    local_goal_error_rad: float
    traversability_score: float
    obstacle_risk: float
    turning_cost: float
    score: float
    debug: dict = field(default_factory=dict)


@dataclass(frozen=True)
class LocalTraversability:
    """Planner-facing summary of a source-ID segmentation mask.

    Scores are normalized to ``[0, 1]``. Obstacle ratios describe each
    direction's configured driving ROI. ``near_obstacle_ratio`` describes the
    lower center ROI, where an obstacle most directly blocks forward motion.
    ``free_space_center`` is the center score before goal preference is added.
    The recommendation is perception-only and never represents mission intent.
    """

    left_score: float
    center_score: float
    right_score: float
    left_obstacle_ratio: float
    center_obstacle_ratio: float
    right_obstacle_ratio: float
    near_obstacle_ratio: float
    mean_confidence: float
    free_space_center: float
    recommended_direction: str
    recommended_heading: float
    stop_recommended: bool
    reason: str
    sector_debug: dict = field(default_factory=dict)


@dataclass(frozen=True)
class LocalPlan:
    """Semantic local target selected before controller command conversion.

    ``steering_target`` is a local heading offset in radians and
    ``speed_target`` is a normalized ``[0, 1]`` speed scale. The existing
    controller converts these values into bounded rover commands.
    """

    selected_direction: str
    steering_target: float
    speed_target: float
    candidate_scores: dict[str, float]
    mode: str
    stop_requested: bool
    reason: str


@dataclass(frozen=True)
class TraversabilityOutput:
    """Validated continuous traversability evidence in source-frame geometry.

    ``score_map`` is float32 in ``[0, 1]`` where larger values mean stronger
    SAM-TP traversability evidence. ``confidence`` is the valid-pixel fraction,
    a data-quality indicator rather than a calibrated model probability.
    """

    score_map: np.ndarray
    valid_mask: np.ndarray
    confidence: float
    frame_timestamp: float
    inference_time_ms: float
    model_version: str


@dataclass(frozen=True)
class CandidateTrajectory:
    """Constant-curvature path sampled in the rover base frame.

    The base frame uses ``+x`` forward and ``+y`` left. Positive curvature
    turns left. Distances are path arc lengths from zero through
    ``horizon_m``. Boundaries include rover width and the configured safety
    margin, but have not been projected into an image.
    """

    curvature: float
    points_xy: np.ndarray
    headings_rad: np.ndarray
    horizon_m: float
    sample_distances_m: np.ndarray
    left_boundary_xy: np.ndarray
    right_boundary_xy: np.ndarray
    effective_half_width_m: float
