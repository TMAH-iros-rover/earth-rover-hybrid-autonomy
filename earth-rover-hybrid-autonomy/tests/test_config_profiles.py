from pathlib import Path

from earth_rover.autonomy.mission1_controller import Mission1ControlConfig
from earth_rover.perception.camera_calibration import (
    load_calibration,
    validate_for_live_use,
)
from earth_rover.planning.motion_primitive_planner import MotionPrimitivePlannerConfig
from earth_rover.utils.config import load_config


ROOT = Path(__file__).resolve().parents[1]


def test_latency_2s_profile_loads_without_changing_default_file():
    config = load_config(
        ROOT / "configs/default.yaml", ROOT / "research/configs/urban_latency_2s.yaml"
    )

    assert config["project"]["profile"] == "latency_2s"
    assert config["latency"]["sensor_delay_sec"] == 2.0
    assert config["latency"]["frame_delay_sec"] == 2.0
    assert config["latency"]["data_delay_sec"] == 2.0
    assert config["control"]["linear_max"] == 0.22
    assert config["safety"]["frame_timeout_sec"] == 3.0


def test_traversability_replay_profile_is_log_only_and_configurable():
    config = load_config(
        ROOT / "configs/default.yaml", ROOT / "research/configs/urban_replay_v2.yaml"
    )

    assert config["recovery"]["enabled"] is False
    assert config["traversability_adapter"]["sector_boundaries"] == [0.0, 0.34, 0.66, 1.0]
    assert config["goal_aware_planner"]["candidate_heading_offsets_deg"]["LEFT"] == 35


def test_mission1_live_profile_has_bounded_deadzone_compensation():
    config = load_config(
        ROOT / "configs/default.yaml", ROOT / "configs/mission1_live.yaml"
    )

    mission = config["mission1_autonomy"]
    control = config["control"]
    assert 0.0 < mission["minimum_linear"] <= mission["base_linear"]
    assert mission["base_linear"] <= mission["max_linear"] <= 0.30
    assert control["linear_max"] == mission["max_linear"]
    assert control["angular_max"] == mission["max_angular"] <= 0.40
    assert max(abs(value) for value in config["planner"]["candidate_headings_deg"]) <= 30
    assert mission["require_metric_projection"] is True
    assert mission["minimum_linear"] == mission["max_linear"] == 0.12
    assert mission["minimum_rotate_angular"] == 0.40
    assert control["angular_max"] == mission["max_angular"] == 0.40
    assert mission["enable_stop_turn_go"] is True
    assert mission["stop_turn_require_motion_response"] is True
    assert mission["minimum_rotate_angular"] <= mission["stop_turn_rotate_angular"] <= mission["max_angular"]
    assert mission["stop_turn_heading_threshold_deg"] < 10.0
    assert mission["stop_turn_rotate_pulse_sec"] <= mission["stop_turn_settle_sec"]
    assert mission["path_recovery_confirm_frames"] >= 3
    # Live actuator evidence requires at least 70 deg/s; retain a bounded
    # margin below the historical one-sample compass-fault rates.
    assert 70.0 <= config["navigation"]["max_heading_rate_deg_per_sec"] <= 90.0


def test_mission1_live_profile_is_metric_projected_with_live_calibration():
    config = load_config(
        ROOT / "configs/default.yaml", ROOT / "configs/mission1_live.yaml"
    )

    planner = config["planner"]
    assert planner["geometry_mode"] == "metric_projected"
    for key in (
        "curvatures",
        "horizon_m",
        "sample_interval_m",
        "rover_width_m",
        "safety_margin_m",
        "min_projected_coverage_ratio",
        "near_field_horizon_fraction",
    ):
        assert key in planner

    terminal_headings = [
        -value * planner["horizon_m"] * 180.0 / 3.141592653589793
        for value in planner["curvatures"]
    ]
    assert terminal_headings == sorted(terminal_headings)
    assert max(
        after - before for before, after in zip(terminal_headings, terminal_headings[1:])
    ) <= planner["max_candidate_switch_deg"] + 1e-6

    calibration_path = config["camera_calibration"]["path"]
    assert calibration_path
    calibration = load_calibration(ROOT / calibration_path)
    validate_for_live_use(calibration, (576, 1024))
    assert calibration.placeholder is False
    assert calibration.image_shape == (576, 1024)
    assert calibration.camera_matrix.tolist() == [
        [491.1568229675591, 0.0, 509.3800751643276],
        [0.0, 493.9500442214054, 267.4264474521625],
        [0.0, 0.0, 1.0],
    ]
    assert calibration.distortion_coefficients.tolist() == [
        -0.27202200129904086,
        0.0714303303013014,
        -0.000338402093278862,
        0.0010053983916549474,
        -0.008558501568650742,
    ]
    assert calibration.camera_from_rover_transform[1, 3] == 0.145
    assert calibration.camera_from_rover_transform[2, 3] == -0.10


def test_default_profile_stays_image_heuristic_without_mission_config():
    config = load_config(ROOT / "configs/default.yaml")

    assert config["planner"]["geometry_mode"] == "image_heuristic"
    assert config["camera_calibration"]["path"] is None


def test_uncalibrated_live_profile_is_explicitly_speed_and_steering_limited():
    config = load_config(
        ROOT / "configs/default.yaml",
        ROOT / "configs/mission1_uncalibrated_live.yaml",
    )

    mission = config["mission1_autonomy"]
    control = config["control"]
    planner = config["planner"]
    assert mission["require_metric_projection"] is False
    assert mission["enable_search_rotate"] is False
    assert mission["rotate_to_goal_heading_deg"] >= 170.0
    assert (
        0.066 < mission["minimum_rotate_angular"]
        <= mission["rotate_to_goal_angular"]
        <= mission["max_angular"]
    )
    assert mission["minimum_linear"] == mission["max_linear"] == 0.12
    assert mission["minimum_path_score"] == planner["path_score_threshold"]
    assert mission["path_recovery_confirm_frames"] >= 3
    assert control["linear_max"] == mission["max_linear"]
    assert control["angular_max"] == mission["max_angular"] <= 0.15
    assert planner["geometry_mode"] == "image_heuristic"
    assert config["urban"]["waypoint_switch_radius_m"] == 8.0
    assert planner["switch_confirm_count"] >= 3
    assert planner["max_candidate_switch_deg"] <= 10.0
    assert max(abs(value) for value in planner["candidate_headings_deg"]) <= 30
    assert config["navigation"]["max_heading_rate_deg_per_sec"] <= 30.0
    assert config["localization"]["heading_outlier_reject_deg"] == 35.0


def test_default_profile_has_side_sector_and_rotate_escape_disabled():
    config = load_config(ROOT / "configs/default.yaml")

    planner = MotionPrimitivePlannerConfig.from_dict(config["planner"])
    mission = Mission1ControlConfig.from_dict(config)

    assert planner.side_sector_enabled is False
    assert mission.enable_rotate_escape is False
    assert mission.enable_stop_turn_go is False


def test_mission1_live_profile_explicitly_enables_bounded_rotate_escape():
    config = load_config(ROOT / "configs/default.yaml", ROOT / "configs/mission1_live.yaml")

    planner = MotionPrimitivePlannerConfig.from_dict(config["planner"])
    mission = Mission1ControlConfig.from_dict(config)

    assert planner.side_sector_enabled is True
    assert planner.side_sector_top_ratio == 0.55
    assert planner.side_sector_stop_score == 0.15
    assert planner.side_sector_min_traversable_ratio == 0.70
    assert planner.side_sector_margin == 0.05
    assert mission.enable_rotate_escape is True
    assert mission.enable_stop_turn_go is True
    assert mission.enable_search_rotate is False
    assert mission.rotate_escape_direction_confirm_frames >= 2
    assert mission.rotate_escape_stationary_confirm_samples >= 2
    assert mission.rotate_escape_settle_sec >= 0.5
    assert mission.rotate_escape_angular <= mission.max_angular
    assert mission.rotate_escape_motion_response_timeout_sec > mission.rotate_escape_pulse_sec
    assert planner.rover_width_m == 0.156
    assert planner.safety_margin_m == 0.05
    assert planner.rover_width_m + 2.0 * planner.safety_margin_m == 0.256


def test_uncalibrated_live_profile_explicitly_enables_side_sector_and_rotate_escape():
    config = load_config(
        ROOT / "configs/default.yaml",
        ROOT / "configs/mission1_uncalibrated_live.yaml",
    )

    planner = MotionPrimitivePlannerConfig.from_dict(config["planner"])
    mission = Mission1ControlConfig.from_dict(config)

    assert planner.side_sector_enabled is True
    assert mission.enable_rotate_escape is True
    assert mission.rotate_escape_direction_confirm_frames == 2
