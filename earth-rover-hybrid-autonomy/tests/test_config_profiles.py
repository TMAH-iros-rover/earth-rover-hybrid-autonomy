from pathlib import Path

from earth_rover.utils.config import load_config


ROOT = Path(__file__).resolve().parents[1]


def test_latency_2s_profile_loads_without_changing_default_file():
    config = load_config(ROOT / "configs/default.yaml", ROOT / "configs/urban_latency_2s.yaml")

    assert config["project"]["profile"] == "latency_2s"
    assert config["latency"]["sensor_delay_sec"] == 2.0
    assert config["latency"]["frame_delay_sec"] == 2.0
    assert config["latency"]["data_delay_sec"] == 2.0
    assert config["control"]["linear_max"] == 0.22
    assert config["safety"]["frame_timeout_sec"] == 3.0


def test_traversability_replay_profile_is_log_only_and_configurable():
    config = load_config(ROOT / "configs/default.yaml", ROOT / "configs/urban_replay_v2.yaml")

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
    assert mission["base_linear"] <= mission["max_linear"] <= 0.15
    assert control["linear_max"] == mission["max_linear"]
    assert control["angular_max"] == mission["max_angular"] <= 0.30
