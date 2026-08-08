from __future__ import annotations

import numpy as np
import pytest

from earth_rover.perception.traversability_adapter import TraversabilityAdapter


def config() -> dict:
    return {
        "traversability_adapter": {
            "roi_top": 0.25,
            "near_top": 0.70,
            "sector_boundaries": [0.0, 0.33, 0.67, 1.0],
            "near_weight": 2.0,
            "on_road_reward": 1.0,
            "off_road_reward": 0.65,
            "obstacle_penalty": 1.0,
            "uncertainty_penalty": 0.55,
            "low_confidence_threshold": 0.55,
            "missing_confidence": 0.5,
            "near_obstacle_stop_ratio": 0.45,
            "confidence_stop_threshold": 0.25,
        }
    }


def test_all_on_road_prefers_center() -> None:
    result = TraversabilityAdapter(config()).adapt(
        np.ones((60, 90), dtype=np.uint8),
        np.ones((60, 90), dtype=np.float32),
    )

    assert result.recommended_direction == "CENTER"
    assert result.center_score == pytest.approx(1.0)
    assert result.stop_recommended is False


def test_off_road_is_traversable_but_less_preferred_than_on_road() -> None:
    adapter = TraversabilityAdapter(config())
    confidence = np.ones((60, 90), dtype=np.float32)

    on_road = adapter.adapt(np.ones((60, 90), dtype=np.uint8), confidence)
    off_road = adapter.adapt(np.full((60, 90), 2, dtype=np.uint8), confidence)

    assert 0.0 < off_road.center_score < on_road.center_score
    assert off_road.center_obstacle_ratio == 0.0


def test_center_near_obstacle_recommends_stop() -> None:
    mask = np.ones((60, 90), dtype=np.uint8)
    mask[42:, 30:60] = 3

    result = TraversabilityAdapter(config()).adapt(mask, np.ones_like(mask, dtype=np.float32))

    assert result.near_obstacle_ratio == pytest.approx(1.0)
    assert result.stop_recommended is True
    assert result.recommended_direction == "STOP"


def test_missing_confidence_uses_configured_conservative_fallback() -> None:
    result = TraversabilityAdapter(config()).adapt(np.ones((60, 90), dtype=np.uint8))

    assert result.mean_confidence == pytest.approx(0.5)
    assert "CONFIDENCE_UNAVAILABLE" in result.reason
    assert result.center_score < 1.0


def test_invalid_mask_and_confidence_are_rejected() -> None:
    adapter = TraversabilityAdapter(config())
    with pytest.raises(ValueError, match="unsupported IDs"):
        adapter.adapt(np.full((10, 10), 4, dtype=np.uint8))
    with pytest.raises(ValueError, match="shape"):
        adapter.adapt(np.ones((10, 10), dtype=np.uint8), np.ones((9, 10), dtype=np.float32))
