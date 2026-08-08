from research.training.audit_frodobots_2k_alignment import (
    _action_label_mismatches,
    _audit_monotonicity,
    _group_indices,
    _nearest_triplet_center,
    _select_action_strips,
    _select_position_strips,
)
from research.training.datasets.frodobots_2k_dataset import ManifestSample


def make_sample(
    ride_id: str,
    frame_id: int,
    timestamp: float,
    action: str = "FORWARD",
    section_id: int = 0,
) -> ManifestSample:
    angular = 0.4 if action == "LEFT" else -0.4 if action == "RIGHT" else 0.0
    return ManifestSample(
        ride_id=ride_id,
        front_playlist_ref=f"ride_{ride_id}/front.m3u8",
        front_segment_ref=f"ride_{ride_id}/front_20240101000000000.ts",
        front_frame_id=frame_id,
        front_timestamp=timestamp,
        matched_control_timestamp=timestamp,
        control_delta_ms=0.0,
        linear=0.2,
        angular=angular,
        action_class=action,
        timeline_section_id=section_id,
    )


def test_position_selection_covers_early_middle_and_late_for_each_ride() -> None:
    samples = tuple(
        make_sample(ride_id, frame_id, 1000.0 + frame_id * 0.05)
        for ride_id in ("1", "2")
        for frame_id in range(20)
    )
    grouped = _group_indices(samples)

    specs = _select_position_strips(samples, grouped, ["1", "2"])

    assert len(specs) == 6
    assert [spec.name.rsplit("_", 1)[1] for spec in specs] == [
        "early",
        "middle",
        "late",
        "early",
        "middle",
        "late",
    ]


def test_action_selection_uses_distinct_rides_when_available() -> None:
    samples = tuple(
        make_sample(ride_id, frame_id, 1000.0 + frame_id * 0.05, action="LEFT")
        for ride_id in ("1", "2", "3")
        for frame_id in range(5)
    )
    grouped = _group_indices(samples)

    specs = _select_action_strips(samples, grouped, "LEFT", 3)

    assert len(specs) == 3
    assert {spec.ride_id for spec in specs} == {"1", "2", "3"}


def test_temporal_strip_requires_consecutive_frame_ids_and_timestamps() -> None:
    samples = (
        make_sample("1", 0, 1000.00),
        make_sample("1", 4, 1000.20),
        make_sample("1", 5, 1000.25),
        make_sample("1", 6, 1000.30),
    )

    triplet = _nearest_triplet_center(samples, [0, 1, 2, 3], target_position=1, require_same_section=True)

    assert triplet == (1, 2, 3)


def test_monotonicity_reports_reversal_and_hls_boundary() -> None:
    samples = (
        make_sample("1", 0, 1000.0),
        make_sample("1", 2, 1000.1),
        make_sample("1", 1, 1000.05, section_id=1),
    )

    audit = _audit_monotonicity(samples, _group_indices(samples))

    assert audit["frame_id_violation_count"] == 1
    assert audit["timestamp_violation_count"] == 1
    assert audit["hls_section_boundary_count"] == 1
    assert audit["hls_section_boundaries"][0]["monotonic"] is False


def test_action_mismatch_is_recomputed_from_existing_thresholds() -> None:
    valid = make_sample("1", 0, 1000.0, action="LEFT")
    invalid = ManifestSample(**{**valid.__dict__, "front_frame_id": 1, "action_class": "RIGHT"})

    mismatches = _action_label_mismatches((valid, invalid))

    assert len(mismatches) == 1
    assert mismatches[0]["expected"] == "LEFT"
    assert mismatches[0]["actual"] == "RIGHT"
