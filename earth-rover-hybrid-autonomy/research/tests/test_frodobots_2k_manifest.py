from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from research.training.datasets.frodobots_2k_manifest import (
    DatasetFormatError,
    build_manifest,
    load_control_timeline,
    match_nearest_control,
    normalize_timestamp,
    parse_front_hls_playlist,
)


CONTROL_HEADER = ["linear", "angular", "rpm_1", "rpm_2", "rpm_3", "rpm_4", "timestamp"]


def write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def write_playlist(path: Path, starts: list[str], durations: list[float], discontinuity_at: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["#EXTM3U", "#EXT-X-VERSION:3"]
    for index, (start, duration) in enumerate(zip(starts, durations)):
        if discontinuity_at == index:
            lines.append("#EXT-X-DISCONTINUITY")
        segment_name = f"sample_ride_1__uid_s_1000__uid_e_video_{start}.ts"
        lines.extend(
            [
                f"#EXTINF:{duration:.6f}",
                segment_name,
            ]
        )
        (path.parent / segment_name).touch()
    lines.append("#EXT-X-ENDLIST")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_ride(
    dataset_root: Path,
    ride_id: int,
    front_rows: list[list[object]],
    control_rows: list[list[object]],
) -> Path:
    ride = dataset_root / f"ride_{ride_id}_20240101000000"
    write_csv(ride / f"front_camera_timestamps_{ride_id}.csv", ["frame_id", "timestamp"], front_rows)
    write_csv(ride / f"control_data_{ride_id}.csv", CONTROL_HEADER, control_rows)
    write_playlist(
        ride / "recordings" / f"sample_ride_{ride_id}__uid_s_1000__uid_e_video.m3u8",
        ["20240101000000000"],
        [20.0],
    )
    return ride


def control_row(linear: object, angular: object, timestamp: object, rpm: object = 0) -> list[object]:
    return [linear, angular, rpm, rpm, rpm, rpm, timestamp]


def test_normalize_seconds_to_seconds() -> None:
    assert normalize_timestamp("1704067200.125", "seconds") == pytest.approx(1704067200.125)


def test_normalize_milliseconds_to_seconds() -> None:
    assert normalize_timestamp("1704067200125", "milliseconds") == pytest.approx(1704067200.125)


def test_nearest_control_prefers_closest_and_earlier_on_tie(tmp_path: Path) -> None:
    path = tmp_path / "control.csv"
    write_csv(
        path,
        CONTROL_HEADER,
        [control_row(0.2, 0.0, 10.0), control_row(0.4, 0.0, 10.2)],
    )
    timeline = load_control_timeline(path)

    match, reason = match_nearest_control(timeline, 10.1, tolerance_sec=0.1)

    assert reason is None
    assert match is not None
    assert match.control.timestamp == pytest.approx(10.0)
    assert match.delta_ms == pytest.approx(100.0)


def test_nearest_control_rejects_outside_tolerance(tmp_path: Path) -> None:
    path = tmp_path / "control.csv"
    write_csv(path, CONTROL_HEADER, [control_row(0.2, 0.0, 10.0)])

    match, reason = match_nearest_control(load_control_timeline(path), 10.101, tolerance_sec=0.1)

    assert match is None
    assert reason == "control_delta_exceeds_tolerance"


def test_identical_duplicate_control_timestamp_is_collapsed(tmp_path: Path) -> None:
    path = tmp_path / "control.csv"
    row = control_row(0.2, -0.1, 10.0, rpm=4)
    write_csv(path, CONTROL_HEADER, [row, row])

    timeline = load_control_timeline(path)
    match, reason = match_nearest_control(timeline, 10.0, tolerance_sec=0.1)

    assert reason is None
    assert match is not None
    assert timeline.stats["identical_duplicate_groups"] == 1
    assert timeline.stats["identical_duplicate_rows_removed"] == 1


def test_conflicting_duplicate_control_timestamp_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "control.csv"
    write_csv(
        path,
        CONTROL_HEADER,
        [control_row(0.2, 0.0, 10.0), control_row(0.0, 0.5, 10.0)],
    )

    timeline = load_control_timeline(path)
    match, reason = match_nearest_control(timeline, 10.0, tolerance_sec=0.1)

    assert match is None
    assert reason == "conflicting_control_duplicate"
    assert timeline.stats["conflicting_duplicate_groups"] == 1


def test_malformed_control_anchor_is_not_bypassed(tmp_path: Path) -> None:
    path = tmp_path / "control.csv"
    write_csv(
        path,
        CONTROL_HEADER,
        [
            control_row(0.2, 0.0, 9.9),
            control_row(0.2, "bad-angular", 10.0),
            control_row(0.2, 0.0, 10.1),
        ],
    )

    match, reason = match_nearest_control(load_control_timeline(path), 10.0, tolerance_sec=0.1)

    assert match is None
    assert reason == "malformed_control_row"


def test_reversed_control_regions_are_segmented_and_overlap_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "control.csv"
    write_csv(
        path,
        CONTROL_HEADER,
        [
            control_row(0.1, 0.0, 10.0),
            control_row(0.2, 0.0, 11.0),
            control_row(0.3, 0.0, 10.5),
            control_row(0.4, 0.0, 12.0),
        ],
    )

    timeline = load_control_timeline(path)
    match, reason = match_nearest_control(timeline, 10.75, tolerance_sec=0.1)

    assert timeline.stats["timestamp_reversals"] == 1
    assert len(timeline.sections) == 2
    assert match is None
    assert reason == "ambiguous_control_section"


def test_explicit_hls_discontinuity_creates_new_section(tmp_path: Path) -> None:
    playlist = tmp_path / "front.m3u8"
    write_playlist(
        playlist,
        ["20240101000000000", "20240101000005000"],
        [5.0, 5.0],
        discontinuity_at=1,
    )

    timeline = parse_front_hls_playlist(playlist, tmp_path)

    assert [segment.section_id for segment in timeline.segments] == [0, 1]
    assert timeline.stats["explicit_discontinuities"] == 1


def test_inferred_hls_gap_creates_new_section(tmp_path: Path) -> None:
    playlist = tmp_path / "front.m3u8"
    write_playlist(
        playlist,
        ["20240101000000000", "20240101000006000"],
        [5.0, 5.0],
    )

    timeline = parse_front_hls_playlist(playlist, tmp_path)

    assert [segment.section_id for segment in timeline.segments] == [0, 1]
    assert timeline.stats["inferred_discontinuities"] == 1


def test_missing_hls_segment_is_rejected(tmp_path: Path) -> None:
    playlist = tmp_path / "front.m3u8"
    playlist.write_text(
        "\n".join(
            [
                "#EXTM3U",
                "#EXTINF:5.000000",
                "sample_ride_1__uid_s_1000__uid_e_video_20240101000000000.ts",
                "#EXT-X-ENDLIST",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(DatasetFormatError, match="missing HLS segment"):
        parse_front_hls_playlist(playlist, tmp_path)


def test_non_monotonic_hls_segment_starts_are_rejected(tmp_path: Path) -> None:
    playlist = tmp_path / "front.m3u8"
    write_playlist(
        playlist,
        ["20240101000005000", "20240101000000000"],
        [5.0, 5.0],
    )

    with pytest.raises(DatasetFormatError, match="non-monotonic HLS"):
        parse_front_hls_playlist(playlist, tmp_path)


def test_builder_never_matches_control_from_another_ride(tmp_path: Path) -> None:
    dataset_root = tmp_path / "raw"
    output_dir = tmp_path / "artifacts"
    base = 1704067200.0
    make_ride(dataset_root, 1, [[0, base]], [control_row(0.2, 0.0, base + 1.0)])
    make_ride(dataset_root, 2, [[0, base]], [control_row(0.0, 0.5, base)])

    report = build_manifest(dataset_root, output_dir, max_rides=2, tolerance_ms=100)

    assert report["processed_ride_count"] == 2
    assert report["valid_sample_count"] == 1
    assert report["rejected_sample_count"] == 1
    assert report["rejection_reasons"]["control_delta_exceeds_tolerance"] == 1
    with (output_dir / "manifest.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["ride_id"] == "2"
    assert rows[0]["action_class"] == "LEFT"


def test_builder_reports_malformed_front_row(tmp_path: Path) -> None:
    dataset_root = tmp_path / "raw"
    output_dir = tmp_path / "artifacts"
    base = 1704067200.0
    make_ride(dataset_root, 1, [[0, "bad-timestamp"], [1, base]], [control_row(0.2, 0.0, base)])

    report = build_manifest(dataset_root, output_dir, max_rides=1, tolerance_ms=100)

    assert report["total_front_frame_count"] == 2
    assert report["valid_sample_count"] == 1
    assert report["rejection_reasons"]["malformed_front_row"] == 1
    saved_report = json.loads((output_dir / "alignment_report.json").read_text(encoding="utf-8"))
    assert saved_report["manifest_path"] == str((output_dir / "manifest.csv").resolve())


def test_builder_rejects_control_across_hls_section_boundary(tmp_path: Path) -> None:
    dataset_root = tmp_path / "raw"
    output_dir = tmp_path / "artifacts"
    base = 1704067200.0
    ride = make_ride(dataset_root, 1, [[0, base + 5.0]], [control_row(0.2, 0.0, base + 4.95)])
    playlist = next((ride / "recordings").glob("*.m3u8"))
    write_playlist(
        playlist,
        ["20240101000000000", "20240101000005000"],
        [5.0, 5.0],
        discontinuity_at=1,
    )

    report = build_manifest(dataset_root, output_dir, max_rides=1, tolerance_ms=100)

    assert report["valid_sample_count"] == 0
    assert report["rejection_reasons"]["hls_section_mismatch"] == 1


def test_manifest_is_deterministic_for_same_inputs(tmp_path: Path) -> None:
    dataset_root = tmp_path / "raw"
    output_dir = tmp_path / "artifacts"
    base = 1704067200.0
    make_ride(dataset_root, 1, [[0, base]], [control_row(0.2, 0.0, base)])

    first = build_manifest(dataset_root, output_dir, max_rides=1, tolerance_ms=100)
    second = build_manifest(dataset_root, output_dir, max_rides=1, tolerance_ms=100)

    assert first["manifest_sha256"] == second["manifest_sha256"]


def test_builder_refuses_to_write_inside_raw_dataset(tmp_path: Path) -> None:
    dataset_root = tmp_path / "raw"
    base = 1704067200.0
    make_ride(dataset_root, 1, [[0, base]], [control_row(0.2, 0.0, base)])

    with pytest.raises(ValueError, match="outside the immutable raw dataset"):
        build_manifest(dataset_root, dataset_root / "generated", max_rides=1)


def test_builder_rejects_non_finite_tolerance(tmp_path: Path) -> None:
    dataset_root = tmp_path / "raw"
    base = 1704067200.0
    make_ride(dataset_root, 1, [[0, base]], [control_row(0.2, 0.0, base)])

    with pytest.raises(ValueError, match="finite and non-negative"):
        build_manifest(dataset_root, tmp_path / "artifacts", tolerance_ms=float("nan"))


def test_empty_or_malformed_control_csv_is_rejected(tmp_path: Path) -> None:
    empty = tmp_path / "empty.csv"
    write_csv(empty, CONTROL_HEADER, [])
    with pytest.raises(DatasetFormatError, match="no control rows"):
        load_control_timeline(empty)

    malformed = tmp_path / "malformed.csv"
    write_csv(malformed, ["linear", "timestamp"], [[0.2, 10.0]])
    with pytest.raises(DatasetFormatError, match="missing columns"):
        load_control_timeline(malformed)
