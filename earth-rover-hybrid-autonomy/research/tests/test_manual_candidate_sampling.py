import csv
import json
from pathlib import Path

import cv2
import numpy as np

from research.training.datasets.frodobots_2k_manifest import HlsSegment, HlsTimeline
from research.training.manual_candidate_sampling import (
    FrontFrame,
    RideSource,
    deterministic_ride_order,
    frame_options,
    front_video_playlists,
    is_hash_distinct,
    load_existing_candidates,
    minimum_hash_distance,
    write_numbered_contact_sheets,
)
from research.training.traversability_expansion import difference_hash


def make_ride(tmp_path: Path, dataset: str, ride_id: str) -> RideSource:
    root = tmp_path / dataset
    ride_dir = root / f"ride_{ride_id}_20240101000000"
    playlist = ride_dir / "recordings" / f"ride_{ride_id}_uid_s_1000_video.m3u8"
    timeline = HlsTimeline(
        playlist_reference=playlist.relative_to(root).as_posix(),
        segments=(HlsSegment("segment_20240101000000000.ts", 0.0, 101.0, 0),),
        starts=(0.0,),
        stats={},
    )
    return RideSource(
        dataset=dataset,
        dataset_root=root,
        ride_id=ride_id,
        ride_dir=ride_dir,
        playlist_path=playlist,
        timeline=timeline,
        frames=tuple(FrontFrame(index, float(index * 5)) for index in range(21)),
    )


def test_front_playlist_filter_uses_uid_1000_video_only(tmp_path: Path) -> None:
    recordings = tmp_path / "recordings"
    recordings.mkdir()
    front = recordings / "ride_uid_s_1000_video.m3u8"
    rear = recordings / "ride_uid_s_1001_video.m3u8"
    audio = recordings / "ride_uid_s_1000_audio.m3u8"
    for path in (front, rear, audio):
        path.write_text("#EXTM3U\n", encoding="utf-8")

    assert front_video_playlists(recordings) == [front]


def test_frame_options_are_deterministic_exclude_edges_and_separate_pair(tmp_path: Path) -> None:
    ride = make_ride(tmp_path, "output_rides_1", "10")

    first = frame_options(ride, 10.0, 20.0, 17)
    repeated = frame_options(ride, 10.0, 20.0, 17)

    assert first == repeated
    assert len(first) == 2
    assert all(10.0 <= frame.timestamp <= 90.0 for frame in first)
    assert abs(first[0].timestamp - first[1].timestamp) >= 20.0


def test_ride_order_is_deterministic_and_prefers_no_input_order(tmp_path: Path) -> None:
    rides = [
        make_ride(tmp_path, "output_rides_1", "3"),
        make_ride(tmp_path, "output_rides_1", "1"),
        make_ride(tmp_path, "output_rides_2", "2"),
    ]

    first = deterministic_ride_order(rides, 17)
    repeated = deterministic_ride_order(list(reversed(rides)), 17)

    assert first == repeated
    assert {ride.ride_id for ride in first} == {"1", "2", "3"}


def test_existing_metadata_and_report_supply_rides_and_image_hashes(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    image = image_dir / "existing.jpg"
    assert cv2.imwrite(str(image), np.full((40, 60, 3), 127, dtype=np.uint8))
    metadata = tmp_path / "metadata.csv"
    with metadata.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("ride_id", "image_path"))
        writer.writeheader()
        writer.writerow({"ride_id": "10", "image_path": "images/existing.jpg"})
    report = tmp_path / "selection_report.json"
    report.write_text(json.dumps({"ride_distribution": {"20": 3}}), encoding="utf-8")

    existing = load_existing_candidates([metadata], [report])

    assert existing.ride_ids == frozenset({"10", "20"})
    assert existing.image_hashes == (difference_hash(image),)


def test_hash_gate_rejects_identical_and_reports_minimum_distance(tmp_path: Path) -> None:
    first = tmp_path / "first.jpg"
    second = tmp_path / "second.jpg"
    copy = tmp_path / "copy.jpg"
    gradient = np.tile(np.arange(64, dtype=np.uint8), (64, 1))
    assert cv2.imwrite(str(first), gradient)
    assert cv2.imwrite(str(copy), gradient)
    assert cv2.imwrite(str(second), np.fliplr(gradient))
    first_hash = difference_hash(first)
    second_hash = difference_hash(second)

    assert is_hash_distinct(difference_hash(copy), [first_hash], 5) is False
    assert is_hash_distinct(second_hash, [first_hash], 5) is True
    assert minimum_hash_distance([first_hash, second_hash]) > 5


def test_contact_sheets_group_25_without_modifying_raw_candidates(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    entries = []
    original_bytes = {}
    for index in range(26):
        path = images / f"manual_v2_{index + 1:04d}.jpg"
        assert cv2.imwrite(str(path), np.full((90, 160, 3), index, dtype=np.uint8))
        original_bytes[path] = path.read_bytes()
        entries.append(
            {
                "candidate_id": f"manual_v2_{index + 1:04d}",
                "image_path": f"images/{path.name}",
            }
        )

    sheets = write_numbered_contact_sheets(entries, tmp_path)

    assert sheets == [
        "contact_sheets/contact_sheet_001.jpg",
        "contact_sheets/contact_sheet_002.jpg",
    ]
    assert all((tmp_path / path).is_file() for path in sheets)
    assert all(path.read_bytes() == content for path, content in original_bytes.items())
