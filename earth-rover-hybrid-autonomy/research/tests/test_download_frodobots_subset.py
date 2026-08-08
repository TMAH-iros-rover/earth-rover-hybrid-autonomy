import json
import sys

from research.scripts import download_frodobots_subset as subset


def repo_file(path, size=None):
    return subset.RepoFile(path=path, size=size)


def test_tar_part_exclusion():
    assert subset.is_archive_part("frodobots_dataset.tar.gz")
    assert subset.is_archive_part("frodobots_dataset.tar.gz.part_aa")
    assert subset.is_archive_part("data.zip")
    assert not subset.is_archive_part("videos/ride_1_front_camera.mp4")


def test_video_pair_matching_by_ride_id():
    assert subset.ride_id_from_video("videos/ride_123_front_camera.mp4") == "ride_123"
    assert subset.ride_id_from_video("videos/ride_123_rear_camera.mp4") == "ride_123"
    assert subset.ride_id_from_video("videos/ride_123_map.mp4") is None


def test_budget_selection_prefers_front_rear_pairs_under_budget():
    files = [
        repo_file("videos/ride_a_front_camera.mp4", 4),
        repo_file("videos/ride_a_rear_camera.mp4", 4),
        repo_file("videos/ride_b_front_camera.mp4", 8),
        repo_file("videos/ride_b_rear_camera.mp4", 8),
    ]

    selected, warnings = subset.select_video_subset(files, budget_bytes=10, sample_videos=8, prefer_pairs=True)

    assert warnings == []
    assert [item.path for item in selected] == [
        "videos/ride_a_front_camera.mp4",
        "videos/ride_a_rear_camera.mp4",
    ]


def test_zarr_metadata_parsing_estimates_logical_size():
    parsed = subset.parse_zarr_array_metadata(
        "frodobots_dataset/dataset_cache.zarr/action",
        {
            "shape": [10, 2],
            "chunks": [5, 2],
            "dtype": "<f4",
            "compressor": {"id": "blosc"},
        },
    )

    assert parsed["shape"] == [10, 2]
    assert parsed["chunks"] == [5, 2]
    assert parsed["dtype"] == "<f4"
    assert parsed["compressor"] == "blosc"
    assert parsed["estimated_logical_size"] == 80


def test_dry_run_manifest_generation(tmp_path, monkeypatch):
    fake_files = [
        repo_file("frodobots_dataset.tar.gz.part_aa", 100),
        repo_file("meta_data/info.json", 10),
        repo_file("train/dataset_info.json", 10),
        repo_file("frodobots_dataset/dataset_cache.zarr/action/.zarray", 10),
        repo_file("videos/ride_a_front_camera.mp4", 20),
        repo_file("videos/ride_a_rear_camera.mp4", 20),
    ]
    monkeypatch.setattr(subset, "list_repo_dataset_files", lambda repo_id, token=None: fake_files)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "download_frodobots_subset.py",
            "--repo-id",
            "BitRobot/Berkeley-FrodoBots-7K",
            "--output-dir",
            str(tmp_path),
            "--budget-gb",
            "2",
            "--sample-videos",
            "1",
            "--prefer-front-rear-pairs",
            "--dry-run",
            "--no-tar-parts",
        ],
    )

    assert subset.main() == 0

    manifest_path = tmp_path / "reports" / "subset_manifest_dry_run.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected_paths = [item["path"] for item in payload["selected_files"]]
    assert "frodobots_dataset.tar.gz.part_aa" not in selected_paths
    assert "meta_data/info.json" in selected_paths
    assert "videos/ride_a_front_camera.mp4" in selected_paths
    assert "videos/ride_a_rear_camera.mp4" in selected_paths
