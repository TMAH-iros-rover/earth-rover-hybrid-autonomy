import csv
import hashlib
import zipfile
from pathlib import Path

import pytest

from research.training.select_manual_candidates_v2 import (
    SELECTED_FIELDS,
    create_selected_bundle,
    normalized_candidate_ids,
)


def write_source_bundle(root: Path, count: int = 3) -> dict[str, bytes]:
    images = root / "images"
    images.mkdir(parents=True)
    contents = {}
    with (root / "candidates.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SELECTED_FIELDS)
        writer.writeheader()
        for number in range(1, count + 1):
            candidate_id = f"manual_v2_{number:04d}"
            content = b"jpeg-bytes-" + bytes([number])
            image = images / f"{candidate_id}.jpg"
            image.write_bytes(content)
            contents[candidate_id] = content
            writer.writerow(
                {
                    "candidate_id": candidate_id,
                    "dataset": "output_rides_1",
                    "ride_id": str(17000 + number),
                    "camera_uid": "1000",
                    "timestamp_sec": f"{number}.000000",
                    "playlist_path": f"ride_{number}/uid_s_1000_video.m3u8",
                    "image_path": f"images/{candidate_id}.jpg",
                }
            )
    return contents


def test_candidate_numbers_are_normalized_to_four_digits() -> None:
    assert normalized_candidate_ids([1, 33, 180]) == [
        "manual_v2_0001",
        "manual_v2_0033",
        "manual_v2_0180",
    ]


def test_missing_or_duplicate_selection_is_rejected_before_output(tmp_path: Path) -> None:
    source = tmp_path / "source"
    write_source_bundle(source)

    with pytest.raises(ValueError, match="must be unique"):
        create_selected_bundle(source, tmp_path / "duplicate", [1, 1])
    with pytest.raises(ValueError, match="missing from candidates.csv"):
        create_selected_bundle(source, tmp_path / "missing", [1, 4])

    assert not (tmp_path / "duplicate").exists()
    assert not (tmp_path / "missing").exists()


def test_selected_bundle_preserves_bytes_and_zip_has_only_top_level_jpgs(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    contents = write_source_bundle(source)
    before = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source.rglob("*")
        if path.is_file()
    }
    output = tmp_path / "selected"

    result = create_selected_bundle(source, output, [3, 1])

    rows = list(
        csv.DictReader(
            (output / "selected_candidates.csv").open(newline="", encoding="utf-8")
        )
    )
    assert [row["candidate_id"] for row in rows] == ["manual_v2_0003", "manual_v2_0001"]
    assert (output / "selection.txt").read_text(encoding="utf-8").splitlines() == [
        "manual_v2_0003",
        "manual_v2_0001",
    ]
    for candidate_id in ("manual_v2_0003", "manual_v2_0001"):
        assert (output / "images" / f"{candidate_id}.jpg").read_bytes() == contents[candidate_id]

    archive_path = Path(result["archive_path"])
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.namelist() == ["manual_v2_0003.jpg", "manual_v2_0001.jpg"]
        assert archive.read("manual_v2_0003.jpg") == contents["manual_v2_0003"]
        assert archive.read("manual_v2_0001.jpg") == contents["manual_v2_0001"]
    after = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_existing_output_is_not_overwritten(tmp_path: Path) -> None:
    source = tmp_path / "source"
    write_source_bundle(source)
    output = tmp_path / "selected"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(ValueError, match="output path already exists"):
        create_selected_bundle(source, output, [1])

    assert marker.read_text(encoding="utf-8") == "keep"
