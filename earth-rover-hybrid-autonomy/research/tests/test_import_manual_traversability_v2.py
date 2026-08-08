import csv
import hashlib
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from research.training.import_manual_traversability_v2 import import_manual_bundle
from research.training.select_manual_candidates_v2 import SELECTED_FIELDS


LABELMAP = """# label:color_rgb:parts:actions
IGNORE:0,0,0::
OBSTACLE:220,50,47::
OFF_ROAD:43,126,216::
ON_ROAD:38,166,91::
background:0,0,0::
"""


def write_inputs(tmp_path: Path, count: int = 2, labelmap: str = LABELMAP) -> tuple[Path, Path, Path]:
    source = tmp_path / "selected"
    images = source / "images"
    images.mkdir(parents=True)
    ids = [f"manual_v2_{number:04d}" for number in range(1, count + 1)]
    with (source / "selected_candidates.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SELECTED_FIELDS)
        writer.writeheader()
        for number, candidate_id in enumerate(ids, start=1):
            image = np.full((12, 20, 3), 30 * number, dtype=np.uint8)
            assert cv2.imwrite(str(images / f"{candidate_id}.jpg"), image)
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
    (source / "selection.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
    export = source / "export.zip"
    rgb = np.array(
        [
            [(0, 0, 0), (38, 166, 91), (43, 126, 216), (220, 50, 47)],
        ]
        * 12,
        dtype=np.uint8,
    )
    rgb = cv2.resize(rgb, (20, 12), interpolation=cv2.INTER_NEAREST)
    ok, class_png = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    assert ok
    ok, object_png = cv2.imencode(".png", np.full((12, 20), 3, dtype=np.uint8))
    assert ok
    with zipfile.ZipFile(export, "w") as archive:
        archive.writestr("labelmap.txt", labelmap)
        for candidate_id in ids:
            archive.writestr(f"SegmentationClass/{candidate_id}.png", class_png.tobytes())
            archive.writestr(f"SegmentationObject/{candidate_id}.png", object_png.tobytes())
    contract = Path(__file__).resolve().parents[2] / "research/configs/traversability_dataset_v1.yaml"
    return source, export, contract


def test_manual_import_preserves_images_and_uses_segmentation_class_only(tmp_path: Path) -> None:
    source, export, contract = write_inputs(tmp_path)
    before = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (source / "images").glob("*.jpg")
    }
    output = tmp_path / "imported"

    report = import_manual_bundle(source, export, output, contract, expected_count=2)

    assert report["valid"] is True
    assert report["segmentation_class_mask_count"] == 2
    assert report["semantic_mask_source"] == "SegmentationClass"
    assert report["segmentation_object_used"] is False
    rows = list(csv.DictReader((output / "metadata.csv").open(newline="", encoding="utf-8")))
    assert [row["sample_id"] for row in rows] == ["manual_v2_0001", "manual_v2_0002"]
    for row in rows:
        source_image = source / row["image_path"]
        output_image = output / row["image_path"]
        assert output_image.read_bytes() == source_image.read_bytes()
        mask = cv2.imread(str(output / row["mask_path"]), cv2.IMREAD_UNCHANGED)
        assert mask is not None and mask.ndim == 2
        assert set(int(value) for value in np.unique(mask)) == {0, 1, 2, 3}
    after = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (source / "images").glob("*.jpg")
    }
    assert after == before
    assert report["contact_sheets"] == ["contact_sheets/overlay_contact_sheet_001.jpg"]


def test_contact_sheets_are_split_at_25_images(tmp_path: Path) -> None:
    source, export, contract = write_inputs(tmp_path, count=26)

    report = import_manual_bundle(source, export, tmp_path / "imported", contract, expected_count=26)

    assert report["contact_sheets"] == [
        "contact_sheets/overlay_contact_sheet_001.jpg",
        "contact_sheets/overlay_contact_sheet_002.jpg",
    ]


def test_labelmap_contract_mismatch_stops_without_output(tmp_path: Path) -> None:
    bad = LABELMAP.replace("ON_ROAD:38,166,91", "ON_ROAD:1,2,3")
    source, export, contract = write_inputs(tmp_path, labelmap=bad)
    output = tmp_path / "imported"

    with pytest.raises(ValueError, match="uses RGB"):
        import_manual_bundle(source, export, output, contract, expected_count=2)

    assert not output.exists()


def test_missing_mask_and_existing_output_are_rejected(tmp_path: Path) -> None:
    source, export, contract = write_inputs(tmp_path)
    missing_export = source / "missing.zip"
    with zipfile.ZipFile(export) as source_zip, zipfile.ZipFile(missing_export, "w") as target:
        for name in source_zip.namelist():
            if name != "SegmentationClass/manual_v2_0002.png":
                target.writestr(name, source_zip.read(name))
    with pytest.raises(ValueError, match="mask set mismatch"):
        import_manual_bundle(source, missing_export, tmp_path / "missing", contract, expected_count=2)

    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="output path already exists"):
        import_manual_bundle(source, export, output, contract, expected_count=2)
    assert marker.read_text(encoding="utf-8") == "keep"
