import csv
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from research.training.datasets.frodobots_2k_dataset import FrameDecodeError, ManifestSample
from research.training.traversability_review import (
    REVIEW_COLUMNS,
    SelectedFrame,
    decode_selected_frame,
    label_contract,
    letterbox_rgb,
    restore_from_letterbox,
    select_representative_frames,
    semantic_to_traversability,
    validate_review_bundle,
    write_gallery,
    write_review_csv,
)


def make_sample(ride_id: str, frame_id: int, timestamp: float) -> ManifestSample:
    return ManifestSample(
        ride_id=ride_id,
        front_playlist_ref=f"ride_{ride_id}/front.m3u8",
        front_segment_ref=f"ride_{ride_id}/front_20240101000000000.ts",
        front_frame_id=frame_id,
        front_timestamp=timestamp,
        matched_control_timestamp=timestamp,
        control_delta_ms=0.0,
        linear=0.2,
        angular=0.0,
        action_class="FORWARD",
        timeline_section_id=0,
    )


def make_samples(ride_count: int = 4, frames_per_ride: int = 20) -> tuple[ManifestSample, ...]:
    return tuple(
        make_sample(str(ride), frame, ride * 1000.0 + frame)
        for ride in range(ride_count)
        for frame in range(frames_per_ride)
    )


def write_contract(path: Path) -> None:
    value = {
        "ignore_index": 2,
        "classes": [
            {"id": 0, "name": "NON_TRAVERSABLE", "color_rgb": [220, 50, 47]},
            {"id": 1, "name": "TRAVERSABLE", "color_rgb": [38, 166, 91]},
            {"id": 2, "name": "UNKNOWN_OR_IGNORE", "color_rgb": [245, 190, 48]},
        ],
    }
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def test_representative_sampling_is_deterministic_cross_ride_and_time_separated() -> None:
    samples = make_samples()

    first = select_representative_frames(samples, 12, 3, 3.0, seed=7)
    repeated = select_representative_frames(samples, 12, 3, 3.0, seed=7)

    assert first == repeated
    assert len(first) == 12
    assert len({item.sample.ride_id for item in first}) == 3
    assert all(samples[item.manifest_index] == item.sample for item in first)
    for ride_id in {item.sample.ride_id for item in first}:
        timestamps = sorted(item.sample.front_timestamp for item in first if item.sample.ride_id == ride_id)
        assert all(right - left >= 3.0 for left, right in zip(timestamps, timestamps[1:]))


def test_representative_sampling_rejects_invalid_arguments() -> None:
    with pytest.raises(ValueError, match="positive"):
        select_representative_frames(make_samples(), 0, 3, 1.0, seed=7)


def test_decode_failure_is_reported_without_substituting_a_frame() -> None:
    selected = SelectedFrame(4, make_sample("7", 4, 1004.0))

    class BrokenDecoder:
        def decode(self, sample: ManifestSample) -> np.ndarray:
            raise FrameDecodeError("missing segment")

    image, failure = decode_selected_frame(BrokenDecoder(), selected)

    assert image is None
    assert failure == {"manifest_index": 4, "ride_id": "7", "error": "missing segment"}


def test_letterbox_preserves_aspect_ratio_and_restores_original_shape() -> None:
    image = np.zeros((576, 1024, 3), dtype=np.uint8)

    padded, padding = letterbox_rgb(image, 512)
    restored = restore_from_letterbox(
        np.zeros((512, 512), dtype=np.uint8),
        padding,
        image.shape[:2],
        cv2.INTER_NEAREST,
    )

    assert padded.shape == (512, 512, 3)
    assert padding == (112, 112, 0, 0)
    assert restored.shape == (576, 1024)


def test_semantic_mapping_defaults_low_confidence_and_road_to_unknown(tmp_path: Path) -> None:
    contract_path = tmp_path / "contract.yaml"
    write_contract(contract_path)
    name_to_id, _, _ = label_contract(contract_path)
    semantic = np.array([[11, 0, 6, 11]], dtype=np.uint8)
    confidence = np.array([[0.9, 0.9, 0.9, 0.2]], dtype=np.float32)
    mapping = {
        "confidence_threshold": 0.7,
        "default_class": "UNKNOWN_OR_IGNORE",
        "mapping": {
            "TRAVERSABLE": ["sidewalk"],
            "NON_TRAVERSABLE": ["wall"],
            "UNKNOWN_OR_IGNORE": ["road"],
        },
    }

    result = semantic_to_traversability(
        semantic,
        confidence,
        {0: "wall", 6: "road", 11: "sidewalk"},
        mapping,
        name_to_id,
    )

    assert result.tolist() == [[1, 0, 2, 2]]


def test_unreviewed_bundle_is_valid_but_exports_no_training_samples(tmp_path: Path) -> None:
    write_contract(tmp_path / "label_contract.yaml")
    write_bundle_assets(tmp_path)
    entry = review_entry()
    write_review_csv([entry], tmp_path / "review.csv")

    report = validate_review_bundle(tmp_path, tmp_path / "verified.csv")

    assert report["valid"] is True
    assert report["verified_training_sample_count"] == 0
    assert list(csv.DictReader((tmp_path / "verified.csv").open())) == []


def test_corrected_mask_takes_precedence_for_reviewed_export(tmp_path: Path) -> None:
    write_contract(tmp_path / "label_contract.yaml")
    write_bundle_assets(tmp_path)
    (tmp_path / "corrected_masks").mkdir()
    corrected_rgb = np.full((12, 20, 3), (38, 166, 91), dtype=np.uint8)
    cv2.imwrite(
        str(tmp_path / "corrected_masks/sample.png"),
        cv2.cvtColor(corrected_rgb, cv2.COLOR_RGB2BGR),
    )
    entry = review_entry()
    entry.update(
        {
            "review_status": "NEEDS_CORRECTION",
            "usable_for_training": "true",
            "corrected_mask_path": "corrected_masks/sample.png",
            "split": "train",
        }
    )
    write_review_csv([entry], tmp_path / "review.csv")

    report = validate_review_bundle(tmp_path, tmp_path / "verified.csv")
    exported = list(csv.DictReader((tmp_path / "verified.csv").open()))

    assert report["valid"] is True
    assert exported[0]["verified_mask_path"] == "corrected_masks/sample.png"


def test_validator_rejects_missing_files_and_invalid_mask_ids(tmp_path: Path) -> None:
    write_contract(tmp_path / "label_contract.yaml")
    write_bundle_assets(tmp_path)
    cv2.imwrite(str(tmp_path / "traversability_masks/sample.png"), np.full((12, 20), 9, dtype=np.uint8))
    write_review_csv([review_entry()], tmp_path / "review.csv")

    report = validate_review_bundle(tmp_path)

    assert report["valid"] is False
    assert "unsupported class ID" in report["errors"][0]


def test_validator_rejects_a_missing_review_asset(tmp_path: Path) -> None:
    write_contract(tmp_path / "label_contract.yaml")
    write_bundle_assets(tmp_path)
    (tmp_path / "confidence/sample.png").unlink()
    write_review_csv([review_entry()], tmp_path / "review.csv")

    report = validate_review_bundle(tmp_path)

    assert report["valid"] is False
    assert "does not exist" in report["errors"][0]


def test_gallery_uses_only_relative_bundle_assets(tmp_path: Path) -> None:
    entry = review_entry()
    entry.update({"mean_confidence": "0.75", "class_distribution": "UNKNOWN=1.0"})

    write_gallery([entry], tmp_path / "gallery.html")
    text = (tmp_path / "gallery.html").read_text(encoding="utf-8")

    assert "images/sample.jpg" in text
    assert "file://" not in text
    assert "/home/" not in text


def review_entry() -> dict[str, str]:
    entry = {column: "" for column in REVIEW_COLUMNS}
    entry.update(
        {
            "sample_id": "sample",
            "image_path": "images/sample.jpg",
            "semantic_mask_path": "semantic_masks/sample.png",
            "semantic_overlay_path": "semantic_overlays/sample.jpg",
            "traversability_mask_path": "traversability_masks/sample.png",
            "traversability_overlay_path": "traversability_overlays/sample.jpg",
            "confidence_path": "confidence/sample.png",
            "ride_id": "7",
            "frame_id": "4",
            "timestamp": "1004.0",
            "manifest_index": "4",
            "playlist": "ride_7/front.m3u8",
            "action_label": "FORWARD",
            "linear": "0.2",
            "angular": "0.0",
            "selection_method": "ride_balanced_temporal_random",
            "split": "UNASSIGNED",
            "review_status": "UNREVIEWED",
            "usable_for_training": "false",
        }
    )
    return entry


def write_bundle_assets(root: Path) -> None:
    image = np.zeros((12, 20, 3), dtype=np.uint8)
    mask = np.full((12, 20), 2, dtype=np.uint8)
    for directory in (
        "images",
        "semantic_masks",
        "semantic_overlays",
        "traversability_masks",
        "traversability_overlays",
        "confidence",
        "metadata",
    ):
        (root / directory).mkdir()
    cv2.imwrite(str(root / "images/sample.jpg"), image)
    cv2.imwrite(str(root / "semantic_masks/sample.png"), mask)
    cv2.imwrite(str(root / "semantic_overlays/sample.jpg"), image)
    cv2.imwrite(str(root / "traversability_masks/sample.png"), mask)
    cv2.imwrite(str(root / "traversability_overlays/sample.jpg"), image)
    cv2.imwrite(str(root / "confidence/sample.png"), image)
    (root / "metadata/sample.json").write_text('{"sample_id": "sample"}\n', encoding="utf-8")
