import csv
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

from research.training.build_traversability_dataset_v2 import build_dataset_v2
from research.training.import_manual_traversability_v2 import METADATA_FIELDS
from research.training.traversability_dataset_v2 import (
    choose_new_group_split,
    group_id,
    regression_assessment,
)


def test_new_group_split_is_deterministic_and_ride_isolated() -> None:
    rows = [
        {
            "sample_id": f"sample_{index}",
            "dataset": "rides_1",
            "ride_id": str(index // 2),
        }
        for index in range(12)
    ]
    pixels = {
        row["sample_id"]: {
            "IGNORE": 1,
            "ON_ROAD": 10,
            "OFF_ROAD": 10,
            "OBSTACLE": 10,
        }
        for row in rows
    }

    first = choose_new_group_split(rows, pixels, seed=17, trials=500)
    repeated = choose_new_group_split(rows, pixels, seed=17, trials=500)

    assert first == repeated
    assert not set(first["train"]) & set(first["new_holdout"])
    assignment = {
        group: split for split, groups in first.items() for group in groups
    }
    assert all(assignment[group_id(row)] in {"train", "new_holdout"} for row in rows)


def test_regression_gate_flags_fixed_evaluation_degradation() -> None:
    v1 = {"mean_iou": 0.60, "pixel_accuracy": 0.80, "false_traversable_rate": 0.10}
    v2 = {"mean_iou": 0.52, "pixel_accuracy": 0.78, "false_traversable_rate": 0.18}

    result = regression_assessment(v1, v2, 0.05, 0.05, 0.05)

    assert result["pass"] is False
    assert result["failure_reasons"] == [
        "mean_iou_regression",
        "false_traversable_rate_regression",
    ]


def test_v2_builder_preserves_v1_splits_and_separates_new_holdout(tmp_path: Path) -> None:
    v1, manual, contract = write_sources(tmp_path)
    output = tmp_path / "approved_153_v2"

    result = build_dataset_v2(
        v1,
        manual,
        output,
        expected_v1_count=6,
        expected_new_count=6,
        seed=17,
    )

    rows = read_csv(output / "manifest.csv")
    v1_rows = read_csv(v1 / "manifest.csv")
    assignments = {row["sample_id"]: row["split"] for row in rows}
    assert len(rows) == 12
    assert all(assignments[row["sample_id"]] == row["split"] for row in v1_rows)
    new_rows = [row for row in rows if row["source_bundle"] == "manual_v2_33"]
    assert {row["split"] for row in new_rows} == {"train", "new_holdout"}
    new_groups = {}
    for row in new_rows:
        new_groups.setdefault((row["source_dataset"], row["ride_id"]), set()).add(
            row["split"]
        )
    assert all(len(splits) == 1 for splits in new_groups.values())
    assert result["merge"]["label_contract_exact_match"] is True
    assert result["merge"]["exact_image_duplicate_count"] == 0
    assert (output / "label_contract.yaml").read_bytes() == contract.read_bytes()
    assert (output / "fixed_v1_splits/validation.csv").read_bytes() == (
        v1 / "splits/validation.csv"
    ).read_bytes()


def test_v2_builder_rejects_contract_mismatch_and_exact_image_duplicate(
    tmp_path: Path,
) -> None:
    v1, manual, _ = write_sources(tmp_path)
    (manual / "label_contract.yaml").write_text("different: true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not byte-identical"):
        build_dataset_v2(v1, manual, tmp_path / "contract_failure", 6, 6, seed=17)

    shutil.rmtree(v1)
    shutil.rmtree(manual)
    v1, manual, _ = write_sources(tmp_path)
    shutil.copy2(v1 / "images/v1_000.jpg", manual / "images/manual_000.jpg")
    with pytest.raises(ValueError, match="exact v1 image duplicates"):
        build_dataset_v2(v1, manual, tmp_path / "duplicate_failure", 6, 6, seed=17)


def write_sources(tmp_path: Path) -> tuple[Path, Path, Path]:
    repository_contract = (
        Path(__file__).resolve().parents[2] / "research/configs/traversability_dataset_v1.yaml"
    )
    v1 = tmp_path / "v1"
    manual = tmp_path / "manual"
    for root in (v1, manual):
        (root / "images").mkdir(parents=True)
        (root / "masks").mkdir()
        shutil.copy2(repository_contract, root / "label_contract.yaml")
    (v1 / "splits").mkdir()
    (v1 / "classes.yaml").write_text("num_labels: 3\n", encoding="utf-8")

    v1_rows = []
    split_names = ("train", "train", "validation", "validation", "test", "test")
    for index, split in enumerate(split_names):
        sample_id = f"v1_{index:03d}"
        write_pair(v1, sample_id, index + 1)
        v1_rows.append(
            {
                "sample_id": sample_id,
                "image_path": f"images/{sample_id}.jpg",
                "mask_path": f"masks/{sample_id}.png",
                "split": split,
                "ride_id": f"v1_ride_{index}",
                "timestamp": str(1000 + index),
                "frame_id": str(index),
                "manifest_index": str(index),
                "playlist": f"v1/ride_{index}.m3u8",
                "segment": f"v1/segment_{index}.ts",
                "source_bundle": "source",
                "source_image_path": "",
                "source_mask_path": "",
            }
        )
    write_csv(v1 / "manifest.csv", tuple(v1_rows[0]), v1_rows)
    (v1 / "merge_report.json").write_text(
        json.dumps({"valid": True, "sample_count": 6}), encoding="utf-8"
    )
    for split in ("train", "validation", "test"):
        write_csv(
            v1 / "splits" / f"{split}.csv",
            tuple(v1_rows[0]),
            [row for row in v1_rows if row["split"] == split],
        )

    manual_rows = []
    for index in range(6):
        sample_id = f"manual_{index:03d}"
        write_pair(manual, sample_id, index + 20)
        manual_rows.append(
            {
                "sample_id": sample_id,
                "image_path": f"images/{sample_id}.jpg",
                "mask_path": f"masks/{sample_id}.png",
                "dataset": "output_rides_1",
                "ride_id": f"manual_ride_{index // 2}",
                "camera_uid": "1000",
                "timestamp_sec": str(2000 + index * 25),
                "playlist_path": f"manual/ride_{index // 2}.m3u8",
                "source_candidate_id": sample_id,
                "review_status": "APPROVED",
            }
        )
    write_csv(manual / "metadata.csv", METADATA_FIELDS, manual_rows)
    (manual / "validation_report.json").write_text(
        json.dumps(
            {
                "valid": True,
                "sample_count": 6,
                "segmentation_class_mask_count": 6,
                "semantic_mask_source": "SegmentationClass",
                "segmentation_object_used": False,
                "source_image_bytes_preserved": True,
            }
        ),
        encoding="utf-8",
    )
    return v1, manual, repository_contract


def write_pair(root: Path, sample_id: str, value: int) -> None:
    image = np.zeros((12, 20, 3), dtype=np.uint8)
    image[:, : value % 19 + 1] = (value, value * 2, value * 3)
    mask = np.array([[0, 1, 2, 3] * 5] * 12, dtype=np.uint8)
    assert cv2.imwrite(str(root / "images" / f"{sample_id}.jpg"), image)
    assert cv2.imwrite(str(root / "masks" / f"{sample_id}.png"), mask)


def write_csv(
    path: Path,
    fields: tuple[str, ...],
    rows: list[dict[str, str]],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
