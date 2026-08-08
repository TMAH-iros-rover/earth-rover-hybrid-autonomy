import csv
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
import torch.nn.functional as functional
from torch import nn
from torch.utils.data import DataLoader

from research.training.build_approved_traversability_dataset_v1 import FIELDS, _write_dataset_manifests
from research.training.datasets.traversability_dataset_v1 import (
    TraversabilityDataset,
    choose_ride_split,
    letterbox_image_and_mask,
    source_mask_to_training,
    training_prediction_to_source,
)
from research.training.train_traversability_segformer import (
    evaluate,
    load_training_model,
    metrics_from_confusion,
    segmentation_loss_sum,
)


def test_source_training_id_mapping_and_inverse() -> None:
    source = np.array([[0, 1, 2, 3]], dtype=np.uint8)

    training = source_mask_to_training(source)

    assert training.tolist() == [[255, 0, 1, 2]]
    assert training_prediction_to_source(np.array([[0, 1, 2]], dtype=np.uint8)).tolist() == [[1, 2, 3]]
    assert set(int(value) for value in np.unique(training)) == {0, 1, 2, 255}


def test_letterbox_preserves_geometry_and_uses_discrete_mask_ids() -> None:
    image = np.zeros((4, 8, 3), dtype=np.uint8)
    image[:, 4:] = 255
    mask = np.ones((4, 8), dtype=np.uint8)
    mask[:, 4:] = 3

    resized_image, resized_mask = letterbox_image_and_mask(image, mask, 16)

    assert resized_image.shape == (16, 16, 3)
    assert resized_mask.shape == (16, 16)
    assert set(int(value) for value in np.unique(resized_mask)) == {0, 1, 3}
    assert np.all(resized_mask[:4] == 0)
    assert np.all(resized_mask[4:12, :8] == 1)
    assert np.all(resized_mask[4:12, 8:] == 3)


def test_dataset_shapes_dtypes_and_batch_collation(tmp_path: Path) -> None:
    manifest = write_manifest(tmp_path, sample_count=2)
    dataset = TraversabilityDataset(manifest, "train", image_size=32, augment=False, seed=7)

    item = dataset[0]
    batch = next(iter(DataLoader(dataset, batch_size=2, shuffle=False)))

    assert item["pixel_values"].shape == (3, 32, 32)
    assert item["pixel_values"].dtype == torch.float32
    assert item["labels"].shape == (32, 32)
    assert item["labels"].dtype == torch.int64
    assert set(item["labels"].unique().tolist()).issubset({0, 1, 2, 255})
    assert batch["pixel_values"].shape == (2, 3, 32, 32)
    assert batch["labels"].shape == (2, 32, 32)


def test_dataset_accepts_new_holdout_split(tmp_path: Path) -> None:
    manifest = write_manifest(tmp_path, sample_count=1)
    rows = list(csv.DictReader(manifest.open(newline="", encoding="utf-8")))
    rows[0]["split"] = "new_holdout"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)

    dataset = TraversabilityDataset(
        manifest,
        "new_holdout",
        image_size=32,
        augment=False,
        seed=7,
    )

    assert len(dataset) == 1


def test_initial_checkpoint_schema_is_strict(tmp_path: Path) -> None:
    checkpoint = tmp_path / "bad.pt"
    torch.save({"model_state_dict": {}}, checkpoint)

    with pytest.raises(ValueError, match="schema is missing keys"):
        load_training_model(checkpoint, torch.device("cpu"))


def test_approved_dataset_writes_validator_metadata_and_split_manifests(tmp_path: Path) -> None:
    (tmp_path / "splits").mkdir()
    row = {field: "value" for field in FIELDS}
    row.update({"sample_id": "sample", "split": "train"})

    _write_dataset_manifests(tmp_path, [row])

    assert (tmp_path / "metadata.csv").read_text(encoding="utf-8") == (
        tmp_path / "manifest.csv"
    ).read_text(encoding="utf-8")
    train_rows = list(csv.DictReader((tmp_path / "splits/train.csv").open(newline="", encoding="utf-8")))
    assert train_rows == [row]
    assert list(csv.DictReader((tmp_path / "splits/validation.csv").open(newline="", encoding="utf-8"))) == []
    assert list(csv.DictReader((tmp_path / "splits/test.csv").open(newline="", encoding="utf-8"))) == []


def test_ride_split_is_deterministic_and_has_no_overlap() -> None:
    rows = [
        {"sample_id": f"sample_{ride}", "ride_id": str(ride)}
        for ride in range(9)
    ]
    pixels = {
        row["sample_id"]: {"IGNORE": 1, "ON_ROAD": 10, "OFF_ROAD": 10, "OBSTACLE": 10}
        for row in rows
    }

    first = choose_ride_split(rows, pixels, seed=17, trials=500)
    repeated = choose_ride_split(rows, pixels, seed=17, trials=500)

    assert first == repeated
    assert not (set(first["train"]) & set(first["validation"]))
    assert not (set(first["train"]) & set(first["test"]))
    assert not (set(first["validation"]) & set(first["test"]))
    assert set().union(*map(set, first.values())) == {str(ride) for ride in range(9)}


def test_safety_metrics_report_obstacle_as_traversable() -> None:
    confusion = torch.tensor(
        [
            [8, 1, 1],
            [2, 7, 1],
            [3, 2, 5],
        ],
        dtype=torch.int64,
    )

    metrics = metrics_from_confusion(confusion, loss=0.5)

    assert metrics["false_traversable_rate"] == 0.5
    assert metrics["obstacle_to_on_road_rate"] == 0.3
    assert metrics["obstacle_to_off_road_rate"] == 0.2
    assert metrics["false_non_traversable_rate"] == 0.1
    assert metrics["on_road_off_road_confusion_rate"] == 0.15


def test_ignore_pixels_are_excluded_from_loss_and_metrics() -> None:
    class FixedModel(nn.Module):
        def forward(self, images: torch.Tensor) -> SimpleNamespace:
            logits = torch.tensor(
                [[[[10.0, -10.0]], [[-10.0, 10.0]], [[-10.0, -10.0]]]],
                dtype=torch.float32,
            )
            return SimpleNamespace(logits=logits.repeat(images.shape[0], 1, 1, 1))

    batch = {
        "pixel_values": torch.zeros((1, 3, 1, 2), dtype=torch.float32),
        "labels": torch.tensor([[[0, 255]]], dtype=torch.int64),
    }
    metrics = evaluate(FixedModel(), [batch], torch.device("cpu"))

    assert metrics["loss"] < 1e-6
    assert sum(sum(row) for row in metrics["confusion_matrix"]) == 1
    assert metrics["pixel_accuracy"] == 1.0


def test_deterministic_loss_matches_unweighted_cross_entropy() -> None:
    logits = torch.tensor(
        [[[[2.0, -1.0]], [[0.0, 3.0]], [[-2.0, 1.0]]]],
        dtype=torch.float32,
    )
    labels = torch.tensor([[[0, 255]]], dtype=torch.int64)

    actual = segmentation_loss_sum(logits, labels)
    expected = functional.cross_entropy(logits, labels, ignore_index=255, reduction="sum")

    assert torch.allclose(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the deterministic kernel regression")
def test_segmentation_loss_backward_supports_strict_cuda_determinism() -> None:
    torch.use_deterministic_algorithms(True)
    logits = torch.randn((1, 3, 8, 8), device="cuda", requires_grad=True)
    labels = torch.randint(0, 3, (1, 8, 8), device="cuda")
    labels[:, 0] = 255

    loss = segmentation_loss_sum(logits, labels)
    loss.backward()

    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def write_manifest(root: Path, sample_count: int) -> Path:
    (root / "images").mkdir()
    (root / "masks").mkdir()
    fields = (
        "sample_id", "image_path", "mask_path", "split", "ride_id", "timestamp",
        "frame_id", "manifest_index", "playlist", "segment",
    )
    rows = []
    for index in range(sample_count):
        sample_id = f"sample_{index:05d}"
        image = np.zeros((18, 32, 3), dtype=np.uint8)
        image[:, :16] = (20, 80, 160)
        mask = np.array([[0, 1, 2, 3] * 8] * 18, dtype=np.uint8)
        image_path = root / "images" / f"{sample_id}.jpg"
        mask_path = root / "masks" / f"{sample_id}.png"
        assert cv2.imwrite(str(image_path), image)
        assert cv2.imwrite(str(mask_path), mask)
        rows.append(
            {
                "sample_id": sample_id,
                "image_path": f"images/{sample_id}.jpg",
                "mask_path": f"masks/{sample_id}.png",
                "split": "train",
                "ride_id": str(index),
                "timestamp": str(1000 + index),
                "frame_id": str(index),
                "manifest_index": str(index),
                "playlist": f"ride_{index}/front.m3u8",
                "segment": f"ride_{index}/front.ts",
            }
        )
    manifest = root / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return manifest
