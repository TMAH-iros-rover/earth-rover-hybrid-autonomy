from __future__ import annotations

import csv
import hashlib
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


SOURCE_TO_TRAINING = np.array([255, 0, 1, 2], dtype=np.uint8)
TRAINING_TO_SOURCE = np.array([1, 2, 3], dtype=np.uint8)
TRAINING_CLASS_NAMES = ("ON_ROAD", "OFF_ROAD", "OBSTACLE")
IMAGE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(frozen=True)
class TraversabilitySample:
    sample_id: str
    image_path: Path
    mask_path: Path
    ride_id: str
    timestamp: float
    frame_id: int
    manifest_index: int
    playlist: str
    segment: str
    split: str


def load_approved_manifest(path: str | Path) -> tuple[TraversabilitySample, ...]:
    manifest = Path(path).expanduser().resolve()
    root = manifest.parent
    rows = list(csv.DictReader(manifest.open(newline="", encoding="utf-8")))
    samples: list[TraversabilitySample] = []
    for row in rows:
        image_path = (root / row["image_path"]).resolve()
        mask_path = (root / row["mask_path"]).resolve()
        if not image_path.is_file() or not mask_path.is_file():
            raise ValueError(f"missing approved image or mask for {row.get('sample_id', '')}")
        samples.append(
            TraversabilitySample(
                sample_id=row["sample_id"],
                image_path=image_path,
                mask_path=mask_path,
                ride_id=row["ride_id"],
                timestamp=float(row["timestamp"]),
                frame_id=int(row["frame_id"]),
                manifest_index=int(row["manifest_index"]),
                playlist=row["playlist"],
                segment=row["segment"],
                split=row["split"],
            )
        )
    return tuple(samples)


class TraversabilityDataset(Dataset[dict[str, object]]):
    def __init__(
        self,
        manifest: str | Path,
        split: str,
        image_size: int = 512,
        augment: bool = False,
        seed: int = 20260718,
    ) -> None:
        if split not in {"train", "validation", "test", "new_holdout"}:
            raise ValueError(f"unsupported split: {split}")
        if image_size <= 0:
            raise ValueError("image_size must be positive")
        self.samples = tuple(sample for sample in load_approved_manifest(manifest) if sample.split == split)
        if not self.samples:
            raise ValueError(f"manifest contains no {split} samples")
        self.image_size = image_size
        self.augment = augment
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.samples)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.samples[index]
        image_bgr = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
        source_mask = cv2.imread(str(sample.mask_path), cv2.IMREAD_UNCHANGED)
        if image_bgr is None or source_mask is None or source_mask.ndim != 2:
            raise OSError(f"unreadable approved sample: {sample.sample_id}")
        if image_bgr.shape[:2] != source_mask.shape:
            raise ValueError(f"image-mask dimensions differ: {sample.sample_id}")
        values = set(int(value) for value in np.unique(source_mask))
        if not values.issubset({0, 1, 2, 3}):
            raise ValueError(f"invalid source mask IDs for {sample.sample_id}: {sorted(values)}")

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        rng = random.Random(f"{self.seed}:{self.epoch}:{sample.sample_id}")
        if self.augment and rng.random() < 0.5:
            image_rgb = np.ascontiguousarray(image_rgb[:, ::-1])
            source_mask = np.ascontiguousarray(source_mask[:, ::-1])
        image_rgb, source_mask = letterbox_image_and_mask(
            image_rgb,
            source_mask,
            self.image_size,
        )
        if self.augment:
            alpha = rng.uniform(0.9, 1.1)
            beta = rng.uniform(-10.0, 10.0)
            image_rgb = np.clip(image_rgb.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        target = source_mask_to_training(source_mask)
        return {
            "pixel_values": image_rgb_to_tensor(image_rgb),
            "labels": torch.from_numpy(target.astype(np.int64)),
            "sample_id": sample.sample_id,
            "ride_id": sample.ride_id,
            "image_path": str(sample.image_path),
            "mask_path": str(sample.mask_path),
        }


def source_mask_to_training(mask: np.ndarray) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError("source mask must be single-channel")
    values = set(int(value) for value in np.unique(mask))
    if not values.issubset({0, 1, 2, 3}):
        raise ValueError(f"source mask contains invalid IDs: {sorted(values)}")
    return SOURCE_TO_TRAINING[mask]


def training_prediction_to_source(prediction: np.ndarray) -> np.ndarray:
    values = set(int(value) for value in np.unique(prediction))
    if not values.issubset({0, 1, 2}):
        raise ValueError(f"training prediction contains invalid IDs: {sorted(values)}")
    return TRAINING_TO_SOURCE[prediction]


def image_rgb_to_tensor(image_rgb: np.ndarray) -> Tensor:
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"expected HxWx3 RGB image, got {image_rgb.shape}")
    image = image_rgb.astype(np.float32) / 255.0
    image = (image - IMAGE_MEAN) / IMAGE_STD
    return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).to(torch.float32)


def letterbox_image(image_rgb: np.ndarray, size: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"expected HxWx3 RGB image, got {image_rgb.shape}")
    if size <= 0:
        raise ValueError("letterbox size must be positive")
    height, width = image_rgb.shape[:2]
    scale = min(size / width, size / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    resized = cv2.resize(image_rgb, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    left = (size - resized_width) // 2
    top = (size - resized_height) // 2
    canvas[top : top + resized_height, left : left + resized_width] = resized
    return canvas, (left, top, resized_width, resized_height)


def restore_letterbox(
    array: np.ndarray,
    original_size: tuple[int, int],
    interpolation: int,
) -> np.ndarray:
    original_height, original_width = original_size
    size = array.shape[0]
    scale = min(size / original_width, size / original_height)
    resized_width = max(1, round(original_width * scale))
    resized_height = max(1, round(original_height * scale))
    left = (size - resized_width) // 2
    top = (size - resized_height) // 2
    cropped = array[top : top + resized_height, left : left + resized_width]
    return cv2.resize(cropped, (original_width, original_height), interpolation=interpolation)


def letterbox_image_and_mask(
    image_rgb: np.ndarray,
    source_mask: np.ndarray,
    size: int,
) -> tuple[np.ndarray, np.ndarray]:
    image_canvas, (left, top, resized_width, resized_height) = letterbox_image(image_rgb, size)
    mask_resized = cv2.resize(source_mask, (resized_width, resized_height), interpolation=cv2.INTER_NEAREST)
    mask_canvas = np.zeros((size, size), dtype=np.uint8)
    mask_canvas[top : top + resized_height, left : left + resized_width] = mask_resized
    return image_canvas, mask_canvas


def choose_ride_split(
    sample_rows: list[dict[str, str]],
    class_pixels_by_sample: dict[str, dict[str, int]],
    seed: int,
    trials: int = 20000,
) -> dict[str, tuple[str, ...]]:
    rides = sorted({row["ride_id"] for row in sample_rows})
    if len(rides) < 6:
        raise ValueError(f"need at least 6 rides for an isolated three-way split, found {len(rides)}")
    sample_counts = Counter(row["ride_id"] for row in sample_rows)
    ride_classes: dict[str, Counter[str]] = defaultdict(Counter)
    for row in sample_rows:
        ride_classes[row["ride_id"]].update(class_pixels_by_sample[row["sample_id"]])
    for class_name in TRAINING_CLASS_NAMES:
        supporting = [ride for ride in rides if ride_classes[ride][class_name] > 0]
        if len(supporting) < 3:
            raise ValueError(f"{class_name} appears in only {len(supporting)} rides; need at least 3")

    validation_ride_count = max(1, round(len(rides) * 0.15))
    test_ride_count = max(1, round(len(rides) * 0.15))
    if validation_ride_count + test_ride_count >= len(rides):
        raise ValueError("not enough rides remain for training")
    total_samples = len(sample_rows)
    global_pixels = sum((ride_classes[ride] for ride in rides), Counter())
    global_total = sum(global_pixels[class_name] for class_name in TRAINING_CLASS_NAMES)
    rng = random.Random(seed)
    best: tuple[float, tuple[str, ...], tuple[str, ...], tuple[str, ...]] | None = None
    for _ in range(trials):
        shuffled = rides.copy()
        rng.shuffle(shuffled)
        validation = tuple(sorted(shuffled[:validation_ride_count]))
        test = tuple(sorted(shuffled[validation_ride_count : validation_ride_count + test_ride_count]))
        train = tuple(sorted(shuffled[validation_ride_count + test_ride_count :]))
        candidate = {"train": train, "validation": validation, "test": test}
        if any(
            sum(ride_classes[ride][class_name] for ride in split_rides) == 0
            for split_rides in candidate.values()
            for class_name in TRAINING_CLASS_NAMES
        ):
            continue
        counts = {
            name: sum(sample_counts[ride] for ride in split_rides)
            for name, split_rides in candidate.items()
        }
        score = (
            abs(counts["train"] / total_samples - 0.70)
            + abs(counts["validation"] / total_samples - 0.15)
            + abs(counts["test"] / total_samples - 0.15)
        )
        for split_rides in candidate.values():
            split_pixels = sum((ride_classes[ride] for ride in split_rides), Counter())
            split_total = sum(split_pixels[class_name] for class_name in TRAINING_CLASS_NAMES)
            score += 0.25 * sum(
                abs(
                    split_pixels[class_name] / split_total
                    - global_pixels[class_name] / global_total
                )
                for class_name in TRAINING_CLASS_NAMES
            )
        ranked = (score, train, validation, test)
        if best is None or ranked < best:
            best = ranked
    if best is None:
        raise ValueError("no deterministic ride split contains all training classes in every split")
    return {"train": best[1], "validation": best[2], "test": best[3]}


def manifest_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
