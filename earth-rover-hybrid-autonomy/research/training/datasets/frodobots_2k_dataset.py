from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from research.training.datasets.action_labels import ACTION_NAMES
from research.training.datasets.frodobots_2k_manifest import MANIFEST_COLUMNS, parse_hls_segment_timestamp


ACTION_TO_INDEX = {name: index for index, name in enumerate(ACTION_NAMES)}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ManifestFormatError(ValueError):
    pass


class FrameDecodeError(RuntimeError):
    pass


@dataclass(frozen=True)
class ManifestSample:
    ride_id: str
    front_playlist_ref: str
    front_segment_ref: str
    front_frame_id: int
    front_timestamp: float
    matched_control_timestamp: float
    control_delta_ms: float
    linear: float
    angular: float
    action_class: str
    timeline_section_id: int


class HlsFrameDecoder:
    def __init__(self, dataset_root: str | Path) -> None:
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        if not self.dataset_root.is_dir():
            raise FileNotFoundError(f"dataset root does not exist: {self.dataset_root}")

    def decode(self, sample: ManifestSample) -> np.ndarray:
        segment_path = self._segment_path(sample.front_segment_ref)
        segment_start = parse_hls_segment_timestamp(sample.front_segment_ref)
        offset_ms = (sample.front_timestamp - segment_start) * 1000.0
        if not math.isfinite(offset_ms) or offset_ms < -1e-3:
            raise FrameDecodeError(
                f"frame timestamp precedes segment start: {sample.front_segment_ref}"
            )
        offset_ms = max(offset_ms, 0.0)

        capture = cv2.VideoCapture(str(segment_path))
        try:
            if not capture.isOpened():
                raise FrameDecodeError(f"cannot open HLS segment: {sample.front_segment_ref}")
            if offset_ms > 0.0 and not capture.set(cv2.CAP_PROP_POS_MSEC, offset_ms):
                raise FrameDecodeError(
                    f"cannot seek to {offset_ms:.3f} ms in segment: {sample.front_segment_ref}"
                )
            ok, frame_bgr = capture.read()
            if not ok or frame_bgr is None or frame_bgr.size == 0:
                raise FrameDecodeError(
                    f"cannot decode frame at {offset_ms:.3f} ms: {sample.front_segment_ref}"
                )
        finally:
            capture.release()

        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    def _segment_path(self, reference: str) -> Path:
        path = (self.dataset_root / reference).resolve()
        try:
            path.relative_to(self.dataset_root)
        except ValueError as exc:
            raise FrameDecodeError(f"segment reference escapes dataset root: {reference}") from exc
        if not path.is_file():
            raise FrameDecodeError(f"HLS segment does not exist: {reference}")
        return path


class FrodoBotsActionDataset(Dataset[dict[str, object]]):
    def __init__(
        self,
        dataset_root: str | Path,
        manifest_path: str | Path,
        image_size: int = 224,
        decoder: HlsFrameDecoder | None = None,
        transform: Callable[[np.ndarray], Tensor] | None = None,
    ) -> None:
        if image_size <= 0:
            raise ValueError("image_size must be positive")
        self.samples = load_manifest(manifest_path)
        self.image_size = image_size
        self.decoder = decoder or HlsFrameDecoder(dataset_root)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        item, _ = self.load_sample(index)
        return item

    def load_sample(self, index: int) -> tuple[dict[str, object], np.ndarray]:
        sample = self.samples[index]
        frame_rgb = self.decoder.decode(sample)
        image = self.transform(frame_rgb) if self.transform else image_to_tensor(frame_rgb, self.image_size)
        if not isinstance(image, Tensor):
            raise TypeError("transform must return a torch.Tensor")
        item: dict[str, object] = {
            "image": image,
            "target": torch.tensor(ACTION_TO_INDEX[sample.action_class], dtype=torch.long),
            "action_class": sample.action_class,
            "ride_id": sample.ride_id,
            "metadata": {
                "front_frame_id": sample.front_frame_id,
                "front_timestamp": sample.front_timestamp,
                "matched_control_timestamp": sample.matched_control_timestamp,
                "control_delta_ms": sample.control_delta_ms,
                "linear": sample.linear,
                "angular": sample.angular,
                "front_playlist_ref": sample.front_playlist_ref,
                "front_segment_ref": sample.front_segment_ref,
                "timeline_section_id": sample.timeline_section_id,
            },
        }
        return item, frame_rgb


def load_manifest(path: str | Path) -> tuple[ManifestSample, ...]:
    path = Path(path).expanduser().resolve()
    samples: list[ManifestSample] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        available = set(reader.fieldnames or [])
        missing = [name for name in MANIFEST_COLUMNS if name not in available]
        if missing:
            raise ManifestFormatError(f"missing manifest columns in {path}: {', '.join(missing)}")
        for line_number, row in enumerate(reader, start=2):
            try:
                sample = ManifestSample(
                    ride_id=_required_text(row["ride_id"]),
                    front_playlist_ref=_required_text(row["front_playlist_ref"]),
                    front_segment_ref=_required_text(row["front_segment_ref"]),
                    front_frame_id=int(row["front_frame_id"]),
                    front_timestamp=_finite_float(row["front_timestamp"]),
                    matched_control_timestamp=_finite_float(row["matched_control_timestamp"]),
                    control_delta_ms=_finite_float(row["control_delta_ms"]),
                    linear=_finite_float(row["linear"]),
                    angular=_finite_float(row["angular"]),
                    action_class=_required_text(row["action_class"]),
                    timeline_section_id=int(row["timeline_section_id"]),
                )
                if sample.front_frame_id < 0 or sample.timeline_section_id < 0:
                    raise ValueError("negative frame or section ID")
                if sample.control_delta_ms < 0:
                    raise ValueError("negative control delta")
                if sample.action_class not in ACTION_TO_INDEX:
                    raise ValueError(f"unknown action class: {sample.action_class}")
            except (TypeError, ValueError) as exc:
                raise ManifestFormatError(f"invalid manifest row {line_number} in {path}: {exc}") from exc
            samples.append(sample)
    if not samples:
        raise ManifestFormatError(f"manifest has no samples: {path}")
    return tuple(samples)


def image_to_tensor(frame_rgb: np.ndarray, image_size: int) -> Tensor:
    if frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3:
        raise FrameDecodeError(f"expected HxWx3 RGB frame, got shape {frame_rgb.shape}")
    resized = cv2.resize(frame_rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)
    array = np.ascontiguousarray(resized.transpose(2, 0, 1), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array)
    mean = tensor.new_tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = tensor.new_tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor - mean) / std


def _finite_float(value: object) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite value: {value!r}")
    return parsed


def _required_text(value: object) -> str:
    if value is None:
        raise ValueError("missing text value")
    parsed = str(value).strip()
    if not parsed:
        raise ValueError("empty text value")
    return parsed
