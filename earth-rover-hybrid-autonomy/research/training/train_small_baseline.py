#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import cv2
import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.training.datasets.action_labels import ACTION_NAMES
from research.training.datasets.frodobots_2k_dataset import FrameDecodeError, FrodoBotsActionDataset, ManifestSample
from research.training.models.action_resnet18 import build_action_resnet18
from research.training.train_tiny_overfit import compute_class_weights, _set_deterministic


@dataclass(frozen=True)
class RideSplit:
    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]


@dataclass(frozen=True)
class CachedSplit:
    images: Tensor
    targets: Tensor
    manifest_indices: tuple[int, ...]
    ride_ids: tuple[str, ...]
    decode_failures: tuple[dict[str, object], ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the bounded Phase 3 10/2/2 ride-level baseline.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--samples-per-ride", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--video-frames-per-ride", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _validate_args(args)
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir == dataset_root or dataset_root in output_dir.parents:
        raise SystemExit("output-dir must remain outside the immutable raw dataset root")
    output_dir.mkdir(parents=True, exist_ok=True)

    _set_deterministic(args.seed)
    if args.require_cuda and not torch.cuda.is_available():
        raise SystemExit("CUDA is required but torch.cuda.is_available() is false")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    source_dataset = FrodoBotsActionDataset(dataset_root, args.manifest)
    split = choose_ride_split(
        source_dataset.samples,
        args.samples_per_ride,
        args.video_frames_per_ride,
        args.seed,
    )
    selected = {
        "train": select_split_indices(source_dataset.samples, split.train, args.samples_per_ride, args.seed),
        "validation": select_split_indices(
            source_dataset.samples, split.validation, args.samples_per_ride, args.seed + 1
        ),
        "test": select_split_indices(source_dataset.samples, split.test, args.samples_per_ride, args.seed + 2),
    }
    cached = {
        name: cache_split(source_dataset, indices, name)
        for name, indices in selected.items()
    }
    for name, value in cached.items():
        if value.decode_failures:
            raise RuntimeError(f"{name} contains {len(value.decode_failures)} unreadable selected samples")

    train_loader = _make_loader(cached["train"], args.batch_size, shuffle=True, seed=args.seed)
    validation_loader = _make_loader(cached["validation"], args.batch_size, shuffle=False, seed=args.seed)
    test_loader = _make_loader(cached["test"], args.batch_size, shuffle=False, seed=args.seed)

    model = build_action_resnet18(len(ACTION_NAMES), pretrained=True).to(device)
    class_weights = compute_class_weights(cached["train"].targets, len(ACTION_NAMES)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    checkpoint_path = output_dir / "resnet18_small_baseline_best.pt"
    history: list[dict[str, object]] = []
    best_macro_f1 = -1.0
    epochs_without_improvement = 0

    for epoch in range(1, args.max_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        validation_metrics = evaluate_metrics(model, validation_loader, criterion, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation": validation_metrics,
            }
        )
        macro_f1 = float(validation_metrics["macro_f1"])
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.6f} "
            f"val_macro_f1={macro_f1:.4f} val_accuracy={validation_metrics['accuracy']:.4f}",
            flush=True,
        )
        if macro_f1 > best_macro_f1 + 1e-12:
            best_macro_f1 = macro_f1
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "action_names": ACTION_NAMES,
                    "image_size": 224,
                    "ride_split": {
                        "train": split.train,
                        "validation": split.validation,
                        "test": split.test,
                    },
                    "epoch": epoch,
                    "validation_macro_f1": macro_f1,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate_metrics(model, test_loader, criterion, device)
    video_path = output_dir / "held_out_test_predictions.mp4"
    video_report = create_prediction_video(
        model,
        source_dataset,
        split.test,
        video_path,
        args.video_frames_per_ride,
        device,
    )

    report = {
        "execution_success": True,
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "torch_version": torch.__version__,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "seed": args.seed,
        "samples_per_ride_cap": args.samples_per_ride,
        "cached_image_dtype": "torch.float16 on CPU; converted to torch.float32 before model input",
        "ride_split": {
            "train": list(split.train),
            "validation": list(split.validation),
            "test": list(split.test),
        },
        "split_overlap": _split_overlap(split),
        "selected_samples": {
            name: {
                "count": len(value.manifest_indices),
                "action_distribution": _target_distribution(value.targets),
                "ride_distribution": dict(sorted(Counter(value.ride_ids).items())),
                "decode_failure_count": len(value.decode_failures),
            }
            for name, value in cached.items()
        },
        "source_split_action_distribution": {
            "train": _source_action_distribution(source_dataset.samples, split.train),
            "validation": _source_action_distribution(source_dataset.samples, split.validation),
            "test": _source_action_distribution(source_dataset.samples, split.test),
        },
        "class_weights": {
            action: float(weight)
            for action, weight in zip(ACTION_NAMES, class_weights.detach().cpu().tolist())
        },
        "epochs_completed": len(history),
        "best_epoch": int(checkpoint["epoch"]),
        "best_validation_macro_f1": float(checkpoint["validation_macro_f1"]),
        "test_metrics": test_metrics,
        "checkpoint_path": str(checkpoint_path),
        "prediction_video": video_report,
        "history": history,
    }
    report_path = output_dir / "small_baseline_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "best_epoch": report["best_epoch"],
                "best_validation_macro_f1": report["best_validation_macro_f1"],
                "test_metrics": test_metrics,
                "checkpoint_path": str(checkpoint_path),
                "prediction_video": video_report,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def choose_ride_split(
    samples: tuple[ManifestSample, ...],
    minimum_samples_per_ride: int,
    minimum_video_run: int,
    seed: int,
) -> RideSplit:
    counts = _ride_action_counts(samples)
    longest_runs = _ride_longest_runs(samples)
    eligible = sorted(ride for ride, counter in counts.items() if sum(counter.values()) >= minimum_samples_per_ride)
    if len(eligible) < 14:
        raise ValueError(f"need 14 eligible rides, found {len(eligible)}")

    video_eligible = [
        ride
        for ride in eligible
        if longest_runs[ride] >= minimum_video_run
    ]
    if len(video_eligible) < 2:
        raise ValueError(f"need two rides with a {minimum_video_run}-frame aligned sequence")
    test = _best_ride_pair(video_eligible, counts, excluded=set(), seed=seed)
    validation = _best_ride_pair(eligible, counts, excluded=set(test), seed=seed + 1)
    remaining = [ride for ride in eligible if ride not in set(test + validation)]
    ranked = sorted(
        remaining,
        key=lambda ride: (_ride_score(counts[ride]), _stable_tiebreak(ride, seed + 2)),
        reverse=True,
    )
    train = tuple(ranked[:10])
    split = RideSplit(train=train, validation=tuple(validation), test=tuple(test))
    if _split_overlap(split):
        raise RuntimeError("ride split overlap detected")
    for name, rides in (("train", split.train), ("validation", split.validation), ("test", split.test)):
        combined = sum((counts[ride] for ride in rides), Counter())
        missing = [action for action in ACTION_NAMES if combined[action] == 0]
        if missing:
            raise ValueError(f"{name} split lacks actions: {', '.join(missing)}")
    return split


def select_split_indices(
    samples: tuple[ManifestSample, ...],
    ride_ids: tuple[str, ...],
    samples_per_ride: int,
    seed: int,
) -> list[int]:
    ride_id_set = set(ride_ids)
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        if sample.ride_id in ride_id_set:
            grouped[sample.ride_id].append(index)
    selected: list[int] = []
    minimum_per_action = max(1, samples_per_ride // 25)
    for ride_id in ride_ids:
        indices = grouped[ride_id]
        rng = random.Random(seed + _stable_tiebreak(ride_id, seed))
        by_action: dict[str, list[int]] = defaultdict(list)
        for index in indices:
            by_action[samples[index].action_class].append(index)
        ride_selected: list[int] = []
        for action in ACTION_NAMES:
            candidates = by_action[action]
            ride_selected.extend(rng.sample(candidates, min(len(candidates), minimum_per_action)))
        selected_set = set(ride_selected)
        remaining = [index for index in indices if index not in selected_set]
        fill_count = min(samples_per_ride - len(ride_selected), len(remaining))
        ride_selected.extend(rng.sample(remaining, fill_count))
        selected.extend(sorted(ride_selected))
    return selected


def cache_split(dataset: FrodoBotsActionDataset, indices: list[int], name: str) -> CachedSplit:
    images: list[Tensor] = []
    targets: list[Tensor] = []
    accepted_indices: list[int] = []
    ride_ids: list[str] = []
    failures: list[dict[str, object]] = []
    for position, index in enumerate(indices, start=1):
        try:
            item = dataset[index]
        except (FrameDecodeError, OSError) as exc:
            failures.append({"manifest_index": index, "error": str(exc)})
            continue
        images.append(item["image"].to(torch.float16))
        targets.append(item["target"])
        accepted_indices.append(index)
        ride_ids.append(str(item["ride_id"]))
        if position % 100 == 0 or position == len(indices):
            print(f"{name}_decoded={position}/{len(indices)}", flush=True)
    if not images:
        raise RuntimeError(f"no {name} samples decoded")
    return CachedSplit(
        images=torch.stack(images),
        targets=torch.stack(targets),
        manifest_indices=tuple(accepted_indices),
        ride_ids=tuple(ride_ids),
        decode_failures=tuple(failures),
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    for images, targets in loader:
        images = images.to(device=device, dtype=torch.float32, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images), targets)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * targets.size(0)
        total_samples += targets.size(0)
    return total_loss / total_samples


@torch.no_grad()
def evaluate_metrics(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    all_targets: list[Tensor] = []
    all_predictions: list[Tensor] = []
    total_loss = 0.0
    total_samples = 0
    for images, targets in loader:
        images = images.to(device=device, dtype=torch.float32, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        total_loss += float(criterion(logits, targets).item()) * targets.size(0)
        total_samples += targets.size(0)
        all_targets.append(targets.cpu())
        all_predictions.append(logits.argmax(dim=1).cpu())
    metrics = classification_metrics(torch.cat(all_targets), torch.cat(all_predictions), len(ACTION_NAMES))
    metrics["loss"] = total_loss / total_samples
    return metrics


def classification_metrics(targets: Tensor, predictions: Tensor, num_classes: int) -> dict[str, object]:
    if targets.shape != predictions.shape or targets.ndim != 1:
        raise ValueError("targets and predictions must have matching one-dimensional shapes")
    encoded = targets.to(torch.int64) * num_classes + predictions.to(torch.int64)
    confusion = torch.bincount(encoded, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    per_class: dict[str, dict[str, float | int]] = {}
    recalls: list[float] = []
    f1_values: list[float] = []
    for index, action in enumerate(ACTION_NAMES):
        true_positive = int(confusion[index, index].item())
        support = int(confusion[index].sum().item())
        predicted = int(confusion[:, index].sum().item())
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[action] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        if support:
            recalls.append(recall)
            f1_values.append(f1)
    total = int(confusion.sum().item())
    correct = int(confusion.diag().sum().item())
    return {
        "accuracy": correct / total,
        "balanced_accuracy": sum(recalls) / len(recalls),
        "macro_f1": sum(f1_values) / len(f1_values),
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }


@torch.no_grad()
def create_prediction_video(
    model: nn.Module,
    dataset: FrodoBotsActionDataset,
    test_rides: tuple[str, ...],
    output_path: Path,
    frames_per_ride: int,
    device: torch.device,
) -> dict[str, object]:
    selected = select_video_indices(dataset.samples, test_rides, frames_per_ride)
    writer: cv2.VideoWriter | None = None
    frame_size: tuple[int, int] | None = None
    correct = 0
    written = 0
    model.eval()
    try:
        for ride_id in test_rides:
            indices = selected[ride_id]
            for start in range(0, len(indices), 32):
                chunk = indices[start : start + 32]
                items: list[dict[str, object]] = []
                frames: list[np.ndarray] = []
                for index in chunk:
                    item, frame_rgb = dataset.load_sample(index)
                    items.append(item)
                    frames.append(frame_rgb)
                images = torch.stack([item["image"] for item in items]).to(device)
                probabilities = model(images).softmax(dim=1).cpu()
                for item, frame_rgb, probability in zip(items, frames, probabilities):
                    prediction = int(probability.argmax().item())
                    confidence = float(probability[prediction].item())
                    target = int(item["target"].item())
                    correct += int(prediction == target)
                    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                    if writer is None:
                        height, width = frame_bgr.shape[:2]
                        frame_size = (width, height)
                        writer = cv2.VideoWriter(
                            str(output_path),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            20.0,
                            (width, height),
                        )
                        if not writer.isOpened():
                            raise OSError(f"failed to open video writer: {output_path}")
                    elif (frame_bgr.shape[1], frame_bgr.shape[0]) != frame_size:
                        raise ValueError("test video frames do not share one resolution")
                    _overlay_prediction(frame_bgr, item, prediction, confidence)
                    writer.write(frame_bgr)
                    written += 1
    finally:
        if writer is not None:
            writer.release()
    if written == 0 or not output_path.is_file() or output_path.stat().st_size == 0:
        raise OSError("prediction video was not created")
    if frame_size is None:
        raise RuntimeError("prediction video frame size was not recorded")
    return {
        "path": str(output_path),
        "frame_count": written,
        "fps": 20.0,
        "codec": "mp4v",
        "resolution": [frame_size[0], frame_size[1]],
        "ride_ids": list(test_rides),
        "accuracy": correct / written,
    }


def select_video_indices(
    samples: tuple[ManifestSample, ...],
    ride_ids: tuple[str, ...],
    frames_per_ride: int,
) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        if sample.ride_id in ride_ids:
            grouped[sample.ride_id].append(index)
    selected: dict[str, list[int]] = {}
    for ride_id in ride_ids:
        runs: list[list[int]] = []
        current: list[int] = []
        for index in grouped[ride_id]:
            if not current or _is_consecutive(samples[current[-1]], samples[index]):
                current.append(index)
            else:
                runs.append(current)
                current = [index]
        if current:
            runs.append(current)
        longest = max(runs, key=len)
        if len(longest) < frames_per_ride:
            raise ValueError(f"ride {ride_id} has no {frames_per_ride}-frame aligned sequence")
        start = (len(longest) - frames_per_ride) // 2
        selected[ride_id] = longest[start : start + frames_per_ride]
    return selected


def _ride_longest_runs(samples: tuple[ManifestSample, ...]) -> dict[str, int]:
    longest: dict[str, int] = defaultdict(int)
    current: dict[str, int] = defaultdict(int)
    previous: dict[str, ManifestSample] = {}
    for sample in samples:
        ride_id = sample.ride_id
        current[ride_id] = (
            current[ride_id] + 1
            if ride_id in previous and _is_consecutive(previous[ride_id], sample)
            else 1
        )
        longest[ride_id] = max(longest[ride_id], current[ride_id])
        previous[ride_id] = sample
    return dict(longest)


def _ride_action_counts(samples: tuple[ManifestSample, ...]) -> dict[str, Counter[str]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for sample in samples:
        counts[sample.ride_id][sample.action_class] += 1
    return dict(counts)


def _best_ride_pair(
    eligible: list[str],
    counts: dict[str, Counter[str]],
    excluded: set[str],
    seed: int,
) -> tuple[str, str]:
    candidates = [ride for ride in eligible if ride not in excluded]
    valid_pairs: list[tuple[tuple[object, ...], tuple[str, str]]] = []
    for pair in combinations(candidates, 2):
        combined = counts[pair[0]] + counts[pair[1]]
        if any(combined[action] == 0 for action in ACTION_NAMES):
            continue
        score = (
            min(combined[action] for action in ACTION_NAMES),
            sum(min(combined[action], 500) for action in ACTION_NAMES),
            _stable_tiebreak("|".join(pair), seed),
        )
        valid_pairs.append((score, pair))
    if not valid_pairs:
        raise ValueError("cannot find a two-ride split containing every action class")
    return max(valid_pairs, key=lambda item: item[0])[1]


def _ride_score(counts: Counter[str]) -> tuple[int, int, int]:
    coverage = sum(counts[action] > 0 for action in ACTION_NAMES)
    minimum = min((counts[action] for action in ACTION_NAMES if counts[action] > 0), default=0)
    return coverage, minimum, min(sum(counts.values()), 10000)


def _stable_tiebreak(value: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _split_overlap(split: RideSplit) -> list[str]:
    all_rides = list(split.train + split.validation + split.test)
    return sorted(ride for ride, count in Counter(all_rides).items() if count > 1)


def _target_distribution(targets: Tensor) -> dict[str, int]:
    counts = torch.bincount(targets, minlength=len(ACTION_NAMES)).tolist()
    return {action: int(count) for action, count in zip(ACTION_NAMES, counts)}


def _source_action_distribution(
    samples: tuple[ManifestSample, ...],
    ride_ids: tuple[str, ...],
) -> dict[str, int]:
    ride_id_set = set(ride_ids)
    counts = Counter(sample.action_class for sample in samples if sample.ride_id in ride_id_set)
    return {action: counts[action] for action in ACTION_NAMES}


def _make_loader(split: CachedSplit, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(split.images, split.targets),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
    )


def _is_consecutive(previous: ManifestSample, current: ManifestSample) -> bool:
    delta = current.front_timestamp - previous.front_timestamp
    return (
        current.ride_id == previous.ride_id
        and current.timeline_section_id == previous.timeline_section_id
        and current.front_frame_id == previous.front_frame_id + 1
        and 0.0 < delta <= 0.075
    )


def _overlay_prediction(
    frame_bgr: np.ndarray,
    item: dict[str, object],
    prediction: int,
    confidence: float,
) -> None:
    target = int(item["target"].item())
    metadata = item["metadata"]
    correct = prediction == target
    color = (0, 220, 0) if correct else (0, 0, 255)
    lines = [
        f"ride={item['ride_id']} frame={metadata['front_frame_id']} ts={metadata['front_timestamp']:.3f}",
        f"ground_truth={ACTION_NAMES[target]}",
        f"prediction={ACTION_NAMES[prediction]} confidence={confidence:.3f}",
        f"linear={metadata['linear']:.3f} angular={metadata['angular']:.3f}",
    ]
    cv2.rectangle(frame_bgr, (0, 0), (frame_bgr.shape[1], 132), (0, 0, 0), -1)
    for index, line in enumerate(lines):
        cv2.putText(
            frame_bgr,
            line,
            (12, 28 + index * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color if index == 2 else (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def _validate_args(args: argparse.Namespace) -> None:
    if args.samples_per_ride <= 0 or args.batch_size <= 0:
        raise SystemExit("samples-per-ride and batch-size must be positive")
    if args.max_epochs <= 0 or args.patience <= 0:
        raise SystemExit("max-epochs and patience must be positive")
    if args.learning_rate <= 0 or args.video_frames_per_ride <= 0:
        raise SystemExit("learning-rate and video-frames-per-ride must be positive")


if __name__ == "__main__":
    raise SystemExit(main())
