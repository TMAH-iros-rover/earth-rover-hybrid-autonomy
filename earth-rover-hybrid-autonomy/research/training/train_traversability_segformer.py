#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import importlib.metadata
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import cv2
import numpy as np
import torch
import torch.nn.functional as functional
import yaml
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.training.datasets.traversability_dataset_v1 import (
    TRAINING_CLASS_NAMES,
    TraversabilityDataset,
    manifest_sha256,
    restore_letterbox,
    training_prediction_to_source,
)
from research.training.models.traversability_segformer import (
    CHECKPOINT,
    REVISION,
    WEIGHT_FILE,
    build_traversability_segformer,
    validate_three_class_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the approved traversability SegFormer-B0 baseline.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=("overfit", "full"), required=True)
    parser.add_argument(
        "--initial-checkpoint",
        help="Optional approved 3-class checkpoint used instead of ADE initialization.",
    )
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("checkpoint") != CHECKPOINT or config.get("revision") != REVISION:
        raise SystemExit("config checkpoint or revision differs from the approved SegFormer-B0 source")
    if int(config.get("num_labels", -1)) != 3 or int(config.get("ignore_index", -1)) != 255:
        raise SystemExit("config must use num_labels=3 and ignore_index=255")
    manifest = Path(args.manifest).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output / "frozen_config.yaml")
    _set_deterministic(int(config["seed"]))
    if args.require_cuda and not torch.cuda.is_available():
        raise SystemExit("CUDA is required but unavailable")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    if args.mode == "overfit":
        if args.initial_checkpoint:
            raise SystemExit("initial-checkpoint is supported only for full training")
        report = run_overfit(manifest, config, output, device)
        report_name = "overfit_report.json"
    else:
        initial_checkpoint = (
            Path(args.initial_checkpoint).expanduser().resolve()
            if args.initial_checkpoint
            else None
        )
        report = run_full_training(
            manifest,
            config,
            output,
            device,
            initial_checkpoint=initial_checkpoint,
        )
        report_name = "experiment_report.json"
    report.update(_environment_report(device, manifest, time.monotonic() - started))
    (output / report_name).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["success"] else 1


def run_overfit(
    manifest: Path,
    config: dict[str, object],
    output: Path,
    device: torch.device,
) -> dict[str, object]:
    settings = config["overfit"]
    dataset = TraversabilityDataset(
        manifest,
        "train",
        image_size=int(config["image_size"]),
        augment=False,
        seed=int(config["seed"]),
    )
    indices = _select_overfit_indices(dataset, int(settings["sample_count"]))
    subset = Subset(dataset, indices)
    loader = _loader(subset, int(settings["batch_size"]), True, int(config["seed"]))
    eval_loader = _loader(subset, int(settings["batch_size"]), False, int(config["seed"]))
    model = build_traversability_segformer(pretrained=True).to(device)
    optimizer = AdamW(model.parameters(), lr=float(settings["learning_rate"]), weight_decay=0.0)
    initial = evaluate(model, eval_loader, device)
    history: list[dict[str, object]] = []
    for epoch in range(1, int(settings["max_epochs"]) + 1):
        train_loss = train_one_epoch(model, loader, optimizer, device)
        metrics = evaluate(model, eval_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "metrics": metrics})
        print(
            f"overfit epoch={epoch:03d} loss={metrics['loss']:.6f} "
            f"mIoU={metrics['mean_iou']:.4f} pixel_acc={metrics['pixel_accuracy']:.4f}",
            flush=True,
        )
        if (
            metrics["pixel_accuracy"] >= float(settings["target_pixel_accuracy"])
            and metrics["mean_iou"] >= float(settings["target_mean_iou"])
            and metrics["predicted_class_count"] >= 2
        ):
            break
    final = history[-1]["metrics"]
    checkpoint = output / "segformer_b0_overfit.pt"
    save_checkpoint(model, checkpoint, len(history), final)
    reload_delta = verify_checkpoint_reload(model, checkpoint, eval_loader, device)
    review = write_prediction_review(model, eval_loader, output / "review_bundle", device)
    success = bool(
        math.isfinite(float(final["loss"]))
        and final["loss"] < initial["loss"]
        and final["pixel_accuracy"] >= float(settings["target_pixel_accuracy"])
        and final["mean_iou"] >= float(settings["target_mean_iou"])
        and final["predicted_class_count"] >= 2
        and reload_delta <= 1e-6
    )
    return {
        "success": success,
        "mode": "overfit",
        "sample_indices": indices,
        "sample_ids": [dataset.samples[index].sample_id for index in indices],
        "initial_metrics": initial,
        "final_metrics": final,
        "loss_decreased": final["loss"] < initial["loss"],
        "epochs_completed": len(history),
        "checkpoint_path": str(checkpoint),
        "checkpoint_reload_max_abs_logit_delta": reload_delta,
        "prediction_review": review,
        "history": history,
        "full_training_performed": False,
        "live_rover_commands_sent": False,
    }


def run_full_training(
    manifest: Path,
    config: dict[str, object],
    output: Path,
    device: torch.device,
    initial_checkpoint: Path | None = None,
) -> dict[str, object]:
    seed = int(config["seed"])
    image_size = int(config["image_size"])
    train_dataset = TraversabilityDataset(manifest, "train", image_size, augment=True, seed=seed)
    validation_dataset = TraversabilityDataset(manifest, "validation", image_size, augment=False, seed=seed)
    test_dataset = TraversabilityDataset(manifest, "test", image_size, augment=False, seed=seed)
    train_loader = _loader(train_dataset, int(config["batch_size"]), True, seed)
    validation_loader = _loader(validation_dataset, int(config["batch_size"]), False, seed)
    model, initialization = load_training_model(
        initial_checkpoint,
        device,
        initialization_mode="approved_v1_best_checkpoint",
    )
    optimizer = AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    checkpoints = output / "checkpoints"
    checkpoints.mkdir()
    history: list[dict[str, object]] = []
    best_mean_iou = -1.0
    best_epoch = 0
    stale_epochs = 0
    best_checkpoint = output / "segformer_b0_best.pt"
    for epoch in range(1, int(config["max_epochs"]) + 1):
        train_dataset.set_epoch(epoch)
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        validation = evaluate(model, validation_loader, device)
        record = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss,
            "validation": validation,
        }
        history.append(record)
        epoch_checkpoint = checkpoints / f"epoch_{epoch:03d}.pt"
        save_checkpoint(
            model,
            epoch_checkpoint,
            epoch,
            validation,
            parent_checkpoint=initial_checkpoint,
        )
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.6f} val_loss={validation['loss']:.6f} "
            f"val_mIoU={validation['mean_iou']:.4f}",
            flush=True,
        )
        if validation["mean_iou"] > best_mean_iou + 1e-12:
            best_mean_iou = float(validation["mean_iou"])
            best_epoch = epoch
            stale_epochs = 0
            shutil.copy2(epoch_checkpoint, best_checkpoint)
        else:
            stale_epochs += 1
            if stale_epochs >= int(config["patience"]):
                break

    checkpoint = torch.load(best_checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    reload_delta = verify_checkpoint_reload(model, best_checkpoint, validation_loader, device)
    test_loader = _loader(test_dataset, int(config["batch_size"]), False, seed)
    test_metrics = evaluate(model, test_loader, device)
    review = write_prediction_review(model, _loader(test_dataset, 1, False, seed), output / "review_bundle", device)
    (output / "final_metrics.json").write_text(
        json.dumps(
            {
                "best_epoch": best_epoch,
                "best_validation_mean_iou": best_mean_iou,
                "test_metrics": test_metrics,
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    (output / "confusion_matrix.json").write_text(
        json.dumps(
            {
                "class_order": list(TRAINING_CLASS_NAMES),
                "test_confusion_matrix": test_metrics["confusion_matrix"],
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    with (output / "training_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("epoch", "learning_rate", "train_loss", "validation_loss", "validation_mean_iou", "validation_pixel_accuracy"))
        writer.writeheader()
        for item in history:
            writer.writerow(
                {
                    "epoch": item["epoch"],
                    "learning_rate": item["learning_rate"],
                    "train_loss": item["train_loss"],
                    "validation_loss": item["validation"]["loss"],
                    "validation_mean_iou": item["validation"]["mean_iou"],
                    "validation_pixel_accuracy": item["validation"]["pixel_accuracy"],
                }
            )
    return {
        "success": bool(
            math.isfinite(float(test_metrics["loss"]))
            and reload_delta <= 1e-6
            and test_metrics["predicted_class_count"] >= 2
        ),
        "mode": "full",
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_mean_iou": best_mean_iou,
        "best_checkpoint_path": str(best_checkpoint),
        "checkpoint_reload_max_abs_logit_delta": reload_delta,
        "test_evaluation_count": 1,
        "test_metrics": test_metrics,
        "prediction_review": review,
        "split_sample_counts": {
            "train": len(train_dataset),
            "validation": len(validation_dataset),
            "test": len(test_dataset),
        },
        "unweighted_cross_entropy": True,
        "initialization": initialization,
        "history": history,
        "live_rover_commands_sent": False,
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_valid = 0
    for batch in loader:
        images = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = _full_resolution_logits(model(images).logits, labels.shape[-2:])
        valid = int((labels != 255).sum().item())
        loss_sum = segmentation_loss_sum(logits, labels)
        loss = loss_sum / max(valid, 1)
        if not torch.isfinite(loss):
            raise RuntimeError("training loss became NaN or Inf")
        loss.backward()
        optimizer.step()
        total_loss += float(loss_sum.item())
        total_valid += valid
    return total_loss / max(total_valid, 1)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, object]:
    model.eval()
    confusion = torch.zeros((3, 3), dtype=torch.int64)
    total_loss = 0.0
    total_valid = 0
    for batch in loader:
        images = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        logits = _full_resolution_logits(model(images).logits, labels.shape[-2:])
        valid = labels != 255
        loss_sum = segmentation_loss_sum(logits, labels)
        total_loss += float(loss_sum.item())
        total_valid += int(valid.sum().item())
        predictions = logits.argmax(dim=1)
        encoded = labels[valid] * 3 + predictions[valid]
        confusion += torch.bincount(encoded.detach().cpu(), minlength=9).reshape(3, 3)
    return metrics_from_confusion(confusion, total_loss / max(total_valid, 1))


def segmentation_loss_sum(logits: Tensor, labels: Tensor) -> Tensor:
    if logits.ndim != 4 or labels.ndim != 3 or logits.shape[0] != labels.shape[0]:
        raise ValueError("logits and labels have incompatible segmentation shapes")
    valid = labels != 255
    if not torch.any(valid):
        raise ValueError("segmentation batch contains no trainable pixels")
    safe_labels = labels.clamp(0, logits.shape[1] - 1)
    log_probabilities = functional.log_softmax(logits, dim=1)
    selected = log_probabilities.gather(1, safe_labels.unsqueeze(1)).squeeze(1)
    return -(selected * valid.to(selected.dtype)).sum()


def metrics_from_confusion(confusion: Tensor, loss: float) -> dict[str, object]:
    matrix = confusion.to(torch.float64)
    true_pixels = matrix.sum(dim=1)
    predicted_pixels = matrix.sum(dim=0)
    true_positive = matrix.diag()
    unions = true_pixels + predicted_pixels - true_positive
    iou = torch.where(unions > 0, true_positive / unions, torch.zeros_like(unions))
    total = float(matrix.sum().item())
    obstacle_total = float(true_pixels[2].item())
    traversable_total = float((true_pixels[0] + true_pixels[1]).item())
    return {
        "loss": float(loss),
        "pixel_accuracy": float(true_positive.sum().item() / total) if total else 0.0,
        "per_class_iou": {name: float(iou[index].item()) for index, name in enumerate(TRAINING_CLASS_NAMES)},
        "mean_iou": float(iou.mean().item()),
        "confusion_matrix": confusion.tolist(),
        "predicted_class_count": int((predicted_pixels > 0).sum().item()),
        "false_traversable_rate": float((matrix[2, 0] + matrix[2, 1]).item() / obstacle_total) if obstacle_total else 0.0,
        "obstacle_to_on_road_rate": float(matrix[2, 0].item() / obstacle_total) if obstacle_total else 0.0,
        "obstacle_to_off_road_rate": float(matrix[2, 1].item() / obstacle_total) if obstacle_total else 0.0,
        "false_non_traversable_rate": float((matrix[0, 2] + matrix[1, 2]).item() / traversable_total) if traversable_total else 0.0,
        "on_road_off_road_confusion_rate": float((matrix[0, 1] + matrix[1, 0]).item() / traversable_total) if traversable_total else 0.0,
    }


def save_checkpoint(
    model: nn.Module,
    path: Path,
    epoch: int,
    metrics: dict[str, object],
    parent_checkpoint: Path | None = None,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model.config.to_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "source_checkpoint": CHECKPOINT,
            "source_revision": REVISION,
            "parent_checkpoint": str(parent_checkpoint) if parent_checkpoint else None,
            "parent_checkpoint_sha256": (
                _sha256(parent_checkpoint) if parent_checkpoint else None
            ),
        },
        path,
    )


def load_training_model(
    initial_checkpoint: Path | None,
    device: torch.device,
    initialization_mode: str = "three_class_checkpoint",
) -> tuple[nn.Module, dict[str, object]]:
    if initial_checkpoint is None:
        return build_traversability_segformer(pretrained=True).to(device), {
            "mode": "ade20k_pretrained_backbone",
            "checkpoint": CHECKPOINT,
            "revision": REVISION,
        }
    if not initial_checkpoint.is_file():
        raise ValueError(f"initial checkpoint is missing: {initial_checkpoint}")
    checkpoint = torch.load(initial_checkpoint, map_location=device, weights_only=True)
    validate_three_class_checkpoint(checkpoint)
    config = checkpoint["model_config"]
    model = build_traversability_segformer(False, config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, {
        "mode": initialization_mode,
        "checkpoint": str(initial_checkpoint),
        "checkpoint_sha256": _sha256(initial_checkpoint),
        "source_epoch": int(checkpoint["epoch"]),
        "source_metrics": checkpoint["metrics"],
    }


@torch.no_grad()
def verify_checkpoint_reload(model: nn.Module, path: Path, loader: DataLoader, device: torch.device) -> float:
    batch = next(iter(loader))
    images = batch["pixel_values"].to(device)
    model.eval()
    expected = model(images).logits
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    reloaded = build_traversability_segformer(False, checkpoint["model_config"]).to(device)
    reloaded.load_state_dict(checkpoint["model_state_dict"])
    reloaded.eval()
    actual = reloaded(images).logits
    return float((expected - actual).abs().max().item())


@torch.no_grad()
def write_prediction_review(
    model: nn.Module,
    loader: DataLoader,
    output: Path,
    device: torch.device,
) -> dict[str, object]:
    for name in ("images", "ground_truth", "predictions", "confidence", "overlays"):
        (output / name).mkdir(parents=True, exist_ok=True)
    model.eval()
    records: list[dict[str, object]] = []
    for batch in loader:
        images = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)
        logits = _full_resolution_logits(model(images).logits, labels.shape[-2:])
        probabilities = logits.softmax(dim=1)
        confidence, predictions = probabilities.max(dim=1)
        for index, sample_id in enumerate(batch["sample_id"]):
            target = labels[index].cpu().numpy()
            prediction = predictions[index].cpu().numpy()
            valid = target != 255
            prediction_source = training_prediction_to_source(prediction)
            prediction_source[~valid] = 0
            confidence_image = np.round(confidence[index].cpu().numpy() * 255).astype(np.uint8)
            confidence_image[~valid] = 0
            image_bgr = cv2.imread(batch["image_path"][index], cv2.IMREAD_COLOR)
            ground_source = cv2.imread(batch["mask_path"][index], cv2.IMREAD_UNCHANGED)
            if image_bgr is None or ground_source is None:
                raise OSError(f"cannot reload approved source assets for {sample_id}")
            prediction_source = restore_letterbox(
                prediction_source,
                image_bgr.shape[:2],
                cv2.INTER_NEAREST,
            )
            confidence_image = restore_letterbox(
                confidence_image,
                image_bgr.shape[:2],
                cv2.INTER_LINEAR,
            )
            ground_color = _colorize_source_mask(ground_source)
            prediction_color = _colorize_source_mask(prediction_source)
            overlay = cv2.addWeighted(image_bgr, 0.55, cv2.cvtColor(prediction_color, cv2.COLOR_RGB2BGR), 0.45, 0.0)
            cv2.imwrite(str(output / "images" / f"{sample_id}.jpg"), image_bgr)
            cv2.imwrite(str(output / "ground_truth" / f"{sample_id}.png"), cv2.cvtColor(ground_color, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(output / "predictions" / f"{sample_id}.png"), prediction_source)
            cv2.imwrite(str(output / "confidence" / f"{sample_id}.png"), confidence_image)
            cv2.imwrite(str(output / "overlays" / f"{sample_id}.jpg"), overlay)
            obstacle = target == 2
            dangerous = int(np.logical_and(obstacle, prediction != 2).sum())
            records.append({"sample_id": sample_id, "dangerous_false_traversable_pixels": dangerous})
    records.sort(key=lambda item: (-int(item["dangerous_false_traversable_pixels"]), str(item["sample_id"])))
    _write_review_html(output, records)
    _write_contact_sheet(output, records, "contact_sheet.jpg", limit=len(records))
    _write_contact_sheet(output, records, "failure_examples.jpg", limit=min(12, len(records)))
    (output / "review_report.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    return {"path": str(output), "sample_count": len(records), "failure_examples": min(12, len(records))}


def _write_review_html(output: Path, records: list[dict[str, object]]) -> None:
    cards = []
    for item in records:
        sample = html.escape(str(item["sample_id"]))
        cards.append(
            f"<article><h2>{sample}</h2><p>false-traversable pixels={item['dangerous_false_traversable_pixels']}</p>"
            f"<div><img src='images/{sample}.jpg'><img src='ground_truth/{sample}.png'>"
            f"<img src='overlays/{sample}.jpg'><img src='confidence/{sample}.png'></div></article>"
        )
    document = """<!doctype html><html><head><meta charset='utf-8'><title>SegFormer Offline Review</title>
<style>body{font-family:system-ui;margin:20px}article{border-bottom:1px solid #bbb;padding:12px 0}div{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}img{width:100%;height:auto}@media(max-width:900px){div{grid-template-columns:1fr}}</style>
</head><body><h1>SegFormer Offline Review</h1><p>Original / ground truth / prediction overlay / confidence. Predictions are restored from the letterboxed model input to source resolution. Offline research output only.</p>__CARDS__</body></html>""".replace("__CARDS__", "\n".join(cards))
    (output / "review.html").write_text(document, encoding="utf-8")


def _write_contact_sheet(output: Path, records: list[dict[str, object]], name: str, limit: int) -> None:
    tile_width, tile_height = 256, 144
    rows = records[:limit]
    canvas = np.full((len(rows) * tile_height, tile_width * 3, 3), 245, dtype=np.uint8)
    for row_index, item in enumerate(rows):
        sample = str(item["sample_id"])
        assets = (
            cv2.imread(str(output / "images" / f"{sample}.jpg")),
            cv2.imread(str(output / "ground_truth" / f"{sample}.png")),
            cv2.imread(str(output / "overlays" / f"{sample}.jpg")),
        )
        for column, asset in enumerate(assets):
            if asset is None:
                raise OSError(f"cannot read review asset for {sample}")
            canvas[row_index * tile_height : (row_index + 1) * tile_height, column * tile_width : (column + 1) * tile_width] = cv2.resize(asset, (tile_width, tile_height), interpolation=cv2.INTER_NEAREST if column == 1 else cv2.INTER_AREA)
    if not cv2.imwrite(str(output / name), canvas):
        raise OSError(f"cannot write {name}")


def _select_overfit_indices(dataset: TraversabilityDataset, count: int) -> list[int]:
    selected: list[int] = []
    covered_rides: set[str] = set()
    covered_classes: set[int] = set()
    candidates: list[tuple[int, set[int]]] = []
    for index, sample in enumerate(dataset.samples):
        mask = cv2.imread(str(sample.mask_path), cv2.IMREAD_UNCHANGED)
        assert mask is not None
        candidates.append((index, set(int(value) for value in np.unique(mask)) - {0}))
    while candidates and len(selected) < count:
        candidates.sort(
            key=lambda item: (
                -len(item[1] - covered_classes),
                -(dataset.samples[item[0]].ride_id not in covered_rides),
                dataset.samples[item[0]].sample_id,
            )
        )
        index, classes = candidates.pop(0)
        selected.append(index)
        covered_classes.update(classes)
        covered_rides.add(dataset.samples[index].ride_id)
    if len(selected) != count or covered_classes != {1, 2, 3}:
        raise ValueError("cannot select a diverse overfit subset covering all source classes")
    return selected


def _loader(dataset: object, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def _full_resolution_logits(logits: Tensor, size: tuple[int, int]) -> Tensor:
    return functional.interpolate(logits, size=size, mode="bilinear", align_corners=False)


def _colorize_source_mask(mask: np.ndarray) -> np.ndarray:
    colors = np.array([[0, 0, 0], [38, 166, 91], [43, 126, 216], [220, 50, 47]], dtype=np.uint8)
    return colors[mask]


def _environment_report(device: torch.device, manifest: Path, elapsed: float) -> dict[str, object]:
    packages = {}
    for name in ("torch", "torchvision", "transformers", "safetensors", "Pillow"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "peak_vram_bytes": {
            "allocated": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
            "reserved": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0,
        },
        "elapsed_seconds": elapsed,
        "torch_cuda_version": torch.version.cuda,
        "packages": packages,
        "source_checkpoint": CHECKPOINT,
        "source_revision": REVISION,
        "source_weight_file": WEIGHT_FILE,
        "checkpoint_head_replaced": True,
        "num_labels": 3,
        "checkpoint_license": "NVIDIA research/evaluation-only non-commercial use",
        "checkpoint_license_source": "https://github.com/NVlabs/SegFormer/blob/master/LICENSE",
        "manifest_path": str(manifest),
        "manifest_sha256": manifest_sha256(manifest),
        "ignore_index": 255,
        "input_geometry": "aspect-ratio-preserving letterbox to configured square size",
        "image_normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "test_used_for_hyperparameter_selection": False,
        "offline_only": True,
        "git_commit": _git_commit(),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


if __name__ == "__main__":
    raise SystemExit(main())
