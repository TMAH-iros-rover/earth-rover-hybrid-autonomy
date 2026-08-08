#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.training.datasets.action_labels import ACTION_NAMES
from research.training.datasets.frodobots_2k_dataset import FrodoBotsActionDataset, ManifestSample
from research.training.models.action_resnet18 import build_action_resnet18


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Phase 3 ResNet18 tiny-overfit gate.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-count", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--target-accuracy", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _validate_args(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    if output_dir == dataset_root or dataset_root in output_dir.parents:
        raise SystemExit("output-dir must remain outside the immutable raw dataset root")
    output_dir.mkdir(parents=True, exist_ok=True)

    _set_deterministic(args.seed)
    if args.require_cuda and not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this verification but torch.cuda.is_available() is false")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    source_dataset = FrodoBotsActionDataset(dataset_root, args.manifest)
    selected_indices = select_balanced_indices(source_dataset.samples, args.sample_count, args.seed)
    images, targets, selected_rides = cache_selected_samples(source_dataset, selected_indices)
    cached_dataset = TensorDataset(images, targets)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        cached_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        pin_memory=device.type == "cuda",
    )
    eval_loader = DataLoader(cached_dataset, batch_size=args.batch_size, shuffle=False)

    model = build_action_resnet18(len(ACTION_NAMES), pretrained=not args.no_pretrained).to(device)
    class_weights = compute_class_weights(targets, len(ACTION_NAMES)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)

    initial_loss, initial_accuracy = evaluate(model, eval_loader, criterion, device)
    history: list[dict[str, float | int]] = []
    for epoch in range(1, args.max_epochs + 1):
        train_one_epoch(model, train_loader, criterion, optimizer, device)
        loss, accuracy = evaluate(model, eval_loader, criterion, device)
        history.append({"epoch": epoch, "loss": loss, "accuracy": accuracy})
        print(f"epoch={epoch:03d} loss={loss:.6f} accuracy={accuracy:.4f}", flush=True)
        if accuracy >= args.target_accuracy:
            break

    final_loss = float(history[-1]["loss"])
    final_accuracy = float(history[-1]["accuracy"])
    checkpoint_path = output_dir / "resnet18_tiny_overfit.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "action_names": ACTION_NAMES,
            "image_size": 224,
            "seed": args.seed,
            "sample_count": args.sample_count,
        },
        checkpoint_path,
    )
    reload_max_abs_delta = verify_checkpoint_reload(model, checkpoint_path, eval_loader, device)
    success = (
        final_accuracy >= args.target_accuracy
        and final_loss < initial_loss
        and reload_max_abs_delta <= 1e-6
    )
    report = {
        "success": success,
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "torch_version": torch.__version__,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "pretrained": not args.no_pretrained,
        "sample_count": len(selected_indices),
        "sample_manifest_indices": selected_indices,
        "sample_action_distribution": dict(sorted(Counter(ACTION_NAMES[target] for target in targets.tolist()).items())),
        "selected_ride_count": len(set(selected_rides)),
        "selected_ride_ids": sorted(set(selected_rides)),
        "batch_size": args.batch_size,
        "max_epochs": args.max_epochs,
        "epochs_completed": len(history),
        "target_accuracy": args.target_accuracy,
        "initial_loss": initial_loss,
        "initial_accuracy": initial_accuracy,
        "final_loss": final_loss,
        "final_accuracy": final_accuracy,
        "loss_decreased": final_loss < initial_loss,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_reload_max_abs_logit_delta": reload_max_abs_delta,
        "history": history,
    }
    report_path = output_dir / "tiny_overfit_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "history"}, indent=2, sort_keys=True))
    return 0 if success else 1


def select_balanced_indices(
    samples: tuple[ManifestSample, ...],
    sample_count: int,
    seed: int,
) -> list[int]:
    if sample_count <= 0 or sample_count % len(ACTION_NAMES) != 0:
        raise ValueError(f"sample_count must be positive and divisible by {len(ACTION_NAMES)}")
    per_class = sample_count // len(ACTION_NAMES)
    indices_by_action: dict[str, list[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        indices_by_action[sample.action_class].append(index)

    rng = random.Random(seed)
    selected: list[int] = []
    for action in ACTION_NAMES:
        candidates = indices_by_action[action]
        if len(candidates) < per_class:
            raise ValueError(f"action {action} has {len(candidates)} samples; need {per_class}")
        selected.extend(rng.sample(candidates, per_class))
    rng.shuffle(selected)
    return selected


def cache_selected_samples(
    dataset: FrodoBotsActionDataset,
    indices: list[int],
) -> tuple[Tensor, Tensor, list[str]]:
    images: list[Tensor] = []
    targets: list[Tensor] = []
    ride_ids: list[str] = []
    for position, index in enumerate(indices, start=1):
        item = dataset[index]
        images.append(item["image"])
        targets.append(item["target"])
        ride_ids.append(str(item["ride_id"]))
        if position % 20 == 0 or position == len(indices):
            print(f"decoded={position}/{len(indices)}", flush=True)
    return torch.stack(images), torch.stack(targets), ride_ids


def compute_class_weights(targets: Tensor, num_classes: int) -> Tensor:
    counts = torch.bincount(targets, minlength=num_classes).to(torch.float32)
    if counts.numel() != num_classes or torch.any(counts == 0):
        raise ValueError("every action class must be represented in the tiny-overfit subset")
    return targets.numel() / (num_classes * counts)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    model.train()
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images), targets)
        loss.backward()
        optimizer.step()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        total_loss += float(criterion(logits, targets).item()) * targets.size(0)
        total_correct += int((logits.argmax(dim=1) == targets).sum().item())
        total_samples += targets.size(0)
    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def verify_checkpoint_reload(
    model: nn.Module,
    checkpoint_path: Path,
    loader: DataLoader,
    device: torch.device,
) -> float:
    probe_images, _ = next(iter(loader))
    probe_images = probe_images.to(device)
    model.eval()
    expected = model(probe_images)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    reloaded = build_action_resnet18(len(ACTION_NAMES), pretrained=False).to(device)
    reloaded.load_state_dict(checkpoint["model_state_dict"])
    reloaded.eval()
    actual = reloaded(probe_images)
    return float((expected - actual).abs().max().item())


def _set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def _validate_args(args: argparse.Namespace) -> None:
    if args.sample_count <= 0 or args.sample_count % len(ACTION_NAMES) != 0:
        raise SystemExit(f"sample-count must be positive and divisible by {len(ACTION_NAMES)}")
    if args.batch_size <= 0 or args.max_epochs <= 0:
        raise SystemExit("batch-size and max-epochs must be positive")
    if args.learning_rate <= 0:
        raise SystemExit("learning-rate must be positive")
    if not 0.0 < args.target_accuracy <= 1.0:
        raise SystemExit("target-accuracy must be in (0, 1]")


if __name__ == "__main__":
    raise SystemExit(main())
