#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch
import yaml


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.training.datasets.traversability_dataset_v1 import TraversabilityDataset, manifest_sha256
from research.training.train_traversability_segformer import (
    _loader,
    evaluate,
    load_training_model,
    write_prediction_review,
)
from research.training.traversability_dataset_v2 import regression_assessment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare approved SegFormer-B0 v1 and v2 checkpoints offline."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--v1-checkpoint", required=True)
    parser.add_argument("--v2-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = Path(args.manifest).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    v1_checkpoint = Path(args.v1_checkpoint).expanduser().resolve()
    v2_checkpoint = Path(args.v2_checkpoint).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists():
        raise SystemExit(f"output path already exists: {output}")
    temporary = output.parent / f".{output.name}.tmp"
    if temporary.exists():
        raise SystemExit(f"temporary output already exists: {temporary}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    gate = config["regression_gate"]
    if args.require_cuda and not torch.cuda.is_available():
        raise SystemExit("CUDA is required but unavailable")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = int(config["seed"])
    image_size = int(config["image_size"])
    batch_size = int(config["batch_size"])
    datasets = {
        split: TraversabilityDataset(
            manifest,
            split,
            image_size=image_size,
            augment=False,
            seed=seed,
        )
        for split in ("validation", "test", "new_holdout")
    }
    try:
        temporary.mkdir(parents=True)
        shutil.copy2(config_path, temporary / "frozen_config.yaml")
        metrics: dict[str, dict[str, object]] = {}
        for model_name, checkpoint_path in (
            ("v1", v1_checkpoint),
            ("v2", v2_checkpoint),
        ):
            model, initialization = load_training_model(checkpoint_path, device)
            metrics[model_name] = {"initialization": initialization}
            for split, dataset in datasets.items():
                loader = _loader(dataset, batch_size, False, seed)
                metrics[model_name][split] = evaluate(model, loader, device)
            for split in ("test", "new_holdout"):
                write_prediction_review(
                    model,
                    _loader(datasets[split], 1, False, seed),
                    temporary / "reviews" / model_name / split,
                    device,
                )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        assessments = {
            split: regression_assessment(
                metrics["v1"][split],
                metrics["v2"][split],
                float(gate["max_mean_iou_drop"]),
                float(gate["max_pixel_accuracy_drop"]),
                float(gate["max_false_traversable_rate_increase"]),
            )
            for split in ("validation", "test", "new_holdout")
        }
        fixed_evaluation_pass = all(
            assessments[split]["pass"] for split in ("validation", "test")
        )
        report = {
            "success": fixed_evaluation_pass,
            "status": "PASS" if fixed_evaluation_pass else "FAIL_V1_FIXED_EVALUATION_REGRESSION",
            "metrics": metrics,
            "assessments": assessments,
            "v1_fixed_evaluation_splits": ["validation", "test"],
            "new_holdout_split": "new_holdout",
            "v1_checkpoint": str(v1_checkpoint),
            "v1_checkpoint_sha256": _sha256(v1_checkpoint),
            "v2_checkpoint": str(v2_checkpoint),
            "v2_checkpoint_sha256": _sha256(v2_checkpoint),
            "manifest_path": str(manifest),
            "manifest_sha256": manifest_sha256(manifest),
            "git_commit": _git_commit(),
            "packages": _packages(),
            "test_used_for_hyperparameter_selection": False,
            "additional_training_performed": False,
            "planner_or_live_rover_integration": False,
        }
        (temporary / "comparison_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["success"] else 2


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


def _packages() -> dict[str, str | None]:
    result = {}
    for name in ("torch", "torchvision", "transformers", "safetensors", "Pillow"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


if __name__ == "__main__":
    raise SystemExit(main())
