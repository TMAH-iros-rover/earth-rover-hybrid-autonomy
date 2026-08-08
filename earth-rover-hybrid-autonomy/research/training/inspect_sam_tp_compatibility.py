#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.sam_tp_reproduction import (
    OFFICIAL_CHECKPOINT_URL,
    OFFICIAL_COMMIT,
    OFFICIAL_PAPER,
    OFFICIAL_REPOSITORY,
    UPSTREAM_LICENSE_STATUS,
    compare_state_dicts,
    environment_report,
    git_provenance,
    load_reproduction_config,
    resolve_upstream_paths,
    sha256_file,
    write_json,
)


ARCHITECTURE_FIELDS = (
    ("image_encoder", "trunk", "embed_dim"),
    ("image_encoder", "trunk", "num_heads"),
    ("image_encoder", "trunk", "stages"),
    ("image_encoder", "trunk", "global_att_blocks"),
    ("image_encoder", "neck", "d_model"),
    ("image_encoder", "neck", "backbone_channel_list"),
    ("num_maskmem",),
    ("use_high_res_features_in_sam",),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strictly verify official SAM-TP config and checkpoint compatibility."
    )
    parser.add_argument("--reproduction-config", required=True)
    parser.add_argument("--upstream-root", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def nested_value(mapping: dict, path: tuple[str, ...]):
    value = mapping
    for key in path:
        value = value[key]
    return value


def compare_architecture_configs(
    inference_config: dict,
    training_config: dict,
) -> dict[str, object]:
    inference_model = inference_config["model"]
    training_model = training_config["trainer"]["model"]
    comparisons = []
    for field in ARCHITECTURE_FIELDS:
        inference_value = nested_value(inference_model, field)
        training_value = nested_value(training_model, field)
        comparisons.append(
            {
                "field": ".".join(field),
                "inference": inference_value,
                "training": training_value,
                "matches": inference_value == training_value,
            }
        )
    inference_size = inference_model["image_size"]
    training_size = training_config["scratch"]["resolution"]
    comparisons.append(
        {
            "field": "image_size",
            "inference": inference_size,
            "training": training_size,
            "matches": inference_size == training_size,
        }
    )
    return {
        "compatible": all(item["matches"] for item in comparisons),
        "comparisons": comparisons,
        "training_only_drop_path_rate": training_model["image_encoder"]["trunk"].get(
            "drop_path_rate"
        ),
        "note": (
            "drop_path_rate is training-only stochastic behavior and does not alter "
            "checkpoint tensor shapes; all shape-defining fields must match."
        ),
    }


def inspect_compatibility(
    reproduction_config: str | Path,
    upstream_root: str | Path,
    checkpoint_override: str | Path | None,
) -> dict[str, object]:
    config = load_reproduction_config(reproduction_config)
    paths = resolve_upstream_paths(upstream_root, config, checkpoint_override)
    provenance = git_provenance(paths["root"])
    report: dict[str, object] = {
        "success": False,
        "official_repository": OFFICIAL_REPOSITORY,
        "official_paper": OFFICIAL_PAPER,
        "upstream_license_status": UPSTREAM_LICENSE_STATUS,
        "expected_upstream_commit": OFFICIAL_COMMIT,
        "upstream": provenance,
        "paths": {key: str(value) for key, value in paths.items()},
        "official_load_path": "sam2.sam_tp.SAM_TP -> sam2.build_sam.build_sam2",
        "official_load_strictness": (
            "build_sam2._load_checkpoint calls model.load_state_dict and raises "
            "on any missing or unexpected key; shape mismatches raise directly"
        ),
    }
    if provenance["remote_url"] != OFFICIAL_REPOSITORY:
        raise ValueError(f"unexpected upstream remote: {provenance['remote_url']}")
    if provenance["commit"] != OFFICIAL_COMMIT or provenance["dirty"]:
        raise ValueError(
            f"upstream must be clean at {OFFICIAL_COMMIT}; actual={provenance}"
        )
    for key in ("model_config", "training_config", "checkpoint"):
        if not paths[key].is_file():
            raise FileNotFoundError(f"{key} does not exist: {paths[key]}")

    inference_yaml = yaml.safe_load(paths["model_config"].read_text(encoding="utf-8"))
    training_yaml = yaml.safe_load(paths["training_config"].read_text(encoding="utf-8"))
    architecture = compare_architecture_configs(inference_yaml, training_yaml)
    report["architecture_config_comparison"] = architecture
    if not architecture["compatible"]:
        report["failure_reason"] = "training and inference architecture fields differ"
        return report

    sys.path.insert(0, str(paths["root"]))
    import torch
    from sam2.build_sam import build_sam2

    checkpoint = torch.load(paths["checkpoint"], map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model"), dict):
        report["checkpoint_top_level_keys"] = (
            sorted(checkpoint) if isinstance(checkpoint, dict) else None
        )
        report["failure_reason"] = "checkpoint does not contain a model state dict"
        return report
    report["checkpoint_top_level_keys"] = sorted(checkpoint)
    report["checkpoint"] = {
        "filename": paths["checkpoint"].name,
        "sha256": sha256_file(paths["checkpoint"]),
        "size_bytes": paths["checkpoint"].stat().st_size,
        "source_url": OFFICIAL_CHECKPOINT_URL,
    }
    model = build_sam2(
        str(paths["model_config"]),
        ckpt_path=None,
        device="cpu",
        mode="eval",
    )
    comparison = compare_state_dicts(model.state_dict(), checkpoint["model"])
    report["state_dict_comparison"] = comparison
    if not comparison["compatible"]:
        report["failure_reason"] = "checkpoint state dict does not exactly match the model"
        return report
    model.load_state_dict(checkpoint["model"], strict=True)
    report["strict_state_dict_load"] = True
    report["environment"] = environment_report()
    report["success"] = True
    return report


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    try:
        report = inspect_compatibility(
            args.reproduction_config,
            args.upstream_root,
            args.checkpoint,
        )
    except Exception as exc:
        report = {
            "success": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    write_json(output, report)
    print(yaml.safe_dump(report, sort_keys=False))
    return 0 if report["success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
