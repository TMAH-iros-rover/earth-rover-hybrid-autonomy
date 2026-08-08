#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.training.datasets.traversability_dataset_v1 import choose_ride_split
from research.training.traversability_annotation import validate_annotation_dataset
from research.training.traversability_review import write_json


FIELDS = (
    "sample_id", "image_path", "mask_path", "split", "ride_id", "timestamp",
    "frame_id", "manifest_index", "playlist", "segment", "source_bundle",
    "source_image_path", "source_mask_path",
)
CLASS_NAMES = ("IGNORE", "ON_ROAD", "OFF_ROAD", "OBSTACLE")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge the approved 20+100 traversability annotations.")
    parser.add_argument("--pilot-bundle", required=True)
    parser.add_argument("--pilot-reviewed", required=True)
    parser.add_argument("--expansion-bundle", required=True)
    parser.add_argument("--expansion-reviewed", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260718)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sources = (
        ("approved_pilot_20", Path(args.pilot_bundle).expanduser().resolve(), Path(args.pilot_reviewed).expanduser().resolve(), 20),
        ("approved_expansion_100", Path(args.expansion_bundle).expanduser().resolve(), Path(args.expansion_reviewed).expanduser().resolve(), 100),
    )
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"output directory is not empty: {output}")
    for _, bundle, reviewed, _ in sources:
        if output == bundle or bundle in output.parents or output == reviewed or reviewed in output.parents:
            raise SystemExit("output directory must remain separate from approved source annotations")

    source_rows: list[tuple[str, Path, Path, dict[str, str]]] = []
    source_reports: dict[str, object] = {}
    class_pixels_by_sample: dict[str, dict[str, int]] = {}
    for source_name, bundle, reviewed, expected_count in sources:
        report_path = reviewed / "validation_report.json"
        if not report_path.is_file():
            raise SystemExit(f"approved validation report is missing: {report_path}")
        recorded = json.loads(report_path.read_text(encoding="utf-8"))
        if recorded.get("valid") is not True or recorded.get("validated_mask_count") != expected_count:
            raise SystemExit(f"source validation report is not approved for {expected_count} masks: {report_path}")
        current = validate_annotation_dataset(bundle, masks_dir=reviewed / "masks")
        if not current["valid"] or current["validated_mask_count"] != expected_count:
            raise SystemExit(json.dumps(current, indent=2, sort_keys=True))
        rows = list(csv.DictReader((bundle / "metadata.csv").open(newline="", encoding="utf-8")))
        if len(rows) != expected_count:
            raise SystemExit(f"expected {expected_count} metadata rows in {bundle}, found {len(rows)}")
        for row in rows:
            source_rows.append((source_name, bundle, reviewed, row))
            class_pixels_by_sample[row["sample_id"]] = {
                name: int(current["per_image"][row["sample_id"]]["class_pixels"][name])
                for name in CLASS_NAMES
            }
        source_reports[source_name] = {
            "bundle": str(bundle),
            "reviewed": str(reviewed),
            "sample_count": expected_count,
            "validation_report": str(report_path),
            "validation_report_sha256": _sha256(report_path),
            "masks_fingerprint": _directory_fingerprint(reviewed / "masks"),
        }

    if len(source_rows) != 120:
        raise SystemExit(f"expected exactly 120 approved pairs, found {len(source_rows)}")
    sample_ids = [row[3]["sample_id"] for row in source_rows]
    if len(set(sample_ids)) != 120:
        raise SystemExit("approved sources contain duplicate sample IDs")
    source_keys = [(row[3]["ride_id"], row[3]["manifest_index"]) for row in source_rows]
    if len(set(source_keys)) != 120:
        raise SystemExit("approved sources contain duplicate ride/manifest provenance")

    split = choose_ride_split([item[3] for item in source_rows], class_pixels_by_sample, args.seed)
    ride_to_split = {
        ride: split_name
        for split_name, rides in split.items()
        for ride in rides
    }
    for name in ("images", "masks", "metadata", "splits"):
        (output / name).mkdir(parents=True, exist_ok=True)

    contract_source = sources[0][1] / "label_contract.yaml"
    expansion_contract = sources[1][1] / "label_contract.yaml"
    if _sha256(contract_source) != _sha256(expansion_contract):
        raise SystemExit("approved source bundles use different label contracts")
    shutil.copy2(contract_source, output / "label_contract.yaml")
    classes = {
        "source_mask": {"IGNORE": 0, "ON_ROAD": 1, "OFF_ROAD": 2, "OBSTACLE": 3},
        "training_target": {"ON_ROAD": 0, "OFF_ROAD": 1, "OBSTACLE": 2, "IGNORE": 255},
        "num_labels": 3,
        "ignore_index": 255,
    }
    (output / "classes.yaml").write_text(yaml.safe_dump(classes, sort_keys=False), encoding="utf-8")

    merged_rows: list[dict[str, str]] = []
    for source_name, bundle, reviewed, row in sorted(source_rows, key=lambda item: item[3]["sample_id"]):
        sample_id = row["sample_id"]
        source_image = (bundle / row["image_path"]).resolve()
        source_mask = (reviewed / "masks" / f"{sample_id}.png").resolve()
        image_suffix = source_image.suffix.lower()
        image_relative = f"images/{sample_id}{image_suffix}"
        mask_relative = f"masks/{sample_id}.png"
        _validate_pair(source_image, source_mask, sample_id)
        shutil.copy2(source_image, output / image_relative)
        shutil.copy2(source_mask, output / mask_relative)
        merged = {
            "sample_id": sample_id,
            "image_path": image_relative,
            "mask_path": mask_relative,
            "split": ride_to_split[row["ride_id"]],
            "ride_id": row["ride_id"],
            "timestamp": row["timestamp"],
            "frame_id": row["frame_id"],
            "manifest_index": row["manifest_index"],
            "playlist": row["playlist"],
            "segment": row["segment"],
            "source_bundle": source_name,
            "source_image_path": str(source_image),
            "source_mask_path": str(source_mask),
        }
        merged_rows.append(merged)
        write_json(output / "metadata" / f"{sample_id}.json", merged)

    _write_dataset_manifests(output, merged_rows)

    validation = validate_annotation_dataset(output, masks_dir=output / "masks")
    if not validation["valid"] or validation["validated_mask_count"] != 120:
        raise SystemExit(json.dumps(validation, indent=2, sort_keys=True))
    statistics = _dataset_statistics(merged_rows, validation["per_image"])
    split_report = {
        "seed": args.seed,
        "target_sample_ratios": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "ride_isolation": True,
        "location_isolation_beyond_ride": "not verifiable from annotation metadata",
        "rides": {name: list(rides) for name, rides in split.items()},
        "statistics": statistics,
    }
    report = {
        "valid": True,
        "dataset_name": "traversability_dataset_v1_approved_120_v1",
        "sample_count": len(merged_rows),
        "source_reports": source_reports,
        "source_masks_used": "approved normalized class-ID masks only",
        "cvat_seed_masks_included": False,
        "pseudo_labels_included": False,
        "segmentation_object_included": False,
        "manifest_path": str(output / "manifest.csv"),
        "manifest_sha256": _sha256(output / "manifest.csv"),
        "split_report_path": str(output / "split_report.json"),
        "model_training_performed": False,
        "live_rover_commands_sent": False,
    }
    write_json(output / "validation_report.json", validation)
    write_json(output / "dataset_statistics.json", statistics)
    write_json(output / "split_report.json", split_report)
    write_json(output / "merge_report.json", report)
    print(json.dumps({"merge": report, "split": split_report}, indent=2, sort_keys=True))
    return 0


def _validate_pair(image_path: Path, mask_path: Path, sample_id: str) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if image is None or mask is None or mask.ndim != 2 or mask.dtype != np.uint8:
        raise ValueError(f"unreadable image or non-class-ID mask: {sample_id}")
    if image.shape[:2] != mask.shape:
        raise ValueError(f"image-mask size mismatch: {sample_id}")
    values = set(int(value) for value in np.unique(mask))
    if not values.issubset({0, 1, 2, 3}):
        raise ValueError(f"invalid source mask IDs for {sample_id}: {sorted(values)}")


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _write_dataset_manifests(output: Path, rows: list[dict[str, str]]) -> None:
    _write_csv(output / "metadata.csv", rows)
    _write_csv(output / "manifest.csv", rows)
    for split_name in ("train", "validation", "test"):
        _write_csv(
            output / "splits" / f"{split_name}.csv",
            [row for row in rows if row["split"] == split_name],
        )


def _dataset_statistics(rows: list[dict[str, str]], per_image: dict[str, object]) -> dict[str, object]:
    result: dict[str, object] = {}
    for split_name in ("overall", "train", "validation", "test"):
        selected = rows if split_name == "overall" else [row for row in rows if row["split"] == split_name]
        pixels: Counter[str] = Counter()
        for row in selected:
            pixels.update(per_image[row["sample_id"]]["class_pixels"])
        total = sum(pixels.values())
        result[split_name] = {
            "sample_count": len(selected),
            "ride_count": len({row["ride_id"] for row in selected}),
            "ride_distribution": dict(sorted(Counter(row["ride_id"] for row in selected).items())),
            "class_pixel_counts": {name: pixels[name] for name in CLASS_NAMES},
            "class_pixel_fractions": {name: pixels[name] / total for name in CLASS_NAMES},
        }
    return result


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _directory_fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.glob("*.png")):
        digest.update(path.name.encode())
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
