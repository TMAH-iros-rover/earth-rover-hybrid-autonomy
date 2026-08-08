#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.training.traversability_dataset_v2 import (
    TRAINING_CLASSES,
    choose_new_group_split,
    group_id,
)


FIELDS = (
    "sample_id",
    "image_path",
    "mask_path",
    "split",
    "ride_id",
    "timestamp",
    "frame_id",
    "manifest_index",
    "playlist",
    "segment",
    "source_bundle",
    "source_image_path",
    "source_mask_path",
    "source_dataset",
    "source_camera_uid",
    "source_candidate_id",
    "provenance_complete",
)
CLASS_NAMES = ("IGNORE", *TRAINING_CLASSES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build immutable traversability dataset v2 from approved v1 and manual v2."
    )
    parser.add_argument("--approved-v1", required=True)
    parser.add_argument("--manual-v2", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-v1-count", type=int, default=120)
    parser.add_argument("--expected-new-count", type=int, default=33)
    parser.add_argument("--new-holdout-ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=20260723)
    return parser.parse_args()


def build_dataset_v2(
    approved_v1: str | Path,
    manual_v2: str | Path,
    output_dir: str | Path,
    expected_v1_count: int = 120,
    expected_new_count: int = 33,
    new_holdout_ratio: float = 0.20,
    seed: int = 20260723,
) -> dict[str, object]:
    v1 = Path(approved_v1).expanduser().resolve()
    manual = Path(manual_v2).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise ValueError(f"output path already exists: {output}")
    temporary = output.parent / f".{output.name}.tmp"
    if temporary.exists():
        raise ValueError(f"temporary output already exists: {temporary}")
    if any(output == source or source in output.parents for source in (v1, manual)):
        raise ValueError("output directory must remain separate from immutable inputs")

    v1_rows = _read_csv(v1 / "manifest.csv")
    manual_rows = _read_csv(manual / "metadata.csv")
    if len(v1_rows) != expected_v1_count:
        raise ValueError(f"expected {expected_v1_count} v1 samples, found {len(v1_rows)}")
    if len(manual_rows) != expected_new_count:
        raise ValueError(f"expected {expected_new_count} manual samples, found {len(manual_rows)}")
    _approved_report(v1, expected_v1_count)
    manual_report = _manual_report(manual, expected_new_count)
    contract_v1 = v1 / "label_contract.yaml"
    contract_manual = manual / "label_contract.yaml"
    if _sha256(contract_v1) != _sha256(contract_manual):
        raise ValueError("v1 and manual-v2 label contracts are not byte-identical")
    _validate_contract(contract_v1)
    _validate_v1_splits(v1_rows)

    v1_ids = {row["sample_id"] for row in v1_rows}
    manual_ids = {row["sample_id"] for row in manual_rows}
    if len(v1_ids) != len(v1_rows) or len(manual_ids) != len(manual_rows):
        raise ValueError("source dataset contains duplicate sample IDs")
    overlap = sorted(v1_ids & manual_ids)
    if overlap:
        raise ValueError(f"v1 and manual-v2 sample IDs overlap: {overlap}")
    v1_rides = {row["ride_id"] for row in v1_rows}
    manual_rides = {row["ride_id"] for row in manual_rows}
    if v1_rides & manual_rides:
        raise ValueError(f"manual-v2 reuses approved-v1 rides: {sorted(v1_rides & manual_rides)}")

    v1_assets, v1_pixels = _validate_v1_assets(v1, v1_rows)
    manual_assets, manual_pixels = _validate_manual_assets(manual, manual_rows)
    exact_duplicates = _exact_image_duplicates(v1_assets, manual_assets)
    if exact_duplicates:
        raise ValueError(f"manual-v2 contains exact v1 image duplicates: {exact_duplicates}")
    minimum_hash_distance = _minimum_cross_hash_distance(v1_assets, manual_assets)
    group_split = choose_new_group_split(
        manual_rows,
        manual_pixels,
        seed,
        holdout_ratio=new_holdout_ratio,
    )
    group_to_split = {
        group: split_name for split_name, groups in group_split.items() for group in groups
    }

    try:
        for directory in ("images", "masks", "metadata", "splits", "fixed_v1_splits"):
            (temporary / directory).mkdir(parents=True, exist_ok=True)
        shutil.copy2(contract_v1, temporary / "label_contract.yaml")
        shutil.copy2(v1 / "classes.yaml", temporary / "classes.yaml")
        for split_name in ("train", "validation", "test"):
            shutil.copy2(
                v1 / "splits" / f"{split_name}.csv",
                temporary / "fixed_v1_splits" / f"{split_name}.csv",
            )
        merged: list[dict[str, str]] = []
        for row in v1_rows:
            merged.append(
                _copy_row(
                    row,
                    v1_assets[row["sample_id"]],
                    temporary,
                    source_bundle="approved_v1_120",
                    source_dataset="",
                    source_camera_uid="",
                    source_candidate_id="",
                    provenance_complete="true",
                )
            )
        for row in manual_rows:
            sample_id = row["sample_id"]
            source_image, source_mask = manual_assets[sample_id]
            normalized = {
                "sample_id": sample_id,
                "split": group_to_split[group_id(row)],
                "ride_id": row["ride_id"],
                "timestamp": row["timestamp_sec"],
                "frame_id": "-1",
                "manifest_index": "-1",
                "playlist": row["playlist_path"],
                "segment": "",
            }
            merged.append(
                _copy_row(
                    normalized,
                    (source_image, source_mask),
                    temporary,
                    source_bundle="manual_v2_33",
                    source_dataset=row["dataset"],
                    source_camera_uid=row["camera_uid"],
                    source_candidate_id=row["source_candidate_id"],
                    provenance_complete="false",
                )
            )
        merged.sort(key=lambda row: row["sample_id"])
        _write_manifests(temporary, merged)
        statistics = _statistics(merged, {**v1_pixels, **manual_pixels})
        split_report = {
            "seed": seed,
            "existing_v1_split_preserved": True,
            "existing_v1_split_manifest_sha256": {
                name: _sha256(v1 / "splits" / f"{name}.csv")
                for name in ("train", "validation", "test")
            },
            "new_group_definition": "ride_id; all timestamps from one source ride stay together",
            "new_holdout_ratio_target": new_holdout_ratio,
            "new_groups": {name: list(groups) for name, groups in group_split.items()},
            "new_group_details": _group_details(manual_rows, group_to_split),
            "new_group_leakage": [],
            "statistics": statistics,
        }
        _write_json(temporary / "split_report.json", split_report)
        _write_json(temporary / "dataset_statistics.json", statistics)
        manifest_hash = _sha256(temporary / "manifest.csv")
        report = {
            "valid": True,
            "dataset_name": "traversability_dataset_v2_approved_153",
            "sample_count": len(merged),
            "approved_v1_sample_count": len(v1_rows),
            "manual_v2_sample_count": len(manual_rows),
            "label_contract_sha256": _sha256(contract_v1),
            "label_contract_exact_match": True,
            "sample_id_overlap": [],
            "approved_v1_ride_overlap": [],
            "exact_image_duplicate_count": 0,
            "minimum_cross_dataset_dhash_distance": minimum_hash_distance,
            "manifest_path": str(output / "manifest.csv"),
            "manifest_sha256": manifest_hash,
            "git_commit": _git_commit(),
            "manual_validation_report_sha256": _sha256(
                manual / "validation_report.json"
            ),
            "manual_source_image_bytes_preserved": manual_report[
                "source_image_bytes_preserved"
            ],
            "existing_v1_modified": False,
            "manual_v2_modified": False,
            "model_training_performed": False,
            "live_rover_commands_sent": False,
        }
        _write_json(temporary / "merge_report.json", report)
        os.replace(temporary, output)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return {"merge": report, "split": split_report}


def _approved_report(root: Path, expected_count: int) -> None:
    report = json.loads((root / "merge_report.json").read_text(encoding="utf-8"))
    if report.get("valid") is not True or report.get("sample_count") != expected_count:
        raise ValueError("approved v1 merge report is not valid")


def _manual_report(root: Path, expected_count: int) -> dict[str, object]:
    report = json.loads((root / "validation_report.json").read_text(encoding="utf-8"))
    required = (
        report.get("valid") is True
        and report.get("sample_count") == expected_count
        and report.get("segmentation_class_mask_count") == expected_count
        and report.get("semantic_mask_source") == "SegmentationClass"
        and report.get("segmentation_object_used") is False
        and report.get("source_image_bytes_preserved") is True
    )
    if not required:
        raise ValueError("manual-v2 validation report is not approved")
    return report


def _validate_contract(path: Path) -> None:
    contract = yaml.safe_load(path.read_text(encoding="utf-8"))
    expected = {0: "IGNORE", 1: "ON_ROAD", 2: "OFF_ROAD", 3: "OBSTACLE"}
    actual = {int(item["id"]): str(item["name"]) for item in contract["classes"]}
    if actual != expected or int(contract["ignore_index"]) != 0:
        raise ValueError(f"unexpected traversability label contract: {actual}")


def _validate_v1_splits(rows: list[dict[str, str]]) -> None:
    by_ride: dict[str, set[str]] = {}
    for row in rows:
        if row["split"] not in {"train", "validation", "test"}:
            raise ValueError(f"unexpected v1 split: {row['split']}")
        by_ride.setdefault(row["ride_id"], set()).add(row["split"])
    leaking = sorted(ride for ride, splits in by_ride.items() if len(splits) != 1)
    if leaking:
        raise ValueError(f"approved v1 ride leakage: {leaking}")
    if set(row["split"] for row in rows) != {"train", "validation", "test"}:
        raise ValueError("approved v1 must contain train, validation, and test")


def _validate_v1_assets(
    root: Path,
    rows: list[dict[str, str]],
) -> tuple[dict[str, tuple[Path, Path]], dict[str, dict[str, int]]]:
    return _validate_assets(root, rows, "timestamp")


def _validate_manual_assets(
    root: Path,
    rows: list[dict[str, str]],
) -> tuple[dict[str, tuple[Path, Path]], dict[str, dict[str, int]]]:
    return _validate_assets(root, rows, "timestamp_sec")


def _validate_assets(
    root: Path,
    rows: list[dict[str, str]],
    timestamp_field: str,
) -> tuple[dict[str, tuple[Path, Path]], dict[str, dict[str, int]]]:
    assets: dict[str, tuple[Path, Path]] = {}
    pixels: dict[str, dict[str, int]] = {}
    for row in rows:
        sample_id = row["sample_id"]
        image = (root / row["image_path"]).resolve()
        mask = (root / row["mask_path"]).resolve()
        if root not in image.parents or root not in mask.parents:
            raise ValueError(f"asset path escapes source bundle: {sample_id}")
        image_data = cv2.imread(str(image), cv2.IMREAD_COLOR)
        mask_data = cv2.imread(str(mask), cv2.IMREAD_UNCHANGED)
        if image_data is None or mask_data is None or mask_data.ndim != 2:
            raise ValueError(f"unreadable image or class-ID mask: {sample_id}")
        if image_data.shape[:2] != mask_data.shape or mask_data.dtype != np.uint8:
            raise ValueError(f"invalid image-mask pair: {sample_id}")
        values, counts = np.unique(mask_data, return_counts=True)
        if not set(int(value) for value in values).issubset({0, 1, 2, 3}):
            raise ValueError(f"invalid mask class IDs: {sample_id}")
        float(row[timestamp_field])
        assets[sample_id] = (image, mask)
        by_id = {int(value): int(count) for value, count in zip(values, counts)}
        pixels[sample_id] = {
            name: by_id.get(class_id, 0)
            for class_id, name in enumerate(CLASS_NAMES)
        }
    return assets, pixels


def _copy_row(
    row: dict[str, str],
    assets: tuple[Path, Path],
    output: Path,
    source_bundle: str,
    source_dataset: str,
    source_camera_uid: str,
    source_candidate_id: str,
    provenance_complete: str,
) -> dict[str, str]:
    source_image, source_mask = assets
    sample_id = row["sample_id"]
    image_relative = f"images/{sample_id}{source_image.suffix.lower()}"
    mask_relative = f"masks/{sample_id}.png"
    shutil.copy2(source_image, output / image_relative)
    shutil.copy2(source_mask, output / mask_relative)
    if (
        _sha256(source_image) != _sha256(output / image_relative)
        or _sha256(source_mask) != _sha256(output / mask_relative)
    ):
        raise ValueError(f"copied image or mask bytes changed: {sample_id}")
    merged = {
        "sample_id": sample_id,
        "image_path": image_relative,
        "mask_path": mask_relative,
        "split": row["split"],
        "ride_id": row["ride_id"],
        "timestamp": row["timestamp"],
        "frame_id": row["frame_id"],
        "manifest_index": row["manifest_index"],
        "playlist": row["playlist"],
        "segment": row["segment"],
        "source_bundle": source_bundle,
        "source_image_path": str(source_image),
        "source_mask_path": str(source_mask),
        "source_dataset": source_dataset,
        "source_camera_uid": source_camera_uid,
        "source_candidate_id": source_candidate_id,
        "provenance_complete": provenance_complete,
    }
    _write_json(output / "metadata" / f"{sample_id}.json", merged)
    return merged


def _write_manifests(output: Path, rows: list[dict[str, str]]) -> None:
    _write_csv(output / "metadata.csv", rows)
    _write_csv(output / "manifest.csv", rows)
    for split in ("train", "validation", "test", "new_holdout"):
        _write_csv(
            output / "splits" / f"{split}.csv",
            [row for row in rows if row["split"] == split],
        )


def _group_details(
    rows: list[dict[str, str]],
    group_to_split: dict[str, str],
) -> dict[str, object]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(group_id(row), []).append(row)
    return {
        group: {
            "split": group_to_split[group],
            "sample_ids": sorted(row["sample_id"] for row in selected),
            "timestamp_min": min(float(row["timestamp_sec"]) for row in selected),
            "timestamp_max": max(float(row["timestamp_sec"]) for row in selected),
        }
        for group, selected in sorted(grouped.items())
    }


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _statistics(
    rows: list[dict[str, str]],
    pixels: dict[str, dict[str, int]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for split in ("overall", "train", "validation", "test", "new_holdout"):
        selected = rows if split == "overall" else [row for row in rows if row["split"] == split]
        counts = sum((Counter(pixels[row["sample_id"]]) for row in selected), Counter())
        total = sum(counts.values())
        result[split] = {
            "sample_count": len(selected),
            "ride_count": len({row["ride_id"] for row in selected}),
            "source_distribution": dict(
                sorted(Counter(row["source_bundle"] for row in selected).items())
            ),
            "class_pixel_counts": {name: counts[name] for name in CLASS_NAMES},
            "class_pixel_fractions": {
                name: counts[name] / total if total else 0.0 for name in CLASS_NAMES
            },
        }
    return result


def _exact_image_duplicates(
    left: dict[str, tuple[Path, Path]],
    right: dict[str, tuple[Path, Path]],
) -> list[dict[str, str]]:
    by_hash: dict[str, str] = {}
    for sample_id, (image, _) in left.items():
        by_hash[_sha256(image)] = sample_id
    return [
        {"v1_sample_id": by_hash[digest], "manual_sample_id": sample_id}
        for sample_id, (image, _) in right.items()
        if (digest := _sha256(image)) in by_hash
    ]


def _difference_hash(path: Path) -> int:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    resized = cv2.resize(image, (9, 8), interpolation=cv2.INTER_AREA)
    bits = resized[:, 1:] > resized[:, :-1]
    value = 0
    for bit in bits.ravel():
        value = (value << 1) | int(bit)
    return value


def _minimum_cross_hash_distance(
    left: dict[str, tuple[Path, Path]],
    right: dict[str, tuple[Path, Path]],
) -> int | None:
    left_hashes = [_difference_hash(image) for image, _ in left.values()]
    right_hashes = [_difference_hash(image) for image, _ in right.values()]
    return min(
        bin(left_hash ^ right_hash).count("1")
        for left_hash in left_hashes
        for right_hash in right_hashes
    ) if left_hashes and right_hashes else None


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise ValueError(f"required manifest is missing: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def main() -> int:
    args = parse_args()
    try:
        report = build_dataset_v2(
            args.approved_v1,
            args.manual_v2,
            args.output_dir,
            args.expected_v1_count,
            args.expected_new_count,
            args.new_holdout_ratio,
            args.seed,
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise SystemExit(f"traversability v2 dataset build failed: {exc}") from exc
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
