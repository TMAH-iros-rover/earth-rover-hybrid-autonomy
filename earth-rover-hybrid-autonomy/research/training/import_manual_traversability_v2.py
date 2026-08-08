#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
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

from research.training.cvat_segmentation_masks import (
    EXPECTED_LABELS,
    cvat_label_mapping,
    normalize_cvat_class_mask,
    read_cvat_segmentation_export,
)
from research.training.select_manual_candidates_v2 import SELECTED_FIELDS


METADATA_FIELDS = (
    "sample_id",
    "image_path",
    "mask_path",
    "dataset",
    "ride_id",
    "camera_uid",
    "timestamp_sec",
    "playlist_path",
    "source_candidate_id",
    "review_status",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import reviewed manual-v2 CVAT masks into a validated standalone bundle."
    )
    parser.add_argument("--source-bundle", required=True)
    parser.add_argument("--cvat-export", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--label-contract",
        default=str(ROOT / "research/configs/traversability_dataset_v1.yaml"),
    )
    parser.add_argument("--expected-count", type=int, default=33)
    return parser.parse_args()


def import_manual_bundle(
    source_bundle: str | Path,
    cvat_export: str | Path,
    output_dir: str | Path,
    contract_path: str | Path,
    expected_count: int = 33,
) -> dict[str, object]:
    source = Path(source_bundle).expanduser().resolve()
    export = Path(cvat_export).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    contract = Path(contract_path).expanduser().resolve()
    if expected_count <= 0:
        raise ValueError("expected count must be positive")
    if output.exists():
        raise ValueError(f"output path already exists: {output}")
    temporary = output.parent / f".{output.name}.tmp"
    if temporary.exists():
        raise ValueError(f"temporary output already exists: {temporary}")
    if output == source or source in output.parents:
        raise ValueError("output directory must be outside the source bundle")

    rows = _read_selected_candidates(source / "selected_candidates.csv")
    if len(rows) != expected_count:
        raise ValueError(f"expected {expected_count} selected candidates, found {len(rows)}")
    ids = [row["candidate_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("selected_candidates.csv contains duplicate candidate IDs")
    _validate_selection_file(source / "selection.txt", ids)
    source_images = {row["candidate_id"]: _source_image(source, row) for row in rows}
    image_files = {path.resolve() for path in (source / "images").glob("*.jpg")}
    if image_files != set(source_images.values()):
        missing = sorted(path.name for path in set(source_images.values()) - image_files)
        extra = sorted(path.name for path in image_files - set(source_images.values()))
        raise ValueError(f"source image set mismatch; missing={missing}, extras={extra}")

    name_to_id, colors, ignore_index = _label_contract(contract)
    if name_to_id != EXPECTED_LABELS or ignore_index != EXPECTED_LABELS["IGNORE"]:
        raise ValueError(
            f"label contract mismatch: labels={name_to_id}, ignore_index={ignore_index}"
        )
    label_entries, cvat_masks = read_cvat_segmentation_export(export)
    _validate_labelmap_against_contract(label_entries, name_to_id, colors)
    if set(cvat_masks) != set(ids):
        missing = sorted(set(ids) - set(cvat_masks))
        extra = sorted(set(cvat_masks) - set(ids))
        raise ValueError(f"SegmentationClass mask set mismatch; missing={missing}, extras={extra}")
    if len(cvat_masks) != expected_count:
        raise ValueError(
            f"expected {expected_count} SegmentationClass masks, found {len(cvat_masks)}"
        )

    index_to_id, color_to_id = cvat_label_mapping(label_entries)
    masks: dict[str, np.ndarray] = {}
    image_shapes: dict[str, tuple[int, int]] = {}
    source_hashes: dict[str, str] = {}
    for sample_id in ids:
        image = cv2.imread(str(source_images[sample_id]), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"source image is unreadable: {sample_id}")
        mask = normalize_cvat_class_mask(
            cvat_masks[sample_id], index_to_id, color_to_id
        )
        if mask.shape != image.shape[:2]:
            raise ValueError(f"image and SegmentationClass mask dimensions differ: {sample_id}")
        invalid = set(int(value) for value in np.unique(mask)) - set(name_to_id.values())
        if invalid:
            raise ValueError(f"normalized mask contains invalid class IDs: {sample_id}: {invalid}")
        masks[sample_id] = mask
        image_shapes[sample_id] = image.shape[:2]
        source_hashes[sample_id] = _sha256(source_images[sample_id])

    try:
        for directory in ("images", "masks", "overlays"):
            (temporary / directory).mkdir(parents=True, exist_ok=True)
        shutil.copy2(contract, temporary / "label_contract.yaml")
        metadata_rows: list[dict[str, str]] = []
        pixel_counts: Counter[int] = Counter()
        per_image: dict[str, dict[str, object]] = {}
        for row in rows:
            sample_id = row["candidate_id"]
            source_image = source_images[sample_id]
            destination_image = temporary / "images" / source_image.name
            shutil.copy2(source_image, destination_image)
            if _sha256(destination_image) != source_hashes[sample_id]:
                raise ValueError(f"copied image bytes changed: {sample_id}")
            mask_path = temporary / "masks" / f"{sample_id}.png"
            if not cv2.imwrite(str(mask_path), masks[sample_id]):
                raise OSError(f"cannot write normalized mask: {sample_id}")
            written_mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if written_mask is None or written_mask.ndim != 2:
                raise ValueError(f"normalized mask is not a single-channel PNG: {sample_id}")
            image_counts = Counter(
                {
                    int(value): int(count)
                    for value, count in zip(*np.unique(written_mask, return_counts=True))
                }
            )
            pixel_counts.update(image_counts)
            total = int(written_mask.size)
            per_image[sample_id] = {
                "class_pixel_counts": {
                    name: image_counts[class_id]
                    for name, class_id in sorted(name_to_id.items(), key=lambda item: item[1])
                },
                "class_pixel_fractions": {
                    name: image_counts[class_id] / total
                    for name, class_id in sorted(name_to_id.items(), key=lambda item: item[1])
                },
                "image_height": image_shapes[sample_id][0],
                "image_width": image_shapes[sample_id][1],
            }
            overlay = _overlay(destination_image, written_mask, colors)
            if not cv2.imwrite(
                str(temporary / "overlays" / f"{sample_id}.jpg"), overlay
            ):
                raise OSError(f"cannot write overlay: {sample_id}")
            metadata_rows.append(
                {
                    "sample_id": sample_id,
                    "image_path": f"images/{source_image.name}",
                    "mask_path": f"masks/{sample_id}.png",
                    "dataset": row["dataset"],
                    "ride_id": row["ride_id"],
                    "camera_uid": row["camera_uid"],
                    "timestamp_sec": row["timestamp_sec"],
                    "playlist_path": row["playlist_path"],
                    "source_candidate_id": sample_id,
                    "review_status": "HUMAN_REVIEWED_PENDING_OVERLAY_APPROVAL",
                }
            )
        _write_metadata(temporary / "metadata.csv", metadata_rows)
        contact_sheets = _write_contact_sheets(metadata_rows, temporary, colors)
        total_pixels = sum(pixel_counts.values())
        class_counts = {
            name: pixel_counts[class_id]
            for name, class_id in sorted(name_to_id.items(), key=lambda item: item[1])
        }
        warnings = _warnings(per_image)
        report = {
            "valid": True,
            "sample_count": len(metadata_rows),
            "image_count": len(metadata_rows),
            "segmentation_class_mask_count": len(masks),
            "semantic_mask_source": "SegmentationClass",
            "segmentation_object_used": False,
            "source_image_bytes_preserved": True,
            "selected_candidates_exact_match": True,
            "label_contract_exact_match": True,
            "ignore_class_id": ignore_index,
            "allowed_class_ids": sorted(name_to_id.values()),
            "class_pixel_counts": class_counts,
            "class_pixel_fractions": {
                name: count / total_pixels if total_pixels else 0.0
                for name, count in class_counts.items()
            },
            "per_image": per_image,
            "warnings": warnings,
            "errors": [],
            "contact_sheets": contact_sheets,
            "cvat_labelmap_entries": [
                {"name": name, "color_rgb": list(color)}
                for name, color in label_entries
            ],
            "fine_tuning_performed": False,
            "existing_approved_dataset_merged": False,
            "live_rover_commands_sent": False,
        }
        _write_json(temporary / "validation_report.json", report)
        (temporary / "README.md").write_text(
            f"# Traversability Manual v2 - {len(metadata_rows)} Imported Masks\n\n"
            "This standalone bundle contains human-reviewed CVAT `SegmentationClass` "
            "masks normalized to the existing traversability v1 class IDs. "
            "`SegmentationObject` masks were not used. Original JPG bytes are preserved. "
            "Review every overlay contact sheet before approving these samples for a "
            "future dataset merge or training run.\n",
            encoding="utf-8",
        )
        os.replace(temporary, output)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return report


def _read_selected_candidates(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise ValueError(f"selected_candidates.csv is missing: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != SELECTED_FIELDS:
            raise ValueError(f"unexpected selected_candidates.csv fields: {reader.fieldnames}")
        return list(reader)


def _validate_selection_file(path: Path, candidate_ids: list[str]) -> None:
    if not path.is_file():
        raise ValueError(f"selection.txt is missing: {path}")
    selected = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if selected != candidate_ids:
        raise ValueError("selection.txt does not match selected_candidates.csv order")


def _source_image(source: Path, row: dict[str, str]) -> Path:
    image = (source / row["image_path"]).resolve()
    try:
        image.relative_to(source)
    except ValueError as exc:
        raise ValueError(f"image path escapes source bundle: {row['image_path']}") from exc
    if image.name != f"{row['candidate_id']}.jpg" or not image.is_file():
        raise ValueError(f"candidate image mismatch or missing: {row['candidate_id']}")
    return image


def _validate_labelmap_against_contract(
    entries: list[tuple[str, tuple[int, int, int]]],
    name_to_id: dict[str, int],
    colors: dict[int, tuple[int, int, int]],
) -> None:
    for name, color in entries:
        contract_name = "IGNORE" if name == "background" else name
        if contract_name not in name_to_id:
            raise ValueError(f"CVAT label is not in the v1 contract: {name}")
        if colors[name_to_id[contract_name]] != color:
            raise ValueError(f"CVAT label color differs from the v1 contract: {name}")


def _write_metadata(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METADATA_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _label_contract(
    path: Path,
) -> tuple[dict[str, int], dict[int, tuple[int, int, int]], int]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("classes"), list):
        raise ValueError(f"invalid label contract: {path}")
    name_to_id: dict[str, int] = {}
    colors: dict[int, tuple[int, int, int]] = {}
    for item in config["classes"]:
        if not isinstance(item, dict):
            raise ValueError("label contract class entries must be mappings")
        class_id = int(item["id"])
        color = tuple(int(channel) for channel in item["color_rgb"])
        if len(color) != 3:
            raise ValueError(f"invalid label contract color: {item}")
        name_to_id[str(item["name"])] = class_id
        colors[class_id] = color
    return name_to_id, colors, int(config["ignore_index"])


def _colorize_mask(
    mask: np.ndarray,
    colors: dict[int, tuple[int, int, int]],
) -> np.ndarray:
    result = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id, color in colors.items():
        result[mask == class_id] = color
    return result


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _overlay(
    image_path: Path,
    mask: np.ndarray,
    colors: dict[int, tuple[int, int, int]],
) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"copied image is unreadable: {image_path}")
    color_rgb = _colorize_mask(mask, colors)
    color_bgr = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)
    return cv2.addWeighted(image, 0.55, color_bgr, 0.45, 0.0)


def _write_contact_sheets(
    rows: list[dict[str, str]],
    root: Path,
    colors: dict[int, tuple[int, int, int]],
) -> list[str]:
    output = root / "contact_sheets"
    output.mkdir()
    paths: list[str] = []
    for start in range(0, len(rows), 25):
        group = rows[start : start + 25]
        canvas = np.full((len(group) * 212, 960, 3), 245, dtype=np.uint8)
        for row_index, row in enumerate(group):
            sample_id = row["sample_id"]
            image = cv2.imread(str(root / row["image_path"]), cv2.IMREAD_COLOR)
            mask = cv2.imread(str(root / row["mask_path"]), cv2.IMREAD_GRAYSCALE)
            overlay = cv2.imread(
                str(root / "overlays" / f"{sample_id}.jpg"), cv2.IMREAD_COLOR
            )
            if image is None or mask is None or overlay is None:
                raise ValueError(f"cannot read contact-sheet assets: {sample_id}")
            mask_bgr = cv2.cvtColor(_colorize_mask(mask, colors), cv2.COLOR_RGB2BGR)
            y = row_index * 212
            for column, (asset, interpolation) in enumerate(
                ((image, cv2.INTER_AREA), (mask_bgr, cv2.INTER_NEAREST), (overlay, cv2.INTER_AREA))
            ):
                canvas[y : y + 180, column * 320 : (column + 1) * 320] = _fit(
                    asset, 320, 180, interpolation
                )
            cv2.putText(
                canvas,
                f"{sample_id} | original / mask / overlay",
                (6, y + 202),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (30, 30, 30),
                1,
                cv2.LINE_AA,
            )
        relative = f"contact_sheets/overlay_contact_sheet_{start // 25 + 1:03d}.jpg"
        if not cv2.imwrite(str(root / relative), canvas):
            raise OSError(f"cannot write contact sheet: {relative}")
        paths.append(relative)
    return paths


def _fit(image: np.ndarray, width: int, height: int, interpolation: int) -> np.ndarray:
    source_height, source_width = image.shape[:2]
    scale = min(width / source_width, height / source_height)
    resized_width = max(1, round(source_width * scale))
    resized_height = max(1, round(source_height * scale))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=interpolation)
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    left = (width - resized_width) // 2
    top = (height - resized_height) // 2
    canvas[top : top + resized_height, left : left + resized_width] = resized
    return canvas


def _warnings(per_image: dict[str, dict[str, object]]) -> list[str]:
    warnings: list[str] = []
    for class_name in EXPECTED_LABELS:
        missing = [
            sample_id
            for sample_id, stats in per_image.items()
            if stats["class_pixel_counts"][class_name] == 0
        ]
        if missing:
            warnings.append(f"{class_name} absent from {len(missing)} masks: {missing}")
    return warnings


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    args = parse_args()
    try:
        report = import_manual_bundle(
            args.source_bundle,
            args.cvat_export,
            args.output_dir,
            args.label_contract,
            args.expected_count,
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(f"manual v2 import failed: {exc}") from exc
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
