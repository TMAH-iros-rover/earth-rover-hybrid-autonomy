from __future__ import annotations

import csv
import html
import json
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import yaml

from research.training.datasets.frodobots_2k_dataset import FrameDecodeError, ManifestSample


REVIEW_STATUSES = {"UNREVIEWED", "ACCEPT", "REJECT", "NEEDS_CORRECTION", "AMBIGUOUS"}
REVIEW_COLUMNS = (
    "sample_id",
    "image_path",
    "semantic_mask_path",
    "semantic_overlay_path",
    "traversability_mask_path",
    "traversability_overlay_path",
    "confidence_path",
    "ride_id",
    "frame_id",
    "timestamp",
    "manifest_index",
    "playlist",
    "action_label",
    "linear",
    "angular",
    "selection_method",
    "split",
    "review_status",
    "reviewer_notes",
    "dominant_failure_type",
    "usable_for_training",
    "corrected_mask_path",
)


@dataclass(frozen=True)
class SelectedFrame:
    manifest_index: int
    sample: ManifestSample
    selection_method: str = "ride_balanced_temporal_random"


def decode_selected_frame(decoder: object, selected: SelectedFrame) -> tuple[np.ndarray | None, dict[str, object] | None]:
    try:
        return decoder.decode(selected.sample), None
    except (FrameDecodeError, OSError) as exc:
        return None, {
            "manifest_index": selected.manifest_index,
            "ride_id": selected.sample.ride_id,
            "error": str(exc),
        }


def load_yaml(path: str | Path) -> dict[str, object]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a YAML mapping: {path}")
    return value


def label_contract(path: str | Path) -> tuple[dict[str, int], dict[int, tuple[int, int, int]], int]:
    config = load_yaml(path)
    classes = config.get("classes")
    if not isinstance(classes, list):
        raise ValueError("label contract must contain a classes list")
    name_to_id: dict[str, int] = {}
    colors: dict[int, tuple[int, int, int]] = {}
    for item in classes:
        if not isinstance(item, dict):
            raise ValueError("label class entries must be mappings")
        class_id = int(item["id"])
        color = tuple(int(channel) for channel in item["color_rgb"])
        if len(color) != 3 or any(channel < 0 or channel > 255 for channel in color):
            raise ValueError(f"invalid label color for class {class_id}")
        name_to_id[str(item["name"])] = class_id
        colors[class_id] = color
    ignore_index = int(config["ignore_index"])
    if ignore_index not in colors:
        raise ValueError("ignore_index is not defined in label classes")
    return name_to_id, colors, ignore_index


def select_representative_frames(
    samples: tuple[ManifestSample, ...],
    requested_count: int,
    maximum_rides: int,
    minimum_separation_seconds: float,
    seed: int,
) -> list[SelectedFrame]:
    if requested_count <= 0 or maximum_rides <= 0 or minimum_separation_seconds < 0:
        raise ValueError("sample count and rides must be positive; separation cannot be negative")
    grouped: dict[str, list[tuple[int, ManifestSample]]] = defaultdict(list)
    for index, sample in enumerate(samples):
        grouped[sample.ride_id].append((index, sample))
    if not grouped:
        return []

    rng = random.Random(seed)
    ride_ids = sorted(grouped)
    rng.shuffle(ride_ids)
    ride_ids = ride_ids[: min(maximum_rides, requested_count, len(ride_ids))]
    candidates: dict[str, list[tuple[int, ManifestSample]]] = {}
    for ride_id in ride_ids:
        chronological = sorted(grouped[ride_id], key=lambda item: (item[1].front_timestamp, item[0]))
        spaced: list[tuple[int, ManifestSample]] = []
        last_timestamp: float | None = None
        for item in chronological:
            timestamp = item[1].front_timestamp
            if last_timestamp is None or timestamp - last_timestamp >= minimum_separation_seconds - 1e-9:
                spaced.append(item)
                last_timestamp = timestamp
        ride_rng = random.Random(f"{seed}:{ride_id}")
        ride_rng.shuffle(spaced)
        candidates[ride_id] = spaced

    selected: list[SelectedFrame] = []
    positions = {ride_id: 0 for ride_id in ride_ids}
    while len(selected) < requested_count:
        progressed = False
        for ride_id in ride_ids:
            position = positions[ride_id]
            if position >= len(candidates[ride_id]):
                continue
            index, sample = candidates[ride_id][position]
            positions[ride_id] += 1
            selected.append(SelectedFrame(index, sample))
            progressed = True
            if len(selected) == requested_count:
                break
        if not progressed:
            break
    return selected


def letterbox_rgb(image_rgb: np.ndarray, size: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3 or size <= 0:
        raise ValueError("letterbox input must be an HxWx3 image and size must be positive")
    height, width = image_rgb.shape[:2]
    scale = min(size / width, size / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    resized = cv2.resize(image_rgb, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    left = (size - resized_width) // 2
    top = (size - resized_height) // 2
    right = size - resized_width - left
    bottom = size - resized_height - top
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return padded, (top, bottom, left, right)


def restore_from_letterbox(
    array: np.ndarray,
    padding: tuple[int, int, int, int],
    output_shape: tuple[int, int],
    interpolation: int,
) -> np.ndarray:
    top, bottom, left, right = padding
    height, width = array.shape[:2]
    cropped = array[top : height - bottom if bottom else height, left : width - right if right else width]
    return cv2.resize(cropped, (output_shape[1], output_shape[0]), interpolation=interpolation)


def semantic_to_traversability(
    semantic_mask: np.ndarray,
    confidence: np.ndarray,
    id_to_label: dict[int, str],
    mapping_config: dict[str, object],
    name_to_id: dict[str, int],
) -> np.ndarray:
    if semantic_mask.shape != confidence.shape:
        raise ValueError("semantic mask and confidence must have matching shapes")
    default_id = name_to_id[str(mapping_config["default_class"])]
    result = np.full(semantic_mask.shape, default_id, dtype=np.uint8)
    mapping = mapping_config.get("mapping")
    if not isinstance(mapping, dict):
        raise ValueError("semantic mapping must contain a mapping section")
    normalized_labels = {class_id: label.strip() for class_id, label in id_to_label.items()}
    available_names = set(normalized_labels.values())
    assigned_names: set[str] = set()
    for target_name, semantic_names in mapping.items():
        if str(target_name) not in name_to_id:
            raise ValueError(f"unknown traversability class in semantic mapping: {target_name}")
        target_id = name_to_id[str(target_name)]
        expected = {str(name).strip() for name in semantic_names}
        missing = expected - available_names
        duplicate = expected & assigned_names
        if missing:
            raise ValueError(f"semantic mapping references unavailable labels: {sorted(missing)}")
        if duplicate:
            raise ValueError(f"semantic labels map to multiple traversability classes: {sorted(duplicate)}")
        assigned_names.update(expected)
        source_ids = [class_id for class_id, label in normalized_labels.items() if label in expected]
        if source_ids:
            result[np.isin(semantic_mask, source_ids)] = target_id
    threshold = float(mapping_config["confidence_threshold"])
    result[confidence < threshold] = default_id
    return result


def colorize_mask(mask: np.ndarray, colors: dict[int, tuple[int, int, int]]) -> np.ndarray:
    output = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id, color in colors.items():
        output[mask == class_id] = color
    return output


def semantic_colors(class_ids: Iterable[int]) -> dict[int, tuple[int, int, int]]:
    return {
        class_id: (
            (37 * class_id + 53) % 256,
            (97 * class_id + 101) % 256,
            (17 * class_id + 199) % 256,
        )
        for class_id in class_ids
    }


def overlay_rgb(image_rgb: np.ndarray, color_mask_rgb: np.ndarray, alpha: float = 0.48) -> np.ndarray:
    if image_rgb.shape != color_mask_rgb.shape:
        raise ValueError("image and color mask must have matching shapes")
    return cv2.addWeighted(image_rgb, 1.0 - alpha, color_mask_rgb, alpha, 0.0)


def class_area_statistics(mask: np.ndarray, id_to_name: dict[int, str]) -> dict[str, object]:
    total = int(mask.size)
    counts = Counter(int(value) for value in mask.reshape(-1))
    return {
        id_to_name[class_id]: {
            "pixels": counts[class_id],
            "fraction": counts[class_id] / total,
        }
        for class_id in sorted(id_to_name)
    }


def write_contact_sheet(entries: list[dict[str, str]], bundle_root: Path, output_path: Path) -> None:
    if not entries:
        raise ValueError("cannot create a contact sheet without entries")
    tile_width, tile_height, caption_height = 320, 180, 44
    columns = 4
    rows = (len(entries) + columns - 1) // columns
    canvas = np.full((rows * (tile_height + caption_height), columns * tile_width, 3), 245, dtype=np.uint8)
    for position, entry in enumerate(entries):
        image = cv2.imread(str(bundle_root / entry["image_path"]), cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f"cannot read contact-sheet image: {entry['image_path']}")
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tile = _fit_rgb(rgb, tile_width, tile_height)
        row, column = divmod(position, columns)
        y = row * (tile_height + caption_height)
        x = column * tile_width
        canvas[y : y + tile_height, x : x + tile_width] = cv2.cvtColor(tile, cv2.COLOR_RGB2BGR)
        caption = f"{entry['sample_id']} ride={entry['ride_id']} frame={entry['frame_id']}"
        cv2.putText(canvas, caption, (x + 6, y + tile_height + 27), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1, cv2.LINE_AA)
    if not cv2.imwrite(str(output_path), canvas):
        raise OSError(f"cannot write contact sheet: {output_path}")


def write_gallery(entries: list[dict[str, str]], output_path: Path) -> None:
    cards = []
    for entry in entries:
        cards.append(
            "<article><h2>{sample}</h2><p>ride={ride} frame={frame} confidence={confidence}</p>"
            "<div class='grid'><figure><img src='{image}'><figcaption>Original</figcaption></figure>"
            "<figure><img src='{semantic}'><figcaption>Semantic overlay</figcaption></figure>"
            "<figure><img src='{traversability}'><figcaption>Traversability overlay</figcaption></figure>"
            "<figure><img src='{uncertainty}'><figcaption>Confidence</figcaption></figure></div>"
            "<p>{distribution}</p></article>".format(
                sample=html.escape(entry["sample_id"]),
                ride=html.escape(entry["ride_id"]),
                frame=html.escape(entry["frame_id"]),
                confidence=html.escape(entry["mean_confidence"]),
                image=html.escape(entry["image_path"]),
                semantic=html.escape(entry["semantic_overlay_path"]),
                traversability=html.escape(entry["traversability_overlay_path"]),
                uncertainty=html.escape(entry["confidence_path"]),
                distribution=html.escape(entry["class_distribution"]),
            )
        )
    document = """<!doctype html><html><head><meta charset='utf-8'><title>Traversability Pilot Review</title>
<style>body{font-family:system-ui,sans-serif;margin:24px;background:#f4f5f6;color:#17191c}article{background:white;border:1px solid #ccd0d5;margin:0 0 24px;padding:16px}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}figure{margin:0}img{width:100%;height:auto;display:block}figcaption{padding-top:6px;font-weight:600}@media(max-width:800px){.grid{grid-template-columns:1fr}}</style>
</head><body><h1>Traversability Pilot Review</h1><p>All masks are unverified pseudo-label drafts. Edit review.csv after inspection.</p><p><strong>Overlay legend:</strong> red = proposed non-traversable, green = proposed traversable, yellow = unknown/ignore. Confirm every region manually.</p>__CARDS__</body></html>""".replace("__CARDS__", "\n".join(cards))
    output_path.write_text(document, encoding="utf-8")


def copy_review_configs(label_contract_path: Path, mapping_path: Path, bundle_root: Path) -> None:
    shutil.copy2(label_contract_path, bundle_root / "label_contract.yaml")
    shutil.copy2(mapping_path, bundle_root / "semantic_mapping.yaml")


def write_cvat_labelmap(
    name_to_id: dict[str, int],
    colors: dict[int, tuple[int, int, int]],
    output_path: Path,
) -> None:
    id_to_name = {class_id: name for name, class_id in name_to_id.items()}
    lines = ["# label : color (RGB) : body parts : actions"]
    for class_id in sorted(id_to_name):
        red, green, blue = colors[class_id]
        lines.append(f"{id_to_name[class_id]}:{red},{green},{blue}::")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_review_csv(entries: list[dict[str, str]], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        writer.writerows({column: entry.get(column, "") for column in REVIEW_COLUMNS} for entry in entries)


def validate_review_bundle(bundle_root: str | Path, verified_output: str | Path | None = None) -> dict[str, object]:
    root = Path(bundle_root).expanduser().resolve()
    name_to_id, colors, _ = label_contract(root / "label_contract.yaml")
    allowed_ids = set(name_to_id.values())
    rows = list(csv.DictReader((root / "review.csv").open(newline="", encoding="utf-8")))
    if not rows:
        raise ValueError("review.csv has no samples")
    errors: list[str] = []
    seen: set[str] = set()
    ride_splits: dict[str, set[str]] = defaultdict(set)
    verified: list[dict[str, str]] = []
    status_counts: Counter[str] = Counter()
    for row in rows:
        sample_id = row.get("sample_id", "")
        if not sample_id or sample_id in seen:
            errors.append(f"missing or duplicate sample_id: {sample_id!r}")
            continue
        seen.add(sample_id)
        status = row.get("review_status", "")
        status_counts[status] += 1
        if status not in REVIEW_STATUSES:
            errors.append(f"{sample_id}: invalid review status {status!r}")
        try:
            image_path = _bundle_path(root, row["image_path"])
            semantic_mask_path = _bundle_path(root, row["semantic_mask_path"])
            semantic_overlay_path = _bundle_path(root, row["semantic_overlay_path"])
            pseudo_mask_path = _bundle_path(root, row["traversability_mask_path"])
            traversability_overlay_path = _bundle_path(root, row["traversability_overlay_path"])
            confidence_path = _bundle_path(root, row["confidence_path"])
            metadata_path = _bundle_path(root, f"metadata/{sample_id}.json")
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            semantic_mask = cv2.imread(str(semantic_mask_path), cv2.IMREAD_UNCHANGED)
            semantic_overlay = cv2.imread(str(semantic_overlay_path), cv2.IMREAD_COLOR)
            pseudo_mask = cv2.imread(str(pseudo_mask_path), cv2.IMREAD_UNCHANGED)
            traversability_overlay = cv2.imread(str(traversability_overlay_path), cv2.IMREAD_COLOR)
            confidence_image = cv2.imread(str(confidence_path), cv2.IMREAD_COLOR)
            assets = (semantic_mask, semantic_overlay, pseudo_mask, traversability_overlay, confidence_image)
            if image is None or any(asset is None for asset in assets):
                raise ValueError("one or more review assets are unreadable")
            if any(asset.shape[:2] != image.shape[:2] for asset in assets):
                raise ValueError("review asset dimensions do not match the image")
            pseudo_ids = _class_id_mask(pseudo_mask, colors)
            if not set(int(value) for value in np.unique(pseudo_ids)).issubset(allowed_ids):
                raise ValueError("pseudo-label mask contains an unsupported class ID")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("sample_id") != sample_id:
                raise ValueError("metadata sample ID does not match review.csv")
            corrected_value = row.get("corrected_mask_path", "").strip()
            selected_mask_path = pseudo_mask_path
            if corrected_value:
                corrected_path = _bundle_path(root, corrected_value)
                corrected = cv2.imread(str(corrected_path), cv2.IMREAD_UNCHANGED)
                if corrected is None or corrected.shape[:2] != image.shape[:2]:
                    raise ValueError("corrected mask is missing or has incorrect dimensions")
                corrected_ids = _class_id_mask(corrected, colors)
                if not set(int(value) for value in np.unique(corrected_ids)).issubset(allowed_ids):
                    raise ValueError("corrected mask contains an unsupported class ID")
                selected_mask_path = corrected_path
            usable = row.get("usable_for_training", "").strip().lower() in {"true", "1", "yes"}
            if status == "NEEDS_CORRECTION" and usable and not corrected_value:
                raise ValueError("usable corrected sample has no corrected mask")
            if status in {"UNREVIEWED", "REJECT", "AMBIGUOUS"} and usable:
                raise ValueError(f"{status} sample cannot be marked usable for training")
            split = row.get("split", "UNASSIGNED").strip() or "UNASSIGNED"
            if split != "UNASSIGNED":
                ride_splits[row["ride_id"]].add(split)
            if status in {"ACCEPT", "NEEDS_CORRECTION"} and usable:
                output_row = dict(row)
                output_row["verified_mask_path"] = str(selected_mask_path.relative_to(root))
                verified.append(output_row)
        except (KeyError, OSError, ValueError) as exc:
            errors.append(f"{sample_id}: {exc}")
    for ride_id, splits in ride_splits.items():
        if len(splits) > 1:
            errors.append(f"ride {ride_id} appears in multiple splits: {sorted(splits)}")
    report = {
        "valid": not errors,
        "sample_count": len(rows),
        "status_counts": dict(sorted(status_counts.items())),
        "verified_training_sample_count": len(verified),
        "errors": errors,
    }
    if verified_output is not None and not errors:
        output = Path(verified_output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(REVIEW_COLUMNS) + ["verified_mask_path"]
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(verified)
        report["verified_output"] = str(output)
    return report


def _bundle_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"path escapes review bundle: {relative}") from exc
    if not path.is_file():
        raise ValueError(f"referenced file does not exist: {relative}")
    return path


def _class_id_mask(mask: np.ndarray, colors: dict[int, tuple[int, int, int]]) -> np.ndarray:
    if mask.ndim == 2:
        return mask
    if mask.ndim != 3 or mask.shape[2] not in {3, 4}:
        raise ValueError("mask must be a single-channel ID PNG or an RGB contract-color PNG")
    rgb = cv2.cvtColor(mask[:, :, :3], cv2.COLOR_BGR2RGB)
    class_ids = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    for class_id, color in colors.items():
        class_ids[np.all(rgb == np.asarray(color, dtype=np.uint8), axis=2)] = class_id
    if np.any(class_ids == 255):
        raise ValueError("RGB mask contains a color not defined in the label contract")
    return class_ids


def _fit_rgb(image_rgb: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    height, width = image_rgb.shape[:2]
    scale = min(target_width / width, target_height / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    resized = cv2.resize(image_rgb, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    canvas = np.full((target_height, target_width, 3), 245, dtype=np.uint8)
    left = (target_width - resized_width) // 2
    top = (target_height - resized_height) // 2
    canvas[top : top + resized_height, left : left + resized_width] = resized
    return canvas


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
