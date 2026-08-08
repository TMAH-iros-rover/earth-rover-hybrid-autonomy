from __future__ import annotations

import zipfile
from pathlib import Path, PurePosixPath

import cv2
import numpy as np


EXPECTED_LABELS = {"IGNORE": 0, "ON_ROAD": 1, "OFF_ROAD": 2, "OBSTACLE": 3}
EXPECTED_COLORS = {
    "IGNORE": (0, 0, 0),
    "ON_ROAD": (38, 166, 91),
    "OFF_ROAD": (43, 126, 216),
    "OBSTACLE": (220, 50, 47),
    "background": (0, 0, 0),
}


def read_cvat_segmentation_export(
    export: Path,
) -> tuple[list[tuple[str, tuple[int, int, int]]], dict[str, np.ndarray]]:
    found: dict[str, np.ndarray] = {}
    labelmap_texts: list[str] = []
    if export.is_dir():
        for path in export.rglob("*.png"):
            if "SegmentationClass" in path.parts:
                if path.stem in found:
                    raise ValueError(f"duplicate CVAT mask stem: {path.stem}")
                found[path.stem] = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        labelmap_paths = list(export.rglob("labelmap.txt"))
        labelmap_texts = [path.read_text(encoding="utf-8") for path in labelmap_paths]
    elif export.is_file() and zipfile.is_zipfile(export):
        with zipfile.ZipFile(export) as archive:
            bad = archive.testzip()
            if bad is not None:
                raise ValueError(f"CVAT ZIP contains a corrupt member: {bad}")
            for name in archive.namelist():
                member = PurePosixPath(name)
                if member.suffix.lower() == ".png" and "SegmentationClass" in member.parts:
                    if member.stem in found:
                        raise ValueError(f"duplicate CVAT mask stem: {member.stem}")
                    encoded = np.frombuffer(archive.read(name), dtype=np.uint8)
                    found[member.stem] = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
                elif member.name == "labelmap.txt":
                    labelmap_texts.append(archive.read(name).decode("utf-8"))
    else:
        raise ValueError("CVAT export must be a directory or ZIP archive")
    if len(labelmap_texts) != 1:
        raise ValueError(f"expected exactly one labelmap.txt, found {len(labelmap_texts)}")
    return parse_cvat_labelmap(labelmap_texts[0]), found


def parse_cvat_labelmap(text: str) -> list[tuple[str, tuple[int, int, int]]]:
    entries: list[tuple[str, tuple[int, int, int]]] = []
    seen_names: set[str] = set()
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(":")
        if len(parts) < 2:
            raise ValueError(f"invalid labelmap line {line_number}: {raw_line!r}")
        name = parts[0].strip()
        if name not in EXPECTED_COLORS:
            raise ValueError(f"unknown CVAT label: {name!r}")
        if name in seen_names:
            raise ValueError(f"duplicate CVAT label: {name!r}")
        seen_names.add(name)
        try:
            color = tuple(int(channel.strip()) for channel in parts[1].split(","))
        except ValueError as exc:
            raise ValueError(f"invalid RGB value for CVAT label {name!r}") from exc
        if len(color) != 3 or color != EXPECTED_COLORS[name]:
            raise ValueError(
                f"CVAT label {name!r} uses RGB {color}, expected {EXPECTED_COLORS[name]}"
            )
        entries.append((name, color))
    missing = set(EXPECTED_LABELS) - seen_names
    if missing:
        raise ValueError(f"CVAT labelmap is missing required labels: {sorted(missing)}")
    return entries


def cvat_label_mapping(
    entries: list[tuple[str, tuple[int, int, int]]],
) -> tuple[dict[int, int], dict[tuple[int, int, int], int]]:
    index_to_id: dict[int, int] = {}
    color_to_id: dict[tuple[int, int, int], int] = {}
    for index, (name, color) in enumerate(entries):
        final_id = 0 if name == "background" else EXPECTED_LABELS[name]
        index_to_id[index] = final_id
        existing = color_to_id.get(color)
        if existing is not None and existing != final_id:
            raise ValueError(f"RGB {color} maps to conflicting final class IDs")
        color_to_id[color] = final_id
    return index_to_id, color_to_id


def normalize_cvat_class_mask(
    mask: np.ndarray | None,
    index_to_id: dict[int, int],
    color_to_id: dict[tuple[int, int, int], int],
) -> np.ndarray:
    if mask is None:
        raise ValueError("SegmentationClass mask is unreadable")
    result = np.full(mask.shape[:2], 255, dtype=np.uint8)
    if mask.ndim == 2:
        values = set(int(value) for value in np.unique(mask))
        unknown = values - set(index_to_id)
        if unknown:
            raise ValueError(f"SegmentationClass mask uses unknown labelmap indices: {sorted(unknown)}")
        for source_index, final_id in index_to_id.items():
            result[mask == source_index] = final_id
    elif mask.ndim == 3 and mask.shape[2] in {3, 4}:
        rgb = cv2.cvtColor(mask[:, :, :3], cv2.COLOR_BGR2RGB)
        for color, final_id in color_to_id.items():
            result[np.all(rgb == np.asarray(color, dtype=np.uint8), axis=2)] = final_id
    else:
        raise ValueError("SegmentationClass mask must be a 1-channel or RGB PNG")
    if np.any(result == 255):
        raise ValueError("SegmentationClass mask contains an unknown index or RGB color")
    return result
