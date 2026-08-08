from __future__ import annotations

import csv
import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2

from research.training.traversability_annotation import AnnotationCandidate, TARGET_SCENE_CATEGORIES


CATEGORY_TARGETS = {
    "PAVED_GROUND": 35,
    "OFF_ROAD_GROUND": 25,
    "GROUND_BOUNDARY": 15,
    "PERSON": 8,
    "VEHICLE": 10,
    "STREET_FURNITURE_OR_POLE": 10,
    "STRUCTURE_OBSTACLE": 15,
    "CURB_OR_STAIRS": 5,
    "TURNING": 15,
    "SHADOW_CANDIDATE": 10,
    "BACKLIGHT_CANDIDATE": 5,
    "BLUR_CANDIDATE": 5,
    "REFLECTION_CANDIDATE": 5,
    "NARROW_PASSAGE_CANDIDATE": 8,
}


@dataclass(frozen=True)
class ExistingAnnotationSet:
    source_keys: frozenset[tuple[str, int]]
    frame_keys: frozenset[tuple[str, int, int]]
    timestamps_by_ride: dict[str, tuple[float, ...]]
    image_hashes: tuple[int, ...]
    sample_ids: frozenset[str]


def load_existing_annotations(bundle_root: str | Path) -> ExistingAnnotationSet:
    root = Path(bundle_root).expanduser().resolve()
    rows = list(csv.DictReader((root / "metadata.csv").open(newline="", encoding="utf-8")))
    timestamps: dict[str, list[float]] = defaultdict(list)
    source_keys: set[tuple[str, int]] = set()
    frame_keys: set[tuple[str, int, int]] = set()
    hashes: list[int] = []
    sample_ids: set[str] = set()
    for row in rows:
        sample_id = row["sample_id"]
        ride_id = row["ride_id"]
        manifest_index = int(row["manifest_index"])
        frame_id = int(row["frame_id"])
        timestamp_ms = round(float(row["timestamp"]) * 1000)
        image_path = (root / row["image_path"]).resolve()
        if not image_path.is_file():
            raise ValueError(f"existing pilot image is missing: {row['image_path']}")
        if sample_id in sample_ids:
            raise ValueError(f"duplicate existing sample ID: {sample_id}")
        sample_ids.add(sample_id)
        source_keys.add((ride_id, manifest_index))
        frame_keys.add((ride_id, frame_id, timestamp_ms))
        timestamps[ride_id].append(float(row["timestamp"]))
        hashes.append(difference_hash(image_path))
    return ExistingAnnotationSet(
        source_keys=frozenset(source_keys),
        frame_keys=frozenset(frame_keys),
        timestamps_by_ride={ride: tuple(sorted(values)) for ride, values in timestamps.items()},
        image_hashes=tuple(hashes),
        sample_ids=frozenset(sample_ids),
    )


def select_expansion_candidates(
    candidates: list[AnnotationCandidate],
    existing: ExistingAnnotationSet,
    requested_count: int,
    minimum_separation_seconds: float,
    maximum_per_ride: int,
    hash_distance_threshold: int,
    seed: int,
) -> tuple[list[AnnotationCandidate], dict[str, object]]:
    if requested_count <= 0 or minimum_separation_seconds < 0 or maximum_per_ride <= 0:
        raise ValueError("count and ride cap must be positive; separation cannot be negative")
    if hash_distance_threshold < 0 or hash_distance_threshold > 64:
        raise ValueError("hash distance threshold must be between 0 and 64")

    hashes: dict[str, int] = {}
    eligible: list[AnnotationCandidate] = []
    excluded: list[dict[str, object]] = []
    exclusion_counts: Counter[str] = Counter()
    for candidate in candidates:
        image_hash = difference_hash(candidate.image_path)
        hashes[candidate.source_sample_id] = image_hash
        reason = _initial_exclusion_reason(
            candidate,
            image_hash,
            existing,
            minimum_separation_seconds,
            hash_distance_threshold,
        )
        if reason is None:
            eligible.append(candidate)
            continue
        exclusion_counts[reason] += 1
        excluded.append(_excluded_record(candidate, reason))

    selected: list[AnnotationCandidate] = []
    selected_rides: Counter[str] = Counter()
    selected_categories: Counter[str] = Counter()
    while eligible and len(selected) < requested_count:
        selectable = [
            candidate
            for candidate in eligible
            if selected_rides[candidate.ride_id] < maximum_per_ride
            and _far_enough_in_time(candidate, selected, minimum_separation_seconds)
            and _far_enough_in_hash(
                hashes[candidate.source_sample_id],
                (hashes[item.source_sample_id] for item in selected),
                hash_distance_threshold,
            )
        ]
        if not selectable:
            break
        selectable.sort(
            key=lambda candidate: (
                -_selection_score(candidate, selected_rides, selected_categories),
                _stable_candidate_key(candidate, seed),
            )
        )
        chosen = selectable[0]
        selected.append(chosen)
        selected_rides[chosen.ride_id] += 1
        selected_categories.update(chosen.scene_categories)
        eligible.remove(chosen)

    remaining_reasons: Counter[str] = Counter()
    for candidate in eligible:
        if selected_rides[candidate.ride_id] >= maximum_per_ride:
            remaining_reasons["ride_cap"] += 1
        elif not _far_enough_in_time(candidate, selected, minimum_separation_seconds):
            remaining_reasons["near_selected_time"] += 1
        elif not _far_enough_in_hash(
            hashes[candidate.source_sample_id],
            (hashes[item.source_sample_id] for item in selected),
            hash_distance_threshold,
        ):
            remaining_reasons["near_selected_visual_hash"] += 1
        else:
            remaining_reasons["not_needed_after_target_count"] += 1

    selected_hashes = [hashes[item.source_sample_id] for item in selected]
    selected_hash_distances = [
        hamming_distance(left, right)
        for index, left in enumerate(selected_hashes)
        for right in selected_hashes[index + 1 :]
    ]
    existing_hash_distances = [
        hamming_distance(value, existing_hash)
        for value in selected_hashes
        for existing_hash in existing.image_hashes
    ]
    selected_time_deltas = [
        abs(left.timestamp - right.timestamp)
        for index, left in enumerate(selected)
        for right in selected[index + 1 :]
        if left.ride_id == right.ride_id
    ]
    selected_exact_overlap_count = sum(
        1
        for item in selected
        if (item.ride_id, item.manifest_index) in existing.source_keys
        or (item.ride_id, item.frame_id, round(item.timestamp * 1000)) in existing.frame_keys
    )

    report = {
        "source_candidate_count": len(candidates),
        "eligible_candidate_count": len(candidates) - sum(exclusion_counts.values()),
        "selected_sample_count": len(selected),
        "selected_ride_count": len(selected_rides),
        "ride_distribution": dict(sorted(selected_rides.items())),
        "scene_category_distribution": dict(sorted(selected_categories.items())),
        "target_categories_not_found": sorted(set(TARGET_SCENE_CATEGORIES) - set(selected_categories)),
        "initial_exclusion_counts": dict(sorted(exclusion_counts.items())),
        "remaining_candidate_reasons": dict(sorted(remaining_reasons.items())),
        "abnormal_camera_candidates": [
            item for item in excluded if str(item["reason"]).startswith("abnormal_")
        ],
        "candidate_existing_exact_exclusion_count": sum(
            1 for item in excluded if item["reason"] in {"existing_source", "existing_frame"}
        ),
        "selected_existing_exact_overlap_count": selected_exact_overlap_count,
        "existing_near_time_exclusion_count": exclusion_counts["near_existing_time"],
        "existing_visual_duplicate_exclusion_count": exclusion_counts["near_existing_visual_hash"],
        "minimum_selected_same_ride_time_delta_seconds": (
            min(selected_time_deltas) if selected_time_deltas else None
        ),
        "minimum_selected_pair_hash_distance": (
            min(selected_hash_distances) if selected_hash_distances else None
        ),
        "minimum_selected_to_existing_hash_distance": (
            min(existing_hash_distances) if existing_hash_distances else None
        ),
        "minimum_separation_seconds": minimum_separation_seconds,
        "maximum_per_ride": maximum_per_ride,
        "visual_hash": {
            "algorithm": "64-bit difference hash on 9x8 grayscale image",
            "maximum_hamming_distance_for_rejection": hash_distance_threshold,
        },
        "seed": seed,
    }
    return selected, report


def difference_hash(image_path: str | Path) -> int:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise OSError(f"cannot read image for difference hash: {image_path}")
    resized = cv2.resize(image, (9, 8), interpolation=cv2.INTER_AREA)
    bits = resized[:, 1:] > resized[:, :-1]
    value = 0
    for bit in bits.reshape(-1):
        value = (value << 1) | int(bit)
    return value


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def _initial_exclusion_reason(
    candidate: AnnotationCandidate,
    image_hash: int,
    existing: ExistingAnnotationSet,
    minimum_separation_seconds: float,
    hash_distance_threshold: int,
) -> str | None:
    source_key = (candidate.ride_id, candidate.manifest_index)
    frame_key = (candidate.ride_id, candidate.frame_id, round(candidate.timestamp * 1000))
    if source_key in existing.source_keys:
        return "existing_source"
    if frame_key in existing.frame_keys:
        return "existing_frame"
    if any(
        abs(candidate.timestamp - timestamp) < minimum_separation_seconds - 1e-9
        for timestamp in existing.timestamps_by_ride.get(candidate.ride_id, ())
    ):
        return "near_existing_time"
    if not _far_enough_in_hash(image_hash, existing.image_hashes, hash_distance_threshold):
        return "near_existing_visual_hash"
    evidence = candidate.category_evidence
    semantic = evidence.get("semantic_fraction", {})
    luminance = evidence.get("image_luminance", {})
    top_semantic = evidence.get("top_semantic_fraction", {})
    ground = float(semantic.get("paved", 0.0)) + float(semantic.get("off_road", 0.0))
    unknown = float(evidence.get("pseudo_unknown_fraction", 0.0))
    if float(top_semantic.get("sky", 0.0)) >= 0.80:
        return "abnormal_sky_dominant"
    if unknown >= 0.90 and ground < 0.05:
        return "abnormal_high_unknown_without_ground"
    if float(luminance.get("laplacian_variance", 0.0)) < 8.0:
        return "abnormal_severe_blur"
    if float(luminance.get("p90", 0.0)) - float(luminance.get("p10", 0.0)) < 8.0:
        return "abnormal_low_information"
    return None


def _selection_score(
    candidate: AnnotationCandidate,
    selected_rides: Counter[str],
    selected_categories: Counter[str],
) -> float:
    category_score = sum(
        20.0 * max(CATEGORY_TARGETS.get(category, 3) - selected_categories[category], 0)
        / CATEGORY_TARGETS.get(category, 3)
        for category in candidate.scene_categories
    )
    semantic = candidate.category_evidence.get("semantic_fraction", {})
    ground = float(semantic.get("paved", 0.0)) + float(semantic.get("off_road", 0.0))
    new_ride_bonus = 30.0 if selected_rides[candidate.ride_id] == 0 else 0.0
    ride_penalty = 6.0 * selected_rides[candidate.ride_id]
    return category_score + min(ground, 0.60) * 80.0 + new_ride_bonus - ride_penalty


def _far_enough_in_time(
    candidate: AnnotationCandidate,
    selected: list[AnnotationCandidate],
    minimum_separation_seconds: float,
) -> bool:
    return all(
        candidate.ride_id != item.ride_id
        or abs(candidate.timestamp - item.timestamp) >= minimum_separation_seconds - 1e-9
        for item in selected
    )


def _far_enough_in_hash(value: int, others: Iterable[int], threshold: int) -> bool:
    return all(hamming_distance(value, int(other)) > threshold for other in others)


def _stable_candidate_key(candidate: AnnotationCandidate, seed: int) -> str:
    value = f"{seed}:{candidate.ride_id}:{candidate.timestamp:.9f}:{candidate.frame_id}"
    return hashlib.sha256(value.encode()).hexdigest()


def _excluded_record(candidate: AnnotationCandidate, reason: str) -> dict[str, object]:
    return {
        "source_sample_id": candidate.source_sample_id,
        "ride_id": candidate.ride_id,
        "timestamp": candidate.timestamp,
        "frame_id": candidate.frame_id,
        "manifest_index": candidate.manifest_index,
        "reason": reason,
    }
