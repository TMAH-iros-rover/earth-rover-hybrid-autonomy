from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from dataclasses import replace

import cv2
import numpy as np

from research.training.traversability_annotation import AnnotationCandidate
from research.training.traversability_expansion import (
    ExistingAnnotationSet,
    difference_hash,
    hamming_distance,
)


HARD_CATEGORIES = (
    "CURB_HARD_NEGATIVE",
    "TRUE_OFF_ROAD",
    "PAVED_HARD_CASE",
)
HARD_CATEGORY_TARGETS = {
    "CURB_HARD_NEGATIVE": 12,
    "TRUE_OFF_ROAD": 6,
    "PAVED_HARD_CASE": 6,
}


def temporal_prefilter(
    rows: list[dict[str, str]],
    minimum_separation_seconds: float,
    maximum_count: int,
    seed: int,
) -> list[dict[str, object]]:
    if minimum_separation_seconds <= 0 or maximum_count <= 0:
        raise ValueError("prefilter separation and maximum count must be positive")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("status") == "OK":
            grouped[row["ride_id"]].append(row)
    candidates: list[dict[str, object]] = []
    for ride_id, ride_rows in grouped.items():
        chronological = sorted(ride_rows, key=lambda row: (float(row["timestamp"]), int(row["frame_id"])))
        previous: dict[str, str] | None = None
        last_selected = float("-inf")
        for row in chronological:
            timestamp = float(row["timestamp"])
            on_road = float(row["on_road_ratio"])
            off_road = float(row["off_road_ratio"])
            confidence = float(row["mean_confidence"])
            previous_on = float(previous["on_road_ratio"]) if previous else on_road
            previous_off = float(previous["off_road_ratio"]) if previous else off_road
            off_road_delta = off_road - previous_off
            on_off_churn = abs(off_road_delta) + abs(on_road - previous_on)
            confidence_drop = _optional_float(row.get("mean_confidence_drop")) or 0.0
            flicker = _optional_float(row.get("pixel_flicker_fraction")) or 0.0
            reasons = []
            if off_road_delta >= 0.015:
                reasons.append("off_road_ratio_increase")
            if on_off_churn >= 0.04 or flicker >= 0.06:
                reasons.append("on_road_off_road_transition")
            if confidence_drop >= 0.05:
                reasons.append("confidence_drop")
            if off_road >= 0.08 and confidence >= 0.60:
                reasons.append("high_confidence_off_road")
            if reasons and timestamp - last_selected >= minimum_separation_seconds - 1e-9:
                score = 3.0 * max(off_road_delta, 0.0) + on_off_churn + flicker + off_road * confidence
                candidates.append(
                    {
                        "row": row,
                        "candidate_reasons": reasons,
                        "off_road_delta": off_road_delta,
                        "on_off_churn": on_off_churn,
                        "temporal_score": score,
                        "stable_key": _stable_key(seed, ride_id, timestamp, row["frame_id"]),
                    }
                )
                last_selected = timestamp
            previous = row
    candidates.sort(key=lambda item: (-float(item["temporal_score"]), str(item["stable_key"])))
    return candidates[:maximum_count]


def classify_hard_example(
    training_prediction: np.ndarray,
    confidence: np.ndarray,
    temporal: dict[str, object],
) -> tuple[str | None, dict[str, float]]:
    if training_prediction.shape != confidence.shape or training_prediction.ndim != 2:
        raise ValueError("prediction and confidence must be matching 2D arrays")
    values = set(int(value) for value in np.unique(training_prediction))
    if not values.issubset({0, 1, 2}):
        raise ValueError(f"unsupported training prediction IDs: {sorted(values)}")
    off_road = training_prediction == 1
    obstacle = training_prediction == 2
    off_road_ratio = float(off_road.mean())
    obstacle_ratio = float(obstacle.mean())
    mean_confidence = float(confidence.mean())
    kernel = np.ones((5, 5), dtype=np.uint8)
    obstacle_near = cv2.dilate(obstacle.astype(np.uint8), kernel, iterations=1).astype(bool)
    adjacency = float(np.logical_and(off_road, obstacle_near).sum() / max(int(off_road.sum()), 1))
    lower_start = training_prediction.shape[0] // 3
    lower_off_road_fraction = float(off_road[lower_start:].mean())
    off_road_delta = float(temporal["off_road_delta"])
    on_off_churn = float(temporal["on_off_churn"])
    flicker = _optional_float(temporal["row"].get("pixel_flicker_fraction")) or 0.0
    curb_score = 3.0 * adjacency + 0.5 * obstacle_ratio + min(off_road_ratio, 0.35)
    paved_score = 3.0 * max(off_road_delta, 0.0) + on_off_churn + flicker
    true_off_road_score = 1.5 * off_road_ratio + mean_confidence - adjacency - on_off_churn
    category: str | None
    if 0.02 <= off_road_ratio <= 0.55 and adjacency >= 0.02 and lower_off_road_fraction >= 0.02:
        category = "CURB_HARD_NEGATIVE"
    elif (off_road_delta >= 0.015 or on_off_churn >= 0.04 or flicker >= 0.06) and off_road_ratio >= 0.03:
        category = "PAVED_HARD_CASE"
    elif off_road_ratio >= 0.10 and mean_confidence >= 0.55:
        category = "TRUE_OFF_ROAD"
    else:
        category = None
    evidence = {
        "off_road_ratio": off_road_ratio,
        "obstacle_ratio": obstacle_ratio,
        "mean_confidence": mean_confidence,
        "off_road_obstacle_adjacency": adjacency,
        "lower_image_off_road_fraction": lower_off_road_fraction,
        "off_road_delta": off_road_delta,
        "on_off_churn": on_off_churn,
        "pixel_flicker_fraction": flicker,
        "curb_candidate_score": curb_score,
        "paved_candidate_score": paved_score,
        "true_off_road_candidate_score": true_off_road_score,
    }
    return category, evidence


def select_hard_examples(
    candidates: list[AnnotationCandidate],
    existing: ExistingAnnotationSet,
    category_targets: dict[str, int],
    minimum_separation_seconds: float,
    maximum_per_ride: int,
    hash_distance_threshold: int,
    seed: int,
) -> tuple[list[AnnotationCandidate], dict[str, object]]:
    if set(category_targets) != set(HARD_CATEGORIES) or any(
        count <= 0 for count in category_targets.values()
    ):
        raise ValueError("category targets must define a positive count for every hard category")
    requested_count = sum(category_targets.values())
    hashes = {candidate.source_sample_id: difference_hash(candidate.image_path) for candidate in candidates}
    eligible: list[AnnotationCandidate] = []
    exclusions: Counter[str] = Counter()
    for candidate in candidates:
        frame_key = (candidate.ride_id, candidate.frame_id, round(candidate.timestamp * 1000))
        if frame_key in existing.frame_keys:
            exclusions["approved_exact_frame"] += 1
        elif any(
            hamming_distance(hashes[candidate.source_sample_id], existing_hash) <= hash_distance_threshold
            for existing_hash in existing.image_hashes
        ):
            exclusions["approved_visual_duplicate"] += 1
        else:
            eligible.append(candidate)

    category_pools = {
        category: sorted(
            [candidate for candidate in eligible if category in candidate.scene_categories],
            key=lambda candidate: (-_category_score(candidate, category), _candidate_key(candidate, seed)),
        )
        for category in HARD_CATEGORIES
    }
    selected: list[AnnotationCandidate] = []
    ride_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    selection_order = ("CURB_HARD_NEGATIVE", "PAVED_HARD_CASE", "TRUE_OFF_ROAD")
    for category in selection_order:
        while category_counts[category] < category_targets[category]:
            choices = [
                candidate
                for candidate in category_pools[category]
                if candidate not in selected
                and ride_counts[candidate.ride_id] < maximum_per_ride
                and _separated(candidate, selected, minimum_separation_seconds)
                and _hash_separated(candidate, selected, hashes, hash_distance_threshold)
            ]
            if not choices:
                break
            chosen = min(
                choices,
                key=lambda candidate: (
                    ride_counts[candidate.ride_id],
                    -_category_score(candidate, category),
                    _candidate_key(candidate, seed),
                ),
            )
            selected.append(chosen)
            ride_counts[chosen.ride_id] += 1
            category_counts[category] += 1

    shortfalls = {
        category: max(0, category_targets[category] - category_counts[category])
        for category in HARD_CATEGORIES
    }
    rides = sorted({candidate.ride_id for candidate in selected})
    validation_ride = (
        max(
            rides,
            key=lambda ride: (
                sum(
                    _category_score(candidate, "CURB_HARD_NEGATIVE")
                    for candidate in selected
                    if candidate.ride_id == ride
                    and candidate.scene_categories[0] == "CURB_HARD_NEGATIVE"
                ),
                _stable_key(seed, ride, 0.0, "validation"),
            ),
        )
        if len(rides) >= 2
        else None
    )
    with_splits = [
        replace(
            candidate,
            annotation_metadata={
                **(candidate.annotation_metadata or {}),
                "candidate_split": (
                    "hard_validation_candidates"
                    if candidate.ride_id == validation_ride
                    else "hard_train_candidates"
                ),
            },
        )
        for candidate in selected
    ]
    split_rides = {
        "hard_validation_candidates": [validation_ride] if validation_ride else [],
        "hard_train_candidates": sorted(set(rides) - {validation_ride}),
    }
    split_sample_counts = Counter(
        "hard_validation_candidates"
        if candidate.ride_id == validation_ride
        else "hard_train_candidates"
        for candidate in selected
    )
    split_category_distribution = {
        split: dict(
            sorted(
                Counter(
                    candidate.scene_categories[0]
                    for candidate in selected
                    if (candidate.ride_id == validation_ride)
                    == (split == "hard_validation_candidates")
                ).items()
            )
        )
        for split in ("hard_train_candidates", "hard_validation_candidates")
    }
    selected_hashes = [hashes[candidate.source_sample_id] for candidate in selected]
    report = {
        "source_candidate_count": len(candidates),
        "eligible_candidate_count": len(eligible),
        "selected_sample_count": len(selected),
        "category_distribution": {
            category: category_counts[category] for category in HARD_CATEGORIES
        },
        "category_targets": dict(category_targets),
        "category_shortfalls": shortfalls,
        "ride_distribution": dict(sorted(ride_counts.items())),
        "split_rides": split_rides,
        "split_sample_counts": dict(sorted(split_sample_counts.items())),
        "split_category_distribution": split_category_distribution,
        "ride_leakage": [],
        "approved_exact_overlap_count": exclusions["approved_exact_frame"],
        "approved_visual_duplicate_exclusion_count": exclusions["approved_visual_duplicate"],
        "minimum_selected_hash_distance": min(
            (
                hamming_distance(left, right)
                for index, left in enumerate(selected_hashes)
                for right in selected_hashes[index + 1:]
            ),
            default=None,
        ),
        "minimum_same_ride_time_delta_seconds": min(
            (
                abs(left.timestamp - right.timestamp)
                for index, left in enumerate(selected)
                for right in selected[index + 1:]
                if left.ride_id == right.ride_id
            ),
            default=None,
        ),
        "minimum_separation_seconds": minimum_separation_seconds,
        "maximum_per_ride": maximum_per_ride,
        "hash_distance_threshold": hash_distance_threshold,
        "seed": seed,
        "category_suggestions_are_ground_truth": False,
        "high_confidence_curb_candidates_for_priority_review": [
            {
                "source_sample_id": candidate.source_sample_id,
                "ride_id": candidate.ride_id,
                "frame_id": candidate.frame_id,
                "timestamp": candidate.timestamp,
                "mean_confidence": float(candidate.category_evidence["mean_confidence"]),
                "curb_candidate_score": float(candidate.category_evidence["curb_candidate_score"]),
            }
            for candidate in sorted(
                [item for item in selected if item.scene_categories[0] == "CURB_HARD_NEGATIVE"],
                key=lambda item: (
                    -float(item.category_evidence["mean_confidence"]),
                    -float(item.category_evidence["curb_candidate_score"]),
                ),
            )[:12]
        ],
    }
    category_target_fulfilled = (
        len(selected) == requested_count
        and all(category_counts[name] == target for name, target in category_targets.items())
    )
    partial_bundle_allowed = (
        bool(selected)
        and bool(split_rides["hard_train_candidates"])
        and bool(split_rides["hard_validation_candidates"])
    )
    report["category_target_fulfilled"] = category_target_fulfilled
    report["partial_bundle_allowed"] = partial_bundle_allowed
    report["ready_for_annotation_bundle"] = partial_bundle_allowed
    return with_splits, report


def _category_score(candidate: AnnotationCandidate, category: str) -> float:
    evidence = candidate.category_evidence
    if category == "CURB_HARD_NEGATIVE":
        return float(evidence["curb_candidate_score"])
    if category == "PAVED_HARD_CASE":
        return float(evidence["paved_candidate_score"])
    return float(evidence["true_off_road_candidate_score"])


def _separated(
    candidate: AnnotationCandidate,
    selected: list[AnnotationCandidate],
    minimum_seconds: float,
) -> bool:
    return all(
        candidate.ride_id != other.ride_id
        or abs(candidate.timestamp - other.timestamp) >= minimum_seconds - 1e-9
        for other in selected
    )


def _hash_separated(
    candidate: AnnotationCandidate,
    selected: list[AnnotationCandidate],
    hashes: dict[str, int],
    threshold: int,
) -> bool:
    value = hashes[candidate.source_sample_id]
    return all(hamming_distance(value, hashes[other.source_sample_id]) > threshold for other in selected)


def _candidate_key(candidate: AnnotationCandidate, seed: int) -> str:
    return _stable_key(seed, candidate.ride_id, candidate.timestamp, candidate.frame_id)


def _stable_key(seed: int, ride_id: str, timestamp: float, frame_id: object) -> str:
    return hashlib.sha256(f"{seed}:{ride_id}:{timestamp:.6f}:{frame_id}".encode()).hexdigest()


def _optional_float(value: object) -> float | None:
    if value in (None, "", "None"):
        return None
    return float(value)
