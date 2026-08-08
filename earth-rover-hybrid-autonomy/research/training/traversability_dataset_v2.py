from __future__ import annotations

import hashlib
import random
from collections import Counter, defaultdict


TRAINING_CLASSES = ("ON_ROAD", "OFF_ROAD", "OBSTACLE")


def choose_new_group_split(
    rows: list[dict[str, str]],
    class_pixels: dict[str, dict[str, int]],
    seed: int,
    holdout_ratio: float = 0.20,
    trials: int = 5000,
) -> dict[str, tuple[str, ...]]:
    if not 0.0 < holdout_ratio < 1.0:
        raise ValueError("holdout ratio must be between zero and one")
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[group_id(row)].append(row)
    group_ids = sorted(groups)
    if len(group_ids) < 2:
        raise ValueError("at least two source ride groups are required")
    target_samples = max(1, round(len(rows) * holdout_ratio))
    global_pixels = _pixels(rows, class_pixels)
    global_total = sum(global_pixels[name] for name in TRAINING_CLASSES)
    if global_total == 0 or any(global_pixels[name] == 0 for name in TRAINING_CLASSES):
        raise ValueError("new manual samples must contain every training class")
    rng = random.Random(seed)
    best: tuple[float, tuple[str, ...]] | None = None
    for _ in range(trials):
        shuffled = group_ids.copy()
        rng.shuffle(shuffled)
        selected: list[str] = []
        count = 0
        for candidate in shuffled:
            candidate_count = len(groups[candidate])
            if not selected or abs(count + candidate_count - target_samples) <= abs(
                count - target_samples
            ):
                selected.append(candidate)
                count += candidate_count
        holdout = tuple(sorted(selected))
        train = tuple(group for group in group_ids if group not in set(holdout))
        if not train or not holdout:
            continue
        train_rows = [row for group in train for row in groups[group]]
        holdout_rows = [row for group in holdout for row in groups[group]]
        train_pixels = _pixels(train_rows, class_pixels)
        holdout_pixels = _pixels(holdout_rows, class_pixels)
        if any(
            pixels[name] == 0
            for pixels in (train_pixels, holdout_pixels)
            for name in TRAINING_CLASSES
        ):
            continue
        holdout_total = sum(holdout_pixels[name] for name in TRAINING_CLASSES)
        score = abs(len(holdout_rows) / len(rows) - holdout_ratio)
        score += 0.25 * sum(
            abs(
                holdout_pixels[name] / holdout_total
                - global_pixels[name] / global_total
            )
            for name in TRAINING_CLASSES
        )
        ranked = (score, holdout)
        if best is None or ranked < best:
            best = ranked
    if best is None:
        raise ValueError("no deterministic group split contains all classes in train and holdout")
    holdout = best[1]
    train = tuple(group for group in group_ids if group not in set(holdout))
    return {"train": train, "new_holdout": holdout}


def group_id(row: dict[str, str]) -> str:
    return row["ride_id"]


def regression_assessment(
    v1_metrics: dict[str, object],
    v2_metrics: dict[str, object],
    max_mean_iou_drop: float,
    max_pixel_accuracy_drop: float,
    max_false_traversable_increase: float,
) -> dict[str, object]:
    differences = {
        "mean_iou": float(v2_metrics["mean_iou"]) - float(v1_metrics["mean_iou"]),
        "pixel_accuracy": float(v2_metrics["pixel_accuracy"])
        - float(v1_metrics["pixel_accuracy"]),
        "false_traversable_rate": float(v2_metrics["false_traversable_rate"])
        - float(v1_metrics["false_traversable_rate"]),
    }
    failures = []
    if differences["mean_iou"] < -max_mean_iou_drop:
        failures.append("mean_iou_regression")
    if differences["pixel_accuracy"] < -max_pixel_accuracy_drop:
        failures.append("pixel_accuracy_regression")
    if differences["false_traversable_rate"] > max_false_traversable_increase:
        failures.append("false_traversable_rate_regression")
    return {
        "pass": not failures,
        "differences_v2_minus_v1": differences,
        "failure_reasons": failures,
        "thresholds": {
            "max_mean_iou_drop": max_mean_iou_drop,
            "max_pixel_accuracy_drop": max_pixel_accuracy_drop,
            "max_false_traversable_rate_increase": max_false_traversable_increase,
        },
    }


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _pixels(
    rows: list[dict[str, str]],
    class_pixels: dict[str, dict[str, int]],
) -> Counter[str]:
    return sum((Counter(class_pixels[row["sample_id"]]) for row in rows), Counter())
