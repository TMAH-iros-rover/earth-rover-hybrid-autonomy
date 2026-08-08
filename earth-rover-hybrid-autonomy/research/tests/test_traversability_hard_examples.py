from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from research.training.traversability_annotation import (
    AnnotationCandidate,
    build_annotation_bundle,
    validate_annotation_dataset,
)
from research.training.traversability_expansion import ExistingAnnotationSet
from research.training.traversability_hard_examples import (
    HARD_CATEGORY_TARGETS,
    classify_hard_example,
    select_hard_examples,
    temporal_prefilter,
)


def test_temporal_prefilter_selects_off_road_transitions_deterministically() -> None:
    rows = []
    for index in range(20):
        rows.append(
            {
                "status": "OK",
                "ride_id": "ride_a",
                "timestamp": str(1000.0 + index * 0.1),
                "frame_id": str(index),
                "on_road_ratio": str(0.8 - index * 0.02),
                "off_road_ratio": str(0.1 + index * 0.02),
                "mean_confidence": "0.8",
                "mean_confidence_drop": "0.0",
                "pixel_flicker_fraction": "0.12",
            }
        )

    first = temporal_prefilter(rows, 0.2, 10, 17)
    repeated = temporal_prefilter(rows, 0.2, 10, 17)

    assert first == repeated
    assert 1 <= len(first) <= 10
    assert all(item["candidate_reasons"] for item in first)
    timestamps = sorted(float(item["row"]["timestamp"]) for item in first)
    assert all(right - left >= 0.2 - 1e-9 for left, right in zip(timestamps, timestamps[1:]))


def test_spatial_heuristic_suggests_curb_candidate_without_claiming_truth() -> None:
    prediction = np.zeros((40, 60), dtype=np.uint8)
    prediction[22:30] = 1
    prediction[30:] = 2
    confidence = np.full((40, 60), 0.85, dtype=np.float32)
    temporal = {
        "off_road_delta": 0.05,
        "on_off_churn": 0.08,
        "row": {"pixel_flicker_fraction": "0.10"},
    }

    category, evidence = classify_hard_example(prediction, confidence, temporal)

    assert category == "CURB_HARD_NEGATIVE"
    assert evidence["off_road_obstacle_adjacency"] > 0.0
    assert evidence["mean_confidence"] > 0.8


def test_hard_selection_meets_exact_category_targets_and_isolates_validation_ride(
    tmp_path: Path,
) -> None:
    candidates = []
    categories = ("CURB_HARD_NEGATIVE", "TRUE_OFF_ROAD", "PAVED_HARD_CASE")
    for index in range(72):
        category = categories[index % 3]
        ride = str(index % 3)
        path = tmp_path / f"image_{index:03d}.jpg"
        image = np.random.default_rng(index).integers(0, 256, (72, 128, 3), dtype=np.uint8)
        assert cv2.imwrite(str(path), image)
        candidates.append(make_candidate(path, category, ride, 1000.0 + index * 2.0, index))
    existing = ExistingAnnotationSet(frozenset(), frozenset(), {}, (), frozenset())

    selected, report = select_hard_examples(
        candidates,
        existing,
        HARD_CATEGORY_TARGETS,
        0.75,
        24,
        4,
        17,
    )

    assert len(selected) == 24
    assert report["ready_for_annotation_bundle"] is True
    assert report["category_target_fulfilled"] is True
    assert report["category_distribution"] == HARD_CATEGORY_TARGETS
    train = set(report["split_rides"]["hard_train_candidates"])
    validation = set(report["split_rides"]["hard_validation_candidates"])
    assert train
    assert validation
    assert not train & validation
    assert report["ride_leakage"] == []
    assert sum(report["split_sample_counts"].values()) == 24
    assert set(report["split_category_distribution"]) == {
        "hard_train_candidates",
        "hard_validation_candidates",
    }


def test_hard_selection_reports_single_ride_shortfall(tmp_path: Path) -> None:
    candidates = []
    categories = ("CURB_HARD_NEGATIVE", "TRUE_OFF_ROAD", "PAVED_HARD_CASE")
    for index in range(24):
        path = tmp_path / f"single_ride_{index:03d}.jpg"
        image = np.random.default_rng(index).integers(0, 256, (72, 128, 3), dtype=np.uint8)
        assert cv2.imwrite(str(path), image)
        candidates.append(
            make_candidate(path, categories[index % 3], "only_ride", 1000.0 + index, index)
        )

    _, report = select_hard_examples(
        candidates,
        ExistingAnnotationSet(frozenset(), frozenset(), {}, (), frozenset()),
        HARD_CATEGORY_TARGETS,
        0.75,
        24,
        4,
        17,
    )

    assert report["ready_for_annotation_bundle"] is False
    assert report["partial_bundle_allowed"] is False
    assert report["split_rides"]["hard_validation_candidates"] == []


def test_hard_selection_does_not_fill_a_missing_category_from_other_categories(
    tmp_path: Path,
) -> None:
    candidates = []
    categories = ["CURB_HARD_NEGATIVE"] * 12 + ["TRUE_OFF_ROAD"] * 12 + ["PAVED_HARD_CASE"] * 3
    for index, category in enumerate(categories):
        path = tmp_path / f"shortfall_{index:03d}.jpg"
        image = np.random.default_rng(index).integers(0, 256, (72, 128, 3), dtype=np.uint8)
        assert cv2.imwrite(str(path), image)
        candidates.append(make_candidate(path, category, str(index % 3), 1000.0 + index * 2, index))

    selected, report = select_hard_examples(
        candidates,
        ExistingAnnotationSet(frozenset(), frozenset(), {}, (), frozenset()),
        HARD_CATEGORY_TARGETS,
        0.75,
        24,
        4,
        17,
    )

    assert len(selected) == 21
    assert report["category_distribution"] == {
        "CURB_HARD_NEGATIVE": 12,
        "TRUE_OFF_ROAD": 6,
        "PAVED_HARD_CASE": 3,
    }
    assert report["category_shortfalls"]["PAVED_HARD_CASE"] == 3
    assert report["category_target_fulfilled"] is False
    assert report["partial_bundle_allowed"] is True
    assert report["ready_for_annotation_bundle"] is True


def test_v1_source_seed_preserves_off_road_and_obstacle_ids(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    image_path = source / "image.jpg"
    mask_path = source / "mask.png"
    assert cv2.imwrite(str(image_path), np.zeros((18, 32, 3), dtype=np.uint8))
    source_mask = np.array([[0, 1, 2, 3] * 8] * 18, dtype=np.uint8)
    assert cv2.imwrite(str(mask_path), source_mask)
    candidate = make_candidate(image_path, "CURB_HARD_NEGATIVE", "ride", 1000.0, 4)
    candidate = replace(candidate, pseudo_mask_path=mask_path, manifest_index=-1)
    second = replace(
        candidate,
        source_sample_id="source_ride_5",
        frame_id=5,
        timestamp=1000.05,
    )
    output = tmp_path / "bundle"

    report = build_annotation_bundle(
        [candidate, second],
        output,
        Path(__file__).parents[2] / "research/configs/traversability_dataset_v1.yaml",
        source,
        17,
        1.0,
        sample_id_prefix="hard_",
        seed_mask_contract="v1_source",
    )

    imported = cv2.imread(str(output / "initial_masks/hard_00000.png"), cv2.IMREAD_UNCHANGED)
    assert np.array_equal(imported, source_mask)
    assert report["seed_mask_contract"] == "v1_source"
    assert validate_annotation_dataset(output, require_masks=False)["valid"] is True


def make_candidate(
    image_path: Path,
    category: str,
    ride: str,
    timestamp: float,
    frame_id: int,
) -> AnnotationCandidate:
    return AnnotationCandidate(
        source_sample_id=f"source_{ride}_{frame_id}",
        image_path=image_path,
        pseudo_mask_path=image_path.with_suffix(".png"),
        ride_id=ride,
        timestamp=timestamp,
        frame_id=frame_id,
        manifest_index=frame_id,
        playlist=f"ride_{ride}/front.m3u8",
        segment=f"ride_{ride}/front.ts",
        action_label="FORWARD",
        linear=0.2,
        angular=0.0,
        scene_categories=(category,),
        category_evidence={
            "curb_candidate_score": 1.0 + frame_id / 1000,
            "paved_candidate_score": 1.0 + frame_id / 1000,
            "true_off_road_candidate_score": 1.0 + frame_id / 1000,
            "mean_confidence": 0.8,
        },
        annotation_metadata={"scene_category_suggestion": category},
    )
