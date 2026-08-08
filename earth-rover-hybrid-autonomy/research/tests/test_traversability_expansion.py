from pathlib import Path

import cv2
import numpy as np

from research.training.traversability_annotation import AnnotationCandidate
from research.training.traversability_expansion import (
    ExistingAnnotationSet,
    difference_hash,
    hamming_distance,
    select_expansion_candidates,
)


def test_difference_hash_is_deterministic_and_detects_identical_images(tmp_path: Path) -> None:
    first = write_image(tmp_path / "first.jpg", seed=1)
    copy = tmp_path / "copy.jpg"
    copy.write_bytes(first.read_bytes())
    different = write_image(tmp_path / "different.jpg", seed=2)

    first_hash = difference_hash(first)

    assert first_hash == difference_hash(first)
    assert first_hash == difference_hash(copy)
    assert hamming_distance(first_hash, difference_hash(different)) > 5


def test_expansion_selection_excludes_existing_near_duplicate_and_abnormal_frames(tmp_path: Path) -> None:
    existing_image = write_image(tmp_path / "existing.jpg", seed=10)
    existing = ExistingAnnotationSet(
        source_keys=frozenset({("ride_existing", 10)}),
        frame_keys=frozenset({("ride_existing", 100, 1000000)}),
        timestamps_by_ride={"ride_existing": (1000.0,)},
        image_hashes=(difference_hash(existing_image),),
        sample_ids=frozenset({"trav_v1_00000"}),
    )
    candidates = [
        make_candidate(tmp_path, "exact", "ride_existing", 1000.0, 100, 10, 11),
        make_candidate(tmp_path, "near_time", "ride_existing", 1005.0, 200, 20, 12),
        make_candidate(tmp_path, "visual", "ride_visual", 2000.0, 300, 30, 10),
        make_candidate(tmp_path, "abnormal", "ride_sky", 3000.0, 400, 40, 13, sky=0.9),
        make_candidate(tmp_path, "eligible_a", "ride_a", 4000.0, 500, 50, 20, category="PAVED_GROUND"),
        make_candidate(tmp_path, "eligible_b", "ride_b", 5000.0, 600, 60, 21, category="OFF_ROAD_GROUND"),
        make_candidate(tmp_path, "eligible_c", "ride_c", 6000.0, 700, 70, 22, category="TURNING"),
        make_candidate(tmp_path, "eligible_d", "ride_d", 7000.0, 800, 80, 23, category="VEHICLE"),
    ]

    selected, report = select_expansion_candidates(candidates, existing, 3, 10.0, 1, 5, seed=7)
    repeated, repeated_report = select_expansion_candidates(candidates, existing, 3, 10.0, 1, 5, seed=7)

    assert selected == repeated
    assert report == repeated_report
    assert len(selected) == 3
    assert len({item.ride_id for item in selected}) == 3
    assert report["selected_existing_exact_overlap_count"] == 0
    assert report["candidate_existing_exact_exclusion_count"] == 1
    assert report["existing_near_time_exclusion_count"] == 1
    assert report["existing_visual_duplicate_exclusion_count"] == 1
    assert report["initial_exclusion_counts"]["abnormal_sky_dominant"] == 1
    assert report["minimum_selected_pair_hash_distance"] > 5
    assert report["minimum_selected_to_existing_hash_distance"] > 5


def make_candidate(
    root: Path,
    name: str,
    ride_id: str,
    timestamp: float,
    frame_id: int,
    manifest_index: int,
    image_seed: int,
    category: str = "PAVED_GROUND",
    sky: float = 0.0,
) -> AnnotationCandidate:
    image_path = write_image(root / f"{name}.jpg", image_seed)
    return AnnotationCandidate(
        source_sample_id=name,
        image_path=image_path,
        pseudo_mask_path=root / f"{name}.png",
        ride_id=ride_id,
        timestamp=timestamp,
        frame_id=frame_id,
        manifest_index=manifest_index,
        playlist=f"ride_{ride_id}/front.m3u8",
        segment=f"ride_{ride_id}/front.ts",
        action_label="FORWARD",
        linear=0.2,
        angular=0.0,
        scene_categories=(category,),
        category_evidence={
            "semantic_fraction": {
                "paved": 0.3 if category == "PAVED_GROUND" else 0.0,
                "off_road": 0.3 if category == "OFF_ROAD_GROUND" else 0.0,
            },
            "pseudo_unknown_fraction": 0.1,
            "top_semantic_fraction": {"sky": sky},
            "image_luminance": {
                "p10": 20.0,
                "p90": 230.0,
                "laplacian_variance": 100.0,
            },
        },
    )


def write_image(path: Path, seed: int) -> Path:
    image = np.random.default_rng(seed).integers(0, 256, size=(72, 128, 3), dtype=np.uint8)
    assert cv2.imwrite(str(path), image)
    return path
