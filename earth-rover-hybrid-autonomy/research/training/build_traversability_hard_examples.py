#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import json
import shutil
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as functional


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.training.datasets.frodobots_2k_dataset import HlsFrameDecoder, ManifestSample
from research.training.datasets.traversability_dataset_v1 import (
    image_rgb_to_tensor,
    letterbox_image,
    restore_letterbox,
    training_prediction_to_source,
)
from research.training.models.traversability_segformer import build_traversability_segformer
from research.training.traversability_annotation import (
    AnnotationCandidate,
    build_annotation_bundle,
    validate_annotation_dataset,
)
from research.training.traversability_expansion import load_existing_annotations
from research.training.traversability_hard_examples import (
    HARD_CATEGORY_TARGETS,
    classify_hard_example,
    select_hard_examples,
    temporal_prefilter,
)
from research.training.traversability_review import colorize_mask, label_contract, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a targeted v1 hard-example CVAT bundle.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--temporal-bundle", required=True)
    parser.add_argument("--approved-dataset", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label-contract", default=str(ROOT / "research/configs/traversability_dataset_v1.yaml"))
    parser.add_argument("--prefilter-count", type=int, default=360)
    parser.add_argument("--prefilter-separation-seconds", type=float, default=0.20)
    parser.add_argument("--selection-separation-seconds", type=float, default=0.75)
    parser.add_argument("--maximum-per-ride", type=int, default=24)
    parser.add_argument("--hash-distance-threshold", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    temporal = Path(args.temporal_bundle).expanduser().resolve()
    approved = Path(args.approved_dataset).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    contract = Path(args.label_contract).expanduser().resolve()
    for path in (dataset_root, temporal, approved, checkpoint_path, contract):
        if not path.exists():
            raise SystemExit(f"required input does not exist: {path}")
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"output directory is not empty: {output}")
    if output == dataset_root or dataset_root in output.parents:
        raise SystemExit("output directory must remain outside the immutable raw dataset")
    statistics_path = temporal / "per_frame_statistics.csv"
    report_path = temporal / "temporal_inference_report.json"
    if not statistics_path.is_file() or not report_path.is_file():
        raise SystemExit("temporal bundle is missing per-frame statistics or its report")
    temporal_report = json.loads(report_path.read_text(encoding="utf-8"))
    if temporal_report.get("success") is not True or temporal_report.get("approved_ride_overlap") != []:
        raise SystemExit("temporal source did not pass its unseen-ride gate")

    rows = list(csv.DictReader(statistics_path.open(newline="", encoding="utf-8")))
    prefiltered = temporal_prefilter(
        rows,
        args.prefilter_separation_seconds,
        args.prefilter_count,
        args.seed,
    )
    if not prefiltered:
        raise SystemExit("temporal statistics contain no relevant hard-example candidates")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_cuda and device.type != "cuda":
        raise SystemExit("CUDA is required but unavailable")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = build_traversability_segformer(False, checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    decoder = HlsFrameDecoder(dataset_root)
    cache = output.parent / f".{output.name}_candidate_cache"
    if cache.exists() and any(cache.iterdir()):
        raise SystemExit(f"candidate cache is not empty: {cache}")
    (cache / "images").mkdir(parents=True, exist_ok=True)
    (cache / "masks").mkdir()
    (cache / "confidence").mkdir()

    inferred: list[AnnotationCandidate] = []
    failures: list[dict[str, object]] = []
    for position, temporal_candidate in enumerate(prefiltered):
        row = temporal_candidate["row"]
        source_id = f"hard_source_{row['ride_id']}_{int(row['frame_id']):08d}"
        sample = _manifest_sample(row)
        try:
            image_rgb = decoder.decode(sample)
            prediction, source_prediction, confidence = infer_frame(
                model,
                image_rgb,
                args.image_size,
                device,
            )
            suggestion, evidence = classify_hard_example(prediction, confidence, temporal_candidate)
            if suggestion is None:
                continue
            image_path = cache / "images" / f"{source_id}.jpg"
            mask_path = cache / "masks" / f"{source_id}.png"
            confidence_path = cache / "confidence" / f"{source_id}.png"
            if not cv2.imwrite(str(image_path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)):
                raise OSError("cannot write candidate image")
            if not cv2.imwrite(str(mask_path), source_prediction):
                raise OSError("cannot write candidate mask")
            if not cv2.imwrite(
                str(confidence_path),
                np.clip(confidence * 255.0, 0, 255).astype(np.uint8),
            ):
                raise OSError("cannot write candidate confidence")
            ratios = {
                "ON_ROAD": float(np.mean(prediction == 0)),
                "OFF_ROAD": float(np.mean(prediction == 1)),
                "OBSTACLE": float(np.mean(prediction == 2)),
            }
            reasons = list(temporal_candidate["candidate_reasons"])
            evidence.update(
                {
                    "candidate_reasons": reasons,
                    "scene_category_suggestion": suggestion,
                    "warning": "Automatic hard-example category suggestion; human confirmation required.",
                }
            )
            inferred.append(
                AnnotationCandidate(
                    source_sample_id=source_id,
                    image_path=image_path,
                    pseudo_mask_path=mask_path,
                    ride_id=row["ride_id"],
                    timestamp=float(row["timestamp"]),
                    frame_id=int(row["frame_id"]),
                    manifest_index=int(row["manifest_index"]),
                    playlist=row["playlist"],
                    segment=row["hls_segment"],
                    action_label=row["action_reference"],
                    linear=float(row["linear_reference"]),
                    angular=float(row["angular_reference"]),
                    scene_categories=(suggestion,),
                    category_evidence=evidence,
                    annotation_metadata={
                        "scene_category_source": "v1_prediction_temporal_and_spatial_heuristics_unverified",
                        "scene_category_suggestion": suggestion,
                        "candidate_reason": "|".join(reasons),
                        "prediction_on_road_ratio": f"{ratios['ON_ROAD']:.6f}",
                        "prediction_off_road_ratio": f"{ratios['OFF_ROAD']:.6f}",
                        "prediction_obstacle_ratio": f"{ratios['OBSTACLE']:.6f}",
                        "mean_confidence": f"{float(confidence.mean()):.6f}",
                        "confidence_path": "",
                    },
                )
            )
            if (position + 1) % 25 == 0:
                print(f"inferred={position + 1}/{len(prefiltered)} eligible={len(inferred)}", flush=True)
        except (OSError, RuntimeError, ValueError) as exc:
            failures.append(
                {
                    "ride_id": row["ride_id"],
                    "frame_id": row["frame_id"],
                    "timestamp": row["timestamp"],
                    "error": str(exc),
                }
            )

    existing = load_existing_annotations(approved)
    selected, selection = select_hard_examples(
        inferred,
        existing,
        category_targets=HARD_CATEGORY_TARGETS,
        minimum_separation_seconds=args.selection_separation_seconds,
        maximum_per_ride=args.maximum_per_ride,
        hash_distance_threshold=args.hash_distance_threshold,
        seed=args.seed,
    )
    selection.update(
        {
            "temporal_prefilter_count": len(prefiltered),
            "inferred_candidate_count": len(inferred),
            "inference_failure_count": len(failures),
            "inference_failures": failures,
            "approved_dataset": str(approved),
            "approved_ride_overlap": sorted(
                {candidate.ride_id for candidate in selected}
                & {sample_ride for sample_ride in _approved_rides(approved)}
            ),
            "existing_v1_test_reused_for_tuning": False,
        }
    )
    shortfall_report = output.parent / f"{output.name}_candidate_shortfall.json"
    if not selection["category_target_fulfilled"] or not selection["partial_bundle_allowed"]:
        write_json(shortfall_report, selection)
    if not selection["partial_bundle_allowed"]:
        raise SystemExit(f"insufficient relevant candidates; inspect {shortfall_report}")
    if selection["approved_ride_overlap"]:
        raise SystemExit("selected hard-example rides overlap approved_120_v1 rides")

    selected = [
        replace(
            candidate,
            annotation_metadata={
                **(candidate.annotation_metadata or {}),
                "confidence_path": f"confidence/trav_v2_hard_{position:05d}.png",
            },
        )
        for position, candidate in enumerate(selected)
    ]

    build = build_annotation_bundle(
        selected,
        output,
        contract,
        temporal,
        args.seed,
        args.selection_separation_seconds,
        sample_id_prefix="trav_v2_hard_",
        seed_mask_contract="v1_source",
    )
    build.update(
        {
            "dataset_name": "traversability_dataset_v2_hard_examples_annotation_v1",
            "source_model_checkpoint": str(checkpoint_path),
            "source_temporal_bundle": str(temporal),
            "temporal_prefilter_count": len(prefiltered),
            "inferred_candidate_count": len(inferred),
            "selected_sample_count": len(selected),
            "requested_category_targets": HARD_CATEGORY_TARGETS,
            "category_target_fulfilled": selection["category_target_fulfilled"],
            "category_shortfalls": selection["category_shortfalls"],
            "category_suggestions_are_ground_truth": False,
            "all_seed_masks_require_human_correction": True,
            "additional_training_performed": False,
            "temporal_smoothing_performed": False,
            "planner_or_live_rover_integration_performed": False,
        }
    )
    write_json(output / "build_report.json", build)
    write_json(output / "selection_report.json", selection)
    copy_confidence_maps(output, selected, cache)
    create_context_strips(output, selected, rows, decoder)
    write_hard_review_html(output)
    write_hard_readme(output)
    validation = validate_annotation_dataset(output, require_masks=False)
    if not validation["valid"]:
        raise SystemExit(json.dumps(validation, indent=2, sort_keys=True))
    shutil.rmtree(cache)
    print(json.dumps({"build": build, "selection": selection, "validation": validation}, indent=2, sort_keys=True))
    return 0


@torch.inference_mode()
def infer_frame(
    model: torch.nn.Module,
    image_rgb: np.ndarray,
    image_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    padded, _ = letterbox_image(image_rgb, image_size)
    tensor = image_rgb_to_tensor(padded).unsqueeze(0).to(device)
    logits = model(tensor).logits
    logits = functional.interpolate(logits, (image_size, image_size), mode="bilinear", align_corners=False)
    probabilities = logits.softmax(dim=1)
    confidence_tensor, prediction_tensor = probabilities.max(dim=1)
    prediction = prediction_tensor[0].cpu().numpy().astype(np.uint8)
    confidence = confidence_tensor[0].cpu().numpy().astype(np.float32)
    prediction = restore_letterbox(prediction, image_rgb.shape[:2], cv2.INTER_NEAREST)
    confidence = restore_letterbox(confidence, image_rgb.shape[:2], cv2.INTER_LINEAR)
    return prediction, training_prediction_to_source(prediction), confidence


def create_context_strips(
    output: Path,
    selected: list[AnnotationCandidate],
    rows: list[dict[str, str]],
    decoder: HlsFrameDecoder,
) -> None:
    context_dir = output / "context"
    context_dir.mkdir()
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("status") == "OK":
            grouped[row["ride_id"]].append(row)
    for ride_rows in grouped.values():
        ride_rows.sort(key=lambda row: (float(row["timestamp"]), int(row["frame_id"])))
    metadata_rows = list(csv.DictReader((output / "metadata.csv").open(newline="", encoding="utf-8")))
    output_by_source = {row["source_pseudo_sample_id"]: row["sample_id"] for row in metadata_rows}
    for candidate in selected:
        ride_rows = grouped[candidate.ride_id]
        center = next(index for index, row in enumerate(ride_rows) if int(row["frame_id"]) == candidate.frame_id)
        if len(ride_rows) < 5:
            raise ValueError(f"ride {candidate.ride_id} has fewer than five temporal context frames")
        context_start = max(0, min(center - 2, len(ride_rows) - 5))
        indices = list(range(context_start, context_start + 5))
        panels = []
        context_frames = []
        for index in indices:
            row = ride_rows[index]
            frame = cv2.cvtColor(decoder.decode(_manifest_sample(row)), cv2.COLOR_RGB2BGR)
            resized = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
            color = (0, 255, 255) if index == center else (255, 255, 255)
            cv2.putText(
                resized,
                f"frame={row['frame_id']} ts={float(row['timestamp']):.3f}",
                (6, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )
            panels.append(resized)
            context_frames.append(
                {
                    "frame_id": int(row["frame_id"]),
                    "timestamp": float(row["timestamp"]),
                    "playlist": row["playlist"],
                    "hls_segment": row["hls_segment"],
                    "mean_confidence": float(row["mean_confidence"]),
                    "prediction_on_road_ratio": float(row["on_road_ratio"]),
                    "prediction_off_road_ratio": float(row["off_road_ratio"]),
                    "prediction_obstacle_ratio": float(row["obstacle_ratio"]),
                    "anomaly_reasons": [
                        reason for reason in row.get("anomaly_reasons", "").split("|") if reason
                    ],
                    "is_annotation_center": index == center,
                }
            )
        sample_id = output_by_source[candidate.source_sample_id]
        context_relative = f"context/{sample_id}.jpg"
        if not cv2.imwrite(str(output / context_relative), np.hstack(panels)):
            raise OSError(f"cannot write context strip for {sample_id}")
        metadata_path = output / "metadata" / f"{sample_id}.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["context_path"] = context_relative
        metadata["context_frames"] = context_frames
        write_json(metadata_path, metadata)


def copy_confidence_maps(
    output: Path,
    selected: list[AnnotationCandidate],
    cache: Path,
) -> None:
    confidence_dir = output / "confidence"
    confidence_dir.mkdir()
    for position, candidate in enumerate(selected):
        source = cache / "confidence" / f"{candidate.source_sample_id}.png"
        destination = confidence_dir / f"trav_v2_hard_{position:05d}.png"
        shutil.copy2(source, destination)


def write_hard_review_html(output: Path) -> None:
    rows = list(csv.DictReader((output / "metadata.csv").open(newline="", encoding="utf-8")))
    rows.sort(
        key=lambda row: (
            row["scene_category_suggestion"] != "CURB_HARD_NEGATIVE",
            -float(row["mean_confidence"]),
            row["sample_id"],
        )
    )
    _, colors, _ = label_contract(output / "label_contract.yaml")
    seed_visuals = output / "seed_visualizations"
    seed_visuals.mkdir()
    cards = []
    for row in rows:
        sample_id = row["sample_id"]
        image = cv2.imread(str(output / row["image_path"]), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(output / "initial_masks" / f"{sample_id}.png"), cv2.IMREAD_UNCHANGED)
        if image is None or mask is None:
            raise OSError(f"cannot read review assets for {sample_id}")
        color = cv2.cvtColor(colorize_mask(mask, colors), cv2.COLOR_RGB2BGR)
        overlay = cv2.addWeighted(image, 0.55, color, 0.45, 0.0)
        confidence = cv2.imread(str(output / row["confidence_path"]), cv2.IMREAD_GRAYSCALE)
        if confidence is None:
            raise OSError(f"cannot read confidence map for {sample_id}")
        confidence_color = cv2.applyColorMap(confidence, cv2.COLORMAP_VIRIDIS)
        confidence_relative = f"confidence_visualizations/{sample_id}.jpg"
        (output / "confidence_visualizations").mkdir(exist_ok=True)
        if not cv2.imwrite(str(output / confidence_relative), confidence_color):
            raise OSError(f"cannot write confidence visualization for {sample_id}")
        if not cv2.imwrite(str(seed_visuals / f"{sample_id}.jpg"), overlay):
            raise OSError(f"cannot write seed overlay for {sample_id}")
        cards.append(
            f"<article><h2>{html.escape(sample_id)} · {html.escape(row['scene_category_suggestion'])}</h2>"
            f"<p>split={html.escape(row['candidate_split'])} ride={html.escape(row['ride_id'])} "
            f"frame={html.escape(row['frame_id'])} confidence={html.escape(row['mean_confidence'])}</p>"
            f"<p>reason={html.escape(row['candidate_reason'])}</p><div>"
            f"<figure><img src='{html.escape(row['image_path'])}'><figcaption>Original</figcaption></figure>"
            f"<figure><img src='seed_visualizations/{sample_id}.jpg'><figcaption>Unverified v1 seed overlay</figcaption></figure>"
            f"<figure><img src='{confidence_relative}'><figcaption>8-bit max-softmax confidence</figcaption></figure>"
            f"</div><img class='context' src='context/{sample_id}.jpg'></article>"
        )
    document = """<!doctype html><html><head><meta charset='utf-8'><title>Hard-example Annotation Review</title>
<style>body{font-family:system-ui;margin:20px}article{border-bottom:1px solid #bbb;padding:14px 0}div{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}figure{margin:0}img{width:100%}.context{margin-top:8px}@media(max-width:800px){div{grid-template-columns:1fr}}</style>
</head><body><h1>Targeted Hard-example Candidates</h1><p>Category suggestions and v1 masks are unverified. Confirm curb geometry and correct every mask in CVAT.</p>__CARDS__</body></html>""".replace("__CARDS__", "\n".join(cards))
    (output / "review.html").write_text(document, encoding="utf-8")


def write_hard_readme(output: Path) -> None:
    (output / "README.md").write_text(
        """# Traversability v2 Hard-example Annotation\n\nThis bundle contains unverified category suggestions from the approved v1 model. `CURB_HARD_NEGATIVE`, `TRUE_OFF_ROAD`, and `PAVED_HARD_CASE` are sampling hints, not labels. Open `review.html`, then import `images/` and `cvat_seed_annotations.zip` into CVAT using Segmentation Mask 1.1. Correct every pixel mask. Curbs, vertical curb faces, high ledges, and stairs are OBSTACLE; traversable gravel, dirt, and short grass are OFF_ROAD; paved traversable ground is ON_ROAD; uncertain height boundaries are IGNORE. Do not add a CURB class. No sample is approved for training before Dell validation and human overlay approval.\n""",
        encoding="utf-8",
    )


def _manifest_sample(row: dict[str, str]) -> ManifestSample:
    return ManifestSample(
        ride_id=row["ride_id"],
        front_playlist_ref=row["playlist"],
        front_segment_ref=row["hls_segment"],
        front_frame_id=int(row["frame_id"]),
        front_timestamp=float(row["timestamp"]),
        matched_control_timestamp=float(row["timestamp"]),
        control_delta_ms=0.0,
        linear=float(row["linear_reference"]),
        angular=float(row["angular_reference"]),
        action_class=row["action_reference"],
        timeline_section_id=0,
    )


def _approved_rides(approved: Path) -> set[str]:
    return {
        row["ride_id"]
        for row in csv.DictReader((approved / "metadata.csv").open(newline="", encoding="utf-8"))
    }


if __name__ == "__main__":
    raise SystemExit(main())
