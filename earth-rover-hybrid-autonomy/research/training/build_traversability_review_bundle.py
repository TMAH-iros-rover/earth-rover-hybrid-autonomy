#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as functional


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.training.datasets.frodobots_2k_dataset import HlsFrameDecoder, load_manifest
from research.training.traversability_review import (
    SelectedFrame,
    class_area_statistics,
    colorize_mask,
    copy_review_configs,
    decode_selected_frame,
    label_contract,
    letterbox_rgb,
    load_yaml,
    overlay_rgb,
    restore_from_letterbox,
    select_representative_frames,
    semantic_colors,
    semantic_to_traversability,
    write_contact_sheet,
    write_cvat_labelmap,
    write_gallery,
    write_json,
    write_review_csv,
)


CHECKPOINT_LICENSE_SOURCE = "https://github.com/NVlabs/SegFormer/blob/master/LICENSE"
CHECKPOINT_SOURCE = "https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a human-review bundle from conservative SegFormer pseudo-labels.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label-contract", default=str(ROOT / "research/configs/traversability_label_contract.yaml"))
    parser.add_argument("--semantic-mapping", default=str(ROOT / "research/configs/traversability_semantic_mapping.yaml"))
    parser.add_argument("--sample-count", type=int, default=40)
    parser.add_argument("--max-rides", type=int, default=8)
    parser.add_argument("--minimum-separation-seconds", type=float, default=5.0)
    parser.add_argument("--input-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _validate_args(args)
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    contract_path = Path(args.label_contract).expanduser().resolve()
    mapping_path = Path(args.semantic_mapping).expanduser().resolve()
    if output_dir == dataset_root or dataset_root in output_dir.parents:
        raise SystemExit("output-dir must remain outside the immutable raw dataset root")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"output-dir is not empty; use a new review bundle directory: {output_dir}")

    mapping_config = load_yaml(mapping_path)
    name_to_id, traversability_colors, _ = label_contract(contract_path)
    samples = load_manifest(manifest_path)
    selected = select_representative_frames(
        samples,
        requested_count=args.sample_count,
        maximum_rides=args.max_rides,
        minimum_separation_seconds=args.minimum_separation_seconds,
        seed=args.seed,
    )
    if not selected:
        raise SystemExit("representative sampler returned no frames")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_cuda and device.type != "cuda":
        raise SystemExit("CUDA is required but torch.cuda.is_available() is false")
    model_bundle = _load_model(mapping_config, device)
    output_dir.mkdir(parents=True, exist_ok=True)
    _create_bundle_directories(output_dir)
    decoder = HlsFrameDecoder(dataset_root)
    entries: list[dict[str, str]] = []
    failures: list[dict[str, object]] = []
    model_latencies_ms: list[float] = []
    end_to_end_latencies_ms: list[float] = []
    traversability_pixels: Counter[str] = Counter()
    semantic_pixels: Counter[str] = Counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for position, selected_frame in enumerate(selected):
        sample_id = f"sample_{position:05d}"
        image_rgb, decode_failure = decode_selected_frame(decoder, selected_frame)
        if decode_failure is not None:
            failures.append({"sample_id": sample_id, **decode_failure})
            continue
        try:
            assert image_rgb is not None
            result = _infer_frame(
                image_rgb,
                model_bundle,
                mapping_config,
                name_to_id,
                traversability_colors,
                args.input_size,
                device,
            )
            model_latencies_ms.append(float(result["model_latency_ms"]))
            end_to_end_latencies_ms.append(float(result["end_to_end_latency_ms"]))
            for class_id, count in Counter(int(value) for value in result["traversability_mask"].reshape(-1)).items():
                traversability_pixels[result["traversability_id_to_name"][class_id]] += count
            for class_id, count in Counter(int(value) for value in result["semantic_mask"].reshape(-1)).items():
                semantic_pixels[result["id_to_label"][class_id]] += count
            entry = _write_sample(output_dir, sample_id, selected_frame, image_rgb, result)
            entries.append(entry)
            print(f"processed={len(entries)}/{len(selected)} sample={sample_id} ride={selected_frame.sample.ride_id}", flush=True)
        except (OSError, RuntimeError, ValueError) as exc:
            failures.append(
                {
                    "sample_id": sample_id,
                    "manifest_index": selected_frame.manifest_index,
                    "ride_id": selected_frame.sample.ride_id,
                    "error": str(exc),
                }
            )
    if not entries:
        raise SystemExit("all selected frames failed decoding or inference")

    copy_review_configs(contract_path, mapping_path, output_dir)
    write_review_csv(entries, output_dir / "review.csv")
    write_gallery(entries, output_dir / "gallery.html")
    write_contact_sheet(entries, output_dir, output_dir / "contact_sheet.jpg")
    write_cvat_labelmap(name_to_id, traversability_colors, output_dir / "cvat_labelmap.txt")
    _write_bundle_readme(output_dir)
    _write_sampling_metadata(entries, output_dir / "sampling_metadata.csv")

    latency_report = {
        "gpu_model_and_logits": _latency_report(model_latencies_ms),
        "end_to_end_per_frame": _latency_report(end_to_end_latencies_ms),
    }
    peak_allocated = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    peak_reserved = torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0
    report = {
        "pipeline_status": "REVIEW_REQUIRED",
        "pseudo_labels_are_ground_truth": False,
        "fine_tuning_performed": False,
        "live_rover_commands_sent": False,
        "dataset_root": str(dataset_root),
        "manifest_path": str(manifest_path),
        "output_dir": str(output_dir),
        "seed": args.seed,
        "requested_sample_count": args.sample_count,
        "selected_sample_count": len(selected),
        "successful_frame_count": len(entries),
        "failed_frame_count": len(failures),
        "processed_ride_count": len({entry["ride_id"] for entry in entries}),
        "ride_distribution": dict(sorted(Counter(entry["ride_id"] for entry in entries).items())),
        "action_reference_distribution": dict(sorted(Counter(entry["action_label"] for entry in entries).items())),
        "minimum_separation_seconds": args.minimum_separation_seconds,
        "input_geometry": {
            "method": "aspect-ratio-preserving resize plus square letterbox; padding removed before output",
            "model_input_size": [args.input_size, args.input_size],
            "outputs_restored_to_original_dimensions": True,
        },
        "model": model_bundle["provenance"],
        "installed_packages": _package_versions(),
        "inference_latency_ms": latency_report,
        "peak_vram_bytes": {
            "allocated": peak_allocated,
            "reserved": peak_reserved,
        },
        "pseudo_label_pixel_distribution": _counter_distribution(traversability_pixels),
        "top_semantic_pixel_distribution": _counter_distribution(semantic_pixels, limit=15),
        "failures": failures,
        "review_status_counts": {"UNREVIEWED": len(entries)},
    }
    write_json(output_dir / "bundle_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _load_model(mapping_config: dict[str, object], device: torch.device) -> dict[str, object]:
    try:
        from huggingface_hub import hf_hub_download
        from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
    except ImportError as exc:
        raise SystemExit(
            f"cannot import the SegFormer runtime; install requirements-segmentation.txt on Dell: {exc}"
        ) from exc

    checkpoint = str(mapping_config["checkpoint"])
    revision = str(mapping_config["revision"])
    weight_file = str(mapping_config["weight_file"])
    weight_path = Path(hf_hub_download(repo_id=checkpoint, filename=weight_file, revision=revision))
    processor = AutoImageProcessor.from_pretrained(checkpoint, revision=revision)
    model = SegformerForSemanticSegmentation.from_pretrained(
        checkpoint,
        revision=revision,
        use_safetensors=True,
    ).to(device)
    model.eval()
    id_to_label = {int(class_id): str(name).strip() for class_id, name in model.config.id2label.items()}
    return {
        "processor": processor,
        "model": model,
        "id_to_label": id_to_label,
        "semantic_colors": semantic_colors(id_to_label),
        "provenance": {
            "checkpoint": checkpoint,
            "revision": revision,
            "weight_file": weight_file,
            "weight_sha256": _sha256(weight_path),
            "weight_size_bytes": weight_path.stat().st_size,
            "checkpoint_source": CHECKPOINT_SOURCE,
            "license_source": CHECKPOINT_LICENSE_SOURCE,
            "license_restriction": "NVIDIA research/evaluation-only non-commercial use",
            "pretrained_dataset": str(mapping_config["pretrained_dataset"]),
            "semantic_class_count": len(id_to_label),
            "purpose": "unverified human-review pseudo-label draft only",
        },
    }


@torch.inference_mode()
def _infer_frame(
    image_rgb: np.ndarray,
    model_bundle: dict[str, object],
    mapping_config: dict[str, object],
    name_to_id: dict[str, int],
    traversability_colors: dict[int, tuple[int, int, int]],
    input_size: int,
    device: torch.device,
) -> dict[str, object]:
    end_to_end_started = time.perf_counter()
    padded, padding = letterbox_rgb(image_rgb, input_size)
    processor = model_bundle["processor"]
    model = model_bundle["model"]
    inputs = processor(images=padded, return_tensors="pt", do_resize=False)
    pixel_values = inputs["pixel_values"].to(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    logits = model(pixel_values=pixel_values).logits
    logits = functional.interpolate(logits, size=(input_size, input_size), mode="bilinear", align_corners=False)
    probabilities = logits.softmax(dim=1)
    confidence_tensor, semantic_tensor = probabilities.max(dim=1)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    model_latency_ms = (time.perf_counter() - started) * 1000.0

    semantic_padded = semantic_tensor[0].to(torch.uint8).cpu().numpy()
    confidence_padded = confidence_tensor[0].to(torch.float32).cpu().numpy()
    output_shape = image_rgb.shape[:2]
    semantic_mask = restore_from_letterbox(semantic_padded, padding, output_shape, cv2.INTER_NEAREST).astype(np.uint8)
    confidence = restore_from_letterbox(confidence_padded, padding, output_shape, cv2.INTER_LINEAR).astype(np.float32)
    id_to_label = model_bundle["id_to_label"]
    traversability_mask = semantic_to_traversability(
        semantic_mask,
        confidence,
        id_to_label,
        mapping_config,
        name_to_id,
    )
    semantic_color = colorize_mask(semantic_mask, model_bundle["semantic_colors"])
    traversability_color = colorize_mask(traversability_mask, traversability_colors)
    return {
        "semantic_mask": semantic_mask,
        "semantic_overlay": overlay_rgb(image_rgb, semantic_color),
        "traversability_mask": traversability_mask,
        "traversability_overlay": overlay_rgb(image_rgb, traversability_color),
        "confidence": confidence,
        "model_latency_ms": model_latency_ms,
        "end_to_end_latency_ms": (time.perf_counter() - end_to_end_started) * 1000.0,
        "padding": list(padding),
        "id_to_label": id_to_label,
        "traversability_id_to_name": {class_id: name for name, class_id in name_to_id.items()},
    }


def _write_sample(
    output_dir: Path,
    sample_id: str,
    selected_frame: SelectedFrame,
    image_rgb: np.ndarray,
    result: dict[str, object],
) -> dict[str, str]:
    sample = selected_frame.sample
    paths = {
        "image_path": f"images/{sample_id}.jpg",
        "semantic_mask_path": f"semantic_masks/{sample_id}.png",
        "semantic_overlay_path": f"semantic_overlays/{sample_id}.jpg",
        "traversability_mask_path": f"traversability_masks/{sample_id}.png",
        "traversability_overlay_path": f"traversability_overlays/{sample_id}.jpg",
        "confidence_path": f"confidence/{sample_id}.png",
    }
    _write_rgb(output_dir / paths["image_path"], image_rgb)
    _write_mask(output_dir / paths["semantic_mask_path"], result["semantic_mask"])
    _write_rgb(output_dir / paths["semantic_overlay_path"], result["semantic_overlay"])
    _write_mask(output_dir / paths["traversability_mask_path"], result["traversability_mask"])
    _write_rgb(output_dir / paths["traversability_overlay_path"], result["traversability_overlay"])
    confidence_uint8 = np.clip(result["confidence"] * 255.0, 0, 255).astype(np.uint8)
    confidence_color = cv2.applyColorMap(confidence_uint8, cv2.COLORMAP_VIRIDIS)
    if not cv2.imwrite(str(output_dir / paths["confidence_path"]), confidence_color):
        raise OSError(f"cannot write confidence image for {sample_id}")

    stats = class_area_statistics(result["traversability_mask"], result["traversability_id_to_name"])
    semantic_counts = Counter(int(value) for value in result["semantic_mask"].reshape(-1))
    top_semantic = [
        {
            "class_id": class_id,
            "class_name": result["id_to_label"][class_id],
            "pixels": pixels,
            "fraction": pixels / result["semantic_mask"].size,
        }
        for class_id, pixels in semantic_counts.most_common(10)
    ]
    metadata = {
        "sample_id": sample_id,
        "source": {
            "ride_id": sample.ride_id,
            "frame_id": sample.front_frame_id,
            "timestamp": sample.front_timestamp,
            "manifest_index": selected_frame.manifest_index,
            "playlist": sample.front_playlist_ref,
            "segment": sample.front_segment_ref,
            "action_label_reference_only": sample.action_class,
            "linear": sample.linear,
            "angular": sample.angular,
        },
        "selection_method": selected_frame.selection_method,
        "original_shape": list(image_rgb.shape),
        "letterbox_padding_top_bottom_left_right": result["padding"],
        "mean_semantic_confidence": float(np.mean(result["confidence"])),
        "traversability_class_area": stats,
        "top_semantic_classes": top_semantic,
        "review_required": True,
    }
    write_json(output_dir / "metadata" / f"{sample_id}.json", metadata)
    entry = {
        "sample_id": sample_id,
        **paths,
        "ride_id": sample.ride_id,
        "frame_id": str(sample.front_frame_id),
        "timestamp": f"{sample.front_timestamp:.6f}",
        "manifest_index": str(selected_frame.manifest_index),
        "playlist": sample.front_playlist_ref,
        "action_label": sample.action_class,
        "linear": f"{sample.linear:.6f}",
        "angular": f"{sample.angular:.6f}",
        "selection_method": selected_frame.selection_method,
        "split": "UNASSIGNED",
        "review_status": "UNREVIEWED",
        "reviewer_notes": "",
        "dominant_failure_type": "",
        "usable_for_training": "false",
        "corrected_mask_path": "",
        "mean_confidence": f"{metadata['mean_semantic_confidence']:.4f}",
        "class_distribution": json.dumps(stats, sort_keys=True),
    }
    return entry


def _create_bundle_directories(output_dir: Path) -> None:
    for name in (
        "images",
        "semantic_masks",
        "semantic_overlays",
        "traversability_masks",
        "traversability_overlays",
        "confidence",
        "metadata",
        "corrected_masks",
    ):
        (output_dir / name).mkdir(parents=True, exist_ok=True)


def _write_rgb(path: Path, image_rgb: np.ndarray) -> None:
    if not cv2.imwrite(str(path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)):
        raise OSError(f"cannot write image: {path}")


def _write_mask(path: Path, mask: np.ndarray) -> None:
    if mask.ndim != 2 or not cv2.imwrite(str(path), mask):
        raise OSError(f"cannot write mask: {path}")


def _write_sampling_metadata(entries: list[dict[str, str]], path: Path) -> None:
    fields = [
        "sample_id", "image_path", "ride_id", "frame_id", "timestamp", "manifest_index",
        "playlist", "action_label", "linear", "angular", "selection_method", "split", "review_status",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: entry[field] for field in fields} for entry in entries)


def _write_bundle_readme(output_dir: Path) -> None:
    text = """# Traversability Pilot Review Bundle

This directory is self-contained and can be inspected without Python or CUDA. Open `gallery.html`, then edit only the reviewer columns in `review.csv`.

All masks are unverified pseudo-label drafts. They are not ground truth and must not enter training until validation after human review.

Review statuses:

- `UNREVIEWED`: not inspected.
- `ACCEPT`: pseudo-label is usable; set `usable_for_training` to `true`.
- `REJECT`: unusable; keep `usable_for_training` false.
- `NEEDS_CORRECTION`: requires a class-ID PNG under `corrected_masks/`; record its relative path.
- `AMBIGUOUS`: cannot be resolved safely; exclude from research.training.

Corrected masks must match the source image dimensions. The validator accepts single-channel PNG values 0, 1, and 2, or RGB PNGs using the exact colors in `cvat_labelmap.txt`. CVAT's `Segmentation Mask` export is compatible with the included label map. Copy the resulting class masks into `corrected_masks/` and record each relative path in `review.csv`. Do not edit raw semantic masks.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def _latency_report(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def _counter_distribution(counts: Counter[str], limit: int | None = None) -> dict[str, dict[str, float | int]]:
    total = sum(counts.values())
    ordered = counts.most_common(limit)
    return {
        name: {"pixels": count, "fraction": count / total}
        for name, count in ordered
    }


def _package_versions() -> dict[str, str | None]:
    packages = ("torch", "torchvision", "transformers", "tokenizers", "safetensors", "huggingface-hub", "Pillow")
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_args(args: argparse.Namespace) -> None:
    if args.sample_count <= 0 or args.max_rides <= 0 or args.input_size <= 0:
        raise SystemExit("sample-count, max-rides, and input-size must be positive")
    if args.minimum_separation_seconds < 0:
        raise SystemExit("minimum-separation-seconds cannot be negative")


if __name__ == "__main__":
    raise SystemExit(main())
