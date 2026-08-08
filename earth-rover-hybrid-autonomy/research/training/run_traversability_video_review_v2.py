#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.training.datasets.frodobots_2k_manifest import RIDE_PATTERN
from research.training.models.traversability_segformer import (
    ID2LABEL,
    build_traversability_segformer,
    validate_three_class_checkpoint,
)
from research.training.traversability_video_review_v2 import (
    H264VideoWriter,
    ReviewFrame,
    process_dataset_review,
    select_review_segments,
    write_json,
)


DATASET_DEFAULTS = {
    "0": Path("~/datasets/output_rides_0"),
    "1": Path("~/datasets/output_rides_1"),
    "2": Path("~/datasets/output_rides_2"),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run offline SegFormer-B0 v2 inference and create QuickTime review videos."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("all", "0", "1", "2"),
        default=("all",),
        help="Dataset indexes to process; use all for output_rides_0, 1, and 2.",
    )
    parser.add_argument("--dataset-root-0", default=str(DATASET_DEFAULTS["0"]))
    parser.add_argument("--dataset-root-1", default=str(DATASET_DEFAULTS["1"]))
    parser.add_argument("--dataset-root-2", default=str(DATASET_DEFAULTS["2"]))
    parser.add_argument("--rides-per-dataset", type=int, default=5)
    parser.add_argument("--seconds-per-ride", type=float, default=60.0)
    parser.add_argument("--output-fps", type=float, default=10.0)
    parser.add_argument("--edge-margin-seconds", type=float, default=10.0)
    parser.add_argument("--maximum-frame-gap-seconds", type=float, default=0.25)
    parser.add_argument("--panel-width", type=int, default=480)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument(
        "--low-confidence-threshold",
        type=float,
        help="Visualization only: render pixels below this softmax confidence as black.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args(argv)


def selected_dataset_indexes(values: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    if "all" in values:
        if len(values) != 1:
            raise ValueError("'all' cannot be combined with individual dataset indexes")
        return ("0", "1", "2")
    return tuple(dict.fromkeys(values))


class SegFormerV2Predictor:
    def __init__(
        self,
        checkpoint_path: Path,
        image_size: int,
        require_cuda: bool,
    ) -> None:
        import torch
        import torch.nn.functional as functional

        from research.training.datasets.traversability_dataset_v1 import (
            image_rgb_to_tensor,
            letterbox_image,
            restore_letterbox,
            training_prediction_to_source,
        )

        if require_cuda and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required but unavailable")
        self.torch = torch
        self.functional = functional
        self.image_rgb_to_tensor = image_rgb_to_tensor
        self.letterbox_image = letterbox_image
        self.restore_letterbox = restore_letterbox
        self.training_prediction_to_source = training_prediction_to_source
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        validate_three_class_checkpoint(checkpoint)
        self.model = build_traversability_segformer(False, checkpoint["model_config"]).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.image_size = image_size
        digest = sha256_file(checkpoint_path)
        self.checkpoint_version = f"v2:{digest[:12]}:epoch-{checkpoint['epoch']}"
        self.checkpoint_sha256 = digest
        self.checkpoint_epoch = int(checkpoint["epoch"])
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def predict(self, frame_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        torch = self.torch
        padded, _ = self.letterbox_image(frame_rgb, self.image_size)
        tensor = self.image_rgb_to_tensor(padded).unsqueeze(0).to(self.device)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = torch.cuda.Event(enable_timing=True) if self.device.type == "cuda" else None
        ended = torch.cuda.Event(enable_timing=True) if self.device.type == "cuda" else None
        if started is not None:
            started.record()
        else:
            import time

            cpu_started = time.perf_counter()
        with torch.inference_mode():
            logits = self.model(tensor).logits
            logits = self.functional.interpolate(
                logits,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
            confidence_tensor, prediction_tensor = logits.softmax(dim=1).max(dim=1)
        if ended is not None:
            ended.record()
            torch.cuda.synchronize(self.device)
            latency_ms = float(started.elapsed_time(ended))
        else:
            import time

            latency_ms = (time.perf_counter() - cpu_started) * 1000.0
        prediction = prediction_tensor[0].cpu().numpy().astype(np.uint8)
        confidence = confidence_tensor[0].cpu().numpy().astype(np.float32)
        prediction = self.restore_letterbox(prediction, frame_rgb.shape[:2], cv2.INTER_NEAREST)
        confidence = self.restore_letterbox(confidence, frame_rgb.shape[:2], cv2.INTER_LINEAR)
        return self.training_prediction_to_source(prediction), confidence, latency_ms

    def runtime_report(self) -> dict[str, object]:
        torch = self.torch
        return {
            "device": str(self.device),
            "cuda_device_name": (
                torch.cuda.get_device_name(0) if self.device.type == "cuda" else None
            ),
            "peak_vram_bytes": {
                "allocated": (
                    torch.cuda.max_memory_allocated(self.device)
                    if self.device.type == "cuda"
                    else 0
                ),
                "reserved": (
                    torch.cuda.max_memory_reserved(self.device)
                    if self.device.type == "cuda"
                    else 0
                ),
            },
            "checkpoint_epoch": self.checkpoint_epoch,
        }


class ExistingHlsDecoder:
    def __init__(self, dataset_root: Path) -> None:
        from research.training.datasets.frodobots_2k_dataset import HlsFrameDecoder

        self.decoder = HlsFrameDecoder(dataset_root)

    def decode(self, frame: ReviewFrame) -> np.ndarray:
        from research.training.datasets.frodobots_2k_dataset import ManifestSample

        return self.decoder.decode(
            ManifestSample(
                ride_id=frame.ride_id,
                front_playlist_ref=frame.playlist_reference,
                front_segment_ref=frame.segment_reference,
                front_frame_id=frame.frame_id,
                front_timestamp=frame.timestamp,
                matched_control_timestamp=frame.timestamp,
                control_delta_ms=0.0,
                linear=0.0,
                angular=0.0,
                action_class="STOP",
                timeline_section_id=frame.timeline_section_id,
            )
        )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    indexes = selected_dataset_indexes(args.datasets)
    if args.rides_per_dataset <= 0:
        raise SystemExit("rides-per-dataset must be positive")
    if args.seconds_per_ride <= 0 or args.output_fps <= 0:
        raise SystemExit("seconds-per-ride and output-fps must be positive")
    if args.panel_width <= 0:
        raise SystemExit("panel-width must be positive")
    if args.low_confidence_threshold is not None and not (
        0.0 <= args.low_confidence_threshold <= 1.0
    ):
        raise SystemExit("low-confidence-threshold must be between 0 and 1")

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    if not checkpoint.is_file() or not config_path.is_file():
        raise SystemExit("checkpoint or config does not exist")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if int(config.get("num_labels", -1)) != 3 or int(config.get("ignore_index", -1)) != 255:
        raise SystemExit("training config must use num_labels=3 and ignore_index=255")
    image_size = int(config["image_size"])
    roots = {
        "0": Path(args.dataset_root_0).expanduser().resolve(),
        "1": Path(args.dataset_root_1).expanduser().resolve(),
        "2": Path(args.dataset_root_2).expanduser().resolve(),
    }
    for index in indexes:
        if not roots[index].is_dir():
            raise SystemExit(f"dataset root does not exist: {roots[index]}")
        if output == roots[index] or roots[index] in output.parents:
            raise SystemExit("output directory must remain outside raw dataset roots")
    if output.exists() and not args.overwrite:
        raise SystemExit(f"output path already exists; pass --overwrite to replace it: {output}")
    temporary = output.parent / f".{output.name}.tmp"
    if temporary.exists():
        if not args.overwrite:
            raise SystemExit(f"temporary output exists: {temporary}")
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)

    predictor = SegFormerV2Predictor(checkpoint, image_size, args.require_cuda)
    reports: dict[str, dict[str, object]] = {}
    try:
        from research.training.manual_candidate_sampling import discover_front_rides

        for index in indexes:
            dataset_root = roots[index]
            rides, discovery = discover_front_rides([dataset_root])
            segments, selection_skips = select_review_segments(
                rides,
                args.rides_per_dataset,
                args.seconds_per_ride,
                args.output_fps,
                args.edge_margin_seconds,
                args.maximum_frame_gap_seconds,
                args.seed,
            )
            skipped = [
                *discovery_skips(dataset_root, {ride.ride_id for ride in rides}),
                *selection_skips,
            ]
            dataset_output = temporary / dataset_root.name
            report = process_dataset_review(
                dataset_root=dataset_root,
                segments=segments,
                skipped_rides=skipped,
                decoder=ExistingHlsDecoder(dataset_root),
                predictor=predictor,
                output_dir=dataset_output,
                output_fps=args.output_fps,
                panel_width=args.panel_width,
                checkpoint_path=checkpoint,
                checkpoint_sha256=predictor.checkpoint_sha256,
                low_confidence_threshold=args.low_confidence_threshold,
                writer_factory=H264VideoWriter,
                discovery=discovery,
                model_metadata={
                    "image_size": image_size,
                    "training_config_path": str(config_path),
                    "training_config_sha256": sha256_file(config_path),
                    "input_geometry": "aspect-ratio-preserving letterbox",
                    "normalization_mean": [0.485, 0.456, 0.406],
                    "normalization_std": [0.229, 0.224, 0.225],
                    "training_class_mapping": {
                        str(key): value for key, value in ID2LABEL.items()
                    },
                    "training_ignore_index": 255,
                },
                reported_output_dir=output / dataset_root.name,
            )
            reports[index] = report
        root_report = {
            "success": all(report["success"] for report in reports.values()),
            "datasets": reports,
            "selected_dataset_indexes": list(indexes),
            "seed": args.seed,
            "rides_per_dataset": args.rides_per_dataset,
            "seconds_per_ride": args.seconds_per_ride,
            "output_fps": args.output_fps,
            "image_size": image_size,
            "preprocessing": "training-identical aspect-ratio-preserving letterbox and ImageNet normalization",
            "model_class_mapping": {str(key): value for key, value in ID2LABEL.items()},
            "display_source_class_ids": {
                "0": "IGNORE_BLACK",
                "1": "ON_ROAD_GREEN",
                "2": "OFF_ROAD_YELLOW",
                "3": "OBSTACLE_RED",
            },
            "runtime": predictor.runtime_report(),
            "training_or_fine_tuning_performed": False,
            "sdk_planner_controller_or_live_rover_used": False,
        }
        write_json(temporary / "review_manifest.json", root_report)
        if output.exists():
            shutil.rmtree(output)
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(f"Review output: {output}")
    return 0 if all(report["success"] for report in reports.values()) else 2


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def discovery_skips(
    dataset_root: Path,
    discovered_ride_ids: set[str],
) -> list[dict[str, object]]:
    skipped: list[dict[str, object]] = []
    for ride_dir in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
        match = RIDE_PATTERN.match(ride_dir.name)
        if match is None or match.group(1) in discovered_ride_ids:
            continue
        skipped.append(
            {
                "dataset": dataset_root.name,
                "ride_id": match.group(1),
                "reason": "front_ride_discovery_failed",
                "detail": "missing, malformed, damaged, duplicate, or incomplete front timestamp/HLS input",
            }
        )
    return skipped


if __name__ == "__main__":
    raise SystemExit(main())
