#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
for import_root in (ROOT, ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from research.training.run_traversability_video_review_v2 import (
    ExistingHlsDecoder,
    discovery_skips,
    selected_dataset_indexes,
)
from training.sam_tp_reproduction import (
    SamTpPredictor,
    git_provenance,
    latency_summary,
    load_reproduction_config,
    resolve_upstream_paths,
    score_to_heatmap,
    sha256_file,
    write_json,
)
from training.sam_tp_phase1_review import (
    SamTpPhase1FrameProcessor,
    draw_image_path_rgb,
    render_trajectory_geometry_rgb,
)
from research.training.traversability_video_review_v2 import (
    H264VideoWriter,
    ReviewFrame,
    ReviewSegment,
    select_review_segments,
)
from earth_rover.planning.trajectory_sampler import (
    DEFAULT_CURVATURES,
    ConstantCurvatureTrajectorySampler,
)


DATASET_DEFAULTS = {
    "0": Path("~/datasets/output_rides_0"),
    "1": Path("~/datasets/output_rides_1"),
    "2": Path("~/datasets/output_rides_2"),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create deterministic official SAM-TP FrodoBots review videos."
    )
    parser.add_argument("--reproduction-config", required=True)
    parser.add_argument("--upstream-root", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--compatibility-report", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=("0",),
        help="Numeric output_rides indexes to process, or all for 0, 1, and 2.",
    )
    parser.add_argument("--dataset-root-0", default=str(DATASET_DEFAULTS["0"]))
    parser.add_argument("--dataset-root-1", default=str(DATASET_DEFAULTS["1"]))
    parser.add_argument("--dataset-root-2", default=str(DATASET_DEFAULTS["2"]))
    parser.add_argument(
        "--dataset-root",
        action="append",
        default=[],
        metavar="INDEX=PATH",
        help="Override a dataset root by numeric index; may be repeated.",
    )
    parser.add_argument("--rides-per-dataset", type=int)
    parser.add_argument("--seconds-per-ride", type=float)
    parser.add_argument("--output-fps", type=float)
    parser.add_argument("--edge-margin-seconds", type=float)
    parser.add_argument("--maximum-frame-gap-seconds", type=float)
    parser.add_argument("--panel-width", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--phase1-trajectories", action="store_true")
    parser.add_argument("--trajectory-horizon-m", type=float, default=2.0)
    parser.add_argument("--trajectory-sample-interval-m", type=float, default=0.1)
    parser.add_argument("--provisional-rover-width-m", type=float, default=0.4)
    parser.add_argument("--provisional-safety-margin-m", type=float, default=0.1)
    parser.add_argument("--minimum-path-score", type=float, default=0.55)
    parser.add_argument(
        "--path-corridor-half-width-ratio",
        type=float,
        default=0.018,
    )
    return parser.parse_args(argv)


def resolve_dataset_roots(
    indexes: tuple[str, ...],
    legacy_roots: dict[str, str | Path],
    overrides: list[str],
) -> dict[str, Path]:
    roots = {
        index: Path(path).expanduser().resolve()
        for index, path in legacy_roots.items()
    }
    seen: set[str] = set()
    for value in overrides:
        index, separator, path = value.partition("=")
        if not separator or not index.isdigit() or not path:
            raise ValueError("dataset-root must use numeric INDEX=PATH syntax")
        if index in seen:
            raise ValueError(f"duplicate dataset-root override: {index}")
        seen.add(index)
        roots[index] = Path(path).expanduser().resolve()
    for index in indexes:
        if not index.isdigit():
            raise ValueError(f"dataset index must be numeric: {index}")
        roots.setdefault(
            index,
            Path(f"~/datasets/output_rides_{index}").expanduser().resolve(),
        )
    return roots


def fit_rgb(image: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = min(width / image.shape[1], height / image.shape[0])
    size = (
        max(1, round(image.shape[1] * scale)),
        max(1, round(image.shape[0] * scale)),
    )
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(image, size, interpolation=interpolation)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    x = (width - size[0]) // 2
    y = (height - size[1]) // 2
    canvas[y : y + size[1], x : x + size[0]] = resized
    return canvas


def compose_sam_tp_panels(
    image_rgb: np.ndarray,
    score: np.ndarray,
    metadata: dict[str, object],
    panel_width: int,
    trajectories: tuple | None = None,
    image_path=None,
    corridor_half_width_ratio: float = 0.018,
) -> np.ndarray:
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3 or image_rgb.dtype != np.uint8:
        raise ValueError("image must be uint8 HxWx3 RGB")
    if score.shape != image_rgb.shape[:2] or not np.isfinite(score).all():
        raise ValueError("score must be finite and match the image")
    if float(score.min()) < 0.0 or float(score.max()) > 1.0:
        raise ValueError("score must remain in [0, 1]")
    heatmap = score_to_heatmap(score)
    overlay = cv2.addWeighted(image_rgb, 0.55, heatmap, 0.45, 0.0)
    original_panel = image_rgb
    if image_path is not None:
        original_panel = draw_image_path_rgb(
            image_rgb,
            image_path,
            corridor_half_width_ratio,
        )
        overlay = draw_image_path_rgb(
            overlay,
            image_path,
            corridor_half_width_ratio,
        )
    panel_height = max(2, round(panel_width * image_rgb.shape[0] / image_rgb.shape[1]))
    panel_height += panel_height % 2
    panel_images = [
        fit_rgb(original_panel, panel_width, panel_height),
        fit_rgb(overlay, panel_width, panel_height),
        fit_rgb(heatmap, panel_width, panel_height),
    ]
    titles = ["IMAGE-SPACE PATH", "SAM-TP OVERLAY + PATH", "TRAVERSABILITY SCORE"]
    if trajectories is not None:
        panel_images.append(
            render_trajectory_geometry_rgb(trajectories, panel_width, panel_height)
        )
        titles.append("CANDIDATE TRAJECTORIES (ROVER FRAME)")
    panels = np.concatenate(panel_images, axis=1)
    body = cv2.cvtColor(panels, cv2.COLOR_RGB2BGR)
    header = 82
    canvas = np.zeros(
        (header + panel_height, panel_width * len(panel_images), 3),
        dtype=np.uint8,
    )
    canvas[header:] = body
    line1 = (
        f"{metadata['dataset']} ride={metadata['ride_id']} frame={metadata['frame_id']} "
        f"timestamp={float(metadata['timestamp']):.3f}"
    )
    line2 = (
        f"SAM-TP {metadata['checkpoint_version']} inference="
        f"{float(metadata['latency_ms']):.1f}ms measured_fps="
        f"{float(metadata['measured_fps']):.2f} red=high blue=low"
    )
    if image_path is not None:
        line2 += (
            f" path={'VALID' if image_path.valid else 'NONE'} "
            f"reason={image_path.reason}"
        )
    cv2.putText(canvas, line1, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.53, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.putText(canvas, line2, (10, 49), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 210, 210), 1, cv2.LINE_AA)
    for index, title in enumerate(titles):
        cv2.putText(
            canvas,
            title,
            (index * panel_width + 10, header - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
    return canvas


def process_dataset(
    dataset_root: Path,
    segments: list[ReviewSegment],
    skipped_rides: list[dict[str, object]],
    decoder: ExistingHlsDecoder,
    predictor: SamTpPredictor,
    output_dir: Path,
    output_fps: float,
    panel_width: int,
    checkpoint_sha256: str,
    phase1_processor: SamTpPhase1FrameProcessor | None = None,
    writer_factory=H264VideoWriter,
    reported_output_dir: Path | None = None,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=False)
    video_path = output_dir / "sam_tp_review.mp4"
    statistics_path = output_dir / "frame_statistics.jsonl"
    writer = None
    processed = 0
    failures: list[dict[str, object]] = []
    latencies: list[float] = []
    requested = sum(len(segment.frames) for segment in segments)
    print(
        f"dataset={dataset_root.name} requested_frames={requested} "
        f"segments={len(segments)}",
        flush=True,
    )
    wall_started = time.monotonic()
    with statistics_path.open("w", encoding="utf-8") as statistics:
        try:
            for segment in segments:
                for frame in segment.frames:
                    try:
                        image = decoder.decode(frame)
                        phase1 = (
                            phase1_processor.process(image, frame.timestamp)
                            if phase1_processor is not None
                            else None
                        )
                        prediction = (
                            phase1.prediction
                            if phase1 is not None
                            else predictor.predict(image)
                        )
                    except Exception as exc:
                        failures.append(
                            {
                                "ride_id": frame.ride_id,
                                "frame_id": frame.frame_id,
                                "timestamp": frame.timestamp,
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                            }
                        )
                        attempted = processed + len(failures)
                        if attempted % 25 == 0 or attempted == requested:
                            print(
                                f"dataset={dataset_root.name} "
                                f"progress={attempted}/{requested} "
                                f"processed={processed} failed={len(failures)} "
                                f"ride={frame.ride_id} frame={frame.frame_id}",
                                flush=True,
                            )
                        continue
                    elapsed = time.monotonic() - wall_started
                    measured_fps = (processed + 1) / elapsed if elapsed else 0.0
                    rendered = compose_sam_tp_panels(
                        image,
                        prediction.traversability_score,
                        {
                            "dataset": frame.dataset,
                            "ride_id": frame.ride_id,
                            "frame_id": frame.frame_id,
                            "timestamp": frame.timestamp,
                            "checkpoint_version": checkpoint_sha256[:12],
                            "latency_ms": prediction.inference_time_ms,
                            "measured_fps": measured_fps,
                        },
                        panel_width,
                        phase1.trajectories if phase1 is not None else None,
                        phase1.image_path if phase1 is not None else None,
                        (
                            phase1_processor.corridor_half_width_ratio
                            if phase1_processor is not None
                            else 0.018
                        ),
                    )
                    if writer is None:
                        writer = writer_factory(
                            video_path,
                            output_fps,
                            (rendered.shape[1], rendered.shape[0]),
                        )
                    writer.write(rendered)
                    statistics.write(
                        json.dumps(
                            {
                                "ride_id": frame.ride_id,
                                "frame_id": frame.frame_id,
                                "timestamp": frame.timestamp,
                                "model_only_latency_ms": prediction.inference_time_ms,
                                "score_min": float(prediction.traversability_score.min()),
                                "score_max": float(prediction.traversability_score.max()),
                                "score_mean": float(prediction.traversability_score.mean()),
                                "image_path_valid": (
                                    phase1.image_path.valid
                                    if phase1 is not None
                                    else None
                                ),
                                "image_path_reason": (
                                    phase1.image_path.reason
                                    if phase1 is not None
                                    else None
                                ),
                                "image_path_mean_score": (
                                    phase1.image_path.mean_score
                                    if phase1 is not None
                                    else None
                                ),
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    processed += 1
                    latencies.append(prediction.inference_time_ms)
                    attempted = processed + len(failures)
                    if attempted % 25 == 0 or attempted == requested:
                        print(
                            f"dataset={dataset_root.name} progress={attempted}/{requested} "
                            f"processed={processed} failed={len(failures)} "
                            f"ride={frame.ride_id} frame={frame.frame_id}",
                            flush=True,
                        )
        finally:
            video_info = writer.close() if writer is not None else {}
    wall_seconds = time.monotonic() - wall_started
    reported_root = reported_output_dir or output_dir
    return {
        "success": processed > 0 and bool(video_info),
        "dataset_path": str(dataset_root),
        "selected_segments": [
            {
                **{key: value for key, value in asdict(segment).items() if key != "frames"},
                "sampled_frame_count": len(segment.frames),
                "first_frame_id": segment.frames[0].frame_id if segment.frames else None,
                "last_frame_id": segment.frames[-1].frame_id if segment.frames else None,
            }
            for segment in segments
        ],
        "skipped_rides": skipped_rides,
        "requested_frame_count": requested,
        "processed_frame_count": processed,
        "skipped_frame_count": len(failures),
        "skipped_frames": failures,
        "sampling_fps": output_fps,
        "wall_clock_seconds": wall_seconds,
        "effective_pipeline_fps": processed / wall_seconds if wall_seconds else 0.0,
        "model_only_latency_ms": latency_summary(latencies),
        "video_encoding_excluded_from_model_latency": True,
        "output_file_path": str(reported_root / video_path.name),
        "output_video": video_info,
        "frame_statistics_path": str(reported_root / statistics_path.name),
        "score_semantics": "red=high traversability, blue=low traversability",
        "temporal_smoothing_applied": False,
        "sdk_or_live_rover_commands_sent": False,
        "phase1_trajectory_geometry": phase1_processor is not None,
        "camera_projection_applied": False,
        "image_path_visualization_only": phase1_processor is not None,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    compatibility_path = Path(args.compatibility_report).expanduser().resolve()
    compatibility = json.loads(compatibility_path.read_text(encoding="utf-8"))
    if compatibility.get("success") is not True:
        raise SystemExit("strict compatibility report is not successful")
    config = load_reproduction_config(args.reproduction_config)
    paths = resolve_upstream_paths(args.upstream_root, config, args.checkpoint)
    checkpoint_sha256 = sha256_file(paths["checkpoint"])
    if compatibility.get("checkpoint", {}).get("sha256") != checkpoint_sha256:
        raise SystemExit("checkpoint differs from the strict compatibility report")
    settings = config["video_review"]
    indexes = selected_dataset_indexes(args.datasets)
    try:
        roots = resolve_dataset_roots(
            indexes,
            {
                "0": args.dataset_root_0,
                "1": args.dataset_root_1,
                "2": args.dataset_root_2,
            },
            args.dataset_root,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    values = {
        "rides_per_dataset": args.rides_per_dataset or int(settings["rides_per_dataset"]),
        "seconds_per_ride": args.seconds_per_ride or float(settings["seconds_per_ride"]),
        "output_fps": args.output_fps or float(settings["output_fps"]),
        "edge_margin_seconds": (
            args.edge_margin_seconds
            if args.edge_margin_seconds is not None
            else float(settings["edge_margin_seconds"])
        ),
        "maximum_frame_gap_seconds": (
            args.maximum_frame_gap_seconds
            if args.maximum_frame_gap_seconds is not None
            else float(settings["maximum_frame_gap_seconds"])
        ),
        "panel_width": args.panel_width or int(settings["panel_width"]),
        "seed": args.seed or int(config["runtime"]["seed"]),
    }
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    for index in indexes:
        if not roots[index].is_dir():
            raise SystemExit(f"dataset root does not exist: {roots[index]}")
        if output == roots[index] or roots[index] in output.parents:
            raise SystemExit("output must remain outside raw dataset roots")

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("official SAM-TP FrodoBots reproduction requires CUDA")
    torch.cuda.reset_peak_memory_stats()
    predictor = SamTpPredictor(
        paths["root"],
        paths["model_config"],
        paths["checkpoint"],
        synchronize=torch.cuda.synchronize,
    )
    phase1_processor = None
    if args.phase1_trajectories:
        sampler = ConstantCurvatureTrajectorySampler(
            DEFAULT_CURVATURES,
            args.trajectory_horizon_m,
            args.trajectory_sample_interval_m,
            args.provisional_rover_width_m,
            args.provisional_safety_margin_m,
        )
        phase1_processor = SamTpPhase1FrameProcessor(
            predictor,
            sampler.sample(),
            checkpoint_sha256[:12],
            minimum_path_score=args.minimum_path_score,
            corridor_half_width_ratio=args.path_corridor_half_width_ratio,
        )
    temporary = output.parent / f".{output.name}.tmp-{os.getpid()}"
    temporary.mkdir(parents=True)
    reports: dict[str, dict[str, object]] = {}
    try:
        from research.training.manual_candidate_sampling import discover_front_rides

        for index in indexes:
            root = roots[index]
            rides, discovery = discover_front_rides([root])
            segments, selection_skips = select_review_segments(
                rides,
                values["rides_per_dataset"],
                values["seconds_per_ride"],
                values["output_fps"],
                values["edge_margin_seconds"],
                values["maximum_frame_gap_seconds"],
                values["seed"],
            )
            skipped = [
                *discovery_skips(root, {ride.ride_id for ride in rides}),
                *selection_skips,
            ]
            report = process_dataset(
                root,
                segments,
                skipped,
                ExistingHlsDecoder(root),
                predictor,
                temporary / root.name,
                values["output_fps"],
                values["panel_width"],
                checkpoint_sha256,
                phase1_processor=phase1_processor,
                reported_output_dir=output / root.name,
            )
            report["discovery"] = discovery
            reports[index] = report
        root_report = {
            "success": all(report["success"] for report in reports.values()),
            "datasets": reports,
            "settings": values,
            "upstream": git_provenance(paths["root"]),
            "checkpoint": compatibility["checkpoint"],
            "compatibility_report": str(compatibility_path),
            "peak_vram_bytes": {
                "allocated": torch.cuda.max_memory_allocated(),
                "reserved": torch.cuda.max_memory_reserved(),
            },
            "score_conversion": "sigmoid(raw_logits)",
            "additional_training_performed": False,
            "sdk_or_live_rover_commands_sent": False,
        }
        write_json(temporary / "review_manifest.json", root_report)
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(f"SAM-TP review output: {output}")
    return 0 if root_report["success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
