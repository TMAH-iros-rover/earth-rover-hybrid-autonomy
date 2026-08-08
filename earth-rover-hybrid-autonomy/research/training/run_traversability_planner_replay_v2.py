#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
for import_root in (ROOT, ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from earth_rover.traversability_replay import (  # noqa: E402
    LogOnlyControlSink,
    OfflineTraversabilityPipeline,
    ReplayObservation,
)
from earth_rover.utils.config import load_config  # noqa: E402
from research.training.run_traversability_video_review_v2 import (  # noqa: E402
    ExistingHlsDecoder,
    SegFormerV2Predictor,
    discovery_skips,
    selected_dataset_indexes,
    sha256_file,
)
from research.training.traversability_planner_replay_v2 import (  # noqa: E402
    compose_planner_review_frame,
    delayed_replay_pairs,
)
from research.training.traversability_video_review_v2 import (  # noqa: E402
    H264VideoWriter,
    latency_summary,
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
        description="Run log-only SegFormer v2 goal-aware planner replay; no SDK commands are sent."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--training-config", required=True)
    parser.add_argument("--autonomy-config", default="research/configs/urban_replay_v2.yaml")
    parser.add_argument("--latency-profile", default="research/configs/urban_latency_2s.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--datasets", nargs="+", choices=("all", "0", "1", "2"), default=("0",))
    parser.add_argument("--dataset-root-0", default=str(DATASET_DEFAULTS["0"]))
    parser.add_argument("--dataset-root-1", default=str(DATASET_DEFAULTS["1"]))
    parser.add_argument("--dataset-root-2", default=str(DATASET_DEFAULTS["2"]))
    parser.add_argument("--ride-id", action="append", help="Restrict selection to one or more ride IDs.")
    parser.add_argument("--ride-count", type=int, default=1)
    parser.add_argument("--duration-seconds", type=float, default=30.0)
    parser.add_argument("--latency-sec", type=float, default=0.0)
    parser.add_argument("--output-fps", type=float, default=10.0)
    parser.add_argument("--edge-margin-seconds", type=float, default=10.0)
    parser.add_argument("--maximum-frame-gap-seconds", type=float, default=0.25)
    parser.add_argument("--panel-width", type=int, default=480)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument(
        "--goal-heading-error-deg",
        type=float,
        required=True,
        help="Explicit synthetic/fixed goal heading error; this is not recorded GPS.",
    )
    parser.add_argument(
        "--low-confidence-threshold",
        type=float,
        help="Visualization only: low-confidence mask pixels are shown as IGNORE.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    indexes = selected_dataset_indexes(args.datasets)
    if args.ride_count <= 0 or args.duration_seconds <= 0.0 or args.output_fps <= 0.0:
        raise SystemExit("ride-count, duration-seconds, and output-fps must be positive")
    if args.latency_sec < 0.0:
        raise SystemExit("latency-sec must be non-negative")
    if args.panel_width <= 0:
        raise SystemExit("panel-width must be positive")
    if args.low_confidence_threshold is not None and not (
        0.0 <= args.low_confidence_threshold <= 1.0
    ):
        raise SystemExit("low-confidence-threshold must be within [0, 1]")

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    training_config_path = Path(args.training_config).expanduser().resolve()
    autonomy_config_path = _rooted(args.autonomy_config)
    latency_profile_path = _rooted(args.latency_profile)
    output = Path(args.output_dir).expanduser().resolve()
    for required in (checkpoint, training_config_path, autonomy_config_path):
        if not required.is_file():
            raise SystemExit(f"required input does not exist: {required}")
    training_config = yaml.safe_load(training_config_path.read_text(encoding="utf-8"))
    if int(training_config.get("num_labels", -1)) != 3:
        raise SystemExit("training config must define the approved three-class model")
    if int(training_config.get("ignore_index", -1)) != 255:
        raise SystemExit("training config must define ignore_index=255")

    config_paths = [ROOT / "configs/default.yaml", autonomy_config_path]
    if args.latency_sec > 0.0:
        if not latency_profile_path.is_file():
            raise SystemExit(f"latency profile does not exist: {latency_profile_path}")
        config_paths.append(latency_profile_path)
    config = load_config(*config_paths)
    if args.latency_sec > float(config["safety"]["frame_timeout_sec"]):
        raise SystemExit("frame timeout is shorter than simulated latency; use a matching latency profile")
    if args.latency_sec > float(config["safety"]["data_timeout_sec"]):
        raise SystemExit("data timeout is shorter than simulated latency; use a matching latency profile")

    roots = {
        "0": Path(args.dataset_root_0).expanduser().resolve(),
        "1": Path(args.dataset_root_1).expanduser().resolve(),
        "2": Path(args.dataset_root_2).expanduser().resolve(),
    }
    for index in indexes:
        if not roots[index].is_dir():
            raise SystemExit(f"dataset root does not exist: {roots[index]}")
        if output == roots[index] or roots[index] in output.parents:
            raise SystemExit("output must remain outside raw dataset roots")
    if output.exists() and not args.overwrite:
        raise SystemExit(f"output already exists; pass --overwrite to replace it: {output}")
    temporary = output.parent / f".{output.name}.tmp"
    if temporary.exists():
        if not args.overwrite:
            raise SystemExit(f"temporary output already exists: {temporary}")
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)

    predictor = SegFormerV2Predictor(
        checkpoint,
        int(training_config["image_size"]),
        args.require_cuda,
    )
    reports: dict[str, dict[str, object]] = {}
    try:
        from research.training.manual_candidate_sampling import discover_front_rides

        for index in indexes:
            root = roots[index]
            rides, discovery = discover_front_rides([root])
            discovered_ride_ids = {ride.ride_id for ride in rides}
            if args.ride_id:
                requested = set(args.ride_id)
                rides = [ride for ride in rides if ride.ride_id in requested]
                missing = sorted(requested - {ride.ride_id for ride in rides})
                if missing:
                    raise ValueError(f"requested ride IDs were not found in {root.name}: {missing}")
            segments, selection_skips = select_review_segments(
                rides,
                args.ride_count,
                args.duration_seconds + args.latency_sec,
                args.output_fps,
                args.edge_margin_seconds,
                args.maximum_frame_gap_seconds,
                args.seed,
            )
            if not segments:
                raise ValueError(f"no replay segment could be selected from {root}")
            dataset_output = temporary / root.name
            dataset_output.mkdir()
            report = process_dataset(
                root,
                segments,
                [
                    *discovery_skips(root, discovered_ride_ids),
                    *selection_skips,
                ],
                ExistingHlsDecoder(root),
                predictor,
                config,
                dataset_output,
                output / root.name,
                checkpoint,
                training_config_path,
                args,
                discovery,
            )
            reports[index] = report
        root_report = {
            "success": all(report["success"] for report in reports.values()),
            "datasets": reports,
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": predictor.checkpoint_sha256,
            "checkpoint_version": predictor.checkpoint_version,
            "training_config_path": str(training_config_path),
            "training_config_sha256": sha256_file(training_config_path),
            "autonomy_config_path": str(autonomy_config_path),
            "latency_profile_path": str(latency_profile_path) if args.latency_sec > 0.0 else None,
            "goal_input_mode": "fixed_heading_error",
            "goal_heading_error_deg": args.goal_heading_error_deg,
            "recorded_gps_used": False,
            "recorded_mission_waypoint_used": False,
            "command_sink": "LogOnlyControlSink",
            "command_transmitted": False,
            "recovery_behavior_evaluated": False,
            "sdk_or_live_rover_used": False,
            "runtime": predictor.runtime_report(),
        }
        write_json(temporary / "review_manifest.json", root_report)
        if output.exists():
            shutil.rmtree(output)
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(f"Planner replay output: {output}")
    return 0 if all(report["success"] for report in reports.values()) else 2


def process_dataset(
    dataset_root: Path,
    segments: list,
    skipped_rides: list[dict[str, object]],
    decoder,
    predictor,
    config: dict,
    output_dir: Path,
    reported_output_dir: Path,
    checkpoint: Path,
    training_config_path: Path,
    args: argparse.Namespace,
    discovery: dict[str, int],
) -> dict[str, object]:
    sink = LogOnlyControlSink(output_dir / "logs")
    writer = None
    writer_info: dict[str, object] = {}
    video_path = output_dir / "planner_replay.mp4"
    processed = 0
    failures: list[dict[str, object]] = []
    latencies: list[float] = []
    started = time.monotonic()
    selected_report = []
    try:
        for segment in segments:
            pairs = delayed_replay_pairs(segment, args.latency_sec, args.duration_seconds)
            selected_report.append(
                {
                    "dataset": segment.dataset,
                    "ride_id": segment.ride_id,
                    "source_start_timestamp": pairs[0].source_timestamp if pairs else None,
                    "source_end_timestamp": pairs[-1].source_timestamp if pairs else None,
                    "observed_start_timestamp": pairs[0].observed_frame.timestamp if pairs else None,
                    "observed_end_timestamp": pairs[-1].observed_frame.timestamp if pairs else None,
                    "simulated_latency_sec": args.latency_sec,
                    "requested_duration_seconds": args.duration_seconds,
                    "control_step_count": len(pairs),
                }
            )
            pipeline = OfflineTraversabilityPipeline(config)
            for pair in pairs:
                frame = pair.observed_frame
                try:
                    frame_rgb = decoder.decode(frame)
                    prediction, confidence, inference_ms = predictor.predict(frame_rgb)
                    observation = ReplayObservation(
                        source_timestamp=pair.source_timestamp,
                        frame_timestamp=frame.timestamp,
                        sensor_timestamp=frame.timestamp,
                        source_mask=prediction,
                        confidence=confidence,
                        inference_latency_ms=inference_ms,
                        heading_error_deg=args.goal_heading_error_deg,
                        goal_input_mode="fixed_heading_error",
                        goal_input_valid=True,
                        gps_valid=False,
                        current_waypoint=None,
                        distance_to_waypoint_m=None,
                        goal_bearing_deg=None,
                        ride_id=frame.ride_id,
                        frame_id=frame.frame_id,
                    )
                    result = pipeline.process(observation)
                    sink.write(result)
                    composed = compose_planner_review_frame(
                        frame_rgb,
                        prediction,
                        confidence,
                        result,
                        dataset_root.name,
                        frame.ride_id,
                        frame.frame_id,
                        predictor.checkpoint_version,
                        args.panel_width,
                        args.low_confidence_threshold,
                    )
                    if writer is None:
                        height, width = composed.shape[:2]
                        writer = H264VideoWriter(video_path, args.output_fps, (width, height))
                    writer.write(composed)
                    processed += 1
                    latencies.append(float(inference_ms))
                except Exception as exc:
                    failures.append(
                        {
                            "ride_id": frame.ride_id,
                            "frame_id": frame.frame_id,
                            "timestamp": frame.timestamp,
                            "reason": type(exc).__name__,
                            "detail": str(exc),
                        }
                    )
    finally:
        sink.close()
        if writer is not None:
            writer_info = writer.close()
    elapsed = time.monotonic() - started
    write_json(output_dir / "selected_segments.json", selected_report)
    report = {
        "success": processed > 0 and bool(writer_info),
        "dataset_name": dataset_root.name,
        "dataset_path": str(dataset_root),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": predictor.checkpoint_sha256,
        "checkpoint_version": predictor.checkpoint_version,
        "training_config_path": str(training_config_path),
        "selected_segments": selected_report,
        "skipped_rides": skipped_rides,
        "discovery": discovery,
        "processed_frame_count": processed,
        "failed_frame_count": len(failures),
        "failed_frames": failures,
        "model_inference_latency_ms": latency_summary(latencies),
        "effective_fps": processed / elapsed if elapsed else 0.0,
        "output_fps": args.output_fps,
        "video_path": str(reported_output_dir / video_path.name),
        "video": writer_info,
        "csv_log_path": str(reported_output_dir / "logs/replay_steps.csv"),
        "jsonl_log_path": str(reported_output_dir / "logs/replay_steps.jsonl"),
        "goal_input_mode": "fixed_heading_error",
        "goal_heading_error_deg": args.goal_heading_error_deg,
        "gps_valid": False,
        "command_transmitted": False,
        "temporal_smoothing_applied": False,
        "recovery_behavior_evaluated": False,
    }
    write_json(output_dir / "review_manifest.json", report)
    return report


def _rooted(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


if __name__ == "__main__":
    raise SystemExit(main())
