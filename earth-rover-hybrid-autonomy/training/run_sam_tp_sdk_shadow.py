#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for import_root in (ROOT, ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from earth_rover.sdk_client import EarthRoverSDKClient  # noqa: E402
from earth_rover.navigation.checkpoint_route import CheckpointRoutePlanner  # noqa: E402
from earth_rover.navigation.localization import GpsHeadingEkf  # noqa: E402
from earth_rover.planning.trajectory_sampler import (  # noqa: E402
    DEFAULT_CURVATURES,
    ConstantCurvatureTrajectorySampler,
)
from earth_rover.utils.config import load_config  # noqa: E402
from earth_rover.planning.motion_primitive_planner import (  # noqa: E402
    MotionPrimitivePlanner,
)
from earth_rover.perception.camera_calibration import (  # noqa: E402
    CalibrationError,
    load_calibration,
)
from training.sam_tp_reproduction import (  # noqa: E402
    OFFICIAL_COMMIT,
    git_provenance,
    sha256_file,
)
from training.sam_tp_checkpoint_format import CheckpointFormat  # noqa: E402
from training.sam_tp_hf_backend import (  # noqa: E402
    HF_SAM2_MODEL_CONFIG_ID,
    build_sam_tp_predictor,
)
from training.sam_tp_sdk_shadow import (  # noqa: E402
    run_shadow_step,
    write_shadow_summary,
)
from training.sam_tp_dashboard_bridge import SamTpDashboardServer  # noqa: E402
from training.sam_tp_event_recorder import (  # noqa: E402
    EventCaptureConfig,
    SamTpEventRecorder,
)
from training.sam_tp_phase1_review import SamTpPhase1FrameProcessor  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run read-only SDK front-camera SAM-TP shadow inference. "
            "No mission or control endpoint is called."
        )
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--mission-config",
        help=(
            "optional profile merged on top of --config (e.g. "
            "configs/mission1_live.yaml), matching run_mission1_autonomy.py's "
            "--config/--mission-config merge; required to apply live planner "
            "overrides such as planner.geometry_mode and camera_calibration.path"
        ),
    )
    parser.add_argument("--upstream-root", required=True)
    parser.add_argument("--model-config")
    parser.add_argument("--checkpoint")
    parser.add_argument("--expected-checkpoint-sha256")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-fps", type=float, default=4.0)
    parser.add_argument("--telemetry-hz", type=float, default=2.0)
    parser.add_argument("--maximum-frame-age-sec", type=float, default=1.0)
    parser.add_argument("--maximum-telemetry-age-sec", type=float, default=1.0)
    parser.add_argument("--request-timeout-sec", type=float, default=2.0)
    parser.add_argument("--panel-width", type=int, default=480)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--maximum-consecutive-failures", type=int, default=5)
    parser.add_argument(
        "--show-window",
        action="store_true",
        help="also open the legacy OpenCV window; browser-only is the default",
    )
    parser.add_argument("--headless", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--route-refresh-hz", type=float, default=1.0)
    parser.add_argument(
        "--mission-route-latest-override",
        type=int,
        help=(
            "read-only route preview override for latest_scanned_checkpoint; "
            "does not call any SDK write endpoint"
        ),
    )
    parser.add_argument("--snapshot-interval", type=int, default=25)
    parser.add_argument("--no-event-capture", action="store_true")
    parser.add_argument("--capture-pre-event-frames", type=int, default=8)
    parser.add_argument("--capture-post-event-frames", type=int, default=8)
    parser.add_argument("--capture-baseline-interval-frames", type=int, default=40)
    parser.add_argument("--capture-queue-size", type=int, default=32)
    parser.add_argument("--dashboard-host", default="127.0.0.1")
    parser.add_argument("--dashboard-port", type=int, default=8001)
    parser.add_argument("--no-browser-bridge", action="store_true")
    parser.add_argument(
        "--planner-mode",
        choices=("connected_path", "motion_primitives", "gps_only"),
        help="override planner.mode from config for A/B testing",
    )
    parser.add_argument(
        "--camera-calibration",
        help=(
            "path to a validated camera calibration file (YAML/JSON), used only "
            "when planner.geometry_mode is metric_projected; overrides "
            "camera_calibration.path from --config"
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.target_fps <= 0.0 or args.telemetry_hz <= 0.0 or args.route_refresh_hz <= 0.0:
        raise SystemExit("target-fps, telemetry-hz, and route-refresh-hz must be positive")
    if (
        args.maximum_frame_age_sec <= 0.0
        or args.maximum_telemetry_age_sec <= 0.0
        or args.request_timeout_sec <= 0.0
    ):
        raise SystemExit("timeouts must be positive")
    if args.panel_width <= 0 or args.snapshot_interval <= 0:
        raise SystemExit("panel-width and snapshot-interval must be positive")
    if any(
        value <= 0
        for value in (
            args.capture_pre_event_frames,
            args.capture_post_event_frames,
            args.capture_baseline_interval_frames,
            args.capture_queue_size,
        )
    ):
        raise SystemExit("event capture frame counts and queue size must be positive")
    if args.max_frames is not None and args.max_frames <= 0:
        raise SystemExit("max-frames must be positive")
    if args.maximum_consecutive_failures <= 0:
        raise SystemExit("maximum-consecutive-failures must be positive")
    if not 1 <= args.dashboard_port <= 65535:
        raise SystemExit("dashboard-port must be in [1, 65535]")
    show_window = args.show_window and not args.headless
    if show_window and sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        raise SystemExit("DISPLAY is unavailable; omit --show-window")

    config_path = _rooted(args.config)
    if not config_path.exists():
        raise SystemExit(f"required input does not exist: {config_path}")
    config_paths = [config_path]
    if args.mission_config:
        mission_config_path = _rooted(args.mission_config)
        if not mission_config_path.exists():
            raise SystemExit(f"required input does not exist: {mission_config_path}")
        config_paths.append(mission_config_path)
    config = load_config(*config_paths)
    sam_tp_cfg = config.get("sam_tp", {})
    if not isinstance(sam_tp_cfg, dict):
        raise SystemExit("config sam_tp section must be a mapping")

    upstream = Path(args.upstream_root).expanduser().resolve()

    def upstream_path(cli_value: str | None, config_key: str) -> Path:
        value = cli_value or sam_tp_cfg.get(config_key)
        if not isinstance(value, str) or not value.strip():
            raise SystemExit(
                f"SAM-TP {config_key} is not configured; set sam_tp.{config_key} "
                f"in {config_path} or pass --{config_key.replace('_', '-')}"
            )
        path = Path(value).expanduser()
        return (path if path.is_absolute() else upstream / path).resolve()

    model_config = upstream_path(args.model_config, "model_config")
    checkpoint = upstream_path(args.checkpoint, "checkpoint")
    expected_checkpoint_sha256 = (
        args.expected_checkpoint_sha256
        or sam_tp_cfg.get("expected_checkpoint_sha256")
    )
    if not isinstance(expected_checkpoint_sha256, str) or len(expected_checkpoint_sha256) != 64:
        raise SystemExit(
            "SAM-TP expected_checkpoint_sha256 must be a 64-character SHA-256 value"
        )
    output = Path(args.output_dir).expanduser().resolve()
    for path in (upstream, model_config, checkpoint):
        if not path.exists():
            raise SystemExit(f"required input does not exist: {path}")
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    provenance = git_provenance(upstream)
    if provenance["commit"] != OFFICIAL_COMMIT or provenance["dirty"]:
        raise SystemExit(f"upstream checkout is not the frozen clean commit: {provenance}")
    checkpoint_sha = sha256_file(checkpoint)
    if checkpoint_sha != expected_checkpoint_sha256:
        raise SystemExit(
            "checkpoint SHA-256 differs from the explicitly approved value: "
            f"expected={expected_checkpoint_sha256} actual={checkpoint_sha}"
        )

    planner_cfg = dict(config.get("planner", {}))
    if args.planner_mode is not None:
        planner_cfg["mode"] = args.planner_mode
    calibration = None
    if planner_cfg.get("geometry_mode") == "metric_projected":
        calibration_path = args.camera_calibration or config.get(
            "camera_calibration", {}
        ).get("path")
        if not calibration_path:
            print(
                "camera calibration path not configured for metric_projected mode; "
                "the local planner will report NO_VALID_CALIBRATION and fail closed "
                "every frame (read-only shadow mode stays observable regardless).",
                flush=True,
            )
        else:
            try:
                calibration = load_calibration(_rooted(str(calibration_path)))
                print(
                    "camera calibration loaded: "
                    f"id={calibration.calibration_id} sha256={calibration.sha256_prefix} "
                    f"resolution={calibration.image_width}x{calibration.image_height}",
                    flush=True,
                )
            except CalibrationError as exc:
                print(
                    f"camera calibration at {calibration_path} is invalid ({exc.reason}); "
                    "the local planner will report NO_VALID_CALIBRATION and fail closed "
                    "every frame (read-only shadow mode stays observable regardless).",
                    flush=True,
                )
    sdk_cfg = config["sdk"]
    navigation_cfg = config.get("navigation", {})
    heading_offset_deg = float(navigation_cfg.get("rover_heading_offset_deg", 0.0))
    if not np.isfinite(heading_offset_deg):
        raise SystemExit("navigation.rover_heading_offset_deg must be finite")
    sdk = EarthRoverSDKClient(
        sdk_cfg["base_url"],
        args.request_timeout_sec,
    )
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("SAM-TP shadow mode requires torch in its independent environment") from exc
    if not torch.cuda.is_available():
        raise SystemExit("SAM-TP shadow mode requires CUDA")
    predictor, checkpoint_format = build_sam_tp_predictor(
        upstream,
        model_config,
        checkpoint,
        synchronize=torch.cuda.synchronize,
    )
    predictor.load()
    checkpoint_metadata = {
        "filename": checkpoint.name,
        "sha256_prefix": checkpoint_sha[:16],
        "backend": checkpoint_format.value,
        "model_config_id": (
            HF_SAM2_MODEL_CONFIG_ID
            if checkpoint_format is CheckpointFormat.HF_SAM2
            else model_config.name
        ),
    }
    print(
        "SAM-TP checkpoint loaded and ready: "
        f"filename={checkpoint_metadata['filename']} "
        f"sha256={checkpoint_metadata['sha256_prefix']} "
        f"backend={checkpoint_metadata['backend']} "
        f"model_config={checkpoint_metadata['model_config_id']} "
        f"load_time_ms={predictor.load_time_ms:.1f}",
        flush=True,
    )
    phase1_processor = SamTpPhase1FrameProcessor(
        predictor,
        ConstantCurvatureTrajectorySampler(
            DEFAULT_CURVATURES,
            horizon_m=2.0,
            sample_interval_m=0.1,
            rover_width_m=0.4,
            safety_margin_m=0.1,
        ).sample(),
        checkpoint_sha[:12],
    )
    local_planner = MotionPrimitivePlanner(planner_cfg)

    # Reuse navigation.max_heading_rate_deg_per_sec (already tuned, already
    # physical-units) for the filter's heading process noise instead of a
    # second independently-tuned bound; localization.yaml can still override
    # it explicitly if ever needed.
    localization_cfg = dict(config.get("localization", {}))
    localization_cfg.setdefault(
        "max_heading_rate_deg_per_sec",
        navigation_cfg.get("max_heading_rate_deg_per_sec", 120.0),
    )
    # Constructed once, outside the route-refresh block below. Route content
    # refreshes preserve the estimate, but a new mission session resets it:
    # direct-bot telemetry can be stale or refer to the rover's pre-positioned
    # location, so carrying that origin into an active mission can make every
    # fresh GPS fix look like a permanent outlier.
    localizer = (
        GpsHeadingEkf(localization_cfg) if localization_cfg.get("enabled", True) else None
    )

    output.mkdir(parents=True)
    jsonl_path = output / "shadow_frames.jsonl"
    records: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    telemetry = None
    telemetry_interval = 1.0 / args.telemetry_hz
    route_interval = 1.0 / args.route_refresh_hz
    next_telemetry = 0.0
    next_route_refresh = 0.0
    route_planner = None
    route_signature = None
    mission_active: bool | None = None
    delay = 1.0 / args.target_fps
    started = time.monotonic()
    consecutive_failures = 0
    window_name = "Earth Rover SAM-TP Read-Only Shadow"
    if show_window:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    dashboard_server = None
    event_recorder = None
    if not args.no_event_capture:
        event_recorder = SamTpEventRecorder(
            output,
            EventCaptureConfig(
                pre_event_frames=args.capture_pre_event_frames,
                post_event_frames=args.capture_post_event_frames,
                baseline_interval_frames=args.capture_baseline_interval_frames,
                queue_size=args.capture_queue_size,
            ),
        )
        print(f"SAM-TP replay capture: {event_recorder.output_dir}", flush=True)
    if not args.no_browser_bridge:
        dashboard_server = SamTpDashboardServer(
            args.dashboard_host,
            args.dashboard_port,
            checkpoint_metadata=checkpoint_metadata,
        )
        dashboard_server.start()
        host, port = dashboard_server.address
        print(f"SAM-TP browser bridge: http://{host}:{port}/status", flush=True)
    print("SAM-TP SDK shadow mode: GET-only, command_transmitted=false", flush=True)
    try:
        with jsonl_path.open("w", encoding="utf-8") as jsonl:
            frame_index = 0
            while args.max_frames is None or frame_index < args.max_frames:
                loop_started = time.monotonic()
                try:
                    if loop_started >= next_route_refresh:
                        try:
                            route = sdk.get_mission_route()
                            latest_scanned = (
                                args.mission_route_latest_override
                                if args.mission_route_latest_override is not None
                                else route["latest_scanned_checkpoint"]
                            )
                            signature = json.dumps(
                                {
                                    "checkpoints": route["checkpoints"],
                                    "latest": latest_scanned,
                                },
                                sort_keys=True,
                            )
                            route_mission_active = bool(route["mission_active"])
                            if (
                                mission_active is False
                                and route_mission_active
                                and localizer is not None
                            ):
                                localizer.reset()
                                print(
                                    "Mission activated: localization reset; "
                                    "reacquiring GPS/heading origin",
                                    flush=True,
                                )
                            mission_active = route_mission_active
                            if route["route_loaded"] and signature != route_signature:
                                route_planner = CheckpointRoutePlanner(
                                    route["checkpoints"],
                                    float(config["urban"]["waypoint_switch_radius_m"]),
                                    latest_scanned_checkpoint=int(latest_scanned or 0),
                                    heading_filter_alpha=float(
                                        navigation_cfg.get("heading_filter_alpha", 1.0)
                                    ),
                                    target_heading_deadband_deg=float(
                                        navigation_cfg.get("target_heading_deadband_deg", 0.0)
                                    ),
                                    large_heading_change_deg=float(
                                        navigation_cfg.get("large_heading_change_deg", 180.0)
                                    ),
                                    max_heading_rate_deg_per_sec=(
                                        float(navigation_cfg["max_heading_rate_deg_per_sec"])
                                        if navigation_cfg.get("max_heading_rate_deg_per_sec")
                                        is not None
                                        else None
                                    ),
                                    max_heading_sample_interval_sec=float(
                                        navigation_cfg.get(
                                            "max_heading_sample_interval_sec",
                                            telemetry_interval,
                                        )
                                    ),
                                )
                                route_signature = signature
                            elif not route["route_loaded"]:
                                route_planner = None
                                route_signature = None
                        except Exception:
                            # Route guidance is optional; perception shadow must
                            # continue when the SDK server has no loaded mission.
                            pass
                        next_route_refresh = loop_started + route_interval
                    fetch_telemetry = loop_started >= next_telemetry
                    step, telemetry = run_shadow_step(
                        sdk,
                        predictor,
                        frame_index,
                        telemetry,
                        fetch_telemetry,
                        started,
                        checkpoint_sha,
                        args.maximum_frame_age_sec,
                        args.maximum_telemetry_age_sec,
                        panel_width=args.panel_width,
                        phase1_processor=phase1_processor,
                        route_planner=route_planner,
                        local_planner=local_planner,
                        heading_offset_deg=heading_offset_deg,
                        localizer=localizer,
                        checkpoint_metadata=checkpoint_metadata,
                        calibration=calibration,
                    )
                    if fetch_telemetry:
                        telemetry_backoff = (
                            3.0 if step.record.get("telemetry_error") else telemetry_interval
                        )
                        next_telemetry = loop_started + telemetry_backoff
                    records.append(step.record)
                    if event_recorder is not None:
                        event_reasons = event_recorder.observe(step)
                        if event_reasons:
                            print(
                                f"capture event frame={frame_index} "
                                f"reasons={','.join(event_reasons)}",
                                flush=True,
                            )
                    if dashboard_server is not None:
                        dashboard_server.store.publish(step.overlay_bgr, step.record)
                    consecutive_failures = 0
                    jsonl.write(json.dumps(step.record, sort_keys=True) + "\n")
                    jsonl.flush()
                    if (
                        frame_index % args.snapshot_interval == 0
                        or args.max_frames == frame_index + 1
                    ):
                        snapshot = output / "latest_dashboard.jpg"
                        if not cv2.imwrite(str(snapshot), step.dashboard_bgr):
                            raise OSError(f"cannot write dashboard snapshot: {snapshot}")
                    print(
                        f"frame={frame_index} state={step.record['shadow_state']} "
                        f"e2e={float(step.record['end_to_end_latency_ms']):.1f}ms "
                        f"fps={float(step.record['effective_fps']):.2f}",
                        flush=True,
                    )
                    if show_window:
                        cv2.imshow(window_name, step.dashboard_bgr)
                        key = cv2.waitKey(1) & 0xFF
                        if key in (27, ord("q")):
                            break
                    frame_index += 1
                except Exception as exc:
                    if dashboard_server is not None:
                        dashboard_server.store.publish_error(exc)
                    failure = {
                        "timestamp": time.time(),
                        "frame_index": frame_index,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                    failures.append(failure)
                    consecutive_failures += 1
                    print(f"shadow frame failed: {failure}", file=sys.stderr, flush=True)
                    if consecutive_failures >= args.maximum_consecutive_failures:
                        print(
                            "maximum consecutive failures reached; stopping shadow mode",
                            file=sys.stderr,
                            flush=True,
                        )
                        break
                    time.sleep(max(delay, 1.0))
                remaining = delay - (time.monotonic() - loop_started)
                if remaining > 0.0:
                    time.sleep(remaining)
    except KeyboardInterrupt:
        pass
    finally:
        if show_window:
            cv2.destroyAllWindows()
        if dashboard_server is not None:
            dashboard_server.close()
        if event_recorder is not None:
            capture_summary = event_recorder.close()
            if capture_summary["write_errors"] or capture_summary["dropped_items"]:
                print(
                    f"WARNING: replay capture incomplete: {capture_summary}",
                    file=sys.stderr,
                    flush=True,
                )
        write_shadow_summary(
            output,
            records,
            failures,
            checkpoint,
            checkpoint_sha,
            checkpoint_metadata=checkpoint_metadata,
        )
    print(f"SAM-TP shadow output: {output}")
    print("No SDK write endpoint or live rover command was used.")
    return 0 if records else 2


def _rooted(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


if __name__ == "__main__":
    raise SystemExit(main())
