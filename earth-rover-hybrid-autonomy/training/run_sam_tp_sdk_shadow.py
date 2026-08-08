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
from training.sam_tp_reproduction import (  # noqa: E402
    OFFICIAL_COMMIT,
    SamTpPredictor,
    git_provenance,
    sha256_file,
)
from training.sam_tp_sdk_shadow import (  # noqa: E402
    run_shadow_step,
    write_shadow_summary,
)
from training.sam_tp_dashboard_bridge import SamTpDashboardServer  # noqa: E402
from training.sam_tp_phase1_review import SamTpPhase1FrameProcessor  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run read-only SDK front-camera SAM-TP shadow inference. "
            "No mission or control endpoint is called."
        )
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--upstream-root", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
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
    parser.add_argument("--dashboard-host", default="127.0.0.1")
    parser.add_argument("--dashboard-port", type=int, default=8001)
    parser.add_argument("--no-browser-bridge", action="store_true")
    parser.add_argument(
        "--planner-mode",
        choices=("connected_path", "motion_primitives", "gps_only"),
        help="override planner.mode from config for A/B testing",
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
    if args.max_frames is not None and args.max_frames <= 0:
        raise SystemExit("max-frames must be positive")
    if args.maximum_consecutive_failures <= 0:
        raise SystemExit("maximum-consecutive-failures must be positive")
    if not 1 <= args.dashboard_port <= 65535:
        raise SystemExit("dashboard-port must be in [1, 65535]")
    show_window = args.show_window and not args.headless
    if show_window and sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        raise SystemExit("DISPLAY is unavailable; omit --show-window")

    upstream = Path(args.upstream_root).expanduser().resolve()
    model_config = Path(args.model_config).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    config_path = _rooted(args.config)
    for path in (upstream, model_config, checkpoint, config_path):
        if not path.exists():
            raise SystemExit(f"required input does not exist: {path}")
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    provenance = git_provenance(upstream)
    if provenance["commit"] != OFFICIAL_COMMIT or provenance["dirty"]:
        raise SystemExit(f"upstream checkout is not the frozen clean commit: {provenance}")
    checkpoint_sha = sha256_file(checkpoint)
    if checkpoint_sha != args.expected_checkpoint_sha256:
        raise SystemExit(
            "checkpoint SHA-256 differs from the explicitly approved value: "
            f"expected={args.expected_checkpoint_sha256} actual={checkpoint_sha}"
        )

    config = load_config(config_path)
    planner_cfg = dict(config.get("planner", {}))
    if args.planner_mode is not None:
        planner_cfg["mode"] = args.planner_mode
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
    predictor = SamTpPredictor(
        upstream,
        model_config,
        checkpoint,
        synchronize=torch.cuda.synchronize,
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
    # Constructed once, outside the route-refresh block below -- the fused
    # position/heading estimate must not reset just because the checkpoint
    # route reloaded.
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
    delay = 1.0 / args.target_fps
    started = time.monotonic()
    consecutive_failures = 0
    window_name = "Earth Rover SAM-TP Read-Only Shadow"
    if show_window:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    dashboard_server = None
    if not args.no_browser_bridge:
        dashboard_server = SamTpDashboardServer(
            args.dashboard_host,
            args.dashboard_port,
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
                    )
                    if fetch_telemetry:
                        telemetry_backoff = (
                            3.0 if step.record.get("telemetry_error") else telemetry_interval
                        )
                        next_telemetry = loop_started + telemetry_backoff
                    records.append(step.record)
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
        write_shadow_summary(
            output,
            records,
            failures,
            checkpoint,
            checkpoint_sha,
        )
    print(f"SAM-TP shadow output: {output}")
    print("No SDK write endpoint or live rover command was used.")
    return 0 if records else 2


def _rooted(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


if __name__ == "__main__":
    raise SystemExit(main())
