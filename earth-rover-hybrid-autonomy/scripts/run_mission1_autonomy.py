#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from earth_rover.autonomy.mission1_controller import (  # noqa: E402
    Mission1Autonomy,
    Mission1ControlConfig,
    SamStatusSource,
)
from earth_rover.autonomy.status_server import (  # noqa: E402
    AutonomyStatusServer,
    AutonomyStatusStore,
)
from earth_rover.sdk_client import EarthRoverSDKClient  # noqa: E402
from earth_rover.utils.config import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mission1 local-path controller; waits for dashboard Start Mission"
    )
    parser.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    parser.add_argument(
        "--mission-config",
        default=str(ROOT / "configs/mission1_live.yaml"),
    )
    parser.add_argument("--enable-live-control", action="store_true")
    parser.add_argument("--status-host", default="127.0.0.1")
    parser.add_argument("--status-port", type=int, default=8002)
    parser.add_argument("--max-ticks", type=int, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config, args.mission_config)
    settings = Mission1ControlConfig.from_dict(config)
    settings.validate()
    sdk_cfg = config.get("sdk", {})
    sdk = EarthRoverSDKClient(
        str(sdk_cfg.get("base_url", "http://127.0.0.1:8000")),
        float(sdk_cfg.get("request_timeout_sec", 0.5)),
    )
    autonomy = Mission1Autonomy(
        sdk,
        SamStatusSource(settings.sam_status_url, settings.sam_timeout_sec),
        settings,
        config,
        live_control_enabled=args.enable_live_control,
    )
    store = AutonomyStatusStore(autonomy.status)
    server = AutonomyStatusServer(
        args.status_host,
        args.status_port,
        store,
        actions={
            "/stop": autonomy.operator_stop,
            "/resume": autonomy.operator_resume,
        },
    )
    server.start()
    host, port = server.address
    mode = "LIVE" if args.enable_live_control else "DRY-RUN"
    print(f"Mission1 autonomy status: http://{host}:{port}/status", flush=True)
    print(f"Mode: {mode}; waiting for dashboard Start Mission", flush=True)
    if not args.enable_live_control:
        print("No rover command will be transmitted without --enable-live-control", flush=True)
    period = 1.0 / settings.loop_hz
    ticks = 0
    try:
        while args.max_ticks is None or ticks < args.max_ticks:
            started = time.monotonic()
            try:
                status = autonomy.tick()
            except Exception as exc:
                status = autonomy.fail_safe(exc)
            store.publish(status)
            print(
                f"state={status['state']} linear={status['linear']:.3f} "
                f"sdk_angular={status['sdk_angular']:.3f} "
                f"(internal_angular={status['angular']:.3f}) "
                f"reason={status['reason']}",
                flush=True,
            )
            ticks += 1
            time.sleep(max(0.0, period - (time.monotonic() - started)))
    except KeyboardInterrupt:
        print("Stopping Mission1 autonomy", flush=True)
    finally:
        autonomy.shutdown()
        server.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
