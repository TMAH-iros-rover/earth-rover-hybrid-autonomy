#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from typing import Any

import requests


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose local Earth Rover SDK bridge readiness.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument(
        "--send-zero-control",
        action="store_true",
        help="send a zero linear/angular/lamp command to verify RTM publish",
    )
    args = parser.parse_args()
    base_url = args.base_url.rstrip("/")
    now = time.time()
    mission = get_json(f"{base_url}/mission-status", args.timeout)
    diagnostics = get_json(f"{base_url}/connection-diagnostics", args.timeout)
    data = try_get_json(f"{base_url}/data", args.timeout)
    front = try_get_json(f"{base_url}/v2/front", args.timeout)
    zero_result = None
    if args.send_zero_control:
      zero_result = post_json(
          f"{base_url}/control",
          {"command": {"linear": 0, "angular": 0, "lamp": 0}},
          args.timeout,
      )

    report = {
        "server_time": now,
        "mission_status_age_sec": age(now, mission.get("timestamp")),
        "front_frame_age_sec": age(now, front.get("timestamp") if front else None),
        "telemetry_age_sec": age(now, data.get("timestamp") if data else None),
        "rover_connected": mission.get("rover_connected"),
        "rtc_connected": mission.get("rtc_connected"),
        "rtm_connected": mission.get("rtm_connected"),
        "control_transport_ready": mission.get("rtm_control_transport_ready"),
        "control_bridge_ready": mission.get("control_bridge_ready"),
        "control_bridge_reason": mission.get("control_bridge_reason"),
        "control_command_fresh": mission.get("control_command_fresh"),
        "control_watchdog_active": mission.get("control_watchdog_active"),
        "last_control_command_age_sec": mission.get("last_control_command_age_sec"),
        "front_frame_length": len(front.get("front_frame", "")) if front else 0,
        "telemetry_valid": bool(data and (data.get("latitude") is not None or data.get("gps"))),
        "zero_control_test_result": zero_result,
        "connection_diagnostics": diagnostics.get("control_bridge"),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def get_json(url: str, timeout: float) -> dict[str, Any]:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise TypeError(f"{url} returned non-object JSON")
    return payload


def try_get_json(url: str, timeout: float) -> dict[str, Any] | None:
    try:
        return get_json(url, timeout)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    response = requests.post(url, json=payload, timeout=timeout)
    try:
        body = response.json()
    except Exception:
        body = {"body": response.text}
    return {"status_code": response.status_code, "body": body}


def age(now: float, timestamp: object) -> float | None:
    try:
        value = float(timestamp)
    except (TypeError, ValueError):
        return None
    diff = now - value
    return diff if diff >= 0 else diff


if __name__ == "__main__":
    raise SystemExit(main())
