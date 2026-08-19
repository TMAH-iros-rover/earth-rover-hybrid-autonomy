from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class DashboardSnapshot:
    jpeg: bytes | None
    status: dict[str, Any]


class DashboardSnapshotStore:
    """Thread-safe latest-value store for the browser dashboard bridge."""

    def __init__(self, checkpoint_metadata: dict[str, Any] | None = None) -> None:
        self._lock = threading.Lock()
        initial_status: dict[str, Any] = {
            "service": "sam-tp-shadow",
            "ready": False,
            "state": "STARTING",
            "command_transmitted": False,
        }
        if checkpoint_metadata is not None:
            initial_status["checkpoint"] = dict(checkpoint_metadata)
        self._snapshot = DashboardSnapshot(
            jpeg=None,
            status=initial_status,
        )

    def publish(self, image_bgr: np.ndarray, record: dict[str, Any]) -> None:
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError("dashboard image must have shape HxWx3")
        if image_bgr.dtype != np.uint8:
            raise ValueError("dashboard image must use uint8 pixels")
        encoded, buffer = cv2.imencode(
            ".jpg",
            image_bgr,
            [cv2.IMWRITE_JPEG_QUALITY, 88],
        )
        if not encoded:
            raise ValueError("could not encode SAM-TP dashboard image")
        status = {
            "service": "sam-tp-shadow",
            "ready": True,
            "state": str(record["shadow_state"]),
            "frame_index": int(record["frame_index"]),
            "published_timestamp": time.time(),
            "inference_latency_ms": float(record["inference_latency_ms"]),
            "planner_latency_ms": float(record.get("planner_latency_ms", 0.0)),
            "end_to_end_latency_ms": float(record["end_to_end_latency_ms"]),
            "effective_fps": float(record["effective_fps"]),
            "score_min": float(record["score_min"]),
            "score_mean": float(record["score_mean"]),
            "score_max": float(record["score_max"]),
            "path_valid": bool(record["image_path_valid"]),
            "path_reason": str(record["image_path_reason"]),
            "path_mean_score": record.get("image_path_mean_score"),
            "local_path_length_px": record.get("local_path_length_px"),
            "local_path_goal_alignment_weight": record.get(
                "local_path_goal_alignment_weight"
            ),
            "local_path_smoothing_method": record.get("local_path_smoothing_method"),
            "local_path_smoothing_applied": record.get("local_path_smoothing_applied"),
            "local_path_smoothing_iterations": record.get(
                "local_path_smoothing_iterations"
            ),
            "local_path_selected_heading_deg": record.get(
                "local_path_selected_heading_deg"
            ),
            "local_path_heading_residual_deg": record.get(
                "local_path_heading_residual_deg"
            ),
            "global_target_heading_error_deg": record.get(
                "global_target_heading_error_deg"
            ),
            "planner": record.get("planner"),
            "geometry_mode": record.get("geometry_mode"),
            "camera_projection_applied": record.get("camera_projection_applied"),
            "image_path_metric_calibrated": record.get(
                "image_path_metric_calibrated"
            ),
            "calibration_id": record.get("calibration_id"),
            "calibration_sha256_prefix": record.get(
                "calibration_sha256_prefix"
            ),
            "near_field_safe": record.get("near_field_safe"),
            "near_field_score": record.get("near_field_score"),
            "trajectory_valid": record.get("trajectory_valid"),
            "trajectory_quality": record.get("trajectory_quality"),
            "planner_confidence": record.get("planner_confidence"),
            "plan_age_sec": record.get("plan_age_sec"),
            "using_held_plan": record.get("using_held_plan"),
            "navigation": record.get("navigation"),
            "localization": record.get("localization"),
            # Rover motion telemetry, forwarded as-is so Mission1Autonomy can
            # confirm the rover is physically stationary before a recovery
            # rotation (see ROTATE_ESCAPE's STOP_CONFIRM phase). Previously
            # computed in run_shadow_step() but dropped at this boundary.
            "telemetry": record.get("telemetry"),
            "telemetry_valid": record.get("telemetry_valid"),
            "telemetry_age_sec": record.get("telemetry_age_sec"),
            "capture_event_reasons": record.get("capture_event_reasons", []),
            "sdk_clock_offset_hours": record.get("sdk_clock_offset_hours"),
            "checkpoint": record.get("checkpoint"),
            "command_transmitted": False,
        }
        with self._lock:
            self._snapshot = DashboardSnapshot(buffer.tobytes(), status)

    def publish_error(self, error: Exception) -> None:
        with self._lock:
            previous = self._snapshot
            status = dict(previous.status)
            previous_frame_available = previous.jpeg is not None
            status.update(
                {
                    "ready": previous_frame_available,
                    "state": "STALE_FRAME" if previous_frame_available else "ERROR",
                    "last_error": f"{type(error).__name__}: {error}",
                    "published_timestamp": time.time(),
                    "command_transmitted": False,
                }
            )
            self._snapshot = DashboardSnapshot(previous.jpeg, status)

    def get(self) -> DashboardSnapshot:
        with self._lock:
            return DashboardSnapshot(
                self._snapshot.jpeg,
                dict(self._snapshot.status),
            )


class SamTpDashboardServer:
    """Local GET-only HTTP bridge consumed by the SDK mission dashboard."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8001,
        store: DashboardSnapshotStore | None = None,
        checkpoint_metadata: dict[str, Any] | None = None,
    ) -> None:
        if not 0 <= port <= 65535:
            raise ValueError("port must be in [0, 65535]")
        self.store = store or DashboardSnapshotStore(checkpoint_metadata)
        handler = _handler_for(self.store)
        self._server = ThreadingHTTPServer((host, port), handler)
        self._thread: threading.Thread | None = None

    @property
    def address(self) -> tuple[str, int]:
        host, port = self._server.server_address[:2]
        return str(host), int(port)

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("SAM-TP dashboard server is already running")
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="sam-tp-dashboard",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        if self._thread is None:
            self._server.server_close()
            return
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2.0)
        self._thread = None


def _handler_for(store: DashboardSnapshotStore) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
            try:
                path = self.path.split("?", 1)[0]
                snapshot = store.get()
                if path == "/status":
                    self._send_json(HTTPStatus.OK, snapshot.status)
                    return
                if path == "/overlay.jpg":
                    if snapshot.jpeg is None:
                        self._send_json(
                            HTTPStatus.SERVICE_UNAVAILABLE,
                            {"detail": "SAM-TP has not published a frame yet"},
                        )
                        return
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(snapshot.jpeg)))
                    self._common_headers()
                    self.end_headers()
                    self.wfile.write(snapshot.jpeg)
                    return
                self._send_json(HTTPStatus.NOT_FOUND, {"detail": "Not found"})
            except (BrokenPipeError, ConnectionResetError):
                return

        def do_OPTIONS(self) -> None:  # noqa: N802 - stdlib handler API
            self.send_response(HTTPStatus.NO_CONTENT)
            self._common_headers()
            self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
            self.end_headers()

        def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, sort_keys=True).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self._common_headers()
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                return

        def _common_headers(self) -> None:
            origin = self.headers.get("Origin")
            if origin in {"http://127.0.0.1:8000", "http://localhost:8000"}:
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")
            self.send_header("Cache-Control", "no-store, max-age=0")
            self.send_header("X-Content-Type-Options", "nosniff")

        def log_message(self, _format: str, *_args: object) -> None:
            return

    return Handler
