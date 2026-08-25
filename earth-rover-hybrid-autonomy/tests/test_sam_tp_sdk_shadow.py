from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from earth_rover.core.types import FrameData, RoverData
from earth_rover.navigation.checkpoint_route import CheckpointRoutePlanner
from training.sam_tp_reproduction import SamTpPrediction
from training.run_sam_tp_sdk_shadow import parse_args
from training.sam_tp_sdk_shadow import (
    compose_traversability_overlay,
    corrected_heading_deg,
    run_shadow_step,
    write_shadow_summary,
)
from training.sam_tp_dashboard_bridge import DashboardSnapshotStore


class ReadOnlyFakeSdk:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.image = np.zeros((12, 20, 3), dtype=np.uint8)
        self.image[0, 0] = [10, 20, 30]
        self.frame_timestamp = 99.9
        self.sdk_frame_timestamp = 99.8

    def get_front_frame(self) -> FrameData:
        self.calls.append("get_front_frame")
        return FrameData(
            self.frame_timestamp,
            self.image.copy(),
            "front",
            sdk_timestamp=self.sdk_frame_timestamp,
        )

    def get_data(self) -> RoverData:
        self.calls.append("get_data")
        return RoverData(
            timestamp=99.95,
            latitude=1.0,
            longitude=2.0,
            orientation=3.0,
            speed=0.0,
            rpms=[0.0, 0.0, 0.0, 0.0],
            battery=90.0,
            signal_level=5.0,
            gps_signal=20.0,
            raw={},
            sdk_timestamp=99.85,
        )

    def send_control(self, *_args, **_kwargs) -> None:
        raise AssertionError("shadow mode must not send control")


class RecordingPredictor:
    def __init__(self) -> None:
        self.images: list[np.ndarray] = []

    def predict(self, image_rgb: np.ndarray) -> SamTpPrediction:
        self.images.append(image_rgb.copy())
        score = np.full(image_rgb.shape[:2], 0.75, dtype=np.float32)
        logits = np.full(image_rgb.shape[:2], 1.0, dtype=np.float32)
        return SamTpPrediction(
            raw_logits=logits,
            traversability_score=score,
            heatmap=np.zeros_like(image_rgb),
            input_shape=image_rgb.shape,
            output_shape=score.shape,
            inference_time_ms=10.0,
            device="test",
        )


class SpyLocalizer:
    """Minimal stand-in for GpsHeadingEkf's interface, for wiring tests."""

    def __init__(self, locked_estimate=None) -> None:
        self.gyro_calls: list[object] = []
        self.observe_calls: list[tuple] = []
        self._locked_estimate = locked_estimate

    def observe_gyro(self, raw_samples) -> None:
        self.gyro_calls.append(raw_samples)

    def observe_gps_heading(self, latitude, longitude, heading_deg, gps_signal=None) -> None:
        self.observe_calls.append((latitude, longitude, heading_deg, gps_signal))

    @property
    def is_locked(self) -> bool:
        return self._locked_estimate is not None

    def current_estimate(self):
        return self._locked_estimate if self._locked_estimate is not None else (None, None, None)

    @property
    def gyro_trusted(self) -> bool:
        return True


class TelemetryFailingSdk(ReadOnlyFakeSdk):
    def get_data(self) -> RoverData:
        self.calls.append("get_data")
        raise RuntimeError("telemetry timeout")


def test_shadow_launcher_defaults_to_browser_only_without_opencv_window() -> None:
    args = parse_args(
        [
            "--upstream-root",
            "upstream",
            "--model-config",
            "model.yaml",
            "--checkpoint",
            "checkpoint.pt",
            "--expected-checkpoint-sha256",
            "abc",
            "--output-dir",
            "output",
        ]
    )

    assert args.show_window is False
    assert args.headless is False


def test_shadow_launcher_accepts_read_only_route_latest_override() -> None:
    args = parse_args(
        [
            "--upstream-root",
            "upstream",
            "--model-config",
            "model.yaml",
            "--checkpoint",
            "checkpoint.pt",
            "--expected-checkpoint-sha256",
            "abc",
            "--output-dir",
            "output",
            "--mission-route-latest-override",
            "1",
        ]
    )

    assert args.mission_route_latest_override == 1


def test_shadow_step_uses_read_only_sdk_and_explicit_bgr_to_rgb() -> None:
    sdk = ReadOnlyFakeSdk()
    predictor = RecordingPredictor()
    # Wall-clock correction must not affect measured durations.
    clock_values = iter((100.0, 90.0, 100.1))
    monotonic_values = iter((10.01, 10.03, 10.2))

    step, telemetry = run_shadow_step(
        sdk,
        predictor,
        frame_index=0,
        telemetry=None,
        fetch_telemetry=True,
        started_monotonic=10.0,
        checkpoint_sha256="abc",
        maximum_frame_age_sec=1.0,
        maximum_telemetry_age_sec=1.0,
        clock=lambda: next(clock_values),
        monotonic=lambda: next(monotonic_values),
        panel_width=100,
    )

    assert sdk.calls == ["get_front_frame", "get_data"]
    assert predictor.images[0][0, 0].tolist() == [30, 20, 10]
    assert telemetry is not None
    assert step.record["command_transmitted"] is False
    assert step.record["candidate_trajectory_count"] == 7
    assert step.record["adapter_confidence"] == 1.0
    assert step.record["trajectory_geometry_only"] is True
    assert step.record["camera_projection_applied"] is False
    assert step.record["sdk_allowed_read_endpoints"] == [
        "/v2/front",
        "/front",
        "/data",
        "/mission-route",
    ]
    assert abs(float(step.record["acquisition_latency_ms"]) - 20.0) < 1e-9
    assert abs(float(step.record["end_to_end_latency_ms"]) - 190.0) < 1e-9
    assert step.record["shadow_state"] == "CLEAR"
    assert step.record["telemetry_valid"] is True
    assert step.record["sdk_frame_timestamp_usable"] is True
    assert step.record["sdk_clock_offset_hours"] is None
    assert step.dashboard_bgr.shape == (242, 300, 3)
    assert step.overlay_bgr.shape == sdk.image.shape


def test_shadow_step_marks_old_frame_stale_without_command() -> None:
    sdk = ReadOnlyFakeSdk()
    predictor = RecordingPredictor()
    clock_values = iter((100.0, 100.01, 101.2))
    monotonic_values = iter((10.01, 10.03, 10.2))

    step, _ = run_shadow_step(
        sdk,
        predictor,
        frame_index=0,
        telemetry=None,
        fetch_telemetry=False,
        started_monotonic=10.0,
        checkpoint_sha256="abc",
        maximum_frame_age_sec=1.0,
        maximum_telemetry_age_sec=1.0,
        clock=lambda: next(clock_values),
        monotonic=lambda: next(monotonic_values),
        panel_width=100,
    )

    assert sdk.calls == ["get_front_frame"]
    assert step.record["prediction_valid"] is False
    assert step.record["shadow_state"] == "STALE_FRAME"
    assert step.record["command_transmitted"] is False


def test_shadow_step_rejects_duplicate_source_frame_without_command() -> None:
    sdk = ReadOnlyFakeSdk()
    sdk.get_front_frame = lambda: FrameData(
        timestamp=99.9,
        image=sdk.image.copy(),
        source="front",
        sdk_timestamp=99.9,
        source_frame_id="session:1000:42:10.0",
        source_media_time_sec=10.0,
        source_frame_new=False,
    )
    clock_values = iter((100.0, 100.01, 100.1))
    monotonic_values = iter((10.01, 10.03, 10.2))

    step, _ = run_shadow_step(
        sdk,
        RecordingPredictor(),
        frame_index=1,
        telemetry=None,
        fetch_telemetry=False,
        started_monotonic=10.0,
        checkpoint_sha256="abc",
        maximum_frame_age_sec=1.0,
        maximum_telemetry_age_sec=1.0,
        clock=lambda: next(clock_values),
        monotonic=lambda: next(monotonic_values),
        panel_width=100,
    )

    assert step.record["shadow_state"] == "DUPLICATE_FRAME"
    assert step.record["prediction_valid"] is False
    assert step.record["source_frame_new"] is False
    assert step.record["command_transmitted"] is False


def test_shadow_step_uses_global_heading_to_bias_read_only_local_path() -> None:
    sdk = ReadOnlyFakeSdk()
    sdk.image = np.zeros((120, 200, 3), dtype=np.uint8)
    sdk.frame_timestamp = 99.9
    planner = CheckpointRoutePlanner(
        [{"sequence": 1, "latitude": 1.001, "longitude": 2.0}],
        switch_radius_m=1.0,
    )
    clock_values = iter((100.0, 100.01, 100.1))
    monotonic_values = iter((10.01, 10.03, 10.2))

    step, _ = run_shadow_step(
        sdk,
        RecordingPredictor(),
        frame_index=0,
        telemetry=None,
        fetch_telemetry=True,
        started_monotonic=10.0,
        checkpoint_sha256="abc",
        maximum_frame_age_sec=1.0,
        maximum_telemetry_age_sec=1.0,
        clock=lambda: next(clock_values),
        monotonic=lambda: next(monotonic_values),
        panel_width=100,
        route_planner=planner,
    )

    navigation = step.record["navigation"]
    assert navigation["target_sequence"] == 1
    assert navigation["target_bearing_deg"] == 0.0
    assert navigation["heading_error_deg"] == pytest.approx(-3.0)
    assert step.record["global_target_heading_error_deg"] == pytest.approx(-3.0)
    assert step.record["image_path_reason"] == "GPS_HEADING_ALIGNED_TRAVERSABLE_PATH"
    assert step.record["command_transmitted"] is False


def test_shadow_step_applies_live_rover_heading_offset_for_route_guidance() -> None:
    sdk = ReadOnlyFakeSdk()
    sdk.image = np.zeros((120, 200, 3), dtype=np.uint8)
    sdk.frame_timestamp = 99.9
    sdk.sdk_frame_timestamp = 99.8
    planner = CheckpointRoutePlanner(
        [{"sequence": 2, "latitude": 30.48268318, "longitude": 114.3026047}],
        switch_radius_m=1.0,
    )
    sdk.get_data = lambda: RoverData(
        timestamp=99.95,
        latitude=30.48248291015625,
        longitude=114.3026351928711,
        orientation=167.0,
        speed=0.0,
        rpms=[0.0, 0.0, 0.0, 0.0],
        battery=90.0,
        signal_level=5.0,
        gps_signal=20.0,
        raw={},
        sdk_timestamp=99.85,
    )
    clock_values = iter((100.0, 100.01, 100.1))
    monotonic_values = iter((10.01, 10.03, 10.2))

    step, _ = run_shadow_step(
        sdk,
        RecordingPredictor(),
        frame_index=0,
        telemetry=None,
        fetch_telemetry=True,
        started_monotonic=10.0,
        checkpoint_sha256="abc",
        maximum_frame_age_sec=1.0,
        maximum_telemetry_age_sec=1.0,
        clock=lambda: next(clock_values),
        monotonic=lambda: next(monotonic_values),
        panel_width=100,
        route_planner=planner,
        heading_offset_deg=180.0,
    )

    navigation = step.record["navigation"]
    assert navigation["current_heading_deg"] == pytest.approx(347.0)
    assert abs(navigation["heading_error_deg"]) < 15.0
    assert step.record["navigation_heading_offset_deg"] == 180.0


def test_shadow_step_feeds_fresh_telemetry_into_localizer_when_fetching() -> None:
    sdk = ReadOnlyFakeSdk()
    localizer = SpyLocalizer()
    clock_values = iter((100.0, 100.01, 100.1))
    monotonic_values = iter((10.01, 10.03, 10.2))

    run_shadow_step(
        sdk,
        RecordingPredictor(),
        frame_index=0,
        telemetry=None,
        fetch_telemetry=True,
        started_monotonic=10.0,
        checkpoint_sha256="abc",
        maximum_frame_age_sec=1.0,
        maximum_telemetry_age_sec=1.0,
        clock=lambda: next(clock_values),
        monotonic=lambda: next(monotonic_values),
        panel_width=100,
        localizer=localizer,
    )

    # ReadOnlyFakeSdk.get_data() returns latitude=1.0, longitude=2.0,
    # orientation=3.0, gps_signal=20.0, raw={} (no gyros key).
    assert localizer.observe_calls == [(1.0, 2.0, 3.0, 20.0)]
    assert localizer.gyro_calls == [None]


def test_shadow_step_does_not_refuse_localizer_when_telemetry_not_refetched() -> None:
    sdk = ReadOnlyFakeSdk()
    localizer = SpyLocalizer()
    stale_telemetry = sdk.get_data()
    sdk.calls.clear()
    clock_values = iter((100.0, 100.01, 100.1))
    monotonic_values = iter((10.01, 10.03, 10.2))

    run_shadow_step(
        sdk,
        RecordingPredictor(),
        frame_index=0,
        telemetry=stale_telemetry,
        fetch_telemetry=False,
        started_monotonic=10.0,
        checkpoint_sha256="abc",
        maximum_frame_age_sec=1.0,
        maximum_telemetry_age_sec=1.0,
        clock=lambda: next(clock_values),
        monotonic=lambda: next(monotonic_values),
        panel_width=100,
        localizer=localizer,
    )

    # Re-fusing the same telemetry sample every tick would make the filter
    # overconfident in stale data -- must only fuse on a fresh fetch.
    assert sdk.calls == ["get_front_frame"]
    assert localizer.observe_calls == []
    assert localizer.gyro_calls == []


def test_telemetry_record_only_includes_raw_imu_arrays_when_fresh() -> None:
    sdk = ReadOnlyFakeSdk()
    telemetry_with_gyros = RoverData(
        **{**sdk.get_data().__dict__, "raw": {"gyros": [[0.0, 0.0, 1.0, 5.0]]}}
    )
    sdk.get_data = lambda: telemetry_with_gyros
    clock_values = iter((100.0, 100.01, 100.1, 100.0, 100.01, 100.1))
    monotonic_values = iter((10.01, 10.03, 10.2, 10.4, 10.42, 10.6))

    fresh_step, _ = run_shadow_step(
        sdk,
        RecordingPredictor(),
        frame_index=0,
        telemetry=None,
        fetch_telemetry=True,
        started_monotonic=10.0,
        checkpoint_sha256="abc",
        maximum_frame_age_sec=1.0,
        maximum_telemetry_age_sec=1.0,
        clock=lambda: next(clock_values),
        monotonic=lambda: next(monotonic_values),
        panel_width=100,
    )
    stale_step, _ = run_shadow_step(
        sdk,
        RecordingPredictor(),
        frame_index=1,
        telemetry=telemetry_with_gyros,
        fetch_telemetry=False,
        started_monotonic=10.0,
        checkpoint_sha256="abc",
        maximum_frame_age_sec=1.0,
        maximum_telemetry_age_sec=1.0,
        clock=lambda: next(clock_values),
        monotonic=lambda: next(monotonic_values),
        panel_width=100,
    )

    # Same underlying telemetry sample either way -- only freshness differs.
    assert fresh_step.record["telemetry"]["gyros"] == [[0.0, 0.0, 1.0, 5.0]]
    assert stale_step.record["telemetry"]["gyros"] is None
    assert stale_step.record["telemetry"]["latitude"] == 1.0  # other fields unaffected


def test_shadow_step_routes_the_locked_localizer_estimate_not_raw_telemetry() -> None:
    sdk = ReadOnlyFakeSdk()
    # Deliberately different from ReadOnlyFakeSdk's raw (1.0, 2.0, 3.0), so a
    # pass-through bug (using raw telemetry despite a locked localizer) is
    # visible in the assertions below.
    localizer = SpyLocalizer(locked_estimate=(9.0, 8.0, 45.0))
    planner = CheckpointRoutePlanner(
        [{"sequence": 1, "latitude": 9.0, "longitude": 8.001}],
        switch_radius_m=1.0,
    )
    clock_values = iter((100.0, 100.01, 100.1))
    monotonic_values = iter((10.01, 10.03, 10.2))

    step, _ = run_shadow_step(
        sdk,
        RecordingPredictor(),
        frame_index=0,
        telemetry=None,
        fetch_telemetry=True,
        started_monotonic=10.0,
        checkpoint_sha256="abc",
        maximum_frame_age_sec=1.0,
        maximum_telemetry_age_sec=1.0,
        clock=lambda: next(clock_values),
        monotonic=lambda: next(monotonic_values),
        panel_width=100,
        route_planner=planner,
        localizer=localizer,
    )

    navigation = step.record["navigation"]
    assert navigation["current_heading_deg"] == pytest.approx(45.0)
    assert step.record["localization"] == {
        "locked": True,
        "fused_latitude": 9.0,
        "fused_longitude": 8.0,
        "fused_heading_deg": 45.0,
        "gyro_trusted": True,
    }


def test_shadow_step_reanchors_route_after_localizer_heading_reacquisition() -> None:
    class ReacquiredLocalizer(SpyLocalizer):
        def __init__(self, locked_estimate=None) -> None:
            super().__init__(locked_estimate)
            self.reanchor_pending = True

        @property
        def heading_valid(self) -> bool:
            return True

        def consume_heading_reanchor(self) -> bool:
            pending = self.reanchor_pending
            self.reanchor_pending = False
            return pending

        def status(self) -> dict[str, object]:
            return {
                "heading_valid": True,
                "heading_status_reason": "HEADING_REACQUIRED_GYRO_UNTRUSTED",
            }

    sdk = ReadOnlyFakeSdk()
    localizer = ReacquiredLocalizer(locked_estimate=(9.0, 8.0, 355.0))
    planner = CheckpointRoutePlanner(
        [{"sequence": 1, "latitude": 9.001, "longitude": 8.0}],
        switch_radius_m=1.0,
        max_heading_rate_deg_per_sec=30.0,
    )
    planner.update(9.0, 8.0, 161.0)
    clock_values = iter((100.0, 100.01, 100.1))
    monotonic_values = iter((10.01, 10.03, 10.2))

    step, _ = run_shadow_step(
        sdk,
        RecordingPredictor(),
        frame_index=0,
        telemetry=None,
        fetch_telemetry=True,
        started_monotonic=10.0,
        checkpoint_sha256="abc",
        maximum_frame_age_sec=1.0,
        maximum_telemetry_age_sec=1.0,
        clock=lambda: next(clock_values),
        monotonic=lambda: next(monotonic_values),
        panel_width=100,
        route_planner=planner,
        localizer=localizer,
    )

    assert step.record["navigation"]["heading_valid"] is True
    assert step.record["navigation"]["current_heading_deg"] == pytest.approx(355.0)
    assert step.record["navigation"]["reason"] == "TRACKING"


def test_corrected_heading_wraps_and_rejects_invalid_values() -> None:
    assert corrected_heading_deg(350.0, 20.0) == pytest.approx(10.0)
    assert corrected_heading_deg(None, 180.0) is None
    assert corrected_heading_deg(float("nan"), 180.0) is None


def test_shadow_step_continues_inference_when_telemetry_fetch_fails() -> None:
    sdk = TelemetryFailingSdk()
    predictor = RecordingPredictor()
    clock_values = iter((100.0, 100.01, 100.1))
    monotonic_values = iter((10.01, 10.03, 10.2))

    step, telemetry = run_shadow_step(
        sdk,
        predictor,
        frame_index=0,
        telemetry=None,
        fetch_telemetry=True,
        started_monotonic=10.0,
        checkpoint_sha256="abc",
        maximum_frame_age_sec=1.0,
        maximum_telemetry_age_sec=1.0,
        clock=lambda: next(clock_values),
        monotonic=lambda: next(monotonic_values),
        panel_width=100,
    )

    assert sdk.calls == ["get_front_frame", "get_data"]
    assert predictor.images
    assert telemetry is None
    assert step.record["shadow_state"] == "WAITING_TELEMETRY"
    assert step.record["prediction_valid"] is True
    assert step.record["telemetry_valid"] is False
    assert "telemetry timeout" in step.record["telemetry_error"]
    assert step.record["navigation"] is None


def test_shadow_step_reports_stale_telemetry_separately() -> None:
    sdk = ReadOnlyFakeSdk()
    predictor = RecordingPredictor()
    telemetry = sdk.get_data()
    telemetry = RoverData(
        **{
            **telemetry.__dict__,
            "timestamp": 98.0,
        }
    )
    sdk.calls.clear()
    clock_values = iter((100.0, 100.01, 100.1))
    monotonic_values = iter((10.01, 10.03, 10.2))

    step, _ = run_shadow_step(
        sdk,
        predictor,
        frame_index=0,
        telemetry=telemetry,
        fetch_telemetry=False,
        started_monotonic=10.0,
        checkpoint_sha256="abc",
        maximum_frame_age_sec=1.0,
        maximum_telemetry_age_sec=1.0,
        clock=lambda: next(clock_values),
        monotonic=lambda: next(monotonic_values),
        panel_width=100,
    )

    assert step.record["prediction_valid"] is True
    assert step.record["telemetry_valid"] is False
    assert step.record["shadow_state"] == "STALE_TELEMETRY"
    assert step.record["command_transmitted"] is False


def test_shadow_step_tolerates_explicit_whole_hour_sdk_clock_offset() -> None:
    sdk = ReadOnlyFakeSdk()
    sdk.frame_timestamp = 32500.0
    clock_values = iter((32500.0, 32500.01, 32500.1))
    monotonic_values = iter((10.01, 10.03, 10.2))

    step, _ = run_shadow_step(
        sdk,
        RecordingPredictor(),
        frame_index=0,
        telemetry=None,
        fetch_telemetry=False,
        started_monotonic=10.0,
        checkpoint_sha256="abc",
        maximum_frame_age_sec=1.0,
        maximum_telemetry_age_sec=1.0,
        clock=lambda: next(clock_values),
        monotonic=lambda: next(monotonic_values),
        panel_width=100,
    )

    assert step.record["sdk_frame_age_sec"] == 32400.3
    assert step.record["sdk_clock_offset_hours"] == 9
    assert step.record["sdk_frame_timestamp_usable"] is False
    assert step.record["prediction_valid"] is True


def test_shadow_summary_records_no_sdk_write_or_motion(tmp_path: Path) -> None:
    records = [
        {
            "shadow_state": "CLEAR",
            "end_to_end_latency_ms": 100.0,
            "inference_latency_ms": 80.0,
            "effective_fps": 8.0,
            "prediction_valid": True,
            "telemetry_valid": True,
        }
    ]

    path = write_shadow_summary(tmp_path, records, [], "checkpoint.pt", "abc")
    report = json.loads(path.read_text(encoding="utf-8"))

    assert report["success"]
    assert report["sdk_write_endpoints"] == []
    assert report["command_transmitted"] is False
    assert report["live_motion_command_sent_by_process"] is False
    assert report["processed_frame_count"] == 1


def test_shadow_step_rejects_non_uint8_sdk_frame() -> None:
    sdk = ReadOnlyFakeSdk()
    sdk.image = np.zeros((12, 20, 3), dtype=np.float32)

    try:
        run_shadow_step(
            sdk,
            RecordingPredictor(),
            frame_index=0,
            telemetry=None,
            fetch_telemetry=False,
            started_monotonic=10.0,
            checkpoint_sha256="abc",
            maximum_frame_age_sec=1.0,
            maximum_telemetry_age_sec=1.0,
            panel_width=100,
        )
    except ValueError as exc:
        assert str(exc) == "SDK front frame must use uint8 pixels"
    else:
        raise AssertionError("non-uint8 SDK frame must be rejected")


def test_shadow_launcher_contains_no_sdk_write_call() -> None:
    root = Path(__file__).resolve().parents[1]
    launcher = (root / "training/run_sam_tp_sdk_shadow.py").read_text(
        encoding="utf-8"
    )
    core = (root / "training/sam_tp_sdk_shadow.py").read_text(encoding="utf-8")

    for source in (launcher, core):
        assert ".send_control(" not in source
        assert ".start_mission(" not in source
        assert ".end_mission(" not in source
    assert ".report_checkpoint(" not in source

    assert ".get_mission_route(" in launcher
    assert "--show-window" in launcher
    assert "--mission-route-latest-override" in launcher


def test_browser_bridge_publishes_latest_overlay_and_read_only_status() -> None:
    store = DashboardSnapshotStore()
    initial = store.get()
    assert initial.status["ready"] is False
    assert initial.status["command_transmitted"] is False

    image = np.full((10, 16, 3), 80, dtype=np.uint8)
    store.publish(
        image,
        {
            "shadow_state": "CLEAR",
            "frame_index": 3,
            "frame_id": "frame-source-3",
            "plan_id": "plan-00000003",
            "frame_width": 16,
            "frame_height": 10,
            "inference_latency_ms": 91.0,
            "end_to_end_latency_ms": 130.0,
            "effective_fps": 7.5,
            "score_min": 0.1,
            "score_mean": 0.6,
            "score_max": 0.9,
            "image_path_valid": True,
            "image_path_reason": "CONNECTED_HIGH_TRAVERSABILITY_IMAGE_PATH",
            "geometry_mode": "metric_projected",
            "camera_projection_applied": True,
            "image_path_metric_calibrated": True,
            "calibration_id": "mission1_front_camera_1024x576_2026_08_14",
            "calibration_sha256_prefix": "1a0b3adefbeaea3e",
            "sdk_clock_offset_hours": 9,
        },
    )
    snapshot = store.get()

    assert snapshot.status["ready"] is True
    assert snapshot.status["frame_index"] == 3
    assert snapshot.status["frame_id"] == "frame-source-3"
    assert snapshot.status["plan_id"] == "plan-00000003"
    assert snapshot.status["frame_width"] == 16
    assert snapshot.status["frame_height"] == 10
    assert snapshot.status["command_transmitted"] is False
    assert snapshot.status["sdk_clock_offset_hours"] == 9
    assert snapshot.status["geometry_mode"] == "metric_projected"
    assert snapshot.status["camera_projection_applied"] is True
    assert snapshot.status["image_path_metric_calibrated"] is True
    assert (
        snapshot.status["calibration_id"]
        == "mission1_front_camera_1024x576_2026_08_14"
    )
    assert snapshot.jpeg is not None
    assert snapshot.jpeg.startswith(b"\xff\xd8")

    frame_timestamp = snapshot.status["frame_published_timestamp"]
    store.publish_error(RuntimeError("next inference failed"))
    stale = store.get().status
    assert stale["state"] == "STALE_FRAME"
    assert stale["frame_published_timestamp"] == frame_timestamp
    assert stale["published_timestamp"] == frame_timestamp
    assert stale["status_timestamp"] >= frame_timestamp


def test_shadow_step_and_dashboard_surface_checkpoint_metadata() -> None:
    sdk = ReadOnlyFakeSdk()
    checkpoint_metadata = {
        "filename": "best_sam_tp.pt",
        "sha256_prefix": "99f0efab7b40d532",
        "backend": "hf_sam2",
        "model_config_id": "sam2_hf_tiny",
    }
    clock_values = iter((100.0, 100.01, 100.1))
    monotonic_values = iter((10.01, 10.03, 10.2))

    step, _ = run_shadow_step(
        sdk,
        RecordingPredictor(),
        frame_index=0,
        telemetry=None,
        fetch_telemetry=False,
        started_monotonic=10.0,
        checkpoint_sha256="99f0efab7b40d532e31909a77f1e9c4d1a0be3a534b96c7d769850cc046b951d",
        maximum_frame_age_sec=1.0,
        maximum_telemetry_age_sec=1.0,
        clock=lambda: next(clock_values),
        monotonic=lambda: next(monotonic_values),
        panel_width=100,
        checkpoint_metadata=checkpoint_metadata,
    )

    assert step.record["checkpoint"] == checkpoint_metadata

    store = DashboardSnapshotStore()
    store.publish(step.overlay_bgr, step.record)
    assert store.get().status["checkpoint"] == checkpoint_metadata


def test_dashboard_surfaces_checkpoint_before_first_frame_and_during_error() -> None:
    checkpoint_metadata = {
        "filename": "best_sam_tp.pt",
        "sha256_prefix": "99f0efab7b40d532",
        "backend": "hf_sam2",
        "model_config_id": "sam2_hf_tiny",
    }
    store = DashboardSnapshotStore(checkpoint_metadata)

    assert store.get().status["state"] == "STARTING"
    assert store.get().status["checkpoint"] == checkpoint_metadata

    store.publish_error(RuntimeError("front frame unavailable"))

    status = store.get().status
    assert status["state"] == "ERROR"
    assert status["checkpoint"] == checkpoint_metadata


def test_compose_traversability_overlay_preserves_source_geometry() -> None:
    image = np.zeros((12, 20, 3), dtype=np.uint8)
    score = np.full((12, 20), 0.75, dtype=np.float32)

    overlay = compose_traversability_overlay(image, score)

    assert overlay.shape == image.shape
    assert overlay.dtype == np.uint8


def test_dashboard_displays_shared_candidate_rejection_reason() -> None:
    from training.sam_tp_sdk_shadow import _dashboard_path_reason

    record = {
        "image_path_reason": "MOTION_PRIMITIVE_NEAR_FIELD_UNSAFE",
        "planner": {
            "candidate_scores": [
                {"hard_rejected": True, "reject_reason": "NO_VALID_CALIBRATION"},
                {"hard_rejected": True, "reject_reason": "NO_VALID_CALIBRATION"},
            ]
        },
    }

    assert _dashboard_path_reason(record) == "NO_VALID_CALIBRATION"
