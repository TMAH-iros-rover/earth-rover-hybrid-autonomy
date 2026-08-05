import asyncio
import json
import time
from pathlib import Path

import pytest

import main
from browser_service import BrowserConfigurationError


ROOT = Path(__file__).resolve().parents[1]


def test_mission_status_does_not_expose_identifiers(monkeypatch) -> None:
    monkeypatch.setenv("MISSION_SLUG", "private-mission")
    monkeypatch.setattr(main, "selected_mission_slug", "private-mission")
    monkeypatch.setattr(main, "active_session_mode", "mission")
    monkeypatch.setenv("BOT_SLUG", "private-rover")
    monkeypatch.setattr(main, "auth_response_data", {"CHANNEL_NAME": "private-channel"})
    monkeypatch.setattr(
        main,
        "checkpoints_list_data",
        {
            "checkpoints_list": [{"sequence": 1}],
            "latest_scanned_checkpoint": 0,
        },
    )

    payload = main.mission_status_payload()

    assert payload["mission_configured"] is True
    assert payload["mission_active"] is True
    assert payload["rover_connected"] is True
    assert payload["operation_mode"] == "mission"
    assert payload["start_mission_required"] is True
    assert payload["camera_and_telemetry_allowed"] is True
    assert payload["control_bridge_ready"] is False
    assert payload["checkpoint_count"] == 1
    assert payload["latest_scanned_checkpoint"] == 0
    assert payload["control_watchdog_timeout_sec"] > 0
    serialized = str(payload)
    assert "private-mission" not in serialized
    assert "private-rover" not in serialized
    assert "private-channel" not in serialized


def test_mission_status_timestamp_uses_timezone_neutral_epoch(monkeypatch) -> None:
    monkeypatch.setenv("TZ", "Asia/Seoul")
    if hasattr(time, "tzset"):
        time.tzset()
    monkeypatch.setattr(main, "auth_response_data", {"CHANNEL_NAME": "direct"})
    before = time.time()
    payload = main.mission_status_payload(
        {
            "ready": True,
            "reason": "RTM_CONTROL_TRANSPORT_READY",
            "rtm_connected": True,
            "rtm_control_transport_ready": True,
            "rtc_connected": False,
        }
    )
    after = time.time()

    assert before - 1 <= payload["timestamp"] <= after + 1
    assert payload["server_timestamp"] == pytest.approx(payload["timestamp"], abs=0.01)
    assert payload["control_bridge_ready"] is True
    assert payload["control_bridge_reason"] == "RTM_CONTROL_TRANSPORT_READY"


def test_utcnow_timestamp_pattern_is_not_used() -> None:
    source = Path(main.__file__).read_text(encoding="utf-8")

    assert "datetime.utcnow" not in source
    assert "utcnow().timestamp" not in source


def test_dashboard_is_available_without_active_mission(monkeypatch) -> None:
    monkeypatch.delenv("MISSION_SLUG", raising=False)
    monkeypatch.setattr(main, "selected_mission_slug", "")
    monkeypatch.setattr(main, "active_session_mode", None)
    monkeypatch.setattr(main, "auth_response_data", {})
    monkeypatch.setattr(main, "checkpoints_list_data", {})
    status = asyncio.run(main.mission_status())
    dashboard = asyncio.run(main.mission_dashboard())
    status_payload = json.loads(status.body)

    assert status.status_code == 200
    assert status_payload["mission_active"] is False
    assert status_payload["rover_connected"] is False
    assert status_payload["operation_mode"] == "direct_bot"
    assert status_payload["start_mission_required"] is False
    assert status_payload["camera_and_telemetry_allowed"] is False
    assert dashboard.status_code == 200
    dashboard_source = (ROOT / dashboard.path).read_text(encoding="utf-8")
    assert "Start Mission" in dashboard_source
    assert "Mission API Results" in dashboard_source


def test_mission_route_returns_cached_data_without_authentication(monkeypatch) -> None:
    monkeypatch.setattr(main, "active_session_mode", "mission")
    monkeypatch.setattr(main, "auth_response_data", {"CHANNEL_NAME": "private"})
    monkeypatch.setattr(
        main,
        "checkpoints_list_data",
        {
            "checkpoints_list": [
                {"sequence": 1, "latitude": 30.1, "longitude": 114.1}
            ],
            "latest_scanned_checkpoint": 0,
        },
    )

    response = asyncio.run(main.mission_route())
    payload = json.loads(response.body)

    assert payload["route_loaded"] is True
    assert payload["mission_active"] is True
    assert payload["checkpoints_list"][0]["sequence"] == 1
    assert "private" not in str(payload)


def test_configured_inactive_mission_does_not_poll_camera(monkeypatch) -> None:
    monkeypatch.setenv("MISSION_SLUG", "mission")
    monkeypatch.setattr(main, "selected_mission_slug", "mission")
    monkeypatch.setattr(main, "active_session_mode", None)
    monkeypatch.setattr(main, "auth_response_data", {})

    payload = main.mission_status_payload()

    assert payload["operation_mode"] == "mission"
    assert payload["mission_active"] is False
    assert payload["camera_and_telemetry_allowed"] is False


def test_dashboard_javascript_has_no_control_endpoint() -> None:
    source = (ROOT / "static/mission_dashboard.js").read_text(encoding="utf-8")

    assert '"/start-mission"' in source
    assert '"/select-mission"' in source
    assert '"/connect-rover"' in source
    assert '"/disconnect-rover"' in source
    assert '"/mission-status"' in source
    assert '"/checkpoints-list"' in source
    assert '"/end-mission"' in source
    assert '"/control"' not in source
    assert "send_control" not in source
    assert 'const SAM_TP_STATUS_PATH = "/sam-tp-status"' in source
    assert 'const AUTONOMY_STATUS_PATH = "/autonomy-status"' in source
    assert "http://127.0.0.1:8001" not in source
    assert "http://127.0.0.1:8002" not in source
    assert 'fetch(AUTONOMY_STATUS_PATH' in source
    assert 'fetch(SAM_TP_STATUS_PATH' in source
    assert '`${SAM_TP_OVERLAY_PATH}?t=${Date.now()}`' in source
    assert "samTpStatusInitialized" in source
    assert '"/connection-diagnostics"' in source
    assert "bearingDegrees" in source
    assert "normalizeHeadingError" in source
    assert "globalPathLayer" in source
    assert "TRAIL_MIN_DISTANCE_M = 2.0" in source
    assert 'requestJson("/mission-route")' in source
    assert "global_target_heading_error_deg" in source
    assert "local_path_goal_alignment_weight" in source
    assert "local_path_smoothing_method" in source
    assert "Bridge connected · waiting for rover" in source
    assert "epochAgeLabel" in source
    assert "pollTelemetry();" in source
    assert "pollCamera();" in source
    assert "|| !state.telemetryReady" not in source
    assert "|| (!state.frontCameraReady && !state.samTpAvailable)" not in source
    assert 'active ? "End Mission" : "Reset Mission"' in source
    assert "(!missionRequested || (!active && !configured))" in source
    assert "stale cloud ride" in source

    html = (ROOT / "static/mission_dashboard.html").read_text(encoding="utf-8")
    assert 'id="view-raw"' in html
    assert 'id="view-sam-tp"' in html
    assert 'id="sam-tp-metrics"' in html
    assert "GPS shortest path" in html
    assert "SAM-TP heading-aware local path" in html
    assert "GPS trail (≥2 m)" in html


def test_direct_rover_connect_uses_auth_and_browser_bridge(monkeypatch) -> None:
    calls = []

    async def fake_retrieve_tokens(_headers, _bot_slug):
        calls.append("tokens")
        return {
            "CHANNEL_NAME": "test",
            "RTC_TOKEN": "rtc",
            "RTM_TOKEN": "rtm",
            "USERID": 1,
            "APP_ID": "app",
            "BOT_UID": "bot-uid",
        }

    async def fake_initialize_browser():
        calls.append("browser")

    async def fake_diagnostics():
        return {
            "page": {
                "remoteUserCount": 1,
                "telemetryPresent": True,
                "frontTrackReady": True,
                "frontFramePresent": False,
            }
        }

    monkeypatch.delenv("MISSION_SLUG", raising=False)
    monkeypatch.setattr(main, "selected_mission_slug", "")
    monkeypatch.setattr(main, "active_session_mode", None)
    monkeypatch.setattr(main, "auth_response_data", {})
    monkeypatch.setenv("SDK_API_TOKEN", "token")
    monkeypatch.setenv("BOT_SLUG", "bot")
    monkeypatch.setattr(main, "retrieve_tokens", fake_retrieve_tokens)
    monkeypatch.setattr(
        main.browser_service, "initialize_browser", fake_initialize_browser
    )
    monkeypatch.setattr(main.browser_service, "diagnostics", fake_diagnostics)

    response = asyncio.run(main.connect_rover())

    assert response.status_code == 200
    assert json.loads(response.body)["operation_mode"] == "direct_bot"
    assert json.loads(response.body)["connection_state"] == "ready"
    assert calls == ["tokens", "browser"]


def test_direct_rover_connect_reports_waiting_when_remote_is_absent(monkeypatch) -> None:
    async def fake_initialize_browser():
        return None

    async def fake_diagnostics():
        return {
            "page": {
                "remoteUserCount": 0,
                "telemetryPresent": False,
                "frontTrackReady": False,
                "frontFramePresent": False,
            }
        }

    monkeypatch.setattr(main, "active_session_mode", "direct_bot")
    monkeypatch.setattr(main, "auth_response_data", {"CHANNEL_NAME": "test"})
    monkeypatch.setattr(
        main.browser_service, "initialize_browser", fake_initialize_browser
    )
    monkeypatch.setattr(main.browser_service, "diagnostics", fake_diagnostics)

    response = asyncio.run(main.connect_rover())
    payload = json.loads(response.body)

    assert payload["connection_state"] == "waiting_for_rover"
    assert payload["remote_user_count"] == 0
    assert payload["telemetry_ready"] is False
    assert payload["front_camera_ready"] is False


def test_rtm_control_bridge_propagates_peer_send_failures() -> None:
    source = (ROOT / "static/basicRtm.js").read_text(encoding="utf-8")
    browser_source = (ROOT / "browser_service.py").read_text(encoding="utf-8")

    assert "async function sendMessage" in source
    assert "return rtmClient" in source
    assert "throw err" in source
    assert "return await window.sendMessage(message)" in browser_source
    assert "window.rtm_ready = false" in source
    assert "window.rtm_ready = true" in source
    assert "window.rtm_ready === true" in browser_source
    assert "window.rtm_channel_state === 'JOINED'" in browser_source
    assert "await browser_service.initialize_browser()" in Path(
        main.__file__
    ).read_text(encoding="utf-8")


def test_browser_configuration_failure_returns_service_unavailable(monkeypatch) -> None:
    async def fail_data():
        raise BrowserConfigurationError("invalid Chrome path")

    monkeypatch.delenv("MISSION_SLUG", raising=False)
    monkeypatch.setattr(main.browser_service, "data", fail_data)
    with pytest.raises(BrowserConfigurationError) as caught:
        asyncio.run(main.get_data())
    response = asyncio.run(
        main.browser_service_error_handler(None, caught.value)
    )

    assert response.status_code == 503
    assert json.loads(response.body) == {"detail": "invalid Chrome path"}


def test_control_command_validation_enforces_sdk_bounds() -> None:
    assert main.validate_control_command(
        {"command": {"linear": 0.2, "angular": -0.3, "lamp": 1}}
    ) == {"linear": 0.2, "angular": -0.3, "lamp": 1}

    with pytest.raises(Exception) as caught:
        main.validate_control_command(
            {"command": {"linear": 1.01, "angular": 0.0, "lamp": 0}}
        )

    assert caught.value.status_code == 400


def test_control_watchdog_sends_stop_after_missing_heartbeat(monkeypatch) -> None:
    sent = []

    async def fake_send(command):
        sent.append(command)

    async def run_test():
        monkeypatch.setattr(main, "CONTROL_WATCHDOG_TIMEOUT_SEC", 0.001)
        monkeypatch.setattr(main, "_send_control_message", fake_send)
        monkeypatch.setattr(main, "_control_generation", 7)
        await main._control_watchdog(7)

    asyncio.run(run_test())

    assert sent == [{"linear": 0.0, "angular": 0.0, "lamp": 0}]
    assert main._control_generation == 8


def test_control_bridge_ready_is_separate_from_command_freshness(monkeypatch) -> None:
    monkeypatch.setattr(main, "_last_control_command_epoch", None)
    monkeypatch.setattr(main, "_last_control_command_monotonic", None)
    monkeypatch.setattr(main, "_control_watchdog_active", False)

    payload = main.mission_status_payload(
        {
            "ready": True,
            "reason": "RTM_CONTROL_TRANSPORT_READY",
            "rtm_connected": True,
            "rtm_control_transport_ready": True,
            "rtc_connected": True,
        }
    )

    assert payload["control_bridge_ready"] is True
    assert payload["control_command_fresh"] is False
    assert payload["last_control_command_age_sec"] is None


def test_zero_control_response_reports_publish_result(monkeypatch) -> None:
    class FakeRequest:
        async def json(self):
            return {"command": {"linear": 0, "angular": 0, "lamp": 0}}

    async def fake_need_start_mission():
        return None

    async def fake_send(command):
        assert command == {"linear": 0.0, "angular": 0.0, "lamp": 0}
        return {"result": "COMMAND_PUBLISHED"}

    async def fake_control_status():
        return {
            "ready": True,
            "reason": "RTM_CONTROL_TRANSPORT_READY",
            "rtm_control_transport_ready": True,
        }

    monkeypatch.setattr(main, "auth_response_data", {"CHANNEL_NAME": "direct"})
    monkeypatch.setattr(main, "need_start_mission", fake_need_start_mission)
    monkeypatch.setattr(main, "_send_control_message", fake_send)
    monkeypatch.setattr(main.browser_service, "control_status", fake_control_status)

    response = asyncio.run(main.control(FakeRequest()))

    assert response["result"] == "COMMAND_PUBLISHED"
    assert response["control_bridge_ready"] is True
    assert response["command_has_motion"] is False


def test_start_ride_preserves_safe_upstream_error(monkeypatch) -> None:
    class Response:
        status_code = 409

        @staticmethod
        def json():
            return {"error": "bot is already reserved"}

    monkeypatch.setattr(main.requests, "post", lambda *_args, **_kwargs: Response())

    with pytest.raises(Exception) as caught:
        asyncio.run(main.start_ride({}, "bot", "mission-1"))

    assert caught.value.status_code == 409
    assert caught.value.detail == {
        "message": "Failed to start mission",
        "upstream_status": 409,
        "upstream_error": "bot is already reserved",
    }


class _JsonRequest:
    def __init__(self, payload):
        self.payload = payload

    async def json(self):
        return self.payload


def test_start_mission_failure_preserves_existing_direct_bridge(monkeypatch) -> None:
    existing_auth = {"CHANNEL_NAME": "direct-channel"}
    reset_calls = []

    async def fail_start_ride(*_args):
        raise main.HTTPException(status_code=409, detail="bot unavailable")

    async def unexpected_reset(*_args, **_kwargs):
        reset_calls.append(True)

    monkeypatch.setenv("SDK_API_TOKEN", "token")
    monkeypatch.setenv("BOT_SLUG", "bot")
    monkeypatch.setattr(main, "selected_mission_slug", "mission1")
    monkeypatch.setattr(main, "active_session_mode", "direct_bot")
    monkeypatch.setattr(main, "active_mission_slug", None)
    monkeypatch.setattr(main, "auth_response_data", existing_auth)
    monkeypatch.setattr(main, "start_ride", fail_start_ride)
    monkeypatch.setattr(main, "reset_local_bridge", unexpected_reset)

    with pytest.raises(main.HTTPException, match="409"):
        asyncio.run(main.start_mission(_JsonRequest({"mission_slug": "mission1"})))

    assert reset_calls == []
    assert main.auth_response_data is existing_auth
    assert main.active_session_mode == "direct_bot"


def test_start_mission_swaps_bridge_only_after_tokens_are_ready(monkeypatch) -> None:
    calls = []

    async def fake_start_ride(*_args):
        calls.append("tokens")
        return {
            "CHANNEL_NAME": "mission-channel",
            "RTC_TOKEN": "rtc",
            "RTM_TOKEN": "rtm",
            "USERID": 12,
            "APP_ID": "app",
            "BOT_UID": "bot-uid",
        }

    async def fake_reset(*_args, **_kwargs):
        calls.append("reset")

    async def fake_checkpoints():
        calls.append("checkpoints")
        main.checkpoints_list_data = {"checkpoints_list": [{"sequence": 1}]}

    async def fake_initialize():
        calls.append("browser")

    monkeypatch.setenv("SDK_API_TOKEN", "token")
    monkeypatch.setenv("BOT_SLUG", "bot")
    monkeypatch.setattr(main, "selected_mission_slug", "mission1")
    monkeypatch.setattr(main, "active_session_mode", "direct_bot")
    monkeypatch.setattr(main, "active_mission_slug", None)
    monkeypatch.setattr(main, "auth_response_data", {"CHANNEL_NAME": "direct"})
    monkeypatch.setattr(main, "checkpoints_list_data", {})
    monkeypatch.setattr(main, "start_ride", fake_start_ride)
    monkeypatch.setattr(main, "reset_local_bridge", fake_reset)
    monkeypatch.setattr(main, "get_checkpoints_list", fake_checkpoints)
    monkeypatch.setattr(main.browser_service, "initialize_browser", fake_initialize)

    response = asyncio.run(
        main.start_mission(_JsonRequest({"mission_slug": "mission1"}))
    )

    assert response.status_code == 200
    assert calls == ["tokens", "reset", "checkpoints", "browser"]
    assert main.auth_response_data["CHANNEL_NAME"] == "mission-channel"
    assert main.active_session_mode == "mission"
    assert main.active_mission_slug == "mission1"


def test_start_mission_uses_local_mission_after_status_retry_failure(
    monkeypatch, tmp_path,
) -> None:
    existing_auth = {"CHANNEL_NAME": "direct-channel", "APP_ID": "app"}
    calls = []

    async def status_invalid(*_args):
        calls.append("tokens")
        raise main.HTTPException(
            status_code=422,
            detail={"upstream_error": "Bot status is invalid."},
        )

    async def fake_reset(*_args, **_kwargs):
        calls.append("reset")

    async def fake_initialize():
        calls.append("restore")

    async def no_sleep(_delay):
        return None

    monkeypatch.setenv("SDK_API_TOKEN", "token")
    monkeypatch.setenv("BOT_SLUG", "bot")
    monkeypatch.setattr(main, "LOCAL_MISSION_STATE_PATH", tmp_path / "state.json")
    monkeypatch.setattr(main, "selected_mission_slug", "mission-1")
    monkeypatch.setattr(main, "active_session_mode", "direct_bot")
    monkeypatch.setattr(main, "active_mission_slug", None)
    monkeypatch.setattr(main, "auth_response_data", existing_auth)
    monkeypatch.setattr(
        main,
        "checkpoints_list_data",
        {"checkpoints_list": [{"sequence": 1}]},
    )
    monkeypatch.setattr(main, "start_ride", status_invalid)
    monkeypatch.setattr(main, "reset_local_bridge", fake_reset)
    monkeypatch.setattr(main.browser_service, "initialize_browser", fake_initialize)
    monkeypatch.setattr(main.browser_service, "page", object())
    monkeypatch.setattr(main.asyncio, "sleep", no_sleep)

    response = asyncio.run(
        main.start_mission(_JsonRequest({"mission_slug": "mission-1"}))
    )
    payload = json.loads(response.body)

    assert calls == ["tokens"]
    assert main.auth_response_data == existing_auth
    assert main.active_session_mode == "local_mission"
    assert main.active_mission_slug == "mission-1"
    assert main.checkpoints_list_data["checkpoints_list"][0]["sequence"] == 1
    assert payload["local_mission"] is True
    assert payload["cloud_mission"] is False


def test_local_mission_checkpoint_reached_advances_cached_route(
    monkeypatch, tmp_path,
) -> None:
    monkeypatch.setattr(main, "LOCAL_MISSION_STATE_PATH", tmp_path / "state.json")
    main.clear_local_mission_state()
    monkeypatch.setattr(main, "active_session_mode", "local_mission")
    monkeypatch.setattr(main, "selected_mission_slug", "mission-1")
    monkeypatch.setattr(main, "auth_response_data", {"CHANNEL_NAME": "direct"})
    monkeypatch.setattr(
        main,
        "checkpoints_list_data",
        {
            "checkpoints_list": [
                {"sequence": 1},
                {"sequence": 2},
                {"sequence": 3},
            ],
            "latest_scanned_checkpoint": 0,
        },
    )

    response = asyncio.run(main.checkpoint_reached(_JsonRequest({})))
    payload = json.loads(response.body)

    assert response.status_code == 200
    assert payload["local_mission"] is True
    assert payload["reached_checkpoint_sequence"] == 1
    assert payload["next_checkpoint_sequence"] == 2
    assert main.checkpoints_list_data["latest_scanned_checkpoint"] == 1
    assert main.current_cached_latest_checkpoint() == 1
    route = json.loads(asyncio.run(main.mission_route()).body)
    assert route["latest_scanned_checkpoint"] == 1


def test_auth_common_reuses_existing_bridge_without_starting_a_ride(
    monkeypatch,
) -> None:
    existing_auth = {"CHANNEL_NAME": "direct-channel"}

    async def unexpected_start(*_args):
        raise AssertionError("existing bridge must not start a cloud ride")

    monkeypatch.setattr(main, "auth_response_data", existing_auth)
    monkeypatch.setattr(main, "active_session_mode", "direct_bot")
    monkeypatch.setattr(main, "selected_mission_slug", "mission-1")
    monkeypatch.setattr(main, "start_ride", unexpected_start)

    result = asyncio.run(main.auth_common())

    assert result is existing_auth
    assert main.active_session_mode == "direct_bot"
