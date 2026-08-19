import asyncio
from pathlib import Path

import pytest

import browser_service


def test_resolve_chrome_executable_uses_valid_configured_path(
    monkeypatch, tmp_path: Path
) -> None:
    executable = tmp_path / "chrome"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o755)
    monkeypatch.setenv("CHROME_EXECUTABLE_PATH", str(executable))

    assert browser_service.resolve_chrome_executable() == str(executable)


def test_resolve_chrome_executable_rejects_invalid_configured_path(
    monkeypatch, tmp_path: Path
) -> None:
    missing = tmp_path / "missing-chrome"
    monkeypatch.setenv("CHROME_EXECUTABLE_PATH", str(missing))

    with pytest.raises(
        browser_service.BrowserConfigurationError,
        match="CHROME_EXECUTABLE_PATH is not an executable file",
    ):
        browser_service.resolve_chrome_executable()


def test_resolve_chrome_executable_detects_linux_browser(monkeypatch) -> None:
    monkeypatch.delenv("CHROME_EXECUTABLE_PATH", raising=False)
    monkeypatch.setattr(
        browser_service.shutil,
        "which",
        lambda name: "/usr/bin/chromium" if name == "chromium" else None,
    )

    assert browser_service.resolve_chrome_executable() == "/usr/bin/chromium"


def test_concurrent_initialization_launches_browser_once(monkeypatch) -> None:
    launch_calls = []

    class FakePage:
        async def setViewport(self, _viewport):
            pass

        async def setExtraHTTPHeaders(self, _headers):
            pass

        async def goto(self, _url, _options):
            pass

        async def click(self, _selector):
            pass

        async def waitForSelector(self, _selector, _options=None):
            pass

        async def waitForFunction(self, _expression, _options):
            pass

        async def waitFor(self, _milliseconds):
            pass

        async def evaluate(self, _script):
            pass

    class FakeBrowser:
        async def newPage(self):
            return FakePage()

        async def close(self):
            pass

    async def fake_launch(**options):
        launch_calls.append(options)
        await asyncio.sleep(0)
        return FakeBrowser()

    async def run_test():
        service = browser_service.BrowserService()
        await asyncio.gather(
            service.initialize_browser(),
            service.initialize_browser(),
        )

    monkeypatch.setattr(browser_service, "resolve_chrome_executable", lambda: "/chrome")
    monkeypatch.setattr(browser_service, "launch", fake_launch)

    asyncio.run(run_test())

    assert len(launch_calls) == 1
    assert launch_calls[0]["handleSIGINT"] is False
    assert launch_calls[0]["handleSIGTERM"] is False
    assert launch_calls[0]["handleSIGHUP"] is False
    assert launch_calls[0]["autoClose"] is False


def test_wait_for_frame_awaits_async_browser_result(monkeypatch) -> None:
    class FakePage:
        def __init__(self):
            self.calls = 0

        async def evaluate(self, script, uid):
            assert "await window.getLastBase64Frame(uid)" in script
            assert uid == 1000
            self.calls += 1
            if self.calls < 3:
                return None
            return "data:image/png;base64,frame"

    async def no_sleep(_delay):
        return None

    service = browser_service.BrowserService()
    service.page = FakePage()
    monkeypatch.setattr(browser_service.asyncio, "sleep", no_sleep)

    frame = asyncio.run(service._wait_for_frame(1000))

    assert frame == "data:image/png;base64,frame"
    assert service.page.calls == 3


def test_frame_metadata_returns_source_identity_from_browser_page() -> None:
    expected = {
        "source_frame_id": "session:1000:42:10.0",
        "source_media_time_sec": 10.0,
        "source_total_video_frames": 42,
    }

    class FakePage:
        async def evaluate(self, script, uid):
            assert "getLastFrameMetadata" in script
            assert uid == 1000
            return expected

    service = browser_service.BrowserService()
    service.page = FakePage()

    assert asyncio.run(service.frame_metadata(1000)) == expected


def test_diagnostics_distinguish_channel_users_from_published_tracks() -> None:
    source = Path(browser_service.__file__).read_text(encoding="utf-8")
    rtc_source = Path("static/basicVideoCall.js").read_text(encoding="utf-8")

    assert "client.remoteUsers" in source
    assert "publishedRemoteUserCount" in source
    assert "rtcEventHistory" in source
    assert 'client.on("user-joined"' in rtc_source
    assert 'client.on("user-left"' in rtc_source
    assert 'recordRtcEvent("user-unpublished"' in rtc_source


def test_control_send_rejects_an_unready_rtm_bridge() -> None:
    class FakePage:
        async def evaluate(self, script, *_args):
            if "window.rtm_ready === true" in script:
                return False
            raise AssertionError("sendMessage must not run while RTM is unready")

    async def fake_diagnostics():
        return {
            "page": {
                "rtmConnectionState": "CONNECTING",
                "rtmChannelState": "NOT_JOINED",
            }
        }

    service = browser_service.BrowserService()
    service.browser = object()
    service.page = FakePage()
    service.diagnostics = fake_diagnostics

    with pytest.raises(browser_service.BrowserServiceError, match="not ready"):
        asyncio.run(service.send_message({"linear": 0.0, "angular": 0.0}))


def test_control_status_reports_rtm_transport_ready_without_recent_command() -> None:
    class FakePage:
        async def evaluate(self, script):
            assert "RTM_CONTROL_TRANSPORT_READY" in script
            return {
                "ready": True,
                "reason": "RTM_CONTROL_TRANSPORT_READY",
                "rtm_connected": True,
                "rtm_control_transport_ready": True,
                "rtc_connected": False,
            }

    service = browser_service.BrowserService()
    service.browser = object()
    service.page = FakePage()
    service.initialization_stage = "READY"

    status = asyncio.run(service.control_status())

    assert status["ready"] is True
    assert status["rtm_control_transport_ready"] is True
    assert status["reason"] == "RTM_CONTROL_TRANSPORT_READY"


def test_front_restarts_a_dead_browser_after_repeated_failures(monkeypatch) -> None:
    monkeypatch.setattr(browser_service, "CAMERA_FAILURE_RESTART_THRESHOLD", 2)
    monkeypatch.setattr(browser_service, "CAMERA_FAILURE_RESTART_MIN_WINDOW_SEC", 0.0)

    class FakePage:
        async def evaluate(self, _script, _uid):
            raise RuntimeError("execution context was destroyed")

    class FakeBrowser:
        def __init__(self):
            self.closed = False

        async def close(self):
            self.closed = True

    service = browser_service.BrowserService()
    fake_browser = FakeBrowser()
    service.browser = fake_browser
    service.page = FakePage()
    service.initialization_stage = "READY"

    async def run_test():
        for _ in range(2):
            with pytest.raises(browser_service.BrowserServiceError):
                await service.front(timeout_sec=0.01)

    asyncio.run(run_test())

    assert fake_browser.closed is True
    assert service.browser is None
    assert service.page is None
    assert service.initialization_stage == "NOT_STARTED"
    assert service._consecutive_camera_failures == 0


def test_front_success_resets_the_failure_counter(monkeypatch) -> None:
    monkeypatch.setattr(browser_service, "CAMERA_FAILURE_RESTART_THRESHOLD", 2)
    monkeypatch.setattr(browser_service, "CAMERA_FAILURE_RESTART_MIN_WINDOW_SEC", 0.0)

    class ScriptedPage:
        def __init__(self, results):
            self.results = list(results)

        async def evaluate(self, _script, _uid):
            result = self.results.pop(0)
            if isinstance(result, Exception):
                raise result
            return result

    service = browser_service.BrowserService()
    service.browser = object()
    # fail, succeed (resets the streak), fail again -- if the reset didn't
    # happen this second failure alone would hit the threshold of 2 and
    # restart the browser.
    service.page = ScriptedPage(
        [
            RuntimeError("transient"),
            "data:image/png;base64,frame",
            RuntimeError("transient"),
        ]
    )
    service.initialization_stage = "READY"

    async def run_test():
        with pytest.raises(browser_service.BrowserServiceError):
            await service.front(timeout_sec=0.01)
        frame = await service.front(timeout_sec=0.01)
        with pytest.raises(browser_service.BrowserServiceError):
            await service.front(timeout_sec=0.01)
        return frame

    frame = asyncio.run(run_test())

    assert frame == "data:image/png;base64,frame"
    assert service._consecutive_camera_failures == 1
    assert service.browser is not None


def test_front_does_not_restart_on_a_fast_failure_burst(monkeypatch) -> None:
    # Regression test: a live run hit repeated failures that raced ahead of
    # real elapsed time (evaluate() raising almost instantly once the page
    # was already dead) and tore down a browser whose RTM control channel
    # was still healthy, right as a drive command was in flight. The count
    # alone must not be enough to restart within a burst shorter than
    # CAMERA_FAILURE_RESTART_MIN_WINDOW_SEC.
    monkeypatch.setattr(browser_service, "CAMERA_FAILURE_RESTART_THRESHOLD", 2)
    monkeypatch.setattr(browser_service, "CAMERA_FAILURE_RESTART_MIN_WINDOW_SEC", 60.0)

    class FakePage:
        async def evaluate(self, _script, _uid):
            raise RuntimeError("execution context was destroyed")

    class FakeBrowser:
        def __init__(self):
            self.closed = False

        async def close(self):
            self.closed = True

    service = browser_service.BrowserService()
    fake_browser = FakeBrowser()
    service.browser = fake_browser
    service.page = FakePage()
    service.initialization_stage = "READY"

    async def run_test():
        for _ in range(5):
            with pytest.raises(browser_service.BrowserServiceError):
                await service.front(timeout_sec=0.01)

    asyncio.run(run_test())

    assert fake_browser.closed is False
    assert service.browser is fake_browser
    assert service._consecutive_camera_failures == 5


def test_control_status_reports_uninitialized_publisher() -> None:
    service = browser_service.BrowserService()

    status = asyncio.run(service.control_status())

    assert status["ready"] is False
    assert status["reason"] == "CONTROL_PUBLISHER_NOT_INITIALIZED"
