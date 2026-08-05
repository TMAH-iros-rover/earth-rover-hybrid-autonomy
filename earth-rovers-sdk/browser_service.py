import asyncio
import math
import os
import shutil
import time
from pathlib import Path

from pyppeteer import launch
from pyppeteer.errors import NetworkError
from pyppeteer.errors import TimeoutError as PyppeteerTimeoutError
from dotenv import load_dotenv

load_dotenv()

# Configuration from environment variables with defaults
FORMAT = os.getenv("IMAGE_FORMAT", "png")
QUALITY = float(os.getenv("IMAGE_QUALITY", "1.0"))
HAS_REAR_CAMERA = os.getenv("HAS_REAR_CAMERA", "False").lower() == "true"
STARTUP_TIMEOUT_SEC = float(os.getenv("BROWSER_STARTUP_TIMEOUT_SEC", "20"))
TELEMETRY_TIMEOUT_SEC = float(os.getenv("TELEMETRY_READY_TIMEOUT_SEC", "15"))
CAMERA_TIMEOUT_SEC = float(os.getenv("CAMERA_READY_TIMEOUT_SEC", "20"))
FRAME_REQUEST_TIMEOUT_SEC = float(os.getenv("FRAME_REQUEST_TIMEOUT_SEC", "1.0"))
# initialize_browser() only checks that self.browser/self.page objects exist,
# not that the RTC video call inside the page is still alive. If the rover's
# WebRTC stream stalls or drops, front()/rear() fail forever with no way to
# recover short of restarting this process. After this many consecutive
# camera failures, tear down and relaunch the browser so the next request
# re-runs the full launch-and-join sequence instead of staying stuck.
#
# Restarting closes the *entire* browser, including a currently-healthy RTM
# control channel -- so this must not fire on ordinary short hiccups (e.g.
# the brief gap between RTC join completing and the first video frame
# actually arriving). Both a failure count AND a minimum elapsed time are
# required before restarting, so a fast-failing burst (e.g. right after the
# page itself dies, evaluate() raises almost instantly) can't rack up the
# count in under a second and trigger a restart loop.
CAMERA_FAILURE_RESTART_THRESHOLD = int(os.getenv("CAMERA_FAILURE_RESTART_THRESHOLD", "12"))
CAMERA_FAILURE_RESTART_MIN_WINDOW_SEC = float(
    os.getenv("CAMERA_FAILURE_RESTART_MIN_WINDOW_SEC", "12.0")
)

if FORMAT not in ["png", "jpeg", "webp"]:
    raise ValueError("Invalid image format. Supported formats: png, jpeg, webp")

if QUALITY < 0 or QUALITY > 1:
    raise ValueError("Invalid image quality. Quality should be between 0 and 1")
for name, value in (
    ("BROWSER_STARTUP_TIMEOUT_SEC", STARTUP_TIMEOUT_SEC),
    ("TELEMETRY_READY_TIMEOUT_SEC", TELEMETRY_TIMEOUT_SEC),
    ("CAMERA_READY_TIMEOUT_SEC", CAMERA_TIMEOUT_SEC),
    ("FRAME_REQUEST_TIMEOUT_SEC", FRAME_REQUEST_TIMEOUT_SEC),
):
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
if CAMERA_FAILURE_RESTART_THRESHOLD <= 0:
    raise ValueError("CAMERA_FAILURE_RESTART_THRESHOLD must be a positive integer")
if not math.isfinite(CAMERA_FAILURE_RESTART_MIN_WINDOW_SEC) or CAMERA_FAILURE_RESTART_MIN_WINDOW_SEC < 0:
    raise ValueError("CAMERA_FAILURE_RESTART_MIN_WINDOW_SEC must be finite and non-negative")


class BrowserServiceError(RuntimeError):
    """Raised when the browser-backed SDK bridge is unavailable."""


class BrowserConfigurationError(BrowserServiceError):
    """Raised when no usable Chrome or Chromium executable is configured."""


def resolve_chrome_executable() -> str:
    configured_path = os.getenv("CHROME_EXECUTABLE_PATH", "").strip()
    if configured_path:
        path = Path(configured_path).expanduser()
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
        raise BrowserConfigurationError(
            "CHROME_EXECUTABLE_PATH is not an executable file: "
            f"{configured_path}"
        )

    for executable in (
        "google-chrome-stable",
        "google-chrome",
        "chromium",
        "chromium-browser",
    ):
        detected_path = shutil.which(executable)
        if detected_path:
            return detected_path

    macos_path = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
    if macos_path.is_file() and os.access(macos_path, os.X_OK):
        return str(macos_path)

    raise BrowserConfigurationError(
        "Chrome/Chromium was not found. Set CHROME_EXECUTABLE_PATH to an "
        "executable returned by 'command -v google-chrome-stable', "
        "'command -v google-chrome', or 'command -v chromium'."
    )


class BrowserService:
    def __init__(self):
        self.browser = None
        self.page = None
        self._initialization_lock = asyncio.Lock()
        self.default_viewport = {"width": 3840, "height": 2160}
        self.initialization_stage = "NOT_STARTED"
        self.last_error = None
        self._consecutive_camera_failures = 0
        self._camera_failure_streak_started_monotonic = None

    def status(self) -> dict:
        return {
            "stage": self.initialization_stage,
            "browser_started": self.browser is not None,
            "page_started": self.page is not None,
            "last_error": self.last_error,
        }

    async def control_ready(self) -> bool:
        """Return whether the browser page can currently send RTM control."""

        return bool((await self.control_status()).get("ready"))

    async def control_status(self) -> dict:
        """Return RTM command transport readiness without command freshness.

        This intentionally does not depend on recent /control heartbeats.  A
        connected RTM transport must report ready before autonomy can send its
        first command; otherwise the SDK and autonomy can deadlock waiting on
        each other.
        """

        if self.page is None or self.initialization_stage != "READY":
            return {
                "ready": False,
                "reason": "CONTROL_PUBLISHER_NOT_INITIALIZED",
                "rtm_connected": False,
                "rtm_control_transport_ready": False,
            }
        try:
            state = await self.page.evaluate(
                """() => {
                  const rtmReady = window.rtm_ready === true;
                  const channelJoined = window.rtm_channel_state === "JOINED";
                  const sendMessageReady = typeof window.sendMessage === "function";
                  const rtcState =
                    (typeof client !== "undefined" && client && client.connectionState)
                    || window.rtc_connection_state
                    || "UNKNOWN";
                  const banned = Array.isArray(window.rtc_event_history)
                    && window.rtc_event_history.slice(-5).some(
                      (event) => event
                        && event.type === "connection-state-change"
                        && event.currentState === "DISCONNECTED"
                        && event.reason === "UID_BANNED"
                    );
                  let reason = "RTM_CONTROL_TRANSPORT_READY";
                  if (!sendMessageReady) {
                    reason = "CONTROL_PUBLISHER_NOT_INITIALIZED";
                  } else if (!rtmReady) {
                    reason = "RTM_NOT_CONNECTED";
                  } else if (!channelJoined) {
                    reason = "RTM_CHANNEL_NOT_JOINED";
                  } else if (banned) {
                    reason = "RTC_UID_BANNED";
                  }
                  const ready = rtmReady && channelJoined && sendMessageReady && !banned;
                  return {
                    ready,
                    reason,
                    rtm_connected: rtmReady,
                    rtm_control_transport_ready: ready,
                    rtm_connection_state: window.rtm_connection_state || "UNKNOWN",
                    rtm_channel_state: window.rtm_channel_state || "UNKNOWN",
                    rtm_last_error: window.rtm_last_error || null,
                    rtm_last_send_state: window.rtm_last_send_state || "NOT_SENT",
                    rtc_connected: rtcState === "CONNECTED",
                    rtc_connection_state: rtcState
                  };
                }"""
            )
            return state if isinstance(state, dict) else {"ready": False, "reason": "UNKNOWN"}
        except Exception:
            return {
                "ready": False,
                "reason": "CONTROL_TRANSPORT_DIAGNOSTICS_FAILED",
                "rtm_connected": False,
                "rtm_control_transport_ready": False,
            }

    async def diagnostics(self) -> dict:
        """Return non-sensitive browser/RTC/RTM readiness details."""

        result = self.status()
        if self.page is None:
            return result
        try:
            page_state = await self.page.evaluate(
                """() => {
                  const users =
                    (typeof remoteUsers !== "undefined" && remoteUsers)
                    ? remoteUsers
                    : {};
                  const channelUsers =
                    (typeof client !== "undefined" && client && client.remoteUsers)
                    ? client.remoteUsers
                    : [];
                  const frontUser = channelUsers.find(
                    (user) => String(user.uid) === "1000"
                  ) || users[1000];
                  const frontVideo = document.querySelector("#player-1000 video");
                  return {
                    rtmConnectionState: window.rtm_connection_state || "UNKNOWN",
                    rtmChannelState: window.rtm_channel_state || "UNKNOWN",
                    rtmLastError: window.rtm_last_error || null,
                    rtmLastSendState: window.rtm_last_send_state || "NOT_SENT",
                    rtmReady: window.rtm_ready === true,
                    telemetryPresent: window.rtm_data != null,
                    sendMessageReady: typeof window.sendMessage === "function",
                    rtcConnectionState:
                        (typeof client !== "undefined" && client && client.connectionState)
                        || window.rtc_connection_state
                        || "UNKNOWN",
                    remoteUserCount: channelUsers.length,
                    remoteUserIds: channelUsers.map((user) => String(user.uid)),
                    publishedRemoteUserCount: Object.keys(users).length,
                    publishedRemoteUserIds: Object.keys(users),
                    videoElementCount: document.querySelectorAll("video").length,
                    frontTrackReady: Boolean(
                        frontUser && frontUser.videoTrack
                        && frontUser.videoTrack.captureEnabled
                    ),
                    frontFramePresent:
                        Boolean(window.lastBase64Frames && window.lastBase64Frames[1000]),
                    frontVideoWidth: frontVideo ? frontVideo.videoWidth : 0,
                    frontVideoHeight: frontVideo ? frontVideo.videoHeight : 0,
                    rtcEventHistory: Array.isArray(window.rtc_event_history)
                        ? window.rtc_event_history.slice(-20)
                        : [],
                  };
                }"""
            )
        except NetworkError as exc:
            self.page = None
            self.last_error = f"diagnostics failed: {exc}"
            result.update(
                {
                    "page_started": False,
                    "last_error": self.last_error,
                    "diagnostics_error": str(exc),
                }
            )
            return result
        result["page"] = page_state
        return result

    async def initialize_browser(self):
        if self.browser and self.page:
            return

        async with self._initialization_lock:
            if self.browser and self.page:
                return

            self.last_error = None
            self.initialization_stage = "RESOLVING_CHROME"
            executable_path = resolve_chrome_executable()
            timeout_ms = int(STARTUP_TIMEOUT_SEC * 1000)
            try:
                self.initialization_stage = "LAUNCHING_CHROME"
                self.browser = await launch(
                    executablePath=executable_path,
                    headless=True,
                    handleSIGINT=False,
                    handleSIGTERM=False,
                    handleSIGHUP=False,
                    autoClose=False,
                    dumpio=os.getenv("BROWSER_DUMPIO", "false").lower() == "true",
                    args=[
                        "--ignore-certificate-errors",
                        "--no-sandbox",
                        "--autoplay-policy=no-user-gesture-required",
                        "--use-fake-ui-for-media-stream",
                        f"--window-size={self.default_viewport['width']},{self.default_viewport['height']}",
                    ],
                )
                self.initialization_stage = "OPENING_SDK_PAGE"
                self.page = await self.browser.newPage()
                await self.page.setViewport(self.default_viewport)
                await self.page.setExtraHTTPHeaders(
                    {"Accept-Language": "en-US,en;q=0.9"}
                )
                await self.page.goto(
                    "http://127.0.0.1:8000/sdk",
                    {"waitUntil": "domcontentloaded", "timeout": timeout_ms},
                )
                self.initialization_stage = "JOINING_RTC_RTM"
                await self.page.waitForSelector("#join", {"timeout": timeout_ms})
                await self.page.click("#join")
                await self.page.waitForFunction(
                    "typeof window.sendMessage === 'function'",
                    {"timeout": timeout_ms},
                )
                # basicRtm.js publishes sendMessage before its asynchronous
                # Agora login and channel join complete.  Returning READY at
                # that point races the first /control requests and produces
                # RTM error 102 (client not logged in).
                await self.page.waitForFunction(
                    "window.rtm_ready === true "
                    "&& window.rtm_channel_state === 'JOINED'",
                    {"timeout": timeout_ms},
                )
                await self.page.waitForFunction(
                    "typeof window.getLastBase64Frame === 'function'",
                    {"timeout": timeout_ms},
                )
                await self.page.waitForFunction(
                    "typeof window.initializeImageParams === 'function'",
                    {"timeout": timeout_ms},
                )
                await self.page.setViewport(self.default_viewport)
                call = f"""() => {{
                    window.initializeImageParams({{
                        imageFormat: "{FORMAT}",
                        imageQuality: {QUALITY}
                    }});
                }}"""
                await self.page.evaluate(call)
                self.initialization_stage = "READY"
            except PyppeteerTimeoutError as exc:
                stage = self.initialization_stage
                self.last_error = f"Timed out during {stage}"
                await self._close_browser_unlocked(preserve_status=True)
                raise BrowserServiceError(
                    f"Chrome SDK bridge timed out during {stage}. Check the "
                    "Hypercorn log, internet access, SDK token, and bot availability."
                ) from exc
            except BrowserServiceError:
                await self._close_browser_unlocked(preserve_status=True)
                raise
            except Exception as exc:
                stage = self.initialization_stage
                self.last_error = f"{type(exc).__name__} during {stage}: {exc}"
                await self._close_browser_unlocked(preserve_status=True)
                raise BrowserServiceError(
                    f"Chrome failed during {stage}. Set BROWSER_DUMPIO=true "
                    "to expose Chrome output, then restart the SDK."
                ) from exc

    async def take_screenshot(self, video_output_folder: str, elements: list):
        await self.initialize_browser()

        dimensions = await self.page.evaluate(
            """() => {
            return {
                width: Math.max(document.documentElement.scrollWidth, window.innerWidth),
                height: Math.max(document.documentElement.scrollHeight, window.innerHeight),
            }
        }"""
        )

        if (
            dimensions["width"] > self.default_viewport["width"]
            or dimensions["height"] > self.default_viewport["height"]
        ):
            await self.page.setViewport(dimensions)

        element_map = {"front": "#player-1000", "rear": "#player-1001", "map": "#map"}

        screenshots = {}
        for name in elements:
            if name in element_map:
                element_id = element_map[name]
                output_path = f"{video_output_folder}/{name}.png"
                element = await self.page.querySelector(element_id)
                if element:
                    start_time = time.time()  # Start time
                    await element.screenshot({"path": output_path})
                    end_time = time.time()  # End time
                    elapsed_time = (
                        end_time - start_time
                    ) * 1000  # Convert to milliseconds
                    print(f"Screenshot for {name} took {elapsed_time:.2f} ms")
                    screenshots[name] = output_path
                else:
                    print(f"Element {element_id} not found")
            else:
                print(f"Invalid element name: {name}")

        return screenshots

    async def data(self) -> dict:
        await self.initialize_browser()

        try:
            await self.page.waitForFunction(
                "window.rtm_data != null",
                {"timeout": int(TELEMETRY_TIMEOUT_SEC * 1000)},
            )
            bot_data = await self.page.evaluate(
                """() => {
            return window.rtm_data;
            }"""
            )
        except PyppeteerTimeoutError as exc:
            self.initialization_stage = "READY_NO_TELEMETRY"
            self.last_error = "RTM telemetry timeout"
            raise BrowserServiceError(
                "RTM telemetry did not arrive before the timeout. Confirm that "
                "the bot is online and assigned to this SDK token/BOT_SLUG."
            ) from exc
        except Exception as exc:
            self.initialization_stage = "READY_NO_TELEMETRY"
            self.last_error = f"RTM telemetry bridge error: {type(exc).__name__}"
            raise BrowserServiceError(
                "RTM telemetry bridge became unavailable. Reconnect the rover "
                "and confirm that its remote RTC/RTM user joined the channel."
            ) from exc
        self.initialization_stage = "READY"
        self.last_error = None
        return bot_data

    async def front(self, timeout_sec: float | None = None) -> str:
        await self.initialize_browser()
        try:
            front_frame = await self._wait_for_frame(
                1000,
                timeout_sec=CAMERA_TIMEOUT_SEC if timeout_sec is None else timeout_sec,
            )
        except BrowserServiceError:
            await self._register_camera_failure()
            raise
        if not front_frame:
            self.initialization_stage = "READY_NO_FRONT_CAMERA"
            self.last_error = "Front camera timeout"
            await self._register_camera_failure()
            raise BrowserServiceError(
                "Front camera did not publish a frame before the timeout. "
                "Confirm that the bot is online and its RTC video is connected."
            )
        self._consecutive_camera_failures = 0
        self._camera_failure_streak_started_monotonic = None
        self.initialization_stage = "READY"
        self.last_error = None
        return front_frame

    async def rear(self, timeout_sec: float | None = None) -> str:
        await self.initialize_browser()
        try:
            rear_frame = await self._wait_for_frame(
                1001,
                timeout_sec=CAMERA_TIMEOUT_SEC if timeout_sec is None else timeout_sec,
            )
        except BrowserServiceError:
            await self._register_camera_failure()
            raise
        if not rear_frame:
            self.initialization_stage = "READY_NO_REAR_CAMERA"
            self.last_error = "Rear camera timeout"
            await self._register_camera_failure()
            raise BrowserServiceError(
                "Rear camera did not publish a frame before the timeout."
            )
        self._consecutive_camera_failures = 0
        self._camera_failure_streak_started_monotonic = None
        self.initialization_stage = "READY"
        self.last_error = None
        return rear_frame

    async def _register_camera_failure(self) -> None:
        """Track repeated front()/rear() failures and restart a dead browser.

        initialize_browser() only checks that self.browser/self.page are set,
        not that the RTC session inside the page is still producing video, so
        a dropped WebRTC connection previously meant every future frame
        request failed forever. Restarting tears down the whole browser --
        including a currently-healthy RTM control channel -- so this must not
        fire on an ordinary short hiccup. Require both a failure count AND a
        minimum elapsed time since the streak began before actually
        restarting.
        """

        now = time.monotonic()
        if self._camera_failure_streak_started_monotonic is None:
            self._camera_failure_streak_started_monotonic = now
        self._consecutive_camera_failures += 1
        streak_elapsed = now - self._camera_failure_streak_started_monotonic
        if (
            self._consecutive_camera_failures < CAMERA_FAILURE_RESTART_THRESHOLD
            or streak_elapsed < CAMERA_FAILURE_RESTART_MIN_WINDOW_SEC
        ):
            return
        self._consecutive_camera_failures = 0
        self._camera_failure_streak_started_monotonic = None
        await self.close_browser()

    async def _wait_for_frame(
        self,
        uid: int,
        timeout_sec: float = CAMERA_TIMEOUT_SEC,
    ) -> str | None:
        """Await the async browser frame API until a real data URL arrives."""

        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            try:
                result = await self.page.evaluate(
                    """async (uid) => {
                      try {
                        const frame = await window.getLastBase64Frame(uid);
                        return typeof frame === "string" && frame.length > 0
                          ? frame
                          : null;
                      } catch (error) {
                        return null;
                      }
                    }""",
                    uid,
                )
            except Exception as exc:
                raise BrowserServiceError(
                    "RTC camera bridge became unavailable. Reconnect the rover "
                    "and confirm that its remote video user joined the channel."
                ) from exc
            if result:
                return result
            await asyncio.sleep(0.1)
        return None

    async def send_message(self, message: dict):
        await self.initialize_browser()

        try:
            ready = await self.page.evaluate(
                "() => window.rtm_ready === true "
                "&& window.rtm_channel_state === 'JOINED'"
            )
            if not ready:
                diagnostics = await self.diagnostics()
                page = diagnostics.get("page", {})
                raise BrowserServiceError(
                    "RTM control bridge is not ready "
                    f"(connection={page.get('rtmConnectionState')}, "
                    f"channel={page.get('rtmChannelState')})."
                )
            await self.page.evaluate(
                """async (message) => {
                    return await window.sendMessage(message);
                }""",
                message,
            )
            return {
                "result": "COMMAND_PUBLISHED",
                "rtm_control_transport_ready": True,
            }
        except BrowserServiceError:
            raise
        except Exception as exc:
            raise BrowserServiceError(
                "RTM rejected the rover control command. Check "
                "/connection-diagnostics and reconnect the mission bridge."
            ) from exc

    async def speak(self, audio_url: str):
        await self.initialize_browser()

        result = await self.page.evaluate(
            """async (audioUrl) => {
                return await window.playAudioToRover(audioUrl);
            }""",
            audio_url,
        )

        return result

    async def close_browser(self, preserve_status: bool = False):
        async with self._initialization_lock:
            await self._close_browser_unlocked(preserve_status=preserve_status)

    async def _close_browser_unlocked(self, preserve_status: bool = False):
        if self.browser:
            try:
                await self.browser.close()
            finally:
                self.browser = None
                self.page = None
        if not preserve_status:
            self.initialization_stage = "NOT_STARTED"
            self.last_error = None
