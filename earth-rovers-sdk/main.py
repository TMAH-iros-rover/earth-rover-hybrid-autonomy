import base64
import functools
import json
import logging
import math
import os
import re
import time
import asyncio
from pathlib import Path

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Literal

from browser_service import BrowserService, BrowserServiceError, FRAME_REQUEST_TIMEOUT_SEC
from rtm_client import RtmClient
from tts_service import generate_speech

load_dotenv()

# Configurar el logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("http_logger")

app = FastAPI()
LOCAL_MISSION_STATE_PATH = Path(
    os.getenv("LOCAL_MISSION_STATE_PATH", "/tmp/earth_rover_sdk_local_mission_state.json")
)


@app.exception_handler(BrowserServiceError)
async def browser_service_error_handler(
    _request: Request, exc: BrowserServiceError
):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.on_event("shutdown")
async def close_browser_service() -> None:
    global _control_generation, _control_watchdog_task
    _control_generation += 1
    if _control_watchdog_task and not _control_watchdog_task.done():
        _control_watchdog_task.cancel()
    if browser_service.page is not None:
        try:
            await _send_control_message(
                {"linear": 0.0, "angular": 0.0, "lamp": 0}
            )
        except Exception:
            logger.exception("SDK shutdown stop command failed")
    await browser_service.close_browser()


# Middleware
def log_request(method):
    @functools.wraps(method)
    def wrapper(*args, **kwargs):
        debug_mode = os.getenv("DEBUG") == "true"
        if debug_mode:
            params = kwargs.get("params", {})
            json_data = kwargs.get("json", {})
            data = kwargs.get("data", {})
            logger.info(
                "=== External Request ===\nMethod: %s\nURL: %s\nParams: %s\nJSON: %s\nData: %s",
                method.__name__.upper(),
                args[0],
                params,
                json_data,
                data,
            )

        response = method(*args, **kwargs)

        if debug_mode:
            logger.info(
                "=== External Response ===\nStatus Code: %s\nResponse: %s",
                response.status_code,
                response.text,
            )

        return response

    return wrapper


requests.get = log_request(requests.get)
requests.post = log_request(requests.post)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRODOBOTS_API_URL = os.getenv(
    "FRODOBOTS_API_URL", "https://frodobots-web-api.onrender.com/api/v1"
)


class AuthResponse(BaseModel):
    CHANNEL_NAME: str
    RTC_TOKEN: str
    RTM_TOKEN: str
    USERID: int
    APP_ID: str
    BOT_UID: str


# In-memory storage for the response
auth_response_data = {}
checkpoints_list_data = {}
selected_mission_slug = os.getenv("MISSION_SLUG", "").strip()
active_session_mode = None
active_mission_slug = None

app.mount("/static", StaticFiles(directory="./static"), name="static")

browser_service = BrowserService()

CONTROL_WATCHDOG_TIMEOUT_SEC = float(
    os.getenv("CONTROL_WATCHDOG_TIMEOUT_SEC", "0.75")
)
if not math.isfinite(CONTROL_WATCHDOG_TIMEOUT_SEC) or CONTROL_WATCHDOG_TIMEOUT_SEC <= 0:
    raise ValueError("CONTROL_WATCHDOG_TIMEOUT_SEC must be finite and positive")

_control_lock = asyncio.Lock()
_control_generation = 0
_control_watchdog_task = None
_last_control_command_epoch = None
_last_control_command_monotonic = None
_control_watchdog_active = False


def validate_control_command(body):
    """Return a bounded rover command or reject malformed input."""

    command = body.get("command") if isinstance(body, dict) else None
    if not isinstance(command, dict):
        raise HTTPException(status_code=400, detail="Command not provided")
    normalized = {}
    for name in ("linear", "angular"):
        value = command.get(name, 0.0)
        if isinstance(value, bool):
            raise HTTPException(status_code=400, detail=f"{name} must be numeric")
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=400, detail=f"{name} must be numeric"
            ) from exc
        if not math.isfinite(value) or not -1.0 <= value <= 1.0:
            raise HTTPException(
                status_code=400, detail=f"{name} must be finite and in [-1, 1]"
            )
        normalized[name] = value
    lamp = command.get("lamp", 0)
    if isinstance(lamp, bool):
        lamp = int(lamp)
    if lamp not in (0, 1):
        raise HTTPException(status_code=400, detail="lamp must be 0 or 1")
    normalized["lamp"] = int(lamp)
    return normalized


def _command_has_motion(command):
    return bool(command["linear"] or command["angular"])


def current_mission_slug() -> str:
    return selected_mission_slug


def normalize_mission_slug(value) -> str:
    slug = str(value or "").strip()
    if not slug or not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", slug):
        raise HTTPException(
            status_code=400,
            detail="mission_slug must use 1-100 letters, numbers, '-' or '_'",
        )
    return slug


def is_bot_status_transition_error(exc: HTTPException) -> bool:
    """Return whether start_ride rejected a still-connected direct bot."""

    if exc.status_code != 422 or not isinstance(exc.detail, dict):
        return False
    upstream_error = str(exc.detail.get("upstream_error", "")).lower()
    return "bot status is invalid" in upstream_error


def clear_local_mission_state() -> None:
    try:
        LOCAL_MISSION_STATE_PATH.unlink()
    except FileNotFoundError:
        return
    except Exception:
        logger.warning("Failed to clear local mission state")


def write_local_mission_state(latest_scanned_checkpoint) -> None:
    try:
        LOCAL_MISSION_STATE_PATH.write_text(
            json.dumps(
                {
                    "mission_slug": current_mission_slug(),
                    "latest_scanned_checkpoint": latest_scanned_checkpoint,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    except Exception:
        logger.warning("Failed to persist local mission state")


def read_local_mission_latest():
    if active_session_mode != "local_mission":
        return None
    try:
        payload = json.loads(LOCAL_MISSION_STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None
    if payload.get("mission_slug") != current_mission_slug():
        return None
    return payload.get("latest_scanned_checkpoint")


async def request_mission_slug(request: Request) -> str:
    try:
        body = await request.json()
    except Exception:
        body = {}
    value = body.get("mission_slug") if isinstance(body, dict) else None
    return normalize_mission_slug(value or current_mission_slug())


async def reset_local_bridge(send_stop: bool = True) -> None:
    """Safely clear the current browser/auth session before changing modes."""

    global auth_response_data, checkpoints_list_data, _control_generation
    global active_session_mode, active_mission_slug
    _control_generation += 1
    if send_stop and browser_service.page is not None:
        try:
            await _send_control_message(
                {"linear": 0.0, "angular": 0.0, "lamp": 0}
            )
        except Exception:
            logger.warning("Bridge reset stop command failed")
    await browser_service.close_browser()
    auth_response_data = {}
    checkpoints_list_data = {}
    active_session_mode = None
    active_mission_slug = None
    clear_local_mission_state()


async def _send_control_message(command):
    async with _control_lock:
        return await browser_service.send_message(command)


async def _control_watchdog(generation):
    """Send an explicit stop when command heartbeats cease."""

    global _control_generation, _control_watchdog_active
    await asyncio.sleep(CONTROL_WATCHDOG_TIMEOUT_SEC)
    if generation != _control_generation:
        return
    stop = {"linear": 0.0, "angular": 0.0, "lamp": 0}
    try:
        await _send_control_message(stop)
    except Exception:
        logger.exception("Control watchdog failed to send stop command")
        return
    if generation == _control_generation:
        _control_generation += 1
        _control_watchdog_active = False
        logger.warning("Control heartbeat expired; rover stop command sent")


def _schedule_control_watchdog(command):
    global _control_generation, _control_watchdog_task
    global _last_control_command_epoch, _last_control_command_monotonic
    global _control_watchdog_active
    _control_generation += 1
    _last_control_command_epoch = time.time()
    _last_control_command_monotonic = time.monotonic()
    if _command_has_motion(command):
        _control_watchdog_task = asyncio.create_task(
            _control_watchdog(_control_generation)
        )
        _control_watchdog_active = True
    else:
        _control_watchdog_task = None
        _control_watchdog_active = False


def _age_from_monotonic(value):
    if value is None:
        return None
    age = time.monotonic() - value
    return age if math.isfinite(age) and age >= 0.0 else None


def mission_status_payload(control_bridge_status: dict | bool | None = None):
    mission_configured = bool(current_mission_slug())
    mission_active = bool(auth_response_data) and active_session_mode in {
        "mission",
        "local_mission",
    }
    if isinstance(control_bridge_status, bool):
        control_bridge_ready = control_bridge_status
        control_bridge_status = {
            "ready": control_bridge_ready,
            "reason": (
                "RTM_CONTROL_TRANSPORT_READY"
                if control_bridge_ready
                else "CONTROL_BRIDGE_STATUS_UNAVAILABLE"
            ),
        }
    elif control_bridge_status is None:
        control_bridge_ready = bool(
            browser_service.page is not None
            and browser_service.initialization_stage == "READY"
        )
        control_bridge_status = {
            "ready": control_bridge_ready,
            "reason": (
                "BROWSER_READY_FALLBACK"
                if control_bridge_ready
                else "CONTROL_PUBLISHER_NOT_INITIALIZED"
            ),
        }
    else:
        control_bridge_ready = bool(control_bridge_status.get("ready"))
    checkpoints = checkpoints_list_data.get("checkpoints_list", [])
    if not isinstance(checkpoints, list):
        checkpoints = []
    latest_scanned_checkpoint = current_cached_latest_checkpoint()
    last_command_age = _age_from_monotonic(_last_control_command_monotonic)
    command_fresh = (
        last_command_age is not None
        and last_command_age <= CONTROL_WATCHDOG_TIMEOUT_SEC
    )
    return {
        "mission_configured": mission_configured,
        "mission_active": mission_active,
        "cloud_mission_active": bool(auth_response_data)
        and active_session_mode == "mission",
        "local_mission_active": bool(auth_response_data)
        and active_session_mode == "local_mission",
        "rover_connected": bool(auth_response_data),
        "session_mode": active_session_mode,
        "operation_mode": "mission" if mission_configured else "direct_bot",
        "start_mission_required": mission_configured,
        "camera_and_telemetry_allowed": bool(auth_response_data),
        "control_bridge_ready": control_bridge_ready,
        "rtc_connected": bool(control_bridge_status.get("rtc_connected")),
        "rtm_connected": bool(control_bridge_status.get("rtm_connected")),
        "rtm_control_transport_ready": bool(
            control_bridge_status.get("rtm_control_transport_ready", control_bridge_ready)
        ),
        "control_bridge_reason": control_bridge_status.get("reason"),
        "control_command_fresh": command_fresh,
        "control_watchdog_active": _control_watchdog_active,
        "last_control_command_age_sec": last_command_age,
        "last_control_command_timestamp": _last_control_command_epoch,
        "checkpoint_count": len(checkpoints),
        "latest_scanned_checkpoint": latest_scanned_checkpoint,
        "checkpoints_loaded": bool(checkpoints_list_data),
        "control_watchdog_timeout_sec": CONTROL_WATCHDOG_TIMEOUT_SEC,
        "server_timestamp": time.time(),
        "timestamp": time.time(),
    }


async def auth_common():
    global auth_response_data, active_session_mode, active_mission_slug
    if isinstance(auth_response_data, dict) and auth_response_data:
        return auth_response_data
    env_tokens = get_env_tokens()

    if env_tokens:
        auth_response_data = env_tokens
        active_session_mode = "mission" if current_mission_slug() else "direct_bot"
        active_mission_slug = (
            current_mission_slug() if active_session_mode == "mission" else None
        )
        return auth_response_data

    auth_header = os.getenv("SDK_API_TOKEN")
    bot_slug = os.getenv("BOT_SLUG")
    mission_slug = current_mission_slug()

    if not auth_header:
        raise HTTPException(
            status_code=500, detail="Authorization header not configured"
        )
    if not bot_slug:
        raise HTTPException(status_code=500, detail="Bot name not configured")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {auth_header}",
    }

    if mission_slug:
        response_data = await start_ride(headers, bot_slug, mission_slug)
    else:
        response_data = await retrieve_tokens(headers, bot_slug)

    new_auth_response_data = {
        "CHANNEL_NAME": response_data.get("CHANNEL_NAME"),
        "RTC_TOKEN": response_data.get("RTC_TOKEN"),
        "RTM_TOKEN": response_data.get("RTM_TOKEN"),
        "USERID": response_data.get("USERID"),
        "APP_ID": response_data.get("APP_ID"),
        "BOT_UID": response_data.get("BOT_UID"),
        "SPECTATOR_USERID": response_data.get("SPECTATOR_USERID"),
        "SPECTATOR_RTC_TOKEN": response_data.get("SPECTATOR_RTC_TOKEN"),
        "BOT_TYPE": response_data.get("BOT_TYPE", "mini"),
    }
    missing_token_fields = [
        name
        for name in ("CHANNEL_NAME", "RTC_TOKEN", "RTM_TOKEN", "USERID", "APP_ID", "BOT_UID")
        if not new_auth_response_data.get(name)
    ]
    if missing_token_fields:
        raise HTTPException(
            status_code=502,
            detail="FrodoBots returned incomplete RTC/RTM credentials",
        )
    auth_response_data = new_auth_response_data
    active_session_mode = "mission" if mission_slug else "direct_bot"
    active_mission_slug = mission_slug if mission_slug else None

    return auth_response_data


def get_env_tokens():
    channel_name = os.getenv("CHANNEL_NAME")
    rtc_token = os.getenv("RTC_TOKEN")
    rtm_token = os.getenv("RTM_TOKEN")
    userid = os.getenv("USERID")
    app_id = os.getenv("APP_ID")
    bot_uid = os.getenv("BOT_UID")

    if all([channel_name, rtc_token, rtm_token, userid, app_id, bot_uid]):
        return {
            "CHANNEL_NAME": channel_name,
            "RTC_TOKEN": rtc_token,
            "RTM_TOKEN": rtm_token,
            "USERID": userid,
            "APP_ID": app_id,
            "BOT_UID": bot_uid,
        }
    return None


async def start_ride(headers, bot_slug, mission_slug):
    start_ride_data = {"bot_slug": bot_slug, "mission_slug": mission_slug}
    start_ride_response = requests.post(
        FRODOBOTS_API_URL + "/sdk/start_ride",
        headers=headers,
        json=start_ride_data,
        timeout=15,
    )

    if start_ride_response.status_code != 200:
        try:
            upstream = start_ride_response.json()
        except ValueError:
            upstream = {}
        if not isinstance(upstream, dict):
            upstream = {}
        upstream_error = next(
            (
                upstream.get(name)
                for name in ("error", "message", "detail")
                if upstream.get(name)
            ),
            None,
        )
        raise HTTPException(
            status_code=start_ride_response.status_code,
            detail={
                "message": "Failed to start mission",
                "upstream_status": start_ride_response.status_code,
                "upstream_error": str(upstream_error)[:300]
                if upstream_error is not None
                else "No error detail returned by FrodoBots",
            },
        )

    return start_ride_response.json()


async def end_ride(headers, bot_slug, mission_slug):
    end_ride_data = {"bot_slug": bot_slug, "mission_slug": mission_slug}
    end_ride_response = requests.post(
        FRODOBOTS_API_URL + "/sdk/end_ride",
        headers=headers,
        json=end_ride_data,
        timeout=15,
    )

    if end_ride_response.status_code != 200:
        raise HTTPException(
            status_code=end_ride_response.status_code, detail="Failed to end mission"
        )

    return end_ride_response.json()


async def retrieve_tokens(headers, bot_slug):
    data = {"bot_slug": bot_slug}
    response = requests.post(
        FRODOBOTS_API_URL + "/sdk/token", headers=headers, json=data, timeout=15
    )

    if response.status_code != 200:
        raise HTTPException(
            status_code=response.status_code, detail="Failed to retrieve tokens"
        )

    return response.json()


async def need_start_mission():
    if not current_mission_slug():
        return
    if auth_response_data:
        return
    raise HTTPException(
        status_code=400, detail="Call /start-mission endpoint to start a mission"
    )


@app.post("/checkpoints-list")
@app.get("/checkpoints-list")
async def checkpoints():
    await need_start_mission()
    await get_checkpoints_list()
    return JSONResponse(content=checkpoints_list_data)


async def get_checkpoints_list():
    global checkpoints_list_data
    auth_header = os.getenv("SDK_API_TOKEN")
    bot_slug = os.getenv("BOT_SLUG")
    mission_slug = current_mission_slug()

    if not mission_slug:
        return

    if not auth_header:
        raise HTTPException(
            status_code=500, detail="Authorization header not configured"
        )
    if not bot_slug:
        raise HTTPException(status_code=500, detail="Bot name not configured")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {auth_header}",
    }

    data = {"bot_slug": bot_slug, "mission_slug": mission_slug}

    response = requests.post(
        FRODOBOTS_API_URL + "/sdk/checkpoints_list",
        headers=headers,
        json=data,
        timeout=15,
    )

    if response.status_code != 200:
        raise HTTPException(
            status_code=response.status_code,
            detail="Failed to retrieve checkpoints list",
        )

    # advance_cached_checkpoint() tracks progress from /checkpoint-reached
    # only in this process's memory -- the cloud's checkpoints_list response
    # doesn't carry it back. Overwriting checkpoints_list_data wholesale on
    # every refresh (e.g. the dashboard's periodic poll while a mission is
    # active) previously reset latest_scanned_checkpoint to whatever (or
    # nothing) the cloud response contains, making a rover that had already
    # reported checkpoint 1 look like it was back at 0 and confusing every
    # consumer of /mission-route. Never let a refresh regress progress.
    previous_latest = _safe_int(checkpoints_list_data.get("latest_scanned_checkpoint"))
    checkpoints_list_data = response.json()
    if previous_latest is not None:
        cloud_latest = _safe_int(checkpoints_list_data.get("latest_scanned_checkpoint"))
        if cloud_latest is None or cloud_latest < previous_latest:
            checkpoints_list_data["latest_scanned_checkpoint"] = previous_latest
    return checkpoints_list_data


def _safe_int(value) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


async def auth():
    await auth_common()
    if not checkpoints_list_data:
        await get_checkpoints_list()
    return JSONResponse(
        content={
            "auth_response_data": auth_response_data,
            "checkpoints_list_data": checkpoints_list_data,
        }
    )


@app.post("/select-mission")
async def select_mission(request: Request):
    """Select a mission at runtime and load its checkpoint route."""

    global selected_mission_slug, checkpoints_list_data
    mission_slug = await request_mission_slug(request)
    if (
        active_session_mode in {"mission", "local_mission"}
        and active_mission_slug
        and active_mission_slug != mission_slug
    ):
        raise HTTPException(
            status_code=409,
            detail=(
                f"Mission {active_mission_slug} is active; end it before "
                f"selecting {mission_slug}"
            ),
        )
    selected_mission_slug = mission_slug
    checkpoints_list_data = {}
    await get_checkpoints_list()
    return JSONResponse(
        content={
            "message": "Mission selected",
            "checkpoints": checkpoints_list_data,
        }
    )


@app.post("/start-mission")
async def start_mission(request: Request):
    global auth_response_data, active_session_mode, active_mission_slug
    global selected_mission_slug, checkpoints_list_data
    mission_slug = await request_mission_slug(request)
    if (
        active_session_mode in {"mission", "local_mission"}
        and active_mission_slug
        and active_mission_slug != mission_slug
    ):
        raise HTTPException(
            status_code=409,
            detail=(
                f"Mission {active_mission_slug} is active; end it before "
                f"starting {mission_slug}"
            ),
        )
    required_env_vars = ["SDK_API_TOKEN", "BOT_SLUG"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]

    if missing_vars:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required environment variables: {', '.join(missing_vars)}",
        )

    needs_mission_session = not (
        active_session_mode == "mission"
        and active_mission_slug == mission_slug
        and bool(auth_response_data)
    )
    if needs_mission_session:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('SDK_API_TOKEN')}",
        }
        previous_auth = (
            dict(auth_response_data)
            if isinstance(auth_response_data, dict) and auth_response_data
            else {}
        )
        previous_mode = active_session_mode
        previous_active_mission_slug = active_mission_slug
        previous_checkpoints = dict(checkpoints_list_data)
        bridge_released = False
        try:
            # If the cloud refuses start_ride because the bot is already in a
            # non-startable status, keep the working direct RTC/RTM bridge and
            # run the selected checkpoint route locally. Closing the bridge
            # here causes the live stream to drop and can fail while rejoining.
            try:
                response_data = await start_ride(
                    headers, os.getenv("BOT_SLUG"), mission_slug
                )
            except HTTPException as exc:
                if not (
                    previous_mode == "direct_bot"
                    and previous_auth
                    and is_bot_status_transition_error(exc)
                ):
                    raise
                selected_mission_slug = mission_slug
                auth_response_data = previous_auth
                active_session_mode = "local_mission"
                active_mission_slug = mission_slug
                checkpoints_list_data = previous_checkpoints
                if not checkpoints_list_data:
                    await get_checkpoints_list()
                write_local_mission_state(
                    checkpoints_list_data.get("latest_scanned_checkpoint", 0)
                )
                if browser_service.page is None:
                    await browser_service.initialize_browser()
                return JSONResponse(
                    status_code=200,
                    content={
                        "message": (
                            "Local mission started; FrodoBots cloud "
                            "start_ride rejected the bot status"
                        ),
                        "local_mission": True,
                        "cloud_mission": False,
                        "upstream_error": exc.detail,
                        "checkpoints_list": checkpoints_list_data,
                    },
                )
        except Exception:
            if bridge_released and previous_auth:
                auth_response_data = previous_auth
                active_session_mode = previous_mode
                active_mission_slug = previous_active_mission_slug
                checkpoints_list_data = previous_checkpoints
                selected_mission_slug = mission_slug
                try:
                    await browser_service.initialize_browser()
                except Exception:
                    logger.exception(
                        "Failed to restore direct rover bridge after mission transition"
                    )
            raise
        new_auth = {
            "CHANNEL_NAME": response_data.get("CHANNEL_NAME"),
            "RTC_TOKEN": response_data.get("RTC_TOKEN"),
            "RTM_TOKEN": response_data.get("RTM_TOKEN"),
            "USERID": response_data.get("USERID"),
            "APP_ID": response_data.get("APP_ID"),
            "BOT_UID": response_data.get("BOT_UID"),
            "SPECTATOR_USERID": response_data.get("SPECTATOR_USERID"),
            "SPECTATOR_RTC_TOKEN": response_data.get("SPECTATOR_RTC_TOKEN"),
            "BOT_TYPE": response_data.get("BOT_TYPE", "mini"),
        }
        missing_token_fields = [
            name
            for name in (
                "CHANNEL_NAME", "RTC_TOKEN", "RTM_TOKEN", "USERID", "APP_ID", "BOT_UID"
            )
            if not new_auth.get(name)
        ]
        if missing_token_fields:
            raise HTTPException(
                status_code=502,
                detail="FrodoBots returned incomplete RTC/RTM credentials",
            )
        if not bridge_released:
            await reset_local_bridge()
        selected_mission_slug = mission_slug
        auth_response_data = new_auth
        active_session_mode = "mission"
        active_mission_slug = mission_slug
        checkpoints_list_data = {}
    else:
        selected_mission_slug = mission_slug
    if not checkpoints_list_data:
        await get_checkpoints_list()
    # Do not report a successful Start Mission transition until the hidden
    # browser has completed its RTC/RTM login. The autonomy process polls
    # mission-status concurrently and uses control_bridge_ready as its gate.
    await browser_service.initialize_browser()
    return JSONResponse(
        status_code=200,
        content={
            "message": "Mission started successfully",
            "checkpoints_list": checkpoints_list_data,
        },
    )


@app.post("/connect-rover")
async def connect_rover():
    """Connect a rover directly without starting a tracked mission."""

    global auth_response_data, active_session_mode
    missing_vars = [
        name for name in ("SDK_API_TOKEN", "BOT_SLUG") if not os.getenv(name)
    ]
    if missing_vars:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required environment variables: {', '.join(missing_vars)}",
        )
    if active_session_mode in {"mission", "local_mission"}:
        raise HTTPException(
            status_code=409,
            detail="A mission is active; end it before reconnecting directly",
        )
    if not auth_response_data:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('SDK_API_TOKEN')}",
        }
        response_data = await retrieve_tokens(headers, os.getenv("BOT_SLUG"))
        auth_response_data = {
            "CHANNEL_NAME": response_data.get("CHANNEL_NAME"),
            "RTC_TOKEN": response_data.get("RTC_TOKEN"),
            "RTM_TOKEN": response_data.get("RTM_TOKEN"),
            "USERID": response_data.get("USERID"),
            "APP_ID": response_data.get("APP_ID"),
            "BOT_UID": response_data.get("BOT_UID"),
            "SPECTATOR_USERID": response_data.get("SPECTATOR_USERID"),
            "SPECTATOR_RTC_TOKEN": response_data.get("SPECTATOR_RTC_TOKEN"),
            "BOT_TYPE": response_data.get("BOT_TYPE", "mini"),
        }
        active_session_mode = "direct_bot"
    await browser_service.initialize_browser()
    diagnostics = await browser_service.diagnostics()
    page = diagnostics.get("page", {})
    stream_ready = bool(
        page.get("remoteUserCount", 0)
        and page.get("telemetryPresent")
        and (page.get("frontTrackReady") or page.get("frontFramePresent"))
    )
    return JSONResponse(
        content={
            "message": (
                "Rover connected successfully"
                if stream_ready
                else "SDK bridge connected; waiting for the rover RTC/RTM stream"
            ),
            "operation_mode": "direct_bot",
            "connection_state": "ready" if stream_ready else "waiting_for_rover",
            "remote_user_count": int(page.get("remoteUserCount", 0)),
            "telemetry_ready": bool(page.get("telemetryPresent")),
            "front_camera_ready": bool(
                page.get("frontTrackReady") or page.get("frontFramePresent")
            ),
        }
    )


@app.post("/disconnect-rover")
async def disconnect_rover():
    """Close the local direct-bot bridge without ending a cloud mission."""

    if active_session_mode in {"mission", "local_mission"}:
        raise HTTPException(
            status_code=409,
            detail="A mission is active; use /end-mission instead",
        )
    await reset_local_bridge()
    return JSONResponse(
        content={
            "message": "Rover disconnected",
            "operation_mode": "direct_bot",
        }
    )


@app.get("/mission-status")
async def mission_status():
    """Return non-sensitive local mission state without starting a mission."""

    return JSONResponse(
        content=mission_status_payload(
            control_bridge_status=await browser_service.control_status()
        )
    )


@app.get("/mission-route")
async def mission_route():
    """Return only the already-loaded route without cloud authentication.

    This endpoint is intentionally side-effect free so read-only perception
    processes can consume checkpoint geometry without starting a mission or
    refreshing FrodoBots state.
    """

    checkpoints = checkpoints_list_data.get("checkpoints_list", [])
    if not isinstance(checkpoints, list):
        checkpoints = []
    return JSONResponse(
        content={
            "checkpoints_list": checkpoints,
            "latest_scanned_checkpoint": current_cached_latest_checkpoint(),
            "mission_active": active_session_mode in {"mission", "local_mission"}
            and bool(auth_response_data),
            "cloud_mission_active": active_session_mode == "mission"
            and bool(auth_response_data),
            "local_mission_active": active_session_mode == "local_mission"
            and bool(auth_response_data),
            "route_loaded": bool(checkpoints),
        }
    )


@app.get("/browser-status")
async def browser_status():
    """Report local SDK bridge state without starting Chrome or the bot link."""

    return JSONResponse(content=browser_service.status())


@app.get("/connection-diagnostics")
async def connection_diagnostics():
    """Return non-sensitive auth and browser channel readiness."""

    required_auth_fields = (
        "CHANNEL_NAME",
        "RTC_TOKEN",
        "RTM_TOKEN",
        "USERID",
        "APP_ID",
        "BOT_UID",
    )
    safe_auth = auth_response_data if isinstance(auth_response_data, dict) else {}
    return JSONResponse(
        content={
            "auth_fields_present": {
                name: bool(safe_auth.get(name))
                for name in required_auth_fields
            },
            "bot_type": safe_auth.get("BOT_TYPE"),
            "browser": await browser_service.diagnostics(),
            "control_bridge": await browser_service.control_status(),
        }
    )


@app.post("/end-mission")
async def end_mission(request: Request):
    global active_session_mode, selected_mission_slug
    required_env_vars = ["SDK_API_TOKEN", "BOT_SLUG"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]

    if missing_vars:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required environment variables: {', '.join(missing_vars)}",
        )

    auth_header = os.getenv("SDK_API_TOKEN")
    bot_slug = os.getenv("BOT_SLUG")
    mission_slug = await request_mission_slug(request)
    if not mission_slug:
        raise HTTPException(status_code=400, detail="No mission selected")
    if active_mission_slug and active_mission_slug != mission_slug:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Mission {active_mission_slug} is active; refusing to end "
                f"different mission {mission_slug}"
            ),
        )
    selected_mission_slug = mission_slug

    if active_session_mode == "local_mission":
        await reset_local_bridge(send_stop=True)
        return JSONResponse(
            content={"message": "Local mission ended successfully"}
        )

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {auth_header}",
    }

    try:
        end_ride_response = await end_ride(headers, bot_slug, mission_slug)
        await reset_local_bridge(send_stop=False)
        return JSONResponse(content={"message": "Mission ended successfully"})
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to end mission: {str(e)}")


async def render_index_html(is_spectator: bool):
    await need_start_mission()
    if not auth_response_data:
        await auth()

    token_type: Literal["SPECTATOR_", ""] = "SPECTATOR_" if is_spectator else ""

    template_vars = {
        "appid": auth_response_data.get("APP_ID", ""),
        "rtc_token": auth_response_data.get(f"{token_type}RTC_TOKEN", ""),
        "rtm_token": "" if is_spectator else auth_response_data.get("RTM_TOKEN", ""),
        "channel": auth_response_data.get("CHANNEL_NAME", ""),
        "uid": auth_response_data.get(f"{token_type}USERID", ""),
        "bot_uid": auth_response_data.get("BOT_UID", ""),
        "checkpoints_list": json.dumps(
            checkpoints_list_data.get("checkpoints_list", [])
        ),
        "map_zoom_level": os.getenv("MAP_ZOOM_LEVEL", "18"),
    }

    with open("index.html", "r", encoding="utf-8") as file:
        html_content = file.read()

    for key, value in template_vars.items():
        html_content = html_content.replace(f"{{{{ {key} }}}}", str(value))

    return HTMLResponse(content=html_content, status_code=200)


@app.get("/")
async def get_index(request: Request):
    return await render_index_html(is_spectator=True)


@app.get("/dashboard")
async def mission_dashboard():
    """Serve mission controls without requiring an active mission."""

    return FileResponse("static/mission_dashboard.html")


@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)


async def _local_service_json(url: str, offline_payload: dict, timeout: float = 0.2) -> dict:
    # These same-origin proxies are polled by the dashboard as often as every
    # 250ms. A synchronous requests call here blocks this process's single
    # event loop for up to `timeout`, which starves every other concurrent
    # request this server is handling -- including the SAM-TP shadow
    # process's own GET /front frame fetches, which then time out and fall
    # back to a stale cached frame (observed as SAM-TP getting stuck in
    # STALE_FRAME while the dashboard tab was open).
    try:
        response = await asyncio.to_thread(requests.get, url, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
        return payload if isinstance(payload, dict) else offline_payload
    except Exception:
        return offline_payload


@app.get("/sam-tp-status")
async def sam_tp_status_proxy():
    """Same-origin proxy to avoid browser console noise when SAM-TP is offline."""

    return JSONResponse(
        content=await _local_service_json(
            "http://127.0.0.1:8001/status",
            {
                "service": "sam-tp-shadow",
                "ready": False,
                "state": "OFFLINE",
                "command_transmitted": False,
            },
        )
    )


@app.get("/sam-tp-overlay.jpg")
async def sam_tp_overlay_proxy():
    try:
        response = await asyncio.to_thread(
            requests.get, "http://127.0.0.1:8001/overlay.jpg", timeout=0.5
        )
        response.raise_for_status()
        return Response(content=response.content, media_type="image/jpeg")
    except Exception:
        return JSONResponse(
            status_code=503,
            content={"detail": "SAM-TP overlay is offline"},
        )


@app.get("/autonomy-status")
async def autonomy_status_proxy():
    """Same-origin proxy to avoid browser console noise when autonomy is offline."""

    return JSONResponse(
        content=await _local_service_json(
            "http://127.0.0.1:8002/status",
            {
                "service": "mission1-autonomy",
                "armed": False,
                "state": "OFFLINE",
                "command_transmitted": False,
                "linear": 0.0,
                "angular": 0.0,
                "reason": "Start scripts/run_mission1_autonomy.sh",
            },
        )
    )


async def _local_service_post(url: str, offline_detail: str, timeout: float = 0.5) -> JSONResponse:
    try:
        response = await asyncio.to_thread(requests.post, url, timeout=timeout)
        if response.content:
            payload = response.json()
        else:
            payload = {}
        if not response.ok:
            return JSONResponse(status_code=response.status_code, content=payload)
        return JSONResponse(content=payload)
    except Exception:
        return JSONResponse(status_code=503, content={"detail": offline_detail})


@app.post("/autonomy-stop")
async def autonomy_stop_proxy():
    return await _local_service_post(
        "http://127.0.0.1:8002/stop",
        "Autonomy controller is offline",
    )


@app.post("/autonomy-resume")
async def autonomy_resume_proxy():
    return await _local_service_post(
        "http://127.0.0.1:8002/resume",
        "Autonomy controller is offline",
    )


@app.get("/sdk")
async def sdk(request: Request):
    return await render_index_html(is_spectator=False)


@app.post("/control-legacy")
async def control_legacy(request: Request):
    await need_start_mission()
    if not auth_response_data:
        await auth()

    body = await request.json()
    command = body.get("command")
    if not command:
        raise HTTPException(status_code=400, detail="Command not provided")

    RtmClient(auth_response_data).send_message(command)

    return {"message": "Command sent successfully"}


@app.post("/control")
async def control(request: Request):
    await need_start_mission()
    if not auth_response_data:
        await auth()

    body = await request.json()
    command = validate_control_command(body)

    try:
        publish_result = await _send_control_message(command)
        _schedule_control_watchdog(command)
        control_status = await browser_service.control_status()
        return {
            "message": "Command sent successfully",
            "result": "COMMAND_PUBLISHED",
            "publish_result": publish_result,
            "control_bridge_ready": bool(control_status.get("ready")),
            "control_bridge_reason": control_status.get("reason"),
            "command_has_motion": _command_has_motion(command),
            "server_timestamp": time.time(),
            "watchdog_timeout_sec": CONTROL_WATCHDOG_TIMEOUT_SEC,
        }
    except BrowserServiceError as e:
        logger.error("RTM control bridge unavailable: %s", str(e))
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        logger.error("Error sending control command: %s", str(e))
        raise HTTPException(
            status_code=500, detail="Failed to send control command"
        ) from e


@app.post("/speak")
async def speak(request: Request):
    await need_start_mission()
    if not auth_response_data:
        await auth()

    body = await request.json()
    text = body.get("text")
    if not text:
        raise HTTPException(status_code=400, detail="Text not provided")

    try:
        audio_path = await generate_speech(text, "static/tts_output")
        audio_filename = os.path.basename(audio_path)
        audio_url = f"http://127.0.0.1:8000/static/{audio_filename}"
        await browser_service.speak(audio_url)
        return {"message": "Speech sent to rover"}
    except Exception as e:
        logger.error("Error in /speak: %s", str(e))
        raise HTTPException(status_code=500, detail=f"TTS failed: {str(e)}") from e


@app.get("/screenshot")
async def get_screenshot(view_types: str = "rear,map,front"):
    await need_start_mission()
    if not auth_response_data:
        await auth()

    print("Received request for screenshot with view_types:", view_types)
    valid_views = {"rear", "map", "front"}
    views_list = view_types.split(",")

    for view in views_list:
        if view not in valid_views:
            raise HTTPException(status_code=400, detail=f"Invalid view type: {view}")

    await browser_service.take_screenshot("screenshots", views_list)

    response_content = {}
    for view in views_list:
        file_path = f"screenshots/{view}.png"
        try:
            with open(file_path, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
                response_content[f"{view}_frame"] = encoded_image
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=500, detail=f"Failed to read {view} image"
            ) from exc

    current_timestamp = time.time()
    response_content["timestamp"] = current_timestamp
    response_content["server_timestamp"] = current_timestamp

    return JSONResponse(content=response_content)


@app.get("/data")
async def get_data():
    await need_start_mission()
    data = await browser_service.data()
    data["server_timestamp"] = time.time()
    return JSONResponse(content=data)


@app.post("/checkpoint-reached")
async def checkpoint_reached(request: Request):
    global checkpoints_list_data
    if active_session_mode not in {"mission", "local_mission"}:
        raise HTTPException(
            status_code=400,
            detail="Start the selected mission before reporting a checkpoint",
        )

    if active_session_mode == "local_mission":
        reached_sequence = advance_cached_checkpoint()
        next_sequence = next_cached_checkpoint_sequence()
        return JSONResponse(
            status_code=200,
            content={
                "message": "Local checkpoint reached successfully",
                "local_mission": True,
                "reached_checkpoint_sequence": reached_sequence,
                "next_checkpoint_sequence": next_sequence or "",
            },
        )

    bot_slug = os.getenv("BOT_SLUG")
    mission_slug = current_mission_slug()
    auth_header = os.getenv("SDK_API_TOKEN")

    if not all([bot_slug, mission_slug, auth_header]):
        raise HTTPException(
            status_code=500, detail="Required environment variables not configured"
        )

    data = await browser_service.data()
    latitude = data.get("latitude")
    longitude = data.get("longitude")

    if not all([latitude, longitude]):
        raise HTTPException(status_code=400, detail="Missing latitude or longitude")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {auth_header}",
    }

    payload = {
        "bot_slug": bot_slug,
        "mission_slug": mission_slug,
        "latitude": latitude,
        "longitude": longitude,
    }

    # Run the blocking cloud call off the event loop. This handler is on the
    # same single-threaded loop as /mission-status and /control; a synchronous
    # requests.post() here stalls those endpoints for the full duration of the
    # upstream call, which is what produced the cascading timeouts observed
    # downstream in Mission1's controller.
    response = await asyncio.to_thread(
        requests.post,
        FRODOBOTS_API_URL + "/sdk/checkpoint_reached",
        headers=headers,
        json=payload,
        timeout=15,
    )

    response_data = response.json()

    if response.status_code != 200:
        raise HTTPException(
            status_code=response.status_code,
            detail={
                "error": response_data.get("error", "Failed to send checkpoint data"),
                "proximate_distance_to_checkpoint": response_data.get(
                    "distance_to_checkpoint", "Unknown"
                ),
            },
        )
    # Keep the side-effect-free /mission-route cache in sync so the SAM-TP
    # process and Mission1 controller advance to the next checkpoint without
    # making another cloud request.  The current target is the first sequence
    # after latest_scanned_checkpoint.
    reached_sequence = advance_cached_checkpoint()
    return JSONResponse(
        status_code=200,
        content={
            "message": "Checkpoint reached successfully",
            "reached_checkpoint_sequence": reached_sequence,
            "next_checkpoint_sequence": response_data.get(
                "next_checkpoint_sequence", ""
            ),
        },
    )


def cached_checkpoint_sequences() -> list[int]:
    sequences = []
    for checkpoint in checkpoints_list_data.get("checkpoints_list", []):
        try:
            sequence = int(float(checkpoint.get("sequence")))
        except (AttributeError, TypeError, ValueError):
            continue
        sequences.append(sequence)
    return sorted(sequences)


def current_cached_latest_checkpoint():
    local_latest = read_local_mission_latest()
    if local_latest is not None:
        return local_latest
    return checkpoints_list_data.get("latest_scanned_checkpoint", 0)


def advance_cached_checkpoint() -> int | None:
    latest = current_cached_latest_checkpoint()
    try:
        latest_number = int(float(latest or 0))
    except (TypeError, ValueError):
        latest_number = 0
    pending_sequences = [
        sequence for sequence in cached_checkpoint_sequences()
        if sequence > latest_number
    ]
    if pending_sequences:
        reached_sequence = min(pending_sequences)
        checkpoints_list_data["latest_scanned_checkpoint"] = reached_sequence
        if active_session_mode == "local_mission":
            write_local_mission_state(reached_sequence)
        return reached_sequence
    return None


def next_cached_checkpoint_sequence() -> int | None:
    latest = current_cached_latest_checkpoint()
    try:
        latest_number = int(float(latest or 0))
    except (TypeError, ValueError):
        latest_number = 0
    for sequence in cached_checkpoint_sequences():
        if sequence > latest_number:
            return sequence
    return None


@app.get("/missions-history")
async def missions_history():
    auth_header = os.getenv("SDK_API_TOKEN")
    bot_slug = os.getenv("BOT_SLUG")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {auth_header}",
    }

    data = {"bot_slug": bot_slug}

    try:
        response = requests.post(
            FRODOBOTS_API_URL + "/sdk/rides_history",
            headers=headers,
            json=data,
            timeout=15,
        )

        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail="Failed to retrieve missions history",
            )

        return JSONResponse(content=response.json())
    except requests.RequestException as e:
        raise HTTPException(
            status_code=500, detail=f"Error fetching missions history: {str(e)}"
        )


@app.get("/v2/screenshot")
async def get_screenshot_v2():
    await need_start_mission()
    if not auth_response_data:
        await auth()

    async def get_frame(frame_type):
        frame = await getattr(browser_service, frame_type)()
        _, frame = frame.split(",", 1)
        return {f"{frame_type}_frame": frame}

    front_task = asyncio.create_task(get_frame("front"))
    tasks = [front_task]

    if auth_response_data.get("BOT_TYPE") == "zero":
        rear_task = asyncio.create_task(get_frame("rear"))
        tasks.append(rear_task)

    results = await asyncio.gather(*tasks)

    response_data = {}
    for result in results:
        response_data.update(result)

    if not response_data:
        raise HTTPException(status_code=404, detail="Frames not available")

    current_timestamp = time.time()
    response_data["timestamp"] = current_timestamp
    response_data["server_timestamp"] = current_timestamp

    return JSONResponse(content=response_data)


if __name__ == "__main__":
    from hypercorn.config import Config

    config = Config()
    config.bind = ["0.0.0.0:8000"]


@app.get("/v2/front")
async def get_front_frame():
    await need_start_mission()
    front_frame = await browser_service.front(timeout_sec=FRAME_REQUEST_TIMEOUT_SEC)
    response_data = {}
    if front_frame:
        _, base64_data = front_frame.split(",", 1)
        response_data["front_frame"] = base64_data
        response_data.update(await browser_service.frame_metadata(1000))
        current_timestamp = time.time()
        response_data["timestamp"] = current_timestamp
        response_data["server_timestamp"] = current_timestamp
        return JSONResponse(content=response_data)
    else:
        raise HTTPException(status_code=404, detail="Front frame not available")


@app.get("/v2/rear")
async def get_rear_frame():
    await need_start_mission()
    if not auth_response_data:
        await auth()

    rear_frame = await browser_service.rear()
    response_data = {}
    if rear_frame:
        _, base64_data = rear_frame.split(",", 1)
        response_data["rear_frame"] = base64_data
        current_timestamp = time.time()
        response_data["timestamp"] = current_timestamp
        response_data["server_timestamp"] = current_timestamp
        return JSONResponse(content=response_data)
    else:
        raise HTTPException(status_code=404, detail="Rear frame not available")


@app.post("/interventions/start")
async def start_intervention(request: Request):
    await need_start_mission()

    auth_header = os.getenv("SDK_API_TOKEN")
    bot_slug = os.getenv("BOT_SLUG")

    if not auth_header:
        raise HTTPException(
            status_code=500, detail="Authorization header not configured"
        )
    if not bot_slug:
        raise HTTPException(status_code=500, detail="Bot name not configured")

    data = await browser_service.data()
    latitude = data.get("latitude")
    longitude = data.get("longitude")

    if not all([latitude, longitude]):
        raise HTTPException(status_code=400, detail="Missing latitude or longitude")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {auth_header}",
    }

    payload = {
        "bot_slug": bot_slug,
        "latitude": latitude,
        "longitude": longitude,
    }

    try:
        response = requests.post(
            FRODOBOTS_API_URL + "/sdk/interventions/start",
            headers=headers,
            json=payload,
            timeout=15,
        )

        response_data = response.json()

        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=response_data.get("error", "Failed to start intervention"),
            )

        return JSONResponse(
            status_code=200,
            content={
                "message": "Intervention started successfully",
                "intervention_id": response_data.get("intervention_id"),
            },
        )
    except requests.RequestException as e:
        raise HTTPException(
            status_code=500, detail=f"Error starting intervention: {str(e)}"
        )


@app.post("/interventions/end")
async def end_intervention(request: Request):
    await need_start_mission()

    auth_header = os.getenv("SDK_API_TOKEN")
    bot_slug = os.getenv("BOT_SLUG")

    if not auth_header:
        raise HTTPException(
            status_code=500, detail="Authorization header not configured"
        )
    if not bot_slug:
        raise HTTPException(status_code=500, detail="Bot name not configured")

    data = await browser_service.data()
    latitude = data.get("latitude")
    longitude = data.get("longitude")

    if not all([latitude, longitude]):
        raise HTTPException(status_code=400, detail="Missing latitude or longitude")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {auth_header}",
    }

    payload = {
        "bot_slug": bot_slug,
        "latitude": latitude,
        "longitude": longitude,
    }

    try:
        response = requests.post(
            FRODOBOTS_API_URL + "/sdk/interventions/end",
            headers=headers,
            json=payload,
            timeout=15,
        )

        response_data = response.json()

        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=response_data.get("error", "Failed to end intervention"),
            )

        return JSONResponse(
            status_code=200,
            content={"message": "Intervention ended successfully"},
        )
    except requests.RequestException as e:
        raise HTTPException(
            status_code=500, detail=f"Error ending intervention: {str(e)}"
        )


@app.get("/interventions/history")
async def interventions_history():
    auth_header = os.getenv("SDK_API_TOKEN")
    bot_slug = os.getenv("BOT_SLUG")

    if not auth_header:
        raise HTTPException(
            status_code=500, detail="Authorization header not configured"
        )
    if not bot_slug:
        raise HTTPException(status_code=500, detail="Bot name not configured")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {auth_header}",
    }

    payload = {"bot_slug": bot_slug}

    try:
        response = requests.get(
            FRODOBOTS_API_URL + "/sdk/interventions/history",
            headers=headers,
            params=payload,
            timeout=15,
        )

        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail="Failed to retrieve interventions history",
            )

        return JSONResponse(content=response.json())
    except requests.RequestException as e:
        raise HTTPException(
            status_code=500, detail=f"Error fetching interventions history: {str(e)}"
        )
