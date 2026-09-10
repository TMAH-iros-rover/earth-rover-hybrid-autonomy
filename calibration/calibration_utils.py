"""Shared helpers for Earth Rover camera calibration tools."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests


SUPPORTED_CAMERAS = ("front", "rear")
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "calibration_config.json"


def load_config(path: Path) -> dict[str, Any]:
    """Load and minimally validate the shared calibration configuration."""
    config = load_json(path)
    checkerboard = config.get("checkerboard", {})
    required = (
        "base_url",
        "camera",
        "request_timeout_sec",
        "images_dir",
        "result_file",
        "verification_alpha",
    )
    missing = [key for key in required if key not in config]
    checkerboard_required = (
        "inner_corners_cols",
        "inner_corners_rows",
        "square_size_mm",
    )
    missing += [
        f"checkerboard.{key}" for key in checkerboard_required if key not in checkerboard
    ]
    if missing:
        raise ValueError(f"Missing calibration config values: {', '.join(missing)}")
    if config["camera"] not in SUPPORTED_CAMERAS:
        raise ValueError(f"Unsupported camera in config: {config['camera']}")
    return config


def config_path(value: str, config_file: Path) -> Path:
    """Resolve paths in the config relative to the config file directory."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else config_file.resolve().parent / path


def camera_endpoint(base_url: str, camera: str) -> str:
    if camera not in SUPPORTED_CAMERAS:
        raise ValueError(f"Unsupported camera: {camera}")
    return f"{base_url.rstrip('/')}/v2/{camera}"


def decode_base64_image(encoded: str) -> np.ndarray:
    """Decode either raw Base64 or a data URL into an OpenCV BGR image."""
    if not encoded:
        raise ValueError("The SDK returned an empty image")
    if "," in encoded:
        encoded = encoded.split(",", 1)[1]
    try:
        image_bytes = base64.b64decode(encoded, validate=True)
    except (ValueError, base64.binascii.Error) as exc:
        raise ValueError("The SDK returned invalid Base64 image data") from exc
    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("OpenCV could not decode the SDK image")
    return image


def fetch_frame(
    session: requests.Session,
    base_url: str,
    camera: str,
    timeout: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fetch one frame and return the decoded BGR image plus SDK metadata."""
    response = session.get(camera_endpoint(base_url, camera), timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    key = f"{camera}_frame"
    if key not in payload:
        raise ValueError(f"SDK response does not contain {key!r}")
    image = decode_base64_image(payload[key])
    metadata = {key: None, **payload}
    metadata.pop(key, None)
    return image, metadata


def find_checkerboard(
    image: np.ndarray, pattern_size: tuple[int, int]
) -> tuple[bool, np.ndarray | None]:
    """Find checkerboard inner corners using OpenCV's robust SB detector."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE
    found, corners = cv2.findChessboardCornersSB(gray, pattern_size, flags=flags)
    return bool(found), corners if found else None


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")
