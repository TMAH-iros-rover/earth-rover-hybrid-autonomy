from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

SCHEMA_VERSION = 1
_VALID_DISTORTION_LENGTHS = (4, 5, 8, 12, 14)
_ORTHONORMALITY_TOLERANCE = 1e-3
_HOMOGENEOUS_ROW_TOLERANCE = 1e-6


class CalibrationError(ValueError):
    """Rejected calibration file with a stable machine-readable reason."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(f"{reason}: {detail}")
        self.reason = reason


@dataclass(frozen=True)
class CameraCalibration:
    """Validated camera intrinsic/extrinsic calibration for metric projection.

    ``camera_from_rover_transform`` maps a homogeneous rover-frame point
    ``[x, y, z, 1]`` (rover frame: +x forward, +y left, +z up) into a
    homogeneous OpenCV camera-frame point ``[X, Y, Z, 1]`` (camera frame:
    +x right, +y down, +z forward, i.e. ``p_camera = camera_from_rover_transform
    @ p_rover``). ``camera_matrix``/``distortion_coefficients`` follow the
    OpenCV pinhole convention and apply to ``(image_width, image_height)``
    pixels exactly -- there is no implicit rescaling.
    """

    schema_version: int
    calibration_id: str
    image_width: int
    image_height: int
    camera_matrix: np.ndarray
    distortion_model: str
    distortion_coefficients: np.ndarray
    camera_from_rover_transform: np.ndarray
    capture_date: str
    capture_source: str
    reprojection_error_px: float | None
    placeholder: bool
    content_sha256: str

    @property
    def image_shape(self) -> tuple[int, int]:
        return (self.image_height, self.image_width)

    @property
    def sha256_prefix(self) -> str:
        return self.content_sha256[:16]


def load_calibration(path: str | Path) -> CameraCalibration:
    """Parse and structurally validate a calibration file (YAML or JSON).

    Structural validation runs unconditionally, including for placeholder
    templates -- a template must still be well-formed. Whether it may be
    used live is a separate, stricter question answered by
    ``validate_for_live_use``.
    """

    resolved = Path(path)
    if not resolved.is_file():
        raise CalibrationError("CALIBRATION_FILE_MISSING", str(resolved))
    raw_bytes = resolved.read_bytes()
    content_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    try:
        if resolved.suffix.lower() == ".json":
            payload = json.loads(raw_bytes.decode("utf-8"))
        else:
            payload = yaml.safe_load(raw_bytes.decode("utf-8"))
    except (yaml.YAMLError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise CalibrationError("CALIBRATION_PARSE_ERROR", str(exc)) from exc
    if not isinstance(payload, dict):
        raise CalibrationError(
            "CALIBRATION_MALFORMED", "top-level calibration document must be a mapping"
        )
    return _validate_structure(payload, content_sha256)


def validate_for_live_use(
    calibration: CameraCalibration,
    expected_image_shape: tuple[int, int],
) -> None:
    """Raise unless ``calibration`` is a real, resolution-matched calibration.

    Never silently rescales intrinsics: ``expected_image_shape`` (height,
    width) must equal the calibration's own image size exactly.
    """

    if calibration.placeholder:
        raise CalibrationError(
            "CALIBRATION_PLACEHOLDER",
            f"calibration_id={calibration.calibration_id!r} is a template, not live-ready",
        )
    expected_height, expected_width = expected_image_shape
    if (
        int(expected_height) != calibration.image_height
        or int(expected_width) != calibration.image_width
    ):
        raise CalibrationError(
            "CALIBRATION_RESOLUTION_MISMATCH",
            f"calibration is {calibration.image_width}x{calibration.image_height}, "
            f"frame is {expected_width}x{expected_height}",
        )


def _validate_structure(payload: dict[str, Any], content_sha256: str) -> CameraCalibration:
    schema_version = _require_int(payload, "schema_version")
    if schema_version != SCHEMA_VERSION:
        raise CalibrationError(
            "CALIBRATION_UNSUPPORTED_SCHEMA_VERSION",
            f"expected {SCHEMA_VERSION}, got {schema_version}",
        )
    calibration_id = _require_nonempty_str(payload, "calibration_id")
    placeholder = payload.get("placeholder", False)
    if not isinstance(placeholder, bool):
        raise CalibrationError("CALIBRATION_MALFORMED", "placeholder must be boolean")
    image_width = _require_positive_int(payload, "image_width")
    image_height = _require_positive_int(payload, "image_height")

    camera_matrix = _require_matrix(payload, "camera_matrix", (3, 3))
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    if fx <= 0.0 or fy <= 0.0:
        raise CalibrationError(
            "CALIBRATION_NONPOSITIVE_FOCAL_LENGTH", f"fx={fx}, fy={fy}"
        )
    if not math.isclose(camera_matrix[2, 2], 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise CalibrationError(
            "CALIBRATION_MALFORMED_INTRINSICS", "camera_matrix[2, 2] must be 1.0"
        )
    if not (camera_matrix[2, 0] == 0.0 and camera_matrix[2, 1] == 0.0):
        raise CalibrationError(
            "CALIBRATION_MALFORMED_INTRINSICS",
            "camera_matrix bottom row must be [0, 0, 1]",
        )
    determinant = float(np.linalg.det(camera_matrix))
    if not math.isfinite(determinant) or abs(determinant) < 1e-9:
        raise CalibrationError(
            "CALIBRATION_SINGULAR_INTRINSICS", f"det(camera_matrix)={determinant}"
        )

    distortion_model = _require_nonempty_str(payload, "distortion_model")
    if distortion_model != "opencv_pinhole":
        raise CalibrationError(
            "CALIBRATION_UNSUPPORTED_DISTORTION_MODEL",
            f"expected 'opencv_pinhole', got {distortion_model!r}",
        )
    distortion_coefficients = _require_vector(payload, "distortion_coefficients")
    if len(distortion_coefficients) not in _VALID_DISTORTION_LENGTHS:
        raise CalibrationError(
            "CALIBRATION_INVALID_DISTORTION_LENGTH",
            f"length {len(distortion_coefficients)} not in {_VALID_DISTORTION_LENGTHS}",
        )

    transform = _require_matrix(payload, "camera_from_rover_transform", (4, 4))
    last_row_error = float(np.max(np.abs(transform[3, :] - np.array([0.0, 0.0, 0.0, 1.0]))))
    if last_row_error > _HOMOGENEOUS_ROW_TOLERANCE:
        raise CalibrationError(
            "CALIBRATION_INVALID_HOMOGENEOUS_ROW",
            f"camera_from_rover_transform[3, :] deviates by {last_row_error}",
        )
    rotation = transform[:3, :3]
    orthonormality_error = float(
        np.max(np.abs(rotation.T @ rotation - np.eye(3)))
    )
    if orthonormality_error > _ORTHONORMALITY_TOLERANCE:
        raise CalibrationError(
            "CALIBRATION_IMPROPER_ROTATION",
            f"R^T R deviates from identity by {orthonormality_error}",
        )
    rotation_det = float(np.linalg.det(rotation))
    if abs(rotation_det - 1.0) > _ORTHONORMALITY_TOLERANCE:
        raise CalibrationError(
            "CALIBRATION_IMPROPER_ROTATION", f"det(R)={rotation_det}, expected +1"
        )

    provenance = payload.get("provenance", {})
    if not isinstance(provenance, dict):
        raise CalibrationError("CALIBRATION_MALFORMED", "provenance must be a mapping")
    capture_date = _require_nonempty_str(provenance, "capture_date")
    capture_source = _require_nonempty_str(provenance, "capture_source")
    reprojection_error_px = provenance.get("reprojection_error_px")
    if reprojection_error_px is not None:
        reprojection_error_px = _finite_float(
            reprojection_error_px, "provenance.reprojection_error_px"
        )
        if reprojection_error_px < 0.0:
            raise CalibrationError(
                "CALIBRATION_MALFORMED", "provenance.reprojection_error_px must be non-negative"
            )

    return CameraCalibration(
        schema_version=schema_version,
        calibration_id=calibration_id,
        image_width=image_width,
        image_height=image_height,
        camera_matrix=camera_matrix,
        distortion_model=distortion_model,
        distortion_coefficients=distortion_coefficients,
        camera_from_rover_transform=transform,
        capture_date=capture_date,
        capture_source=capture_source,
        reprojection_error_px=reprojection_error_px,
        placeholder=bool(placeholder),
        content_sha256=content_sha256,
    )


def _require_int(payload: dict[str, Any], key: str) -> int:
    if key not in payload:
        raise CalibrationError("CALIBRATION_MISSING_FIELD", key)
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise CalibrationError("CALIBRATION_MALFORMED", f"{key} must be an integer")
    return value


def _require_positive_int(payload: dict[str, Any], key: str) -> int:
    value = _require_int(payload, key)
    if value <= 0:
        raise CalibrationError("CALIBRATION_MALFORMED", f"{key} must be positive")
    return value


def _require_nonempty_str(payload: dict[str, Any], key: str) -> str:
    if key not in payload:
        raise CalibrationError("CALIBRATION_MISSING_FIELD", key)
    value = payload[key]
    if not isinstance(value, str) or not value.strip():
        raise CalibrationError("CALIBRATION_MALFORMED", f"{key} must be a non-empty string")
    return value


def _finite_float(value: Any, key: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise CalibrationError("CALIBRATION_MALFORMED", f"{key} must be numeric") from exc
    if not math.isfinite(parsed):
        raise CalibrationError("CALIBRATION_NON_FINITE", key)
    return parsed


def _require_matrix(
    payload: dict[str, Any], key: str, shape: tuple[int, int]
) -> np.ndarray:
    if key not in payload:
        raise CalibrationError("CALIBRATION_MISSING_FIELD", key)
    try:
        array = np.array(payload[key], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise CalibrationError("CALIBRATION_MALFORMED", f"{key} must be numeric") from exc
    if array.shape != shape:
        raise CalibrationError(
            "CALIBRATION_MALFORMED_SHAPE", f"{key} must have shape {shape}, got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise CalibrationError("CALIBRATION_NON_FINITE", key)
    array.setflags(write=False)
    return array


def _require_vector(payload: dict[str, Any], key: str) -> np.ndarray:
    if key not in payload:
        raise CalibrationError("CALIBRATION_MISSING_FIELD", key)
    try:
        array = np.array(payload[key], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise CalibrationError("CALIBRATION_MALFORMED", f"{key} must be numeric") from exc
    if array.ndim != 1:
        raise CalibrationError("CALIBRATION_MALFORMED_SHAPE", f"{key} must be a 1D list")
    if not np.isfinite(array).all():
        raise CalibrationError("CALIBRATION_NON_FINITE", key)
    array.setflags(write=False)
    return array
