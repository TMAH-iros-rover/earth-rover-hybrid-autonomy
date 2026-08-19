from __future__ import annotations

import json

import pytest
import yaml

from earth_rover.perception.camera_calibration import (
    CalibrationError,
    load_calibration,
    validate_for_live_use,
)


def _valid_payload(**overrides):
    payload = {
        "schema_version": 1,
        "calibration_id": "test_cal",
        "placeholder": False,
        "image_width": 640,
        "image_height": 480,
        "camera_matrix": [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]],
        "distortion_model": "opencv_pinhole",
        "distortion_coefficients": [0.0, 0.0, 0.0, 0.0, 0.0],
        "camera_from_rover_transform": [
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.3],
            [1.0, 0.0, 0.0, -0.2],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "provenance": {
            "capture_date": "2026-01-01",
            "capture_source": "test",
            "reprojection_error_px": 0.4,
        },
    }
    payload.update(overrides)
    return payload


def _write(tmp_path, payload, name="cal.yaml"):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(payload))
    return path


def test_load_valid_calibration(tmp_path):
    path = _write(tmp_path, _valid_payload())
    cal = load_calibration(path)
    assert cal.calibration_id == "test_cal"
    assert cal.placeholder is False
    assert cal.image_shape == (480, 640)
    assert len(cal.content_sha256) == 64
    assert cal.sha256_prefix == cal.content_sha256[:16]


def test_unsupported_distortion_model_is_rejected(tmp_path):
    path = _write(tmp_path, _valid_payload(distortion_model="opencv_fisheye"))

    with pytest.raises(CalibrationError) as exc_info:
        load_calibration(path)

    assert exc_info.value.reason == "CALIBRATION_UNSUPPORTED_DISTORTION_MODEL"


def test_sha256_is_deterministic_and_content_based(tmp_path):
    path_a = _write(tmp_path, _valid_payload(), "a.yaml")
    path_b = _write(tmp_path, _valid_payload(), "b.yaml")
    cal_a = load_calibration(path_a)
    cal_b = load_calibration(path_b)
    assert cal_a.content_sha256 == cal_b.content_sha256

    path_c = _write(tmp_path, _valid_payload(calibration_id="different"), "c.yaml")
    cal_c = load_calibration(path_c)
    assert cal_c.content_sha256 != cal_a.content_sha256


def test_json_is_supported(tmp_path):
    path = tmp_path / "cal.json"
    path.write_text(json.dumps(_valid_payload()))
    cal = load_calibration(path)
    assert cal.calibration_id == "test_cal"


def test_missing_file(tmp_path):
    with pytest.raises(CalibrationError) as exc_info:
        load_calibration(tmp_path / "missing.yaml")
    assert exc_info.value.reason == "CALIBRATION_FILE_MISSING"


def test_malformed_yaml(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("not: [valid: yaml: :::")
    with pytest.raises(CalibrationError) as exc_info:
        load_calibration(path)
    assert exc_info.value.reason == "CALIBRATION_PARSE_ERROR"


def test_top_level_must_be_a_mapping(tmp_path):
    path = tmp_path / "list.yaml"
    path.write_text(yaml.safe_dump([1, 2, 3]))
    with pytest.raises(CalibrationError) as exc_info:
        load_calibration(path)
    assert exc_info.value.reason == "CALIBRATION_MALFORMED"


def test_missing_required_field(tmp_path):
    payload = _valid_payload()
    del payload["calibration_id"]
    path = _write(tmp_path, payload)
    with pytest.raises(CalibrationError) as exc_info:
        load_calibration(path)
    assert exc_info.value.reason == "CALIBRATION_MISSING_FIELD"


def test_unsupported_schema_version(tmp_path):
    path = _write(tmp_path, _valid_payload(schema_version=2))
    with pytest.raises(CalibrationError) as exc_info:
        load_calibration(path)
    assert exc_info.value.reason == "CALIBRATION_UNSUPPORTED_SCHEMA_VERSION"


def test_nonpositive_image_size(tmp_path):
    path = _write(tmp_path, _valid_payload(image_width=0))
    with pytest.raises(CalibrationError):
        load_calibration(path)


def test_camera_matrix_wrong_shape(tmp_path):
    path = _write(tmp_path, _valid_payload(camera_matrix=[[500.0, 0.0], [0.0, 500.0]]))
    with pytest.raises(CalibrationError) as exc_info:
        load_calibration(path)
    assert exc_info.value.reason == "CALIBRATION_MALFORMED_SHAPE"


def test_camera_matrix_non_finite(tmp_path):
    payload = _valid_payload(
        camera_matrix=[[float("nan"), 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]]
    )
    path = _write(tmp_path, payload)
    with pytest.raises(CalibrationError) as exc_info:
        load_calibration(path)
    assert exc_info.value.reason == "CALIBRATION_NON_FINITE"


@pytest.mark.parametrize(
    "camera_matrix",
    [
        [[-500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]],
        [[0.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]],
        [[500.0, 0.0, 320.0], [0.0, -500.0, 240.0], [0.0, 0.0, 1.0]],
    ],
)
def test_nonpositive_focal_length_rejected(tmp_path, camera_matrix):
    path = _write(tmp_path, _valid_payload(camera_matrix=camera_matrix))
    with pytest.raises(CalibrationError) as exc_info:
        load_calibration(path)
    assert exc_info.value.reason == "CALIBRATION_NONPOSITIVE_FOCAL_LENGTH"


def test_singular_intrinsics_rejected(tmp_path):
    # fx is positive but astronomically small: passes the positive-focal-
    # length check yet leaves the matrix numerically singular.
    payload = _valid_payload(
        camera_matrix=[[1e-12, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]]
    )
    path = _write(tmp_path, payload)
    with pytest.raises(CalibrationError) as exc_info:
        load_calibration(path)
    assert exc_info.value.reason == "CALIBRATION_SINGULAR_INTRINSICS"


@pytest.mark.parametrize(
    "camera_matrix",
    [
        [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 2.0]],
        [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [1.0, 0.0, 1.0]],
        [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 1.0, 1.0]],
    ],
)
def test_malformed_intrinsics_bottom_row_rejected(tmp_path, camera_matrix):
    path = _write(tmp_path, _valid_payload(camera_matrix=camera_matrix))
    with pytest.raises(CalibrationError) as exc_info:
        load_calibration(path)
    assert exc_info.value.reason == "CALIBRATION_MALFORMED_INTRINSICS"


def test_invalid_distortion_length(tmp_path):
    path = _write(tmp_path, _valid_payload(distortion_coefficients=[0.0, 0.0, 0.0]))
    with pytest.raises(CalibrationError) as exc_info:
        load_calibration(path)
    assert exc_info.value.reason == "CALIBRATION_INVALID_DISTORTION_LENGTH"


def test_transform_wrong_shape(tmp_path):
    payload = _valid_payload(
        camera_from_rover_transform=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    path = _write(tmp_path, payload)
    with pytest.raises(CalibrationError) as exc_info:
        load_calibration(path)
    assert exc_info.value.reason == "CALIBRATION_MALFORMED_SHAPE"


def test_transform_bad_homogeneous_row(tmp_path):
    payload = _valid_payload(
        camera_from_rover_transform=[
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.3],
            [1.0, 0.0, 0.0, -0.2],
            [1.0, 0.0, 0.0, 1.0],
        ]
    )
    path = _write(tmp_path, payload)
    with pytest.raises(CalibrationError) as exc_info:
        load_calibration(path)
    assert exc_info.value.reason == "CALIBRATION_INVALID_HOMOGENEOUS_ROW"


def test_transform_non_orthonormal_rotation_rejected(tmp_path):
    payload = _valid_payload(
        camera_from_rover_transform=[
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    path = _write(tmp_path, payload)
    with pytest.raises(CalibrationError) as exc_info:
        load_calibration(path)
    assert exc_info.value.reason == "CALIBRATION_IMPROPER_ROTATION"


def test_transform_reflection_rejected(tmp_path):
    # Orthonormal (R^T R == I) but det(R) == -1: a reflection, not a
    # rotation. Must be caught even though the orthonormality check alone
    # would pass it.
    payload = _valid_payload(
        camera_from_rover_transform=[
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    path = _write(tmp_path, payload)
    with pytest.raises(CalibrationError) as exc_info:
        load_calibration(path)
    assert exc_info.value.reason == "CALIBRATION_IMPROPER_ROTATION"


def test_identity_rotation_is_accepted(tmp_path):
    payload = _valid_payload(
        camera_from_rover_transform=[
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    path = _write(tmp_path, payload)
    cal = load_calibration(path)
    assert cal.calibration_id == "test_cal"


def test_placeholder_loads_but_structural_validation_still_applies(tmp_path):
    payload = _valid_payload(placeholder=True, calibration_id="TEMPLATE")
    path = _write(tmp_path, payload)
    cal = load_calibration(path)
    assert cal.placeholder is True

    broken = _valid_payload(placeholder=True, camera_matrix=[[500.0, 0.0], [0.0, 500.0]])
    broken_path = _write(tmp_path, broken, "broken.yaml")
    with pytest.raises(CalibrationError):
        load_calibration(broken_path)


def test_validate_for_live_use_rejects_placeholder(tmp_path):
    path = _write(tmp_path, _valid_payload(placeholder=True))
    cal = load_calibration(path)
    with pytest.raises(CalibrationError) as exc_info:
        validate_for_live_use(cal, (480, 640))
    assert exc_info.value.reason == "CALIBRATION_PLACEHOLDER"


def test_validate_for_live_use_rejects_resolution_mismatch(tmp_path):
    path = _write(tmp_path, _valid_payload())
    cal = load_calibration(path)
    with pytest.raises(CalibrationError) as exc_info:
        validate_for_live_use(cal, (720, 1280))
    assert exc_info.value.reason == "CALIBRATION_RESOLUTION_MISMATCH"


def test_validate_for_live_use_accepts_matching_real_calibration(tmp_path):
    path = _write(tmp_path, _valid_payload())
    cal = load_calibration(path)
    validate_for_live_use(cal, (480, 640))
