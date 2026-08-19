from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts" / "calibration") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts" / "calibration"))

import calibrate_extrinsics as cx  # noqa: E402


CAMERA_MATRIX = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
DIST_COEFFS = np.zeros(5)


def _known_transform():
    pitch = math.radians(20.0)
    base = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
    pitch_rotation = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(pitch), -math.sin(pitch)],
            [0.0, math.sin(pitch), math.cos(pitch)],
        ]
    )
    rotation = pitch_rotation @ base
    camera_position_in_rover = np.array([0.0, 0.0, 0.3])
    translation = -rotation @ camera_position_in_rover
    return rotation, translation


def _write_intrinsics(tmp_path):
    path = tmp_path / "intrinsics.json"
    path.write_text(
        json.dumps(
            {
                "camera_matrix": CAMERA_MATRIX.tolist(),
                "distortion_coefficients": DIST_COEFFS.tolist(),
            }
        )
    )
    return path


def _write_correspondences(tmp_path, count=8, name="correspondences.json"):
    rotation, translation = _known_transform()
    rvec, _ = cv2.Rodrigues(rotation)
    rover_points = np.array(
        [
            [1.0, 0.3, 0.0],
            [1.0, -0.3, 0.0],
            [1.5, 0.5, 0.0],
            [1.5, -0.5, 0.0],
            [2.0, 0.2, 0.0],
            [2.0, -0.2, 0.0],
            [0.8, 0.0, 0.0],
            [1.8, 0.0, 0.0],
        ][:count],
        dtype=np.float64,
    )
    image_points, _ = cv2.projectPoints(
        rover_points, rvec, translation, CAMERA_MATRIX, DIST_COEFFS
    )
    image_points = image_points.reshape(-1, 2)
    entries = [
        {"rover_xyz_m": p.tolist(), "pixel_uv": uv.tolist()}
        for p, uv in zip(rover_points, image_points)
    ]
    path = tmp_path / name
    path.write_text(json.dumps(entries))
    return path


def test_recovers_known_transform_and_passes_thresholds(tmp_path, capsys):
    correspondences = _write_correspondences(tmp_path)
    intrinsics = _write_intrinsics(tmp_path)
    output = tmp_path / "extrinsics.yaml"
    report = tmp_path / "report.json"

    exit_code = cx.main(
        [
            "--correspondences",
            str(correspondences),
            "--intrinsics",
            str(intrinsics),
            "--output",
            str(output),
            "--report",
            str(report),
        ]
    )

    assert exit_code == 0
    assert output.exists()
    fragment = yaml.safe_load(output.read_text())
    transform = np.array(fragment["camera_from_rover_transform"])
    rotation, translation = _known_transform()
    expected = np.eye(4)
    expected[:3, :3] = rotation
    expected[:3, 3] = translation
    assert transform == pytest.approx(expected, abs=1e-4)

    report_payload = json.loads(report.read_text())
    assert report_payload["passed_thresholds"] is True
    assert report_payload["reprojection_rmse_px"] < 1e-3


def test_insufficient_points_withholds_output(tmp_path):
    correspondences = _write_correspondences(tmp_path, count=4)
    intrinsics = _write_intrinsics(tmp_path)
    output = tmp_path / "extrinsics.yaml"

    exit_code = cx.main(
        [
            "--correspondences",
            str(correspondences),
            "--intrinsics",
            str(intrinsics),
            "--min-points",
            "6",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 1
    assert not output.exists()


def test_noisy_correspondences_exceed_rmse_threshold_and_withhold_output(tmp_path):
    correspondences_path = _write_correspondences(tmp_path)
    entries = json.loads(correspondences_path.read_text())
    rng = np.random.default_rng(0)
    for entry in entries:
        entry["pixel_uv"][0] += float(rng.uniform(20.0, 40.0))
        entry["pixel_uv"][1] += float(rng.uniform(20.0, 40.0))
    correspondences_path.write_text(json.dumps(entries))
    intrinsics = _write_intrinsics(tmp_path)
    output = tmp_path / "extrinsics.yaml"

    exit_code = cx.main(
        [
            "--correspondences",
            str(correspondences_path),
            "--intrinsics",
            str(intrinsics),
            "--max-reprojection-rmse-px",
            "0.5",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 1
    assert not output.exists()


def test_min_points_below_six_rejected(tmp_path):
    correspondences = _write_correspondences(tmp_path)
    intrinsics = _write_intrinsics(tmp_path)
    with pytest.raises(SystemExit):
        cx.main(
            [
                "--correspondences",
                str(correspondences),
                "--intrinsics",
                str(intrinsics),
                "--min-points",
                "3",
                "--output",
                str(tmp_path / "out.yaml"),
            ]
        )
