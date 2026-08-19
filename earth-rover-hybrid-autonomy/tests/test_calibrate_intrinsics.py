from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts" / "calibration") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts" / "calibration"))

import calibrate_intrinsics as ci  # noqa: E402


def _write_charuco_images(directory: Path, count: int = 10) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
    board = cv2.aruco.CharucoBoard((5, 7), 0.03, 0.022, dictionary)
    board_image = board.generateImage((900, 1200), marginSize=40)
    board_color = cv2.cvtColor(board_image, cv2.COLOR_GRAY2BGR)
    rng = np.random.default_rng(0)
    width, height = 640, 480
    for index in range(count):
        source = np.float32([[0, 0], [1200, 0], [1200, 900], [0, 900]])
        jitter = rng.uniform(-60, 60, size=(4, 2)).astype(np.float32)
        destination = np.float32(
            [[80, 60], [560, 40], [600, 420], [40, 440]]
        ) + jitter
        matrix = cv2.getPerspectiveTransform(source, destination)
        warped = cv2.warpPerspective(
            board_color, matrix, (width, height), borderValue=(255, 255, 255)
        )
        cv2.imwrite(str(directory / f"frame_{index:02d}.png"), warped)


def _write_chessboard_images(directory: Path, count: int = 10) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    columns, rows = 7, 5
    square_px = 60
    board_width = (columns + 1) * square_px
    board_height = (rows + 1) * square_px
    board = np.zeros((board_height, board_width), dtype=np.uint8)
    for row in range(rows + 1):
        for col in range(columns + 1):
            if (row + col) % 2 == 0:
                board[
                    row * square_px : (row + 1) * square_px,
                    col * square_px : (col + 1) * square_px,
                ] = 255
    board_color = cv2.cvtColor(board, cv2.COLOR_GRAY2BGR)
    rng = np.random.default_rng(1)
    width, height = 640, 480
    for index in range(count):
        source = np.float32(
            [[0, 0], [board_width, 0], [board_width, board_height], [0, board_height]]
        )
        jitter = rng.uniform(-50, 50, size=(4, 2)).astype(np.float32)
        destination = np.float32(
            [[100, 80], [540, 60], [560, 400], [80, 420]]
        ) + jitter
        matrix = cv2.getPerspectiveTransform(source, destination)
        warped = cv2.warpPerspective(
            board_color, matrix, (width, height), borderValue=(255, 255, 255)
        )
        cv2.imwrite(str(directory / f"frame_{index:02d}.png"), warped)


def test_charuco_calibration_passes_and_writes_fragment(tmp_path):
    images_dir = tmp_path / "images"
    _write_charuco_images(images_dir, count=10)
    output = tmp_path / "intrinsics.yaml"
    report = tmp_path / "report.json"

    exit_code = ci.main(
        [
            "--images-dir",
            str(images_dir),
            "--board",
            "charuco",
            "--columns",
            "5",
            "--rows",
            "7",
            "--square-size-m",
            "0.03",
            "--marker-size-m",
            "0.022",
            "--min-accepted-images",
            "5",
            "--max-reprojection-rmse-px",
            "8.0",
            "--output",
            str(output),
            "--report",
            str(report),
        ]
    )

    assert exit_code == 0
    assert output.exists()
    fragment = yaml.safe_load(output.read_text())
    assert fragment["image_width"] == 640
    assert fragment["image_height"] == 480
    assert len(fragment["camera_matrix"]) == 3
    assert fragment["distortion_model"] == "opencv_pinhole"

    report_payload = json.loads(report.read_text())
    assert report_payload["passed_thresholds"] is True
    assert report_payload["accepted_count"] == 10
    assert report_payload["rejected_count"] == 0


def test_chessboard_calibration_passes_and_writes_fragment(tmp_path):
    images_dir = tmp_path / "images"
    _write_chessboard_images(images_dir, count=10)
    output = tmp_path / "intrinsics.yaml"

    exit_code = ci.main(
        [
            "--images-dir",
            str(images_dir),
            "--board",
            "chessboard",
            "--columns",
            "7",
            "--rows",
            "5",
            "--square-size-m",
            "0.03",
            "--min-accepted-images",
            "5",
            "--max-reprojection-rmse-px",
            "8.0",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    assert output.exists()


def test_threshold_not_met_withholds_output(tmp_path):
    images_dir = tmp_path / "images"
    _write_chessboard_images(images_dir, count=10)
    output = tmp_path / "intrinsics.yaml"

    exit_code = ci.main(
        [
            "--images-dir",
            str(images_dir),
            "--board",
            "chessboard",
            "--columns",
            "7",
            "--rows",
            "5",
            "--square-size-m",
            "0.03",
            "--min-accepted-images",
            "100",
            "--max-reprojection-rmse-px",
            "8.0",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 1
    assert not output.exists()


def test_unreadable_and_mismatched_images_are_rejected_with_reasons(tmp_path):
    images_dir = tmp_path / "images"
    _write_chessboard_images(images_dir, count=5)
    (images_dir / "not_an_image.png").write_bytes(b"not a real png")
    mismatched = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.imwrite(str(images_dir / "wrong_size.png"), mismatched)
    output = tmp_path / "intrinsics.yaml"
    report = tmp_path / "report.json"

    ci.main(
        [
            "--images-dir",
            str(images_dir),
            "--board",
            "chessboard",
            "--columns",
            "7",
            "--rows",
            "5",
            "--square-size-m",
            "0.03",
            "--min-accepted-images",
            "3",
            "--max-reprojection-rmse-px",
            "8.0",
            "--output",
            str(output),
            "--report",
            str(report),
        ]
    )

    report_payload = json.loads(report.read_text())
    reasons = {item["path"].split("/")[-1]: item["reason"] for item in report_payload["per_image"]}
    assert reasons["not_an_image.png"] == "UNREADABLE_IMAGE"
    assert reasons["wrong_size.png"] == "IMAGE_SIZE_MISMATCH"


def test_min_accepted_images_below_three_rejected(tmp_path):
    images_dir = tmp_path / "images"
    _write_chessboard_images(images_dir, count=5)
    with pytest.raises(SystemExit):
        ci.main(
            [
                "--images-dir",
                str(images_dir),
                "--board",
                "chessboard",
                "--columns",
                "7",
                "--rows",
                "5",
                "--square-size-m",
                "0.03",
                "--min-accepted-images",
                "2",
                "--output",
                str(tmp_path / "out.yaml"),
            ]
        )


def test_charuco_requires_marker_size(tmp_path):
    images_dir = tmp_path / "images"
    _write_charuco_images(images_dir, count=3)
    with pytest.raises(SystemExit):
        ci.main(
            [
                "--images-dir",
                str(images_dir),
                "--board",
                "charuco",
                "--columns",
                "5",
                "--rows",
                "7",
                "--square-size-m",
                "0.03",
                "--output",
                str(tmp_path / "out.yaml"),
            ]
        )
