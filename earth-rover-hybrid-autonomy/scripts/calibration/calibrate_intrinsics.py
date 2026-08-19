#!/usr/bin/env python3
"""Offline camera intrinsic calibration from saved chessboard/ChArUco images.

Read-only over local image files on disk. Does not import
``earth_rover.sdk_client`` and never calls any network or SDK endpoint --
this tool only ever reads image files and writes a local YAML/JSON output.

Writes the intrinsics-only YAML fragment at ``--output`` only when both
``--min-accepted-images`` and ``--max-reprojection-rmse-px`` thresholds
pass; otherwise it prints/writes the numerical report and exits non-zero
without producing output, so a bad calibration attempt cannot silently
become a usable file.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml


@dataclass
class ImageReport:
    path: str
    accepted: bool
    reprojection_error_px: float | None = None
    reason: str | None = None


@dataclass
class CalibrationResult:
    accepted_count: int
    rmse_px: float | None
    per_image: list[ImageReport] = field(default_factory=list)
    image_width: int = 0
    image_height: int = 0
    camera_matrix: np.ndarray | None = None
    dist_coeffs: np.ndarray | None = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--glob", default="*.png", help="image filename glob within --images-dir")
    parser.add_argument("--board", choices=("chessboard", "charuco"), default="chessboard")
    parser.add_argument(
        "--columns",
        type=int,
        required=True,
        help="inner corners per row (chessboard) or squares per row (charuco)",
    )
    parser.add_argument(
        "--rows",
        type=int,
        required=True,
        help="inner corners per column (chessboard) or squares per column (charuco)",
    )
    parser.add_argument("--square-size-m", type=float, required=True)
    parser.add_argument(
        "--marker-size-m",
        type=float,
        help="ChArUco marker size in meters; required when --board charuco",
    )
    parser.add_argument(
        "--aruco-dictionary",
        default="DICT_5X5_50",
        help="cv2.aruco predefined dictionary name, ChArUco only",
    )
    parser.add_argument("--min-accepted-images", type=int, default=10)
    parser.add_argument("--max-reprojection-rmse-px", type=float, default=0.75)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="intrinsics-only YAML fragment, written only if thresholds pass",
    )
    parser.add_argument("--report", type=Path, help="optional JSON report path")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.board == "charuco" and args.marker_size_m is None:
        raise SystemExit("--marker-size-m is required when --board charuco")
    if args.min_accepted_images < 3:
        raise SystemExit("--min-accepted-images must be at least 3")
    if not (args.max_reprojection_rmse_px > 0.0):
        raise SystemExit("--max-reprojection-rmse-px must be positive")
    if not args.images_dir.is_dir():
        raise SystemExit(f"--images-dir does not exist: {args.images_dir}")

    images = sorted(args.images_dir.glob(args.glob))
    if not images:
        raise SystemExit(f"no images matched {args.images_dir}/{args.glob}")

    if args.board == "chessboard":
        result = calibrate_chessboard(
            images, args.columns, args.rows, args.square_size_m
        )
    else:
        result = calibrate_charuco(
            images,
            args.columns,
            args.rows,
            args.square_size_m,
            args.marker_size_m,
            args.aruco_dictionary,
        )

    passed = (
        result.accepted_count >= args.min_accepted_images
        and result.rmse_px is not None
        and result.rmse_px <= args.max_reprojection_rmse_px
    )
    report = {
        "board": args.board,
        "images_considered": len(images),
        "accepted_count": result.accepted_count,
        "rejected_count": len(images) - result.accepted_count,
        "per_image": [vars(item) for item in result.per_image],
        "reprojection_rmse_px": result.rmse_px,
        "image_width": result.image_width,
        "image_height": result.image_height,
        "min_accepted_images": args.min_accepted_images,
        "max_reprojection_rmse_px": args.max_reprojection_rmse_px,
        "passed_thresholds": passed,
    }
    report_text = json.dumps(report, indent=2, sort_keys=True)
    print(report_text)
    if args.report:
        args.report.write_text(report_text + "\n", encoding="utf-8")

    if not passed:
        print("Thresholds not met; intrinsics fragment NOT written.", file=sys.stderr)
        return 1

    assert result.camera_matrix is not None and result.dist_coeffs is not None
    fragment = {
        "image_width": result.image_width,
        "image_height": result.image_height,
        "camera_matrix": [[float(v) for v in row] for row in result.camera_matrix],
        "distortion_model": "opencv_pinhole",
        "distortion_coefficients": [float(v) for v in np.asarray(result.dist_coeffs).reshape(-1)],
        "provenance": {
            "capture_source": f"{args.board}_calibration:{args.images_dir}",
            "reprojection_error_px": result.rmse_px,
        },
    }
    args.output.write_text(yaml.safe_dump(fragment, sort_keys=False), encoding="utf-8")
    print(f"Intrinsics fragment written to {args.output}")
    return 0


def calibrate_chessboard(
    images: list[Path],
    columns: int,
    rows: int,
    square_size_m: float,
) -> CalibrationResult:
    pattern_size = (columns, rows)
    object_template = np.zeros((rows * columns, 3), dtype=np.float32)
    object_template[:, :2] = (
        np.mgrid[0:columns, 0:rows].T.reshape(-1, 2).astype(np.float32) * square_size_m
    )

    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    per_image: list[ImageReport] = []
    image_size: tuple[int, int] | None = None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for path in images:
        image = cv2.imread(str(path))
        if image is None:
            per_image.append(ImageReport(str(path), False, reason="UNREADABLE_IMAGE"))
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        size = (gray.shape[1], gray.shape[0])
        if image_size is None:
            image_size = size
        elif size != image_size:
            per_image.append(ImageReport(str(path), False, reason="IMAGE_SIZE_MISMATCH"))
            continue
        found, corners = cv2.findChessboardCorners(
            gray,
            pattern_size,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        if not found:
            per_image.append(ImageReport(str(path), False, reason="CHESSBOARD_NOT_FOUND"))
            continue
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        object_points.append(object_template)
        image_points.append(corners)
        per_image.append(ImageReport(str(path), True))

    return _run_calibrate_camera(object_points, image_points, per_image, image_size)


def calibrate_charuco(
    images: list[Path],
    columns: int,
    rows: int,
    square_size_m: float,
    marker_size_m: float,
    aruco_dictionary_name: str,
) -> CalibrationResult:
    dictionary_id = getattr(cv2.aruco, aruco_dictionary_name, None)
    if dictionary_id is None:
        raise SystemExit(f"unknown --aruco-dictionary: {aruco_dictionary_name}")
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    board = cv2.aruco.CharucoBoard((columns, rows), square_size_m, marker_size_m, dictionary)
    detector = cv2.aruco.CharucoDetector(board)

    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    per_image: list[ImageReport] = []
    image_size: tuple[int, int] | None = None
    minimum_corners = max(6, (columns - 1) * (rows - 1) // 2)

    for path in images:
        image = cv2.imread(str(path))
        if image is None:
            per_image.append(ImageReport(str(path), False, reason="UNREADABLE_IMAGE"))
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        size = (gray.shape[1], gray.shape[0])
        if image_size is None:
            image_size = size
        elif size != image_size:
            per_image.append(ImageReport(str(path), False, reason="IMAGE_SIZE_MISMATCH"))
            continue
        charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)
        if charuco_corners is None or len(charuco_corners) < minimum_corners:
            per_image.append(ImageReport(str(path), False, reason="CHARUCO_BOARD_NOT_FOUND"))
            continue
        obj_points, img_points = board.matchImagePoints(charuco_corners, charuco_ids)
        if obj_points is None or len(obj_points) < minimum_corners:
            per_image.append(ImageReport(str(path), False, reason="CHARUCO_INSUFFICIENT_POINTS"))
            continue
        object_points.append(obj_points)
        image_points.append(img_points)
        per_image.append(ImageReport(str(path), True))

    return _run_calibrate_camera(object_points, image_points, per_image, image_size)


def _run_calibrate_camera(
    object_points: list[np.ndarray],
    image_points: list[np.ndarray],
    per_image: list[ImageReport],
    image_size: tuple[int, int] | None,
) -> CalibrationResult:
    accepted_count = len(object_points)
    if accepted_count < 3 or image_size is None:
        return CalibrationResult(accepted_count, None, per_image, 0, 0, None, None)

    rmse, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None
    )
    accepted_index = 0
    for report in per_image:
        if not report.accepted:
            continue
        projected, _ = cv2.projectPoints(
            object_points[accepted_index],
            rvecs[accepted_index],
            tvecs[accepted_index],
            camera_matrix,
            dist_coeffs,
        )
        error = float(
            np.sqrt(
                np.mean(
                    np.sum(
                        (
                            projected.reshape(-1, 2)
                            - image_points[accepted_index].reshape(-1, 2)
                        )
                        ** 2,
                        axis=1,
                    )
                )
            )
        )
        report.reprojection_error_px = error
        accepted_index += 1

    return CalibrationResult(
        accepted_count,
        float(rmse),
        per_image,
        image_size[0],
        image_size[1],
        camera_matrix,
        dist_coeffs,
    )


if __name__ == "__main__":
    raise SystemExit(main())
