#!/usr/bin/env python3
"""Calculate intrinsic camera parameters from saved checkerboard images."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from calibration_utils import (
    DEFAULT_CONFIG_PATH,
    config_path,
    find_checkerboard,
    load_config,
    save_json,
)


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def calibration_object_points(
    pattern_size: tuple[int, int], square_size_mm: float
) -> np.ndarray:
    points = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    points[:, :2] = np.mgrid[
        0 : pattern_size[0], 0 : pattern_size[1]
    ].T.reshape(-1, 2)
    points[:, :2] *= square_size_mm
    return points


def calibrate_directory(
    image_dir: Path,
    output: Path,
    pattern_size: tuple[int, int],
    square_size_mm: float,
    camera: str,
) -> dict:
    image_paths = sorted(
        path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
    ) if image_dir.exists() else []
    if not image_paths:
        raise RuntimeError(f"No calibration images found in {image_dir}")

    template = calibration_object_points(pattern_size, square_size_mm)
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    used_paths: list[Path] = []
    rejected: list[dict[str, str]] = []
    image_size: tuple[int, int] | None = None

    for path in image_paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            rejected.append({"file": path.name, "reason": "unreadable"})
            continue
        current_size = (image.shape[1], image.shape[0])
        if image_size is None:
            image_size = current_size
        if current_size != image_size:
            rejected.append({"file": path.name, "reason": "resolution_mismatch"})
            continue
        found, corners = find_checkerboard(image, pattern_size)
        if not found or corners is None:
            rejected.append({"file": path.name, "reason": "corners_not_found"})
            continue
        object_points.append(template.copy())
        image_points.append(corners.astype(np.float32))
        used_paths.append(path)

    if len(used_paths) < 10:
        raise RuntimeError(
            f"Only {len(used_paths)} valid images were found; collect at least 10 "
            "(30-50 recommended)"
        )
    assert image_size is not None
    rms, matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None
    )

    per_view_errors: list[dict[str, float | str]] = []
    for path, world, observed, rotation, translation in zip(
        used_paths, object_points, image_points, rvecs, tvecs
    ):
        projected, _ = cv2.projectPoints(
            world, rotation, translation, matrix, distortion
        )
        # Root mean square Euclidean distance per detected corner, in pixels.
        error = cv2.norm(observed, projected, cv2.NORM_L2) / np.sqrt(len(projected))
        per_view_errors.append({"file": path.name, "error_px": float(error)})

    result = {
        "schema_version": 1,
        "camera": camera,
        "image_width": image_size[0],
        "image_height": image_size[1],
        "checkerboard": {
            "inner_corners": [pattern_size[0], pattern_size[1]],
            "square_size_mm": square_size_mm,
        },
        "camera_matrix": matrix.tolist(),
        "distortion_model": "opencv_plumb_bob",
        "dist_coeffs": distortion.reshape(-1).tolist(),
        "rms_error": float(rms),
        "mean_reprojection_error_px": float(
            np.mean([item["error_px"] for item in per_view_errors])
        ),
        "valid_image_count": len(used_paths),
        "rejected_image_count": len(rejected),
        "per_view_errors": per_view_errors,
        "rejected_images": rejected,
    }
    save_json(output, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--images", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--camera", choices=("front", "rear"))
    parser.add_argument("--cols", type=int, help="Horizontal inner corners")
    parser.add_argument("--rows", type=int, help="Vertical inner corners")
    parser.add_argument("--square-mm", type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    checkerboard = config["checkerboard"]
    camera = args.camera or config["camera"]
    cols = args.cols if args.cols is not None else checkerboard["inner_corners_cols"]
    rows = args.rows if args.rows is not None else checkerboard["inner_corners_rows"]
    square_mm = (
        args.square_mm if args.square_mm is not None else checkerboard["square_size_mm"]
    )
    images = args.images or config_path(config["images_dir"], args.config)
    output = args.output or config_path(config["result_file"], args.config)
    result = calibrate_directory(
        images, output, (cols, rows), square_mm, camera
    )
    print(f"Saved calibration: {output}")
    print(f"Valid images: {result['valid_image_count']}")
    print(f"RMS error: {result['rms_error']:.4f}")
    print(f"Mean reprojection error: {result['mean_reprojection_error_px']:.4f} px")


if __name__ == "__main__":
    main()
