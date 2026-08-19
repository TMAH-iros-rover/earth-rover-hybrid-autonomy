#!/usr/bin/env python3
"""Offline ground-plane camera extrinsic calibration via solvePnP.

Read-only over local correspondence/intrinsics files on disk. Does not
import ``earth_rover.sdk_client`` and never calls any network or SDK
endpoint.

Correspondences are explicit measured rover-frame target coordinates
(``rover_xyz_m``, rover frame: +x forward, +y left, +z up -- ground-plane
targets have ``z = 0``) matched by hand to pixel coordinates
(``pixel_uv``) picked from one saved, undistorted-in-intent rover camera
frame. Writes the ``camera_from_rover_transform`` fragment at ``--output``
only when both ``--min-points`` and ``--max-reprojection-rmse-px``
thresholds pass.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--correspondences",
        required=True,
        type=Path,
        help=(
            "YAML/JSON list of {rover_xyz_m: [x, y, z], pixel_uv: [u, v]}, "
            "at least --min-points entries"
        ),
    )
    parser.add_argument(
        "--intrinsics",
        required=True,
        type=Path,
        help=(
            "YAML/JSON containing camera_matrix and distortion_coefficients "
            "-- a full calibration file or the fragment written by "
            "calibrate_intrinsics.py"
        ),
    )
    parser.add_argument("--min-points", type=int, default=6)
    parser.add_argument("--max-reprojection-rmse-px", type=float, default=1.5)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="camera_from_rover_transform fragment, written only if thresholds pass",
    )
    parser.add_argument("--report", type=Path, help="optional JSON report path")
    return parser.parse_args(argv)


def load_intrinsics(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = _load_yaml_or_json(path)
    if "camera_matrix" not in payload or "distortion_coefficients" not in payload:
        raise SystemExit(
            f"{path} must contain camera_matrix and distortion_coefficients"
        )
    camera_matrix = np.array(payload["camera_matrix"], dtype=np.float64)
    dist_coeffs = np.array(payload["distortion_coefficients"], dtype=np.float64)
    if camera_matrix.shape != (3, 3):
        raise SystemExit("camera_matrix must have shape (3, 3)")
    return camera_matrix, dist_coeffs


def load_correspondences(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = _load_yaml_or_json(path)
    if not isinstance(payload, list):
        raise SystemExit("correspondences file must be a YAML/JSON list")
    object_points = []
    image_points = []
    for index, entry in enumerate(payload):
        if "rover_xyz_m" not in entry or "pixel_uv" not in entry:
            raise SystemExit(f"correspondence {index} missing rover_xyz_m/pixel_uv")
        xyz = [float(v) for v in entry["rover_xyz_m"]]
        uv = [float(v) for v in entry["pixel_uv"]]
        if len(xyz) != 3 or len(uv) != 2:
            raise SystemExit(f"correspondence {index} has malformed rover_xyz_m/pixel_uv")
        object_points.append(xyz)
        image_points.append(uv)
    return (
        np.array(object_points, dtype=np.float64),
        np.array(image_points, dtype=np.float64),
    )


def _load_yaml_or_json(path: Path):
    if not path.is_file():
        raise SystemExit(f"required input does not exist: {path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    return yaml.safe_load(text)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.min_points < 6:
        raise SystemExit(
            "--min-points must be at least 6 (solvePnP is well-posed at 4 "
            "non-degenerate points; 6 leaves margin for measurement noise)"
        )
    if not (args.max_reprojection_rmse_px > 0.0):
        raise SystemExit("--max-reprojection-rmse-px must be positive")

    camera_matrix, dist_coeffs = load_intrinsics(args.intrinsics)
    object_points, image_points = load_correspondences(args.correspondences)
    point_count = len(object_points)

    if point_count < args.min_points:
        report = {
            "point_count": point_count,
            "min_points": args.min_points,
            "reprojection_rmse_px": None,
            "passed_thresholds": False,
            "reason": "INSUFFICIENT_POINTS",
        }
        _emit_report(report, args.report)
        print("Thresholds not met; extrinsics fragment NOT written.", file=sys.stderr)
        return 1

    success, rvec, tvec = cv2.solvePnP(
        object_points, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        report = {
            "point_count": point_count,
            "min_points": args.min_points,
            "reprojection_rmse_px": None,
            "passed_thresholds": False,
            "reason": "SOLVE_PNP_FAILED",
        }
        _emit_report(report, args.report)
        print("Thresholds not met; extrinsics fragment NOT written.", file=sys.stderr)
        return 1

    projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
    per_point_error_px = np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1)
    rmse = float(np.sqrt(np.mean(per_point_error_px**2)))
    passed = rmse <= args.max_reprojection_rmse_px

    report = {
        "point_count": point_count,
        "min_points": args.min_points,
        "reprojection_rmse_px": rmse,
        "max_reprojection_rmse_px": args.max_reprojection_rmse_px,
        "per_point_error_px": [float(v) for v in per_point_error_px],
        "passed_thresholds": passed,
    }
    _emit_report(report, args.report)
    if not passed:
        print("Thresholds not met; extrinsics fragment NOT written.", file=sys.stderr)
        return 1

    rotation, _ = cv2.Rodrigues(rvec)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = tvec.reshape(3)
    fragment = {
        "camera_from_rover_transform": [[float(v) for v in row] for row in transform],
        "provenance": {"reprojection_error_px": rmse},
    }
    args.output.write_text(yaml.safe_dump(fragment, sort_keys=False), encoding="utf-8")
    print(f"Extrinsics fragment written to {args.output}")
    return 0


def _emit_report(report: dict, report_path: Path | None) -> None:
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if report_path:
        report_path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
