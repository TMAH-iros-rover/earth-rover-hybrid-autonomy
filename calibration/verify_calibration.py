#!/usr/bin/env python3
"""Compare raw and undistorted live Earth Rover camera frames."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import requests

from calibration_utils import (
    DEFAULT_CONFIG_PATH,
    config_path,
    fetch_frame,
    load_config,
    load_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--base-url")
    parser.add_argument("--camera", choices=("front", "rear"))
    parser.add_argument("--alpha", type=float, help="0=crop, 1=keep full field of view")
    parser.add_argument("--timeout", type=float)
    return parser.parse_args()


def scaled_matrix(data: dict, frame_size: tuple[int, int]) -> np.ndarray:
    matrix = np.asarray(data["camera_matrix"], dtype=np.float64)
    source = (int(data["image_width"]), int(data["image_height"]))
    if source == frame_size:
        return matrix
    source_ratio = source[0] / source[1]
    frame_ratio = frame_size[0] / frame_size[1]
    if abs(source_ratio - frame_ratio) > 1e-3:
        raise RuntimeError(
            f"Aspect ratio changed from {source[0]}x{source[1]} to "
            f"{frame_size[0]}x{frame_size[1]}; recalibrate at the live resolution"
        )
    scale_x, scale_y = frame_size[0] / source[0], frame_size[1] / source[1]
    matrix[0, :] *= scale_x
    matrix[1, :] *= scale_y
    matrix[2, 2] = 1.0
    return matrix


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    calibration = args.calibration or config_path(config["result_file"], args.config)
    base_url = args.base_url or config["base_url"]
    camera = args.camera or config["camera"]
    alpha = args.alpha if args.alpha is not None else config["verification_alpha"]
    timeout = args.timeout if args.timeout is not None else config["request_timeout_sec"]
    data = load_json(calibration)
    distortion = np.asarray(data["dist_coeffs"], dtype=np.float64)
    session = requests.Session()
    maps = None
    map_size = None
    roi = None
    try:
        while True:
            frame, _ = fetch_frame(session, base_url, camera, timeout)
            size = (frame.shape[1], frame.shape[0])
            if maps is None or map_size != size:
                matrix = scaled_matrix(data, size)
                new_matrix, roi = cv2.getOptimalNewCameraMatrix(
                    matrix, distortion, size, alpha, size
                )
                maps = cv2.initUndistortRectifyMap(
                    matrix, distortion, None, new_matrix, size, cv2.CV_16SC2
                )
                map_size = size
                valid_fraction = (roi[2] * roi[3]) / (size[0] * size[1])
                print(
                    f"Live resolution: {size[0]}x{size[1]} | alpha={alpha:.2f} | "
                    f"valid ROI={roi} ({valid_fraction:.1%})"
                )
                if valid_fraction < 0.1:
                    print(
                        "WARNING: The valid undistorted area is very small. "
                        "Set verification_alpha closer to 0.0."
                    )
            corrected = cv2.remap(frame, maps[0], maps[1], cv2.INTER_LINEAR)
            comparison = np.hstack((frame, corrected))
            cv2.putText(comparison, "RAW", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(comparison, "UNDISTORTED", (size[0] + 16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Earth Rover calibration verification (Q to quit)", comparison)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        session.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
