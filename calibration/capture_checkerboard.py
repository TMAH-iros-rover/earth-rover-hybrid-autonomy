#!/usr/bin/env python3
"""Capture checkerboard frames interactively from the Earth Rover SDK."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import requests

from calibrate_camera import calibrate_directory
from calibration_utils import (
    DEFAULT_CONFIG_PATH,
    config_path,
    fetch_frame,
    find_checkerboard,
    load_config,
    save_json,
)


def draw_label(image, text: str, y: int, color: tuple[int, int, int]) -> None:
    cv2.putText(
        image, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA
    )


def next_index(image_dir: Path, camera: str) -> int:
    indexes = []
    for path in image_dir.glob(f"{camera}_*.png"):
        try:
            indexes.append(int(path.stem.rsplit("_", 1)[1]))
        except ValueError:
            continue
    return max(indexes, default=0) + 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--base-url")
    parser.add_argument("--camera", choices=("front", "rear"))
    parser.add_argument("--cols", type=int, help="Horizontal inner corners")
    parser.add_argument("--rows", type=int, help="Vertical inner corners")
    parser.add_argument("--square-mm", type=float)
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--images", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    checkerboard = config["checkerboard"]
    camera = args.camera or config["camera"]
    base_url = args.base_url or config["base_url"]
    timeout = args.timeout if args.timeout is not None else config["request_timeout_sec"]
    cols = args.cols if args.cols is not None else checkerboard["inner_corners_cols"]
    rows = args.rows if args.rows is not None else checkerboard["inner_corners_rows"]
    square_mm = (
        args.square_mm if args.square_mm is not None else checkerboard["square_size_mm"]
    )
    image_dir = args.images or config_path(config["images_dir"], args.config)
    output = args.output or config_path(config["result_file"], args.config)
    image_dir.mkdir(parents=True, exist_ok=True)
    pattern_size = (cols, rows)
    index = next_index(image_dir, camera)
    saved_count = len(list(image_dir.glob(f"{camera}_*.png")))
    session = requests.Session()
    window = f"Earth Rover {camera} calibration"
    last_error = ""
    last_error_at = 0.0

    print("SPACE: save detected frame | D: delete last | C: calibrate | Q: quit")
    try:
        while True:
            try:
                frame, sdk_metadata = fetch_frame(
                    session, base_url, camera, timeout
                )
                found, corners = find_checkerboard(frame, pattern_size)
                display = frame.copy()
                if found and corners is not None:
                    cv2.drawChessboardCorners(display, pattern_size, corners, found)
                status = "DETECTED" if found else "NOT DETECTED"
                color = (0, 220, 0) if found else (0, 0, 255)
                draw_label(display, status, 28, color)
                draw_label(display, f"Captured: {saved_count} (30-50 recommended)", 56, (255, 255, 255))
                draw_label(display, "SPACE save | D delete | C calibrate | Q quit", 84, (255, 255, 255))
                if last_error and time.monotonic() - last_error_at < 3:
                    draw_label(display, last_error, 112, (0, 165, 255))
                cv2.imshow(window, display)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord(" "):
                    if not found:
                        last_error = "Not saved: checkerboard corners were not detected"
                        last_error_at = time.monotonic()
                        continue
                    path = image_dir / f"{camera}_{index:04d}.png"
                    if not cv2.imwrite(str(path), frame):
                        raise RuntimeError(f"Failed to write {path}")
                    metadata = {
                        "camera": camera,
                        "image_file": path.name,
                        "image_width": frame.shape[1],
                        "image_height": frame.shape[0],
                        "checkerboard_inner_corners": [cols, rows],
                        "square_size_mm": square_mm,
                        "captured_at_unix": time.time(),
                        "sdk_metadata": sdk_metadata,
                    }
                    save_json(path.with_suffix(".json"), metadata)
                    print(f"Saved {path}")
                    index += 1
                    saved_count += 1
                elif key == ord("d"):
                    candidates = sorted(image_dir.glob(f"{camera}_*.png"))
                    if candidates:
                        path = candidates[-1]
                        path.unlink()
                        path.with_suffix(".json").unlink(missing_ok=True)
                        saved_count -= 1
                        index = next_index(image_dir, camera)
                        print(f"Deleted {path}")
                elif key == ord("c"):
                    try:
                        result = calibrate_directory(
                            image_dir, output, pattern_size, square_mm, camera
                        )
                        last_error = (
                            f"Calibration saved: RMS={result['rms_error']:.3f}, "
                            f"mean={result['mean_reprojection_error_px']:.3f}px"
                        )
                        print(last_error)
                    except RuntimeError as exc:
                        last_error = str(exc)
                    last_error_at = time.monotonic()
            except (requests.RequestException, ValueError) as exc:
                last_error = f"SDK error: {exc}"
                last_error_at = time.monotonic()
                print(last_error)
                time.sleep(0.2)
    finally:
        session.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
