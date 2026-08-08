#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def draw_line_chart(values: pd.Series, title: str, width: int = 900, height: int = 260) -> np.ndarray:
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    cv2.putText(image, title, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2, cv2.LINE_AA)
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        cv2.putText(image, "no data", (16, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 80), 2, cv2.LINE_AA)
        return image
    min_v = float(clean.min())
    max_v = float(clean.max())
    if abs(max_v - min_v) < 1e-9:
        max_v = min_v + 1.0
    points = []
    left, right, top, bottom = 60, width - 20, 48, height - 34
    for idx, value in enumerate(clean.to_list()):
        x = left + int((right - left) * idx / max(1, len(clean) - 1))
        y = bottom - int((bottom - top) * (float(value) - min_v) / (max_v - min_v))
        points.append((x, y))
    cv2.rectangle(image, (left, top), (right, bottom), (210, 210, 210), 1)
    if len(points) > 1:
        cv2.polylines(image, [np.array(points, dtype=np.int32)], False, (40, 110, 210), 2, cv2.LINE_AA)
    cv2.putText(image, f"min {min_v:.3f}", (left, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1)
    cv2.putText(image, f"max {max_v:.3f}", (right - 110, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1)
    return image


def read_events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    events = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    out_dir = run_dir / "viz"
    out_dir.mkdir(parents=True, exist_ok=True)

    commands = pd.read_csv(run_dir / "commands.csv")
    cv2.imwrite(str(out_dir / "command_linear.png"), draw_line_chart(commands["linear"], "command linear"))
    cv2.imwrite(str(out_dir / "command_angular.png"), draw_line_chart(commands["angular"], "command angular"))
    if "heading_error_deg" in commands:
        cv2.imwrite(str(out_dir / "heading_error.png"), draw_line_chart(commands["heading_error_deg"], "heading error deg"))
    if "obstacle_confidence" in commands:
        cv2.imwrite(
            str(out_dir / "obstacle_confidence.png"),
            draw_line_chart(commands["obstacle_confidence"], "obstacle confidence"),
        )

    candidate_counts = Counter(commands.get("candidate_name", pd.Series(dtype=str)).fillna(""))
    events = read_events(run_dir / "events.jsonl")
    with (out_dir / "summary.txt").open("w", encoding="utf-8") as handle:
        handle.write("candidate counts\n")
        for name, count in candidate_counts.most_common():
            handle.write(f"{name}: {count}\n")
        handle.write("\nevents\n")
        for event in events:
            handle.write(f"{event.get('timestamp')}: {event.get('event')} {event.get('detail')}\n")

    print(f"visualizations written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

