#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from earth_rover.planning.motion_primitive_planner import (  # noqa: E402
    MotionPrimitivePlanner,
    MotionPrimitivePlannerConfig,
    primitive_curve_points,
)


class Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value


def main() -> int:
    parser = argparse.ArgumentParser(description="Synthetic Mission1 planner stability comparison")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["motion_primitives", "gps_only"],
        choices=["motion_primitives", "gps_only"],
    )
    parser.add_argument("--frames", type=int, default=40)
    args = parser.parse_args()
    for mode in args.modes:
        metrics = run_sequence(mode, args.frames)
        print(
            f"{mode}: switches={metrics['switch_count']} "
            f"switches_per_sec={metrics['switches_per_sec']:.2f} "
            f"angular_sign_flips={metrics['angular_sign_flip_count']} "
            f"valid_ratio={metrics['valid_ratio']:.2f} "
            f"held_duration_sec={metrics['held_plan_duration_sec']:.2f} "
            f"full_stops={metrics['full_stop_recommendation_count']} "
            f"near_field_stops={metrics['near_field_stop_count']} "
            f"mean_confidence={metrics['mean_planner_confidence']:.2f}",
            flush=True,
        )
    return 0


def run_sequence(mode: str, frames: int) -> dict[str, float]:
    clock = Clock()
    planner = MotionPrimitivePlanner(
        MotionPrimitivePlannerConfig(mode=mode, candidate_score_ema_alpha=0.35),
        monotonic=clock,
    )
    previous_heading = None
    previous_sign = 0
    switch_count = 0
    sign_flips = 0
    valid_count = 0
    held_frames = 0
    stop_count = 0
    near_stop_count = 0
    confidence = []
    for index in range(frames):
        clock.value = index * 0.25
        score, valid = synthetic_score(index)
        target_heading = math.radians(0.0 if index < frames * 0.75 else 35.0)
        plan = planner.plan(score, valid, target_heading_error_rad=target_heading, checkpoint_sequence=2)
        heading = (
            plan.selected_candidate.heading_deg if plan.selected_candidate is not None else 0.0
        )
        if previous_heading is not None and heading != previous_heading:
            switch_count += 1
        sign = 1 if heading > 0 else -1 if heading < 0 else 0
        if previous_sign and sign and previous_sign != sign:
            sign_flips += 1
        previous_heading = heading
        previous_sign = sign or previous_sign
        valid_count += int(plan.path_valid)
        held_frames += int(plan.using_held_plan)
        stop_count += int(not plan.path_valid)
        near_stop_count += int(not plan.near_field_safe)
        confidence.append(plan.planner_confidence)
    duration = max(1e-6, frames * 0.25)
    return {
        "switch_count": float(switch_count),
        "switches_per_sec": switch_count / duration,
        "angular_sign_flip_count": float(sign_flips),
        "valid_ratio": valid_count / max(1, frames),
        "held_plan_duration_sec": held_frames * 0.25,
        "full_stop_recommendation_count": float(stop_count),
        "near_field_stop_count": float(near_stop_count),
        "mean_planner_confidence": float(np.mean(confidence)) if confidence else 0.0,
    }


def synthetic_score(index: int, shape: tuple[int, int] = (120, 160)) -> tuple[np.ndarray, np.ndarray]:
    score = np.full(shape, 0.42, dtype=np.float32)
    valid = np.ones(shape, dtype=bool)
    # Central and left corridors cross slightly in score, which should not
    # cause frame-by-frame switching once hysteresis is active.
    central = 0.78 + (0.04 if index % 2 == 0 else -0.04)
    left = 0.76 + (-0.04 if index % 2 == 0 else 0.04)
    paint(score, 0.0, central)
    paint(score, 15.0, left)
    if index == 12:
        paint(score, 0.0, 0.10)
    if 24 <= index <= 27:
        score[:, :] = 0.10
    if index >= 32:
        paint(score, 30.0, 0.88)
    return score, valid


def paint(score: np.ndarray, heading_deg: float, value: float) -> None:
    points = primitive_curve_points(score.shape, heading_deg)
    for x, y in points:
        score[max(0, y - 2) : min(score.shape[0], y + 3), max(0, x - 3) : min(score.shape[1], x + 4)] = value


if __name__ == "__main__":
    raise SystemExit(main())
