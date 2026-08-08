#!/usr/bin/env python3
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from earth_rover.control.command_filter import CommandFilter
from earth_rover.core.types import ControlCommand
from earth_rover.navigation.gps_utils import normalize_angle_deg
from earth_rover.autonomy.mission1_controller import mission1_to_sdk_angular
from earth_rover.planning.motion_primitive_planner import (
    MotionPrimitivePlanner,
    MotionPrimitivePlannerConfig,
    image_direction_from_x_offset,
    primitive_curve_points,
    selected_candidate_endpoint_x_offset_px,
)


SDK_CONVENTION_SOURCE = [
    "live rover test: SDK angular < 0 turns physical RIGHT and increases rover heading",
    "live rover test: SDK angular > 0 turns physical LEFT and decreases rover heading",
]


@dataclass(frozen=True)
class Case:
    name: str
    current_heading_deg: float
    target_bearing_deg: float
    expected_physical: str


def direction_from_signed_angle(angle_deg: float, *, deadband_deg: float = 1.0) -> str:
    if angle_deg > deadband_deg:
        return "RIGHT"
    if angle_deg < -deadband_deg:
        return "LEFT"
    return "CENTER"


def internal_direction_from_angular(angular: float, *, deadband: float = 1e-6) -> str:
    if angular > deadband:
        return "RIGHT"
    if angular < -deadband:
        return "LEFT"
    return "CENTER"


def sdk_physical_direction_from_angular(angular: float, *, deadband: float = 1e-6) -> str:
    if angular < -deadband:
        return "RIGHT"
    if angular > deadband:
        return "LEFT"
    return "CENTER"


def score_map_for_heading(
    heading_deg: float,
    *,
    shape: tuple[int, int] = (120, 160),
) -> tuple[np.ndarray, np.ndarray]:
    score = np.full(shape, 0.35, dtype=np.float32)
    valid = np.ones(shape, dtype=bool)
    points = primitive_curve_points(shape, heading_deg)
    for x, y in points:
        score[max(0, y - 2) : min(shape[0], y + 3), max(0, x - 3) : min(shape[1], x + 4)] = 0.95
    return score, valid


def command_from_heading_deg(heading_deg: float) -> ControlCommand:
    heading_rad = math.radians(heading_deg)
    angular = max(-0.22, min(0.22, 0.40 * heading_rad))
    return ControlCommand(0.04, angular, mode="DIAG")


def run_case(case: Case) -> bool:
    nav_error = normalize_angle_deg(case.target_bearing_deg - case.current_heading_deg)
    planner = MotionPrimitivePlanner(
        MotionPrimitivePlannerConfig(
            candidate_score_ema_alpha=1.0,
            min_candidate_commit_sec=0.1,
            switch_confirm_count=1,
            path_score_threshold=0.20,
            near_field_stop_threshold=0.10,
        )
    )
    candidate_heading = max(-45.0, min(45.0, round(nav_error / 15.0) * 15.0))
    score, valid = score_map_for_heading(candidate_heading)
    plan = planner.plan(score, valid, target_heading_error_rad=math.radians(nav_error))
    selected = plan.selected_candidate
    if selected is None:
        raise RuntimeError(f"{case.name}: no selected candidate")
    offset = selected_candidate_endpoint_x_offset_px(selected.points_uv)
    path_direction = image_direction_from_x_offset(offset)
    raw = command_from_heading_deg(selected.heading_deg)
    filtered = CommandFilter(
        {
            "control": {
                "command_smoothing_alpha": 0.0,
                "max_linear_delta_per_sec": 10.0,
                "max_angular_delta_per_sec": 10.0,
            }
        }
    ).apply(raw, 1.0, frame_is_stale=False, data_is_stale=False)
    sdk_angular = mission1_to_sdk_angular(filtered.angular)
    result = {
        "navigation": direction_from_signed_angle(nav_error),
        "planner": direction_from_signed_angle(selected.heading_deg),
        "image": path_direction,
        "internal_command": internal_direction_from_angular(filtered.angular),
        "sdk_command": sdk_physical_direction_from_angular(sdk_angular),
    }
    passed = all(value == case.expected_physical for value in result.values())
    print(f"CASE: {case.name}")
    print(f"Expected physical turn: {case.expected_physical}")
    print(f"GPS bearing:              {case.target_bearing_deg:+.1f} deg compass clockwise")
    print(f"Current heading:          {case.current_heading_deg:+.1f} deg assumed compass")
    print(f"Navigation error:         {nav_error:+.1f} deg = {result['navigation']}")
    print(f"Selected candidate:       {selected.heading_deg:+.1f} deg = {result['planner']}")
    print(f"Candidate endpoint offset:{offset:+d} px = {result['image']}")
    print(f"Internal raw angular:     {raw.angular:+.3f} = {internal_direction_from_angular(raw.angular)}")
    print(f"Internal filtered angular:{filtered.angular:+.3f} = {result['internal_command']}")
    print(f"SDK transmitted angular:  {sdk_angular:+.3f} = {result['sdk_command']}")
    print(f"RESULT: {'PASS' if passed else 'FAIL'}")
    if not passed:
        for stage, value in result.items():
            if value != case.expected_physical:
                print(f"FIRST MISMATCH: {stage} produced {value}")
                break
    print()
    return passed


def print_candidate_geometry() -> None:
    print("MOTION PRIMITIVE GEOMETRY")
    shape = (120, 160)
    for heading in (-45, -30, -15, 0, 15, 30, 45):
        points = primitive_curve_points(shape, heading)
        offset = selected_candidate_endpoint_x_offset_px(points)
        print(
            f"candidate {heading:+4.0f} deg: first={tuple(points[0])} "
            f"mid={tuple(points[len(points)//2])} end={tuple(points[-1])} "
            f"offset={offset:+4d}px direction={image_direction_from_x_offset(offset)}"
        )
    print()


def print_overlay_arrow_formula() -> None:
    width = 160
    center_x = width // 2
    length = 50
    print("OVERLAY ARROW ENDPOINT CHECK")
    for theta_deg in (30.0, -30.0):
        theta = math.radians(theta_deg)
        end_x = round(center_x + length * math.sin(theta))
        print(
            f"theta={theta_deg:+.1f} deg with x=center+sin(theta): "
            f"end_x-center={end_x - center_x:+d}px "
            f"direction={image_direction_from_x_offset(end_x - center_x)}"
        )
    print("Live SAM-TP path overlay draws selected candidate points; legacy goal-behind label is reported separately.")
    print()


def print_sdk_manual_convention() -> None:
    print("SDK MANUAL CONTROL CONVENTION EVIDENCE")
    for line in SDK_CONVENTION_SOURCE:
        print(f"- {line}")
    print("Conclusion used by this diagnostic: SDK angular negative=RIGHT, positive=LEFT.")
    print()


def main() -> int:
    print("Mission1 direction sign diagnostic")
    print("Mission1 internal convention: positive angle/angular = clockwise physical RIGHT.")
    print("Earth Rover SDK command convention: negative angular = physical RIGHT.")
    print()
    print_candidate_geometry()
    print_overlay_arrow_formula()
    print_sdk_manual_convention()
    cases = (
        Case("north_to_east", 0.0, 90.0, "RIGHT"),
        Case("north_to_west", 0.0, 270.0, "LEFT"),
        Case("east_to_north", 90.0, 0.0, "LEFT"),
        Case("wrap_350_to_10", 350.0, 10.0, "RIGHT"),
        Case("wrap_10_to_350", 10.0, 350.0, "LEFT"),
    )
    passed = [run_case(case) for case in cases]
    print(f"SUMMARY: {sum(passed)}/{len(passed)} cases passed")
    return 0 if all(passed) else 1


if __name__ == "__main__":
    raise SystemExit(main())
