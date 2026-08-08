#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from earth_rover.control.command_filter import CommandFilter
from earth_rover.control.hybrid_controller import HybridReactiveController
from earth_rover.core.types import ControlCommand
from earth_rover.perception.dummy_traversability import DummyTraversabilityModel
from earth_rover.planning.candidate_planner import CandidateDirectionPlanner
from earth_rover.replay import ReplayDataset
from earth_rover.utils.config import load_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    config = load_config(ROOT / args.config)
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.run_dir) / "replay"
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = ReplayDataset(args.run_dir)
    perception = DummyTraversabilityModel(config)
    planner = CandidateDirectionPlanner(config)
    controller = HybridReactiveController(config)
    command_filter = CommandFilter(config)
    counts = Counter()

    with (output_dir / "commands.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["timestamp", "linear", "angular", "mode", "candidate_name"])
        writer.writeheader()
        for frame, _data in dataset:
            result = perception.infer(frame)
            candidate = planner.select(heading_error_rad=0.0, perception=result)
            raw = controller.compute(
                heading_error_rad=0.0,
                candidate=candidate,
                perception=result,
                emergency_stop=False,
                recovery_command=None,
                dt=0.2,
            )
            command = command_filter.apply(raw, dt=0.2, frame_is_stale=False, data_is_stale=False)
            counts[candidate.name] += 1
            writer.writerow(
                {
                    "timestamp": frame.timestamp,
                    "linear": command.linear,
                    "angular": command.angular,
                    "mode": command.mode,
                    "candidate_name": candidate.name,
                }
            )

    with (output_dir / "candidate_stats.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["candidate_name", "count"])
        for name, count in counts.most_common():
            writer.writerow([name, count])
    print(f"replay complete: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

