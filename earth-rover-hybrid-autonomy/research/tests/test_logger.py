import csv
import json

import numpy as np

from earth_rover.core.types import CandidateDirection, ControlCommand, FrameData, PerceptionResult, RoverData
from earth_rover.logger import RunLogger


def test_timeline_csv_contains_required_run_fields(tmp_path):
    logger = RunLogger(
        {
            "logging": {
                "enabled": True,
                "save_frames": True,
                "save_debug_images": False,
                "log_dir": str(tmp_path),
            }
        }
    )
    frame = FrameData(timestamp=1.0, image=np.zeros((8, 8, 3), dtype=np.uint8), source="front", sdk_timestamp=0.5)
    data = RoverData(
        timestamp=2.0,
        latitude=30.1,
        longitude=114.2,
        orientation=90.0,
        speed=0.1,
        rpms=[1.0, 2.0],
        battery=95.0,
        signal_level=4.0,
        gps_signal=30.0,
        raw={},
        sdk_timestamp=1.5,
    )
    perception = PerceptionResult(0.7, 0.8, 0.6, 0.2, 0.7)
    candidate = CandidateDirection("CENTER", 0.0, 0.8, 0.2, 0.0, 1.0)
    command = ControlCommand(0.2, -0.1, mode="NORMAL_DRIVE")

    logger.log_step(
        frame=frame,
        data=data,
        perception=perception,
        candidate=candidate,
        raw_command=command,
        command=command,
        extra_debug={
            "target_checkpoint": {"sequence": 1, "latitude": "30.2", "longitude": "114.3"},
            "distance_to_checkpoint": 12.5,
            "heading_error_deg": -15.0,
        },
    )
    logger.close()

    timeline = next(tmp_path.glob("run_*/timeline.csv"))
    with timeline.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 1
    row = rows[0]
    assert row["frame_timestamp"] == "1.0"
    assert row["frame_sdk_timestamp"] == "0.5"
    assert row["data_timestamp"] == "2.0"
    assert row["data_sdk_timestamp"] == "1.5"
    assert row["command_timestamp"]
    assert row["front_frame_path"] == "frames/front_000001.jpg"
    assert row["rear_frame_path"] == ""
    assert row["latitude"] == "30.1"
    assert row["longitude"] == "114.2"
    assert row["orientation"] == "90.0"
    assert row["speed"] == "0.1"
    assert row["rpms"] == "[1.0, 2.0]"
    assert row["battery"] == "95.0"
    assert row["signal_level"] == "4.0"
    assert row["gps_signal"] == "30.0"
    assert '"sequence": 1' in row["target_checkpoint"]
    assert row["distance_to_checkpoint"] == "12.5"
    assert row["heading_error"] == "-15.0"
    assert row["linear_command"] == "0.2"
    assert row["angular_command"] == "-0.1"
    assert row["mode"] == "NORMAL_DRIVE"
    assert row["failure_flag"] == "False"


def test_recovery_event_saves_frames_and_metadata(tmp_path):
    logger = RunLogger(
        {
            "logging": {
                "enabled": True,
                "save_frames": True,
                "save_debug_images": False,
                "log_dir": str(tmp_path),
            }
        }
    )
    frame = FrameData(timestamp=1.0, image=np.zeros((8, 8, 3), dtype=np.uint8), source="front")
    rear = FrameData(timestamp=1.0, image=np.zeros((8, 8, 3), dtype=np.uint8), source="rear")

    metadata = logger.log_recovery_event(
        {
            "reason": "stuck",
            "mode_before": "NORMAL_DRIVE",
            "recovery_action": "reverse_then_rotate",
            "result": "pending",
        },
        front_frame=frame,
        rear_frame=rear,
    )
    logger.close()

    run_dir = next(tmp_path.glob("run_*"))
    assert (run_dir / metadata["front_frame_path"]).exists()
    assert (run_dir / metadata["rear_frame_path"]).exists()
    meta_path = run_dir / "recovery" / "recovery_0001_meta.json"
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    assert payload["reason"] == "stuck"
    assert payload["front_frame_path"] == "recovery/recovery_0001_front_before.jpg"
    assert payload["rear_frame_path"] == "recovery/recovery_0001_rear_before.jpg"
