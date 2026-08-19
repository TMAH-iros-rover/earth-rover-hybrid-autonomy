from __future__ import annotations

import json

import numpy as np

from training.sam_tp_event_recorder import (
    EventCaptureConfig,
    SamTpEventRecorder,
    detect_event_reasons,
    load_replay_capture,
)
from training.sam_tp_sdk_shadow import ShadowStep


def _step(
    frame_index: int,
    *,
    selected_heading: float = 0.0,
    fused_heading: float = 10.0,
    target_bearing: float = 20.0,
    target_sequence: int = 1,
    state: str = "CLEAR",
    near_safe: bool = True,
    switch_reason: str | None = None,
    score_mean: float = 0.5,
) -> ShadowStep:
    source = np.full((4, 6, 3), frame_index, dtype=np.uint8)
    logits = np.full((4, 6), frame_index + 0.25, dtype=np.float32)
    score = np.full((4, 6), score_mean, dtype=np.float32)
    record = {
        "frame_index": frame_index,
        "shadow_state": state,
        "near_field_safe": near_safe,
        "score_mean": score_mean,
        "planner": {
            "selected_candidate_heading_deg": selected_heading,
            "switch_reason": switch_reason,
        },
        "navigation": {
            "target_bearing_deg": target_bearing,
            "target_sequence": target_sequence,
        },
        "localization": {"fused_heading_deg": fused_heading},
    }
    return ShadowStep(source, source, source, logits, score, record)


def test_event_detection_catches_navigation_planner_and_score_jumps() -> None:
    previous = _step(1).record
    current = _step(
        2,
        selected_heading=30.0,
        fused_heading=50.0,
        target_bearing=100.0,
        target_sequence=2,
        switch_reason="immediate_reset",
        score_mean=0.8,
    ).record

    assert set(detect_event_reasons(previous, current)) == {
        "planner_immediate_reset",
        "candidate_jump",
        "target_sequence_change",
        "target_bearing_jump",
        "fused_heading_jump",
        "score_mean_jump",
    }


def test_sustained_nonclear_or_unsafe_state_only_triggers_on_entry() -> None:
    first = _step(1, state="STALE_FRAME", near_safe=False).record
    second = _step(2, state="STALE_FRAME", near_safe=False).record

    assert set(detect_event_reasons(None, first)) == {
        "shadow_state:STALE_FRAME",
        "near_field_unsafe",
    }
    assert detect_event_reasons(first, second) == ()


def test_event_recorder_saves_pre_and_post_window_as_replay_bundle(tmp_path) -> None:
    recorder = SamTpEventRecorder(
        tmp_path,
        EventCaptureConfig(
            pre_event_frames=2,
            post_event_frames=2,
            baseline_interval_frames=100,
            queue_size=32,
        ),
    )
    recorder.observe(_step(0))
    recorder.observe(_step(1))
    reasons = recorder.observe(_step(2, selected_heading=30.0))
    recorder.observe(_step(3, selected_heading=30.0))
    recorder.observe(_step(4, selected_heading=30.0))
    summary = recorder.close()

    assert reasons == ("candidate_jump",)
    assert summary["saved_frames"] == 5
    assert summary["event_count"] == 1
    assert summary["dropped_items"] == 0
    assert summary["write_errors"] == []

    capture_dir = tmp_path / "replay_capture"
    frames = load_replay_capture(capture_dir)
    assert [frame.frame_index for frame in frames] == [0, 1, 2, 3, 4]
    assert np.array_equal(frames[2].source_bgr, _step(2).source_bgr)
    assert np.array_equal(frames[2].raw_logits, _step(2).raw_logits)
    assert np.array_equal(frames[2].score_map, _step(2).score_map)
    assert frames[2].record["capture_event_reasons"] == ["candidate_jump"]

    event = json.loads((capture_dir / "events.jsonl").read_text().strip())
    assert event == {
        "frame_index": 2,
        "reasons": ["candidate_jump"],
        "window_start": 0,
        "window_end": 4,
    }
