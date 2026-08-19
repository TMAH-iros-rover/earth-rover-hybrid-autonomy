from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from training.sam_tp_event_recorder import ReplayCaptureFrame
from training.sam_tp_replay_ab import evaluate_replay_frames, summarize_ab_rows


@dataclass(frozen=True)
class Prediction:
    traversability_score: np.ndarray
    inference_time_ms: float = 10.0


class Predictor:
    def __init__(self, score: float) -> None:
        self.score = score

    def predict(self, image_rgb: np.ndarray) -> Prediction:
        return Prediction(np.full(image_rgb.shape[:2], self.score, dtype=np.float32))


def frame(index: int) -> ReplayCaptureFrame:
    image = np.full((120, 160, 3), index, dtype=np.uint8)
    return ReplayCaptureFrame(
        frame_index=index,
        source_bgr=image,
        raw_logits=np.zeros(image.shape[:2], dtype=np.float32),
        score_map=np.zeros(image.shape[:2], dtype=np.float32),
        record={
            "frame_index": index,
            "navigation": {"heading_error_deg": 0.0, "target_sequence": 1},
        },
    )


def test_same_frame_ab_evaluation_and_summary() -> None:
    rows, score_maps = evaluate_replay_frames(
        [frame(1), frame(2), frame(10)],
        {"baseline": Predictor(0.9), "candidate": Predictor(0.8)},
        {"candidate_score_ema_alpha": 1.0},
    )
    summary = summarize_ab_rows(rows)

    assert [row["contiguous_with_previous"] for row in rows] == [False, True, False]
    assert rows[0]["score_mae"] == pytest.approx(0.1)
    assert rows[0]["mask_iou_at_0_5"] == 1.0
    assert score_maps[1]["baseline"].shape == (120, 160)
    assert summary["frame_count"] == 3
    assert summary["selected_heading_disagreement_fraction"] == 0.0
    assert summary["baseline"]["contiguous_left_right_flip_count"] == 0


def test_summary_counts_only_contiguous_heading_flips() -> None:
    def row(index: int, contiguous: bool, baseline: float, candidate: float):
        def model(heading: float):
            return {
                "inference_time_ms": 10.0,
                "planner": {
                    "selected_candidate_heading_deg": heading,
                    "switch_stop_required": False,
                },
            }

        return {
            "frame_index": index,
            "contiguous_with_previous": contiguous,
            "baseline": model(baseline),
            "candidate": model(candidate),
            "score_mae": 0.1,
            "mask_iou_at_0_5": 0.8,
        }

    summary = summarize_ab_rows(
        [
            row(1, False, -10.0, -10.0),
            row(2, True, 10.0, -10.0),
            row(8, False, -30.0, 30.0),
        ]
    )

    assert summary["baseline"]["contiguous_left_right_flip_count"] == 1
    assert summary["candidate"]["contiguous_left_right_flip_count"] == 0
    assert summary["selected_heading_abs_delta_deg_max"] == 60.0
