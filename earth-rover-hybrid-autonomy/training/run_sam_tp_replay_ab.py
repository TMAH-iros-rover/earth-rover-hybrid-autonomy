#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
for import_root in (ROOT, ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from earth_rover.utils.config import load_config  # noqa: E402
from training.sam_tp_event_recorder import load_replay_capture  # noqa: E402
from training.sam_tp_hf_backend import build_sam_tp_predictor  # noqa: E402
from training.sam_tp_replay_ab import (  # noqa: E402
    evaluate_replay_frames,
    render_ab_overlay,
    summarize_ab_rows,
)
from training.sam_tp_reproduction import (  # noqa: E402
    OFFICIAL_CHECKPOINT,
    OFFICIAL_MODEL_CONFIG,
    sha256_file,
)


BASELINE_SHA256 = "2607fd6049d37f17fe96132cf35459f7e0a895107632410637d812756e3f9adb"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Same-frame SAM-TP checkpoint replay A/B")
    parser.add_argument("--capture-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--upstream-root", default="../external/GENIE-SAMTP")
    parser.add_argument("--baseline-model-config", default=OFFICIAL_MODEL_CONFIG)
    parser.add_argument("--baseline-checkpoint", default=OFFICIAL_CHECKPOINT)
    parser.add_argument("--baseline-sha256", default=BASELINE_SHA256)
    parser.add_argument("--candidate-model-config")
    parser.add_argument("--candidate-checkpoint")
    parser.add_argument("--candidate-sha256")
    parser.add_argument("--nominal-fps", type=float, default=4.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(_rooted(args.config))
    sam_config = config.get("sam_tp", {})
    upstream = _rooted(args.upstream_root)
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    specs = {
        "baseline": {
            "model_config": args.baseline_model_config,
            "checkpoint": args.baseline_checkpoint,
            "sha256": args.baseline_sha256,
        },
        "candidate": {
            "model_config": args.candidate_model_config or sam_config.get("model_config"),
            "checkpoint": args.candidate_checkpoint or sam_config.get("checkpoint"),
            "sha256": args.candidate_sha256 or sam_config.get("expected_checkpoint_sha256"),
        },
    }
    resolved: dict[str, dict[str, object]] = {}
    for name, spec in specs.items():
        model_config = _under_upstream(upstream, spec["model_config"], f"{name} model config")
        checkpoint = _under_upstream(upstream, spec["checkpoint"], f"{name} checkpoint")
        expected_sha = str(spec["sha256"] or "")
        actual_sha = sha256_file(checkpoint)
        if len(expected_sha) != 64 or actual_sha != expected_sha:
            raise SystemExit(
                f"{name} checkpoint SHA mismatch: expected={expected_sha} actual={actual_sha}"
            )
        resolved[name] = {
            "model_config": model_config,
            "checkpoint": checkpoint,
            "sha256": actual_sha,
        }
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("SAM-TP replay A/B requires torch") from exc
    if not torch.cuda.is_available():
        raise SystemExit("SAM-TP replay A/B requires CUDA")
    predictors = {}
    backend_names = {}
    for name, spec in resolved.items():
        predictor, backend = build_sam_tp_predictor(
            upstream,
            spec["model_config"],
            spec["checkpoint"],
            synchronize=torch.cuda.synchronize,
        )
        predictor.load()
        predictors[name] = predictor
        backend_names[name] = backend.value
    frames = load_replay_capture(args.capture_dir)
    if not frames:
        raise SystemExit("replay capture contains no frames")
    rows, score_maps = evaluate_replay_frames(
        frames,
        predictors,
        dict(config.get("planner", {})),
        nominal_fps=args.nominal_fps,
    )
    output.mkdir(parents=True)
    overlays = output / "overlays"
    overlays.mkdir()
    ordered_frames = sorted(frames, key=lambda item: item.frame_index)
    with (output / "frames.jsonl").open("w", encoding="utf-8") as handle:
        for frame, row in zip(ordered_frames, rows):
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            overlay = render_ab_overlay(frame.source_bgr, score_maps[frame.frame_index], row)
            path = overlays / f"frame_{frame.frame_index:08d}.jpg"
            if not cv2.imwrite(str(path), overlay):
                raise OSError(f"could not write A/B overlay: {path}")
    summary = summarize_ab_rows(rows)
    summary["models"] = {
        name: {
            "checkpoint": Path(spec["checkpoint"]).name,
            "sha256": spec["sha256"],
            "backend": backend_names[name],
        }
        for name, spec in resolved.items()
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"A/B replay output: {output}")
    return 0


def _rooted(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _under_upstream(upstream: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise SystemExit(f"{label} is not configured")
    path = Path(value).expanduser()
    resolved = (path if path.is_absolute() else upstream / path).resolve()
    if not resolved.is_file():
        raise SystemExit(f"{label} does not exist: {resolved}")
    return resolved


if __name__ == "__main__":
    raise SystemExit(main())
