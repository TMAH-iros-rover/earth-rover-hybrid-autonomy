#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.sam_tp_reproduction import (
    SamTpPredictor,
    environment_report,
    git_provenance,
    latency_summary,
    load_reproduction_config,
    resolve_upstream_paths,
    score_to_heatmap,
    sha256_file,
    write_json,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run strict single-image SAM-TP inference.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--reproduction-config", required=True)
    parser.add_argument("--upstream-root", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--compatibility-report", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--warmup-iterations", type=int)
    parser.add_argument("--benchmark-iterations", type=int)
    return parser.parse_args(argv)


def array_statistics(array: np.ndarray) -> dict[str, object]:
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "nan_count": int(np.isnan(array).sum()),
        "inf_count": int(np.isinf(array).sum()),
    }


def compose_side_by_side(
    image_rgb: np.ndarray,
    heatmap_rgb: np.ndarray,
    score_heatmap_rgb: np.ndarray,
) -> np.ndarray:
    overlay = cv2.addWeighted(image_rgb, 0.55, heatmap_rgb, 0.45, 0.0)
    return np.concatenate((image_rgb, overlay, score_heatmap_rgb), axis=1)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    compatibility_path = Path(args.compatibility_report).expanduser().resolve()
    compatibility = json.loads(compatibility_path.read_text(encoding="utf-8"))
    if compatibility.get("success") is not True:
        raise SystemExit("strict compatibility report is not successful")
    config = load_reproduction_config(args.reproduction_config)
    paths = resolve_upstream_paths(args.upstream_root, config, args.checkpoint)
    if compatibility.get("checkpoint", {}).get("sha256") != sha256_file(paths["checkpoint"]):
        raise SystemExit("compatibility report checkpoint hash differs from requested checkpoint")
    image_path = Path(args.image).expanduser().resolve()
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise SystemExit(f"cannot read image: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    runtime = config["runtime"]
    benchmark = config["benchmark"]
    warmups = (
        args.warmup_iterations
        if args.warmup_iterations is not None
        else int(benchmark["warmup_iterations"])
    )
    iterations = (
        args.benchmark_iterations
        if args.benchmark_iterations is not None
        else int(benchmark["measured_iterations"])
    )
    if warmups < 0 or iterations <= 0:
        raise SystemExit("benchmark iterations must be non-negative and measured count positive")

    import torch

    if runtime["device"] != "cuda" or not torch.cuda.is_available():
        raise SystemExit("official Dell reproduction requires CUDA")
    torch.cuda.reset_peak_memory_stats()
    predictor = SamTpPredictor(
        paths["root"],
        paths["model_config"],
        paths["checkpoint"],
        device="cuda",
        synchronize=torch.cuda.synchronize,
    )
    first = predictor.predict(image_rgb)
    for _ in range(warmups):
        predictor.predict(image_rgb)
    latencies = [predictor.predict(image_rgb).inference_time_ms for _ in range(iterations)]
    score_heatmap = score_to_heatmap(first.traversability_score)
    side_by_side = compose_side_by_side(image_rgb, first.heatmap, score_heatmap)

    temporary = output.parent / f".{output.name}.tmp-{os.getpid()}"
    if temporary.exists():
        raise SystemExit(f"temporary output already exists: {temporary}")
    temporary.mkdir(parents=True)
    try:
        cv2.imwrite(str(temporary / "original.png"), image_bgr)
        cv2.imwrite(
            str(temporary / "official_heatmap.png"),
            cv2.cvtColor(first.heatmap, cv2.COLOR_RGB2BGR),
        )
        cv2.imwrite(
            str(temporary / "traversability_score_heatmap.png"),
            cv2.cvtColor(score_heatmap, cv2.COLOR_RGB2BGR),
        )
        cv2.imwrite(
            str(temporary / "side_by_side.png"),
            cv2.cvtColor(side_by_side, cv2.COLOR_RGB2BGR),
        )
        np.save(temporary / "raw_logits.npy", first.raw_logits)
        np.save(temporary / "traversability_score.npy", first.traversability_score)
        summary = latency_summary(latencies)
        report = {
            "success": True,
            "source_image": str(image_path),
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "input_resolution": list(image_rgb.shape[:2]),
            "model_input_size": int(runtime["input_size"]),
            "output_resolution": list(first.output_shape),
            "raw_logits": array_statistics(first.raw_logits),
            "traversability_score": array_statistics(first.traversability_score),
            "score_conversion": (
                "sigmoid(raw_logits); high/red means stronger membership in the "
                "bottom-point-prompted traversable region"
            ),
            "official_prompt_policy": {
                "positive_points": ["bottom_left", "bottom_right", "bottom_center"],
                "score_threshold": 0.0,
                "multimask_output": False,
            },
            "model_load_time_ms": predictor.load_time_ms,
            "first_inference_latency_ms": first.inference_time_ms,
            "warmup_count": warmups,
            "benchmark_frame_count": iterations,
            "warm_inference_latency_ms": summary,
            "throughput_fps": 1000.0 / summary["mean"],
            "precision": "fp32",
            "batch_size": 1,
            "peak_vram_bytes": {
                "allocated": torch.cuda.max_memory_allocated(),
                "reserved": torch.cuda.max_memory_reserved(),
            },
            "environment": environment_report(),
            "dependency_setup_method": os.environ.get(
                "SAM_TP_ENVIRONMENT_BACKEND",
                runtime["dependency_setup_method"],
            ),
            "upstream": git_provenance(paths["root"]),
            "checkpoint": compatibility["checkpoint"],
            "compatibility_report": str(compatibility_path),
            "model_loaded_once": predictor.load_count == 1,
        }
        write_json(temporary / "metadata.json", report)
        (temporary / "report.md").write_text(
            "# SAM-TP Single-Image Reproduction\n\n"
            f"- Status: PASS\n"
            f"- Upstream commit: `{report['upstream']['commit']}`\n"
            f"- Checkpoint SHA-256: `{report['checkpoint']['sha256']}`\n"
            f"- Input/output: `{report['input_resolution']}` / `{report['output_resolution']}`\n"
            f"- First inference: `{report['first_inference_latency_ms']:.3f} ms`\n"
            f"- Warm p50/p95: `{summary['p50']:.3f}` / `{summary['p95']:.3f} ms`\n"
            f"- Throughput: `{report['throughput_fps']:.3f} FPS`\n"
            f"- Peak allocated/reserved VRAM: "
            f"`{report['peak_vram_bytes']['allocated']}` / "
            f"`{report['peak_vram_bytes']['reserved']} bytes`\n"
            "- Heatmap semantics: red is higher traversability; blue is lower.\n",
            encoding="utf-8",
        )
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(yaml.safe_dump(report, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
