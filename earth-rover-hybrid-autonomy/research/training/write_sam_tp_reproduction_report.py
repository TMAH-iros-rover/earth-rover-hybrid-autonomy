#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.sam_tp_reproduction import (
    OFFICIAL_CHECKPOINT_URL,
    OFFICIAL_PAPER,
    UPSTREAM_LICENSE_STATUS,
    write_json,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate completed SAM-TP reproduction reports.")
    parser.add_argument("--compatibility-report", required=True)
    parser.add_argument("--smoke-report", required=True)
    parser.add_argument("--video-report", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-markdown", required=True)
    return parser.parse_args(argv)


def read_report(path: str | Path) -> dict:
    value = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"report is not a JSON object: {path}")
    return value


def aggregate_reports(
    compatibility: dict,
    smoke: dict,
    video: dict,
) -> dict[str, object]:
    success = all(
        report.get("success") is True for report in (compatibility, smoke, video)
    )
    return {
        "status": "PASS" if success else "FAIL",
        "success": success,
        "upstream": smoke.get("upstream"),
        "official_paper": OFFICIAL_PAPER,
        "upstream_license_status": UPSTREAM_LICENSE_STATUS,
        "checkpoint": {
            **compatibility.get("checkpoint", {}),
            "source_url": OFFICIAL_CHECKPOINT_URL,
        },
        "compatibility": {
            "success": compatibility.get("success"),
            "architecture_config_comparison": compatibility.get(
                "architecture_config_comparison"
            ),
            "state_dict_comparison": compatibility.get("state_dict_comparison"),
            "strict_state_dict_load": compatibility.get("strict_state_dict_load"),
        },
        "single_image": {
            "source_image": smoke.get("source_image"),
            "input_resolution": smoke.get("input_resolution"),
            "output_resolution": smoke.get("output_resolution"),
            "raw_logits": smoke.get("raw_logits"),
            "traversability_score": smoke.get("traversability_score"),
            "model_load_time_ms": smoke.get("model_load_time_ms"),
            "first_inference_latency_ms": smoke.get("first_inference_latency_ms"),
            "warm_inference_latency_ms": smoke.get("warm_inference_latency_ms"),
            "throughput_fps": smoke.get("throughput_fps"),
            "peak_vram_bytes": smoke.get("peak_vram_bytes"),
        },
        "video_review": {
            "success": video.get("success"),
            "settings": video.get("settings"),
            "datasets": video.get("datasets"),
            "peak_vram_bytes": video.get("peak_vram_bytes"),
        },
        "environment": smoke.get("environment"),
        "dependency_setup_method": smoke.get("dependency_setup_method"),
        "random_initialization_used": False,
        "training_performed": False,
        "planner_or_sdk_integration_performed": False,
        "live_rover_commands_sent": False,
    }


def markdown_report(report: dict[str, object]) -> str:
    single = report["single_image"]
    checkpoint = report["checkpoint"]
    warm = single["warm_inference_latency_ms"]
    video = report["video_review"]
    dataset_lines = []
    for index, item in (video.get("datasets") or {}).items():
        dataset_lines.append(
            f"- Dataset {index}: {item['processed_frame_count']} processed, "
            f"{item['skipped_frame_count']} skipped, "
            f"{item['effective_pipeline_fps']:.3f} end-to-end FPS"
        )
    return (
        "# SAM-TP Reproduction Report\n\n"
        f"- Status: **{report['status']}**\n"
        f"- Upstream commit: `{report['upstream']['commit']}`\n"
        f"- Checkpoint: `{checkpoint['filename']}`\n"
        f"- Checkpoint SHA-256: `{checkpoint['sha256']}`\n"
        f"- Checkpoint size: `{checkpoint['size_bytes']} bytes`\n"
        f"- License status: {report['upstream_license_status']}\n"
        f"- Model load: `{single['model_load_time_ms']:.3f} ms`\n"
        f"- First inference: `{single['first_inference_latency_ms']:.3f} ms`\n"
        f"- Warm p50/p95: `{warm['p50']:.3f}` / `{warm['p95']:.3f} ms`\n"
        f"- Warm throughput: `{single['throughput_fps']:.3f} FPS`\n"
        f"- Peak allocated/reserved VRAM: "
        f"`{single['peak_vram_bytes']['allocated']}` / "
        f"`{single['peak_vram_bytes']['reserved']} bytes`\n\n"
        "## FrodoBots review\n\n"
        + ("\n".join(dataset_lines) if dataset_lines else "- No successful dataset review.")
        + "\n\n"
        "Raw logits and sigmoid scores were preserved without temporal smoothing. "
        "No training, planner/SDK integration, or live rover command was performed.\n"
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_json = Path(args.output_json).expanduser().resolve()
    output_markdown = Path(args.output_markdown).expanduser().resolve()
    if output_json.exists() or output_markdown.exists():
        raise SystemExit("aggregate report output already exists")
    report = aggregate_reports(
        read_report(args.compatibility_report),
        read_report(args.smoke_report),
        read_report(args.video_report),
    )
    write_json(output_json, report)
    output_markdown.write_text(markdown_report(report), encoding="utf-8")
    print(f"Reproduction report: {output_json}")
    return 0 if report["success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
