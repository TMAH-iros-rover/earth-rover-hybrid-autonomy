#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.training.datasets.action_labels import ACTION_NAMES
from research.training.datasets.frodobots_2k_dataset import FrameDecodeError, FrodoBotsActionDataset, ManifestSample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify lazy FrodoBots-2K HLS loading and create a contact sheet.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.num_samples < 20:
        raise SystemExit("num-samples must be at least 20")
    if args.batch_size <= 0:
        raise SystemExit("batch-size must be positive")

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir == dataset_root or dataset_root in output_dir.parents:
        raise SystemExit("output-dir must remain outside the immutable raw dataset root")
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = FrodoBotsActionDataset(dataset_root, args.manifest)
    candidate_indices = _candidate_indices(dataset.samples, args.num_samples * 5)
    decoded: list[tuple[int, dict[str, object], np.ndarray]] = []
    failures: list[dict[str, object]] = []

    for index in candidate_indices:
        try:
            item, frame_rgb = dataset.load_sample(index)
        except (FrameDecodeError, OSError) as exc:
            failures.append({"manifest_index": index, "error": str(exc)})
            continue
        decoded.append((index, item, frame_rgb))
        if len(decoded) == args.num_samples:
            break

    if len(decoded) < args.num_samples:
        report = {
            "requested_sample_count": args.num_samples,
            "decoded_sample_count": len(decoded),
            "unreadable_sample_count": len(failures),
            "unreadable_samples": failures,
        }
        _write_report(output_dir, report)
        raise SystemExit(f"decoded only {len(decoded)} of {args.num_samples} requested samples")

    first_index = decoded[0][0]
    first = dataset[first_index]["image"]
    repeated = dataset[first_index]["image"]
    deterministic = isinstance(first, torch.Tensor) and torch.equal(first, repeated)

    batch_indices = [entry[0] for entry in decoded[: args.batch_size]]
    batch = next(iter(DataLoader(Subset(dataset, batch_indices), batch_size=args.batch_size, shuffle=False)))
    images = batch["image"]
    expected_shape = (len(batch_indices), 3, 224, 224)
    if tuple(images.shape) != expected_shape:
        raise RuntimeError(f"unexpected batch shape: {tuple(images.shape)} != {expected_shape}")
    if not deterministic:
        raise RuntimeError("repeated sample access returned different tensors")

    contact_sheet_path = output_dir / "aligned_samples.jpg"
    _write_contact_sheet(decoded, contact_sheet_path)
    report = {
        "manifest_path": str(Path(args.manifest).expanduser().resolve()),
        "manifest_sample_count": len(dataset),
        "requested_sample_count": args.num_samples,
        "decoded_sample_count": len(decoded),
        "unreadable_sample_count": len(failures),
        "unreadable_samples": failures,
        "deterministic_repeat_access": deterministic,
        "batch_shape": list(images.shape),
        "action_class_distribution": dict(sorted(Counter(item[1]["action_class"] for item in decoded).items())),
        "visualization_path": str(contact_sheet_path),
        "sample_manifest_indices": [entry[0] for entry in decoded],
    }
    _write_report(output_dir, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _candidate_indices(samples: tuple[ManifestSample, ...], spread_count: int) -> list[int]:
    if not samples:
        return []
    groups = {
        action: [index for index, sample in enumerate(samples) if sample.action_class == action]
        for action in ACTION_NAMES
    }
    nonempty_groups = [groups[action] for action in ACTION_NAMES if groups[action]]
    per_class = max(1, (spread_count + len(nonempty_groups) - 1) // len(nonempty_groups))
    selections = [
        np.linspace(0, len(group) - 1, num=min(per_class, len(group)), dtype=int).tolist()
        for group in nonempty_groups
    ]
    prioritized: list[int] = []
    for position in range(max(len(selection) for selection in selections)):
        for group, selection in zip(nonempty_groups, selections):
            if position < len(selection):
                prioritized.append(group[selection[position]])
    seen = set(prioritized)
    return prioritized + [index for index in range(len(samples)) if index not in seen]


def _write_contact_sheet(
    decoded: list[tuple[int, dict[str, object], np.ndarray]],
    path: Path,
) -> None:
    columns = 5
    cell_width = 512
    cell_height = 288
    rows = (len(decoded) + columns - 1) // columns
    canvas = np.zeros((rows * cell_height, columns * cell_width, 3), dtype=np.uint8)
    for position, (manifest_index, item, frame_rgb) in enumerate(decoded):
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        frame_bgr = cv2.resize(frame_bgr, (cell_width, cell_height), interpolation=cv2.INTER_AREA)
        metadata = item["metadata"]
        lines = [
            f"index={manifest_index} ride={item['ride_id']} frame={metadata['front_frame_id']}",
            f"ts={metadata['front_timestamp']:.3f} delta={metadata['control_delta_ms']:.1f}ms",
            f"linear={metadata['linear']:.3f} angular={metadata['angular']:.3f}",
            f"action={item['action_class']}",
        ]
        overlay_height = 18 + len(lines) * 24
        cv2.rectangle(frame_bgr, (0, 0), (cell_width, overlay_height), (0, 0, 0), -1)
        for line_index, line in enumerate(lines):
            cv2.putText(
                frame_bgr,
                line,
                (8, 25 + line_index * 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        row, column = divmod(position, columns)
        canvas[
            row * cell_height : (row + 1) * cell_height,
            column * cell_width : (column + 1) * cell_width,
        ] = frame_bgr
    if not cv2.imwrite(str(path), canvas):
        raise OSError(f"failed to write visualization: {path}")


def _write_report(output_dir: Path, report: dict[str, object]) -> None:
    report_path = output_dir / "hls_verification_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
