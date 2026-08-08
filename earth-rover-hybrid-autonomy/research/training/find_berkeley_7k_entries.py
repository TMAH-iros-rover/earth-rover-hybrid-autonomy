#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan streaming WebDataset entries by key without downloading the full 7K dataset.")
    parser.add_argument("--dataset-id", default="BitRobot/Berkeley-FrodoBots-7K")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-rows", type=int, default=5000)
    parser.add_argument("--contains", default=".zarray,.zattrs,.zgroup,meta_data,info.json,stats")
    parser.add_argument("--output", default="datasets/berkeley_7k_probe/matching_entries.jsonl")
    args = parser.parse_args()

    load_dataset = import_hf_load_dataset()
    needles = [item.strip() for item in args.contains.split(",") if item.strip()]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(args.dataset_id, split=args.split, streaming=True)
    scanned = 0
    matched = 0

    with output.open("w", encoding="utf-8") as handle:
        for row in dataset:
            scanned += 1
            key = str(row.get("__key__", ""))
            if any(needle in key for needle in needles):
                matched += 1
                handle.write(json.dumps(summarize_row(row), ensure_ascii=False) + "\n")
                print(f"match {matched}: {key}")
            if scanned >= args.max_rows:
                break

    print(f"scanned rows: {scanned}")
    print(f"matched rows: {matched}")
    print(f"wrote: {output}")
    return 0


def import_hf_load_dataset():
    removed_modules = {}
    for name in list(sys.modules):
        if name == "datasets" or name.startswith("datasets."):
            removed_modules[name] = sys.modules.pop(name)

    cwd = Path.cwd().resolve()
    filtered_sys_path = []
    for item in sys.path:
        resolved = cwd if item == "" else Path(item).resolve()
        if resolved == ROOT or resolved == cwd or ROOT in resolved.parents:
            continue
        filtered_sys_path.append(item)

    original_sys_path = sys.path[:]
    try:
        sys.path[:] = filtered_sys_path
        from datasets import load_dataset

        return load_dataset
    except ImportError:
        sys.modules.update(removed_modules)
        raise
    finally:
        sys.path[:] = original_sys_path


def summarize_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: summarize_value(value) for key, value in row.items()}


def summarize_value(value: Any) -> Any:
    if isinstance(value, bytes):
        text = try_decode_text(value)
        if text is not None:
            return {"type": "bytes", "length": len(value), "text_prefix": text[:1000]}
        return {"type": "bytes", "length": len(value)}
    if isinstance(value, str):
        return value
    if value is None or isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return {"type": type(value).__name__, "length": len(value)}
    if isinstance(value, dict):
        return {key: summarize_value(item) for key, item in value.items()}
    return {"type": type(value).__name__, "repr": repr(value)[:200]}


def try_decode_text(value: bytes) -> str | None:
    try:
        text = value.decode("utf-8")
    except UnicodeDecodeError:
        return None
    if not text.strip():
        return None
    if any(char in text[:200] for char in ["{", "[", "zarr_format", "shape", "dtype"]):
        return text
    return None


if __name__ == "__main__":
    raise SystemExit(main())
