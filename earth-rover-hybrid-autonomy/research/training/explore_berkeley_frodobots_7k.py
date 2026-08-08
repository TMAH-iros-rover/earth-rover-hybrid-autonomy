#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

_ACTION_LABELS_PATH = ROOT / "training" / "datasets" / "action_labels.py"
_ACTION_LABELS_SPEC = importlib.util.spec_from_file_location("earth_rover_action_labels", _ACTION_LABELS_PATH)
if _ACTION_LABELS_SPEC is None or _ACTION_LABELS_SPEC.loader is None:
    raise RuntimeError(f"Could not load action label helpers from {_ACTION_LABELS_PATH}")
_ACTION_LABELS_MODULE = importlib.util.module_from_spec(_ACTION_LABELS_SPEC)
sys.modules[_ACTION_LABELS_SPEC.name] = _ACTION_LABELS_MODULE
_ACTION_LABELS_SPEC.loader.exec_module(_ACTION_LABELS_MODULE)
action_to_linear_angular = _ACTION_LABELS_MODULE.action_to_linear_angular
classify_action = _ACTION_LABELS_MODULE.classify_action


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect a small streaming sample of BitRobot/Berkeley-FrodoBots-7K without downloading 769GB."
    )
    parser.add_argument("--dataset-id", default="BitRobot/Berkeley-FrodoBots-7K")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-rows", type=int, default=200)
    parser.add_argument("--output-dir", default="datasets/berkeley_7k_probe")
    parser.add_argument("--token", default=None, help="Optional Hugging Face token. Usually `huggingface-cli login` is cleaner.")
    args = parser.parse_args()

    try:
        load_dataset = import_hf_load_dataset()
    except ImportError as exc:
        raise SystemExit(
            "Could not import Hugging Face `datasets`. If it is installed, make sure "
            "the local project `datasets/` directory is not shadowing it. Try running "
            "`python -c \"from datasets import load_dataset; print(load_dataset)\"` "
            "from outside this repo, or reinstall with `python -m pip install -U datasets huggingface_hub`."
        ) from exc

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_kwargs: dict[str, Any] = {"split": args.split, "streaming": True}
    if args.token:
        dataset_kwargs["token"] = args.token

    dataset = load_dataset(args.dataset_id, **dataset_kwargs)

    column_types: dict[str, Counter] = defaultdict(Counter)
    string_lengths: dict[str, list[int]] = defaultdict(list)
    sample_values: dict[str, list[Any]] = defaultdict(list)
    url_counts: Counter = Counter()
    action_counts: Counter = Counter()
    parsed_actions: list[dict[str, Any]] = []

    sample_path = output_dir / "sample_rows.jsonl"
    rows_scanned = 0
    with sample_path.open("w", encoding="utf-8") as sample_file:
        for row_index, row in enumerate(dataset):
            if row_index >= args.max_rows:
                break
            rows_scanned += 1

            slim_row = {}
            for key, value in row.items():
                type_name = type(value).__name__
                column_types[key][type_name] += 1
                slim_value = summarize_value(value)
                slim_row[key] = slim_value
                if len(sample_values[key]) < 5:
                    sample_values[key].append(slim_value)
                if isinstance(value, str):
                    string_lengths[key].append(len(value))

            row_url = row.get("__url__")
            if isinstance(row_url, str):
                url_counts[row_url] += 1

            parsed = first_parseable_action(row)
            if parsed is not None:
                action_key, linear, angular = parsed
                label = classify_action(linear, angular)
                action_counts[label] += 1
                parsed_actions.append(
                    {
                        "row_index": row_index,
                        "key": row.get("__key__"),
                        "url": row_url,
                        "action_key": action_key,
                        "linear": linear,
                        "angular": angular,
                        "label": label,
                    }
                )

            sample_file.write(json.dumps(slim_row, ensure_ascii=False) + "\n")

    summary = {
        "dataset_id": args.dataset_id,
        "split": args.split,
        "max_rows_requested": args.max_rows,
        "rows_scanned": rows_scanned,
        "columns": {
            key: {
                "types": dict(counter),
                "sample_values": sample_values[key],
                "string_length_min": min(string_lengths[key]) if string_lengths[key] else None,
                "string_length_max": max(string_lengths[key]) if string_lengths[key] else None,
            }
            for key, counter in sorted(column_types.items())
        },
        "unique_urls_in_sample": len(url_counts),
        "url_counts": dict(url_counts.most_common(20)),
        "action_label_counts": dict(action_counts),
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    with (output_dir / "parsed_actions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["row_index", "key", "url", "action_key", "linear", "angular", "label"],
        )
        writer.writeheader()
        writer.writerows(parsed_actions)

    print(f"scanned rows: {summary['rows_scanned']}")
    print(f"columns: {', '.join(summary['columns'].keys())}")
    print(f"action labels: {dict(action_counts)}")
    print(f"wrote: {output_dir}")
    return 0


def first_parseable_action(row: dict[str, Any]) -> tuple[str, float, float] | None:
    preferred_keys = ("action_mbra", "action", "action_original")
    for key in preferred_keys:
        parsed = action_to_linear_angular(row.get(key))
        if parsed is not None:
            return key, parsed[0], parsed[1]
    for key, value in row.items():
        if "action" not in key:
            continue
        parsed = action_to_linear_angular(value)
        if parsed is not None:
            return key, parsed[0], parsed[1]
    return None


def import_hf_load_dataset():
    """Import Hugging Face datasets while avoiding this repo's local datasets/ dir."""
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


def summarize_value(value: Any) -> Any:
    if value is None or isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, str):
        if len(value) <= 160:
            return value
        return {"type": "str", "length": len(value), "prefix": value[:120]}
    if isinstance(value, bytes):
        return {"type": "bytes", "length": len(value)}
    if hasattr(value, "shape"):
        return {
            "type": type(value).__name__,
            "shape": list(value.shape),
            "dtype": str(getattr(value, "dtype", "")),
        }
    if isinstance(value, (list, tuple)):
        return {
            "type": type(value).__name__,
            "length": len(value),
            "sample": [summarize_value(item) for item in list(value)[:5]],
        }
    if isinstance(value, dict):
        return {key: summarize_value(item) for key, item in list(value.items())[:20]}
    return {"type": type(value).__name__, "repr": repr(value)[:160]}


if __name__ == "__main__":
    raise SystemExit(main())
