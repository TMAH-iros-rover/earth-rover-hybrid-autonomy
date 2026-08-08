#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


SMALL_METADATA_SUFFIXES = (
    ".json",
    ".zarray",
    ".zattrs",
    ".zgroup",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download only small metadata files from Berkeley-FrodoBots-7K.")
    parser.add_argument("--dataset-id", default="BitRobot/Berkeley-FrodoBots-7K")
    parser.add_argument("--output-dir", default="datasets/berkeley_7k_metadata")
    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:
        raise SystemExit("Missing dependency: huggingface_hub. Install with `python -m pip install -U huggingface_hub`.") from exc

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    files = api.list_repo_files(args.dataset_id, repo_type="dataset")
    metadata_files = [
        path
        for path in files
        if is_small_metadata(path)
    ]

    downloaded = []
    for repo_path in metadata_files:
        local_path = hf_hub_download(
            repo_id=args.dataset_id,
            repo_type="dataset",
            filename=repo_path,
            local_dir=output_dir,
        )
        downloaded.append(str(Path(local_path).relative_to(output_dir)))

    summary = {
        "dataset_id": args.dataset_id,
        "repo_file_count": len(files),
        "metadata_file_count": len(metadata_files),
        "downloaded": downloaded,
        "zarr_arrays": inspect_zarr_arrays(output_dir),
    }
    with (output_dir / "metadata_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(f"repo files: {len(files)}")
    print(f"metadata files downloaded: {len(metadata_files)}")
    print(f"wrote: {output_dir / 'metadata_summary.json'}")
    return 0


def is_small_metadata(path: str) -> bool:
    if path.startswith("."):
        return False
    if path.startswith("videos/"):
        return False
    if path.endswith(".tar.gz") or ".tar.gz.part" in path:
        return False
    return path.endswith(SMALL_METADATA_SUFFIXES)


def inspect_zarr_arrays(root: Path) -> dict[str, dict]:
    arrays = {}
    for zarray_path in root.rglob(".zarray"):
        try:
            payload = json.loads(zarray_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        rel = zarray_path.parent.relative_to(root).as_posix()
        arrays[rel] = {
            "shape": payload.get("shape"),
            "chunks": payload.get("chunks"),
            "dtype": payload.get("dtype"),
            "compressor": payload.get("compressor", {}).get("id") if isinstance(payload.get("compressor"), dict) else payload.get("compressor"),
            "order": payload.get("order"),
        }
    return dict(sorted(arrays.items()))


if __name__ == "__main__":
    raise SystemExit(main())
