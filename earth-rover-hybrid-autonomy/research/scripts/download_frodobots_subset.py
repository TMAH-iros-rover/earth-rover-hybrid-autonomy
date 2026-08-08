#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


GB = 1024**3
DEFAULT_METADATA_FILES = [
    "meta_data/info.json",
    "meta_data/stats.safetensors",
    "meta_data/episode_data_index.safetensors",
    "train/dataset_info.json",
    "train/state.json",
    "frodobots_dataset/dataset_cache.zarr/.zgroup",
]
IMPORTANT_ZARR_ARRAYS = [
    "action_mbra",
    "action_original",
    "action",
    "timestamp",
    "episode_index",
    "frame_index",
    "observation.latitude",
    "observation.longitude",
    "observation.filtered_heading",
    "observation.compass_heading",
    "observation.wheel_rpm",
    "observation.images.front.path",
    "observation.images.front.timestamp",
    "observation.images.rear.path",
    "observation.images.rear.timestamp",
]
ZARR_ROOT = "frodobots_dataset/dataset_cache.zarr"


@dataclass(frozen=True)
class RepoFile:
    path: str
    size: int | None = None


@dataclass(frozen=True)
class SelectedFile:
    path: str
    size: int | None
    role: str


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    token = os.getenv(args.hf_token_env) if args.hf_token_env else None
    repo_files = list_repo_dataset_files(args.repo_id, token=token)
    write_file_reports(repo_files, reports_dir)

    budget_bytes = int(float(args.budget_gb) * GB)
    selected, warnings = build_subset_selection(
        repo_files=repo_files,
        budget_bytes=budget_bytes,
        download_metadata=args.download_metadata,
        sample_videos=args.sample_videos,
        prefer_front_rear_pairs=args.prefer_front_rear_pairs,
        include_zarr_sample=args.include_zarr_sample,
        no_tar_parts=args.no_tar_parts,
    )
    estimated_bytes = sum_known_sizes(selected)
    manifest = {
        "repo_id": args.repo_id,
        "budget_gb": args.budget_gb,
        "budget_bytes": budget_bytes,
        "estimated_bytes": estimated_bytes,
        "estimated_gb": estimated_bytes / GB,
        "dry_run": args.dry_run,
        "selected_files": [asdict(item) for item in selected],
        "warnings": warnings,
    }

    if args.dry_run:
        path = reports_dir / "subset_manifest_dry_run.json"
        write_json(path, manifest)
        write_subset_summary(reports_dir / "subset_summary.md", manifest, actual_bytes=0, zarr_arrays=[])
        print_selection(selected, estimated_bytes, budget_bytes, warnings, dry_run=True)
        return 0

    downloaded, actual_bytes, download_warnings = download_selected_files(
        selected=selected,
        repo_id=args.repo_id,
        output_dir=output_dir,
        budget_bytes=budget_bytes,
        token=token,
    )
    warnings.extend(download_warnings)
    zarr_arrays = parse_downloaded_zarr_metadata(output_dir)
    write_zarr_reports(zarr_arrays, reports_dir)
    manifest["dry_run"] = False
    manifest["downloaded_files"] = downloaded
    manifest["actual_downloaded_bytes"] = actual_bytes
    manifest["actual_downloaded_gb"] = actual_bytes / GB
    manifest["warnings"] = warnings
    write_json(reports_dir / "subset_manifest.json", manifest)
    write_subset_summary(reports_dir / "subset_summary.md", manifest, actual_bytes=actual_bytes, zarr_arrays=zarr_arrays)
    maybe_extract_frame_preview(output_dir, reports_dir, downloaded)
    print_selection(selected, actual_bytes, budget_bytes, warnings, dry_run=False)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Safely download a size-limited Berkeley-FrodoBots-7K subset.")
    parser.add_argument("--repo-id", default="BitRobot/Berkeley-FrodoBots-7K")
    parser.add_argument("--output-dir", default="datasets/frodobots_7k_subset20")
    parser.add_argument("--budget-gb", type=float, default=20.0)
    parser.add_argument("--hf-token-env", default="HF_TOKEN")
    parser.add_argument("--download-metadata", action="store_true", default=True)
    parser.add_argument("--no-download-metadata", dest="download_metadata", action="store_false")
    parser.add_argument("--sample-videos", type=int, default=8)
    parser.add_argument("--prefer-front-rear-pairs", action="store_true")
    parser.add_argument("--include-zarr-sample", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-tar-parts", action="store_true", default=True)
    parser.add_argument("--allow-tar-parts", dest="no_tar_parts", action="store_false")
    return parser.parse_args()


def list_repo_dataset_files(repo_id: str, token: str | None = None) -> list[RepoFile]:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit("Missing dependency: huggingface_hub. Install with `python -m pip install huggingface_hub`.") from exc

    api = HfApi(token=token)
    try:
        tree = api.list_repo_tree(repo_id=repo_id, repo_type="dataset", recursive=True)
        files = []
        for item in tree:
            path = getattr(item, "path", None)
            if not path:
                continue
            if getattr(item, "type", "file") != "file":
                continue
            files.append(RepoFile(path=path, size=getattr(item, "size", None)))
        return sorted(files, key=lambda item: item.path)
    except Exception:
        paths = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
        return sorted([RepoFile(path=path, size=None) for path in paths], key=lambda item: item.path)


def build_subset_selection(
    repo_files: list[RepoFile],
    budget_bytes: int,
    download_metadata: bool,
    sample_videos: int,
    prefer_front_rear_pairs: bool,
    include_zarr_sample: bool,
    no_tar_parts: bool,
) -> tuple[list[SelectedFile], list[str]]:
    warnings = []
    selected: list[SelectedFile] = []
    by_path = {item.path: item for item in repo_files}
    if no_tar_parts:
        repo_files = [item for item in repo_files if not is_archive_part(item.path)]

    if download_metadata:
        for path in metadata_paths(repo_files):
            item = by_path.get(path)
            if item is not None and not is_archive_part(item.path):
                selected.append(SelectedFile(item.path, item.size, "metadata"))
        missing = [path for path in DEFAULT_METADATA_FILES if path not in by_path]
        if missing:
            warnings.append(f"metadata files not exposed as repo files: {', '.join(missing)}")

    zarr_reserved_bytes = 1 * GB
    video_budget = max(0, budget_bytes - zarr_reserved_bytes - sum_known_sizes(selected))
    video_selection, video_warnings = select_video_subset(
        repo_files=repo_files,
        budget_bytes=video_budget,
        sample_videos=sample_videos,
        prefer_pairs=prefer_front_rear_pairs,
    )
    warnings.extend(video_warnings)
    selected.extend(video_selection)

    if include_zarr_sample:
        remaining = budget_bytes - sum_known_sizes(selected)
        zarr_selection, zarr_warnings = select_zarr_sample(repo_files, max(0, remaining))
        warnings.extend(zarr_warnings)
        selected.extend(zarr_selection)

    selected = dedupe_selection(selected)
    known = sum_known_sizes(selected)
    if known > budget_bytes:
        warnings.append(f"known selected size exceeds budget: {known} > {budget_bytes}")
    return selected, warnings


def metadata_paths(repo_files: list[RepoFile]) -> list[str]:
    paths = set(DEFAULT_METADATA_FILES)
    for item in repo_files:
        if item.path.startswith(f"{ZARR_ROOT}/") and item.path.endswith("/.zarray"):
            paths.add(item.path)
    return sorted(paths)


def is_archive_part(path: str) -> bool:
    name = Path(path).name
    return name.endswith(".tar.gz") or ".tar.gz.part" in name or name.endswith(".tgz") or name.endswith(".zip")


def select_video_subset(
    repo_files: list[RepoFile],
    budget_bytes: int,
    sample_videos: int,
    prefer_pairs: bool,
) -> tuple[list[SelectedFile], list[str]]:
    warnings = []
    by_path = {item.path: item for item in repo_files}
    front_by_ride = {}
    rear_by_ride = {}
    for item in repo_files:
        ride_id = ride_id_from_video(item.path)
        if ride_id is None:
            continue
        if item.path.endswith("_front_camera.mp4"):
            front_by_ride[ride_id] = item
        elif item.path.endswith("_rear_camera.mp4"):
            rear_by_ride[ride_id] = item

    selected: list[SelectedFile] = []
    known_total = 0
    unknown_sizes = False
    rides = sorted(front_by_ride)
    for ride_id in rides:
        if len([item for item in selected if item.role == "video_front"]) >= sample_videos:
            break
        front = front_by_ride[ride_id]
        files = [front]
        if prefer_pairs and ride_id in rear_by_ride:
            files.append(rear_by_ride[ride_id])
        projected = known_total + sum(item.size or 0 for item in files)
        if any(item.size is None for item in files):
            unknown_sizes = True
            if len(selected) >= max(1, sample_videos * (2 if prefer_pairs else 1)):
                break
        elif projected > budget_bytes:
            continue
        for item in files:
            role = "video_front" if item.path == front.path else "video_rear"
            selected.append(SelectedFile(item.path, item.size, role))
        known_total = projected

    if unknown_sizes:
        warnings.append("some video file sizes are unavailable; selection is conservative but budget cannot be guaranteed")
    if not selected and any(path.startswith("videos/") for path in by_path):
        warnings.append("videos exist but none fit the budget/selection rules")
    if not front_by_ride:
        warnings.append("no videos/*front_camera.mp4 files found in repo file listing")
    return selected, warnings


def ride_id_from_video(path: str) -> str | None:
    match = re.match(r"^videos/(.+)_(front|rear)_camera\.mp4$", path)
    if not match:
        return None
    return match.group(1)


def select_zarr_sample(repo_files: list[RepoFile], budget_bytes: int) -> tuple[list[SelectedFile], list[str]]:
    warnings = []
    selected: list[SelectedFile] = []
    known_total = 0
    by_array = {array: [] for array in IMPORTANT_ZARR_ARRAYS}
    for item in repo_files:
        array = zarr_array_for_chunk(item.path)
        if array in by_array:
            by_array[array].append(item)

    for array in IMPORTANT_ZARR_ARRAYS:
        chunks = sorted(by_array.get(array, []), key=lambda item: item.path)
        if not chunks:
            continue
        for item in chunks[:2]:
            projected = known_total + (item.size or 0)
            if item.size is not None and projected > budget_bytes:
                break
            selected.append(SelectedFile(item.path, item.size, f"zarr_chunk:{array}"))
            known_total = projected
    if not selected:
        warnings.append("no safe zarr sample chunks found in repo file listing")
    return selected, warnings


def zarr_array_for_chunk(path: str) -> str | None:
    prefix = f"{ZARR_ROOT}/"
    if not path.startswith(prefix):
        return None
    rel = path[len(prefix) :]
    parts = rel.split("/")
    if len(parts) < 2:
        return None
    if parts[-1].startswith("."):
        return None
    array = "/".join(parts[:-1])
    if "/" in array:
        return None
    return array


def dedupe_selection(items: Iterable[SelectedFile]) -> list[SelectedFile]:
    deduped = {}
    for item in items:
        deduped[item.path] = item
    return sorted(deduped.values(), key=lambda item: item.path)


def sum_known_sizes(items: Iterable[SelectedFile]) -> int:
    return sum(item.size or 0 for item in items)


def download_selected_files(
    selected: list[SelectedFile],
    repo_id: str,
    output_dir: Path,
    budget_bytes: int,
    token: str | None,
) -> tuple[list[dict[str, Any]], int, list[str]]:
    from huggingface_hub import hf_hub_download

    warnings = []
    downloaded = []
    actual_bytes = current_selected_local_size(output_dir, selected)
    for item in selected:
        if item.size is not None and actual_bytes + item.size > budget_bytes:
            warnings.append(f"stopped before budget overflow at {item.path}")
            break
        local_path = output_dir / item.path
        if not local_path.exists():
            hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename=item.path,
                local_dir=output_dir,
                token=token,
            )
        if not local_path.exists():
            warnings.append(f"download did not create expected file: {item.path}")
            continue
        size = local_path.stat().st_size
        if item.size is not None and size != item.size:
            warnings.append(f"size mismatch for {item.path}: expected {item.size}, got {size}")
        if actual_bytes + size > budget_bytes:
            warnings.append(f"budget exceeded after {item.path}; stopping")
            break
        actual_bytes += size
        downloaded.append({"path": item.path, "role": item.role, "size": size})
    return downloaded, actual_bytes, warnings


def current_selected_local_size(output_dir: Path, selected: list[SelectedFile]) -> int:
    return 0


def parse_downloaded_zarr_metadata(output_dir: Path) -> list[dict[str, Any]]:
    arrays = []
    for path in sorted(output_dir.rglob(".zarray")):
        rel_array = path.parent.relative_to(output_dir).as_posix()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        arrays.append(parse_zarr_array_metadata(rel_array, payload))
    return arrays


def parse_zarr_array_metadata(array_path: str, payload: dict[str, Any]) -> dict[str, Any]:
    shape = payload.get("shape")
    chunks = payload.get("chunks")
    dtype = payload.get("dtype")
    compressor = payload.get("compressor")
    if isinstance(compressor, dict):
        compressor = compressor.get("id", str(compressor))
    return {
        "array_path": array_path,
        "shape": shape,
        "chunks": chunks,
        "dtype": dtype,
        "compressor": compressor,
        "estimated_logical_size": estimate_logical_size(shape, dtype),
    }


def estimate_logical_size(shape: Any, dtype: Any) -> int | None:
    if not isinstance(shape, list) or dtype is None:
        return None
    itemsize = dtype_itemsize(str(dtype))
    if itemsize is None:
        return None
    count = 1
    for dim in shape:
        try:
            count *= int(dim)
        except (TypeError, ValueError):
            return None
    return count * itemsize


def dtype_itemsize(dtype: str) -> int | None:
    zarr_match = re.fullmatch(r"[<>=|]?[fiu](\d+)", dtype)
    if zarr_match:
        return int(zarr_match.group(1))
    match = re.search(r"(\d+)$", dtype)
    if match:
        bits = int(match.group(1))
        if bits % 8 == 0:
            return bits // 8
    if dtype in {"|b1", "bool"}:
        return 1
    return None


def write_file_reports(repo_files: list[RepoFile], reports_dir: Path) -> None:
    payload = [asdict(item) for item in repo_files]
    write_json(reports_dir / "files_report.json", payload)
    with (reports_dir / "files_report.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "size"])
        writer.writeheader()
        writer.writerows(payload)


def write_zarr_reports(arrays: list[dict[str, Any]], reports_dir: Path) -> None:
    write_json(reports_dir / "zarr_arrays.json", arrays)
    with (reports_dir / "zarr_arrays.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["array_path", "shape", "chunks", "dtype", "compressor", "estimated_logical_size"],
        )
        writer.writeheader()
        for row in arrays:
            writer.writerow({key: json.dumps(value) if isinstance(value, list) else value for key, value in row.items()})


def write_subset_summary(path: Path, manifest: dict[str, Any], actual_bytes: int, zarr_arrays: list[dict[str, Any]]) -> None:
    selected = manifest.get("selected_files", [])
    downloaded = manifest.get("downloaded_files", [])
    front_count = count_role(downloaded or selected, "video_front")
    rear_count = count_role(downloaded or selected, "video_rear")
    metadata_count = count_role(downloaded or selected, "metadata")
    zarr_chunk_count = sum(1 for item in downloaded or selected if str(item.get("role", "")).startswith("zarr_chunk:"))
    lines = [
        "# FrodoBots 7K Subset Summary",
        "",
        f"- budget_gb: {manifest.get('budget_gb')}",
        f"- estimated_download_size_gb: {manifest.get('estimated_gb', 0):.3f}",
        f"- actual_downloaded_size_gb: {actual_bytes / GB:.3f}",
        f"- number_of_front_videos: {front_count}",
        f"- number_of_rear_videos: {rear_count}",
        f"- metadata_files_downloaded: {metadata_count}",
        f"- zarr_arrays_detected: {len(zarr_arrays)}",
        f"- zarr_chunks_downloaded: {zarr_chunk_count}",
        "",
        "## Recommended Next Step",
        "",
        "Inspect the downloaded metadata and video samples. If front/rear videos and matching action/timestamp chunks are present, build a ride-level loader. Otherwise request an extracted ride-level subset from the organizers.",
    ]
    warnings = manifest.get("warnings") or []
    if warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend([f"- {warning}" for warning in warnings])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def count_role(items: list[dict[str, Any]], role: str) -> int:
    return sum(1 for item in items if item.get("role") == role)


def maybe_extract_frame_preview(output_dir: Path, reports_dir: Path, downloaded: list[dict[str, Any]]) -> None:
    try:
        import cv2
        import numpy as np
    except ImportError:
        return
    sample_dir = reports_dir / "frame_samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    images = []
    front_videos = [output_dir / item["path"] for item in downloaded if item.get("role") == "video_front"]
    for video_index, video_path in enumerate(front_videos[:4]):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            continue
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frame_indices = [0, max(0, total // 2), max(0, total - 1)] if total else [0]
        for frame_index in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok:
                continue
            sample_path = sample_dir / f"video{video_index:02d}_frame{frame_index:06d}.jpg"
            cv2.imwrite(str(sample_path), frame)
            thumb = cv2.resize(frame, (240, 135), interpolation=cv2.INTER_AREA)
            images.append(thumb)
        cap.release()
    if not images:
        return
    width = 4
    rows = math.ceil(len(images) / width)
    canvas = np.zeros((rows * 135, width * 240, 3), dtype=np.uint8)
    for index, image in enumerate(images):
        y = (index // width) * 135
        x = (index % width) * 240
        canvas[y : y + 135, x : x + 240] = image
    cv2.imwrite(str(reports_dir / "frame_contact_sheet.jpg"), canvas)


def print_selection(selected: list[SelectedFile], size_bytes: int, budget_bytes: int, warnings: list[str], dry_run: bool) -> None:
    label = "dry-run selected" if dry_run else "downloaded/selected"
    print(f"{label} files: {len(selected)}")
    print(f"size: {size_bytes / GB:.3f} GB / budget {budget_bytes / GB:.3f} GB")
    print(f"budget respected: {size_bytes <= budget_bytes}")
    for item in selected[:50]:
        size = "unknown" if item.size is None else f"{item.size / GB:.3f}GB"
        print(f"  [{item.role}] {item.path} ({size})")
    if len(selected) > 50:
        print(f"  ... {len(selected) - 50} more")
    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    raise SystemExit(main())
