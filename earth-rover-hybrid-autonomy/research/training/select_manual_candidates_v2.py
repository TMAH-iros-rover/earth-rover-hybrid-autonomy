#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import shutil
import zipfile
from pathlib import Path


SELECTED_FIELDS = (
    "candidate_id",
    "dataset",
    "ride_id",
    "camera_uid",
    "timestamp_sec",
    "playlist_path",
    "image_path",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a CVAT upload bundle from reviewed manual candidate numbers."
    )
    parser.add_argument("--source-bundle", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--selection", nargs="+", type=int, required=True)
    return parser.parse_args()


def normalized_candidate_ids(numbers: list[int]) -> list[str]:
    if any(number <= 0 or number > 9999 for number in numbers):
        raise ValueError("selection numbers must be between 1 and 9999")
    if len(set(numbers)) != len(numbers):
        raise ValueError("selection numbers must be unique")
    return [f"manual_v2_{number:04d}" for number in numbers]


def create_selected_bundle(
    source_bundle: str | Path,
    output_dir: str | Path,
    selection: list[int],
) -> dict[str, object]:
    source = Path(source_bundle).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise ValueError(f"output path already exists: {output}")
    temporary = output.parent / f".{output.name}.tmp"
    if temporary.exists():
        raise ValueError(f"temporary output already exists: {temporary}")
    if output == source or source in output.parents:
        raise ValueError("output directory must be outside the source bundle")

    candidate_ids = normalized_candidate_ids(selection)
    rows = _read_candidates(source / "candidates.csv")
    rows_by_id = {row["candidate_id"]: row for row in rows}
    if len(rows_by_id) != len(rows):
        raise ValueError("candidates.csv contains duplicate candidate IDs")
    missing = [candidate_id for candidate_id in candidate_ids if candidate_id not in rows_by_id]
    if missing:
        raise ValueError(f"selected candidates are missing from candidates.csv: {missing}")

    selected_rows = [dict(rows_by_id[candidate_id]) for candidate_id in candidate_ids]
    source_images = [_source_image(source, row) for row in selected_rows]
    archive_name = f"manual_candidates_v2_selected_{len(selected_rows)}.zip"

    try:
        images_dir = temporary / "images"
        images_dir.mkdir(parents=True)
        for row, source_image in zip(selected_rows, source_images):
            destination = images_dir / source_image.name
            shutil.copy2(source_image, destination)
            row["image_path"] = f"images/{destination.name}"

        with (temporary / "selected_candidates.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=SELECTED_FIELDS)
            writer.writeheader()
            writer.writerows(selected_rows)
        (temporary / "selection.txt").write_text(
            "\n".join(candidate_ids) + "\n", encoding="utf-8"
        )
        (temporary / "README.md").write_text(
            f"# Manual Traversability Candidates v2 - Selected {len(selected_rows)}\n\n"
            "This directory contains the manually selected, unannotated front-camera "
            "images for CVAT. The JPG files are byte-for-byte copies of the source "
            "candidate images. Upload the ZIP using CVAT's image task workflow. "
            "Candidate provenance is recorded in `selected_candidates.csv`.\n",
            encoding="utf-8",
        )
        archive_path = temporary / archive_name
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as archive:
            for row in selected_rows:
                image = temporary / row["image_path"]
                archive.write(image, arcname=image.name)
        os.replace(temporary, output)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    return {
        "selected_count": len(selected_rows),
        "selected_candidate_ids": candidate_ids,
        "output_dir": str(output),
        "archive_path": str(output / archive_name),
    }


def _read_candidates(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise ValueError(f"candidates.csv is missing: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != SELECTED_FIELDS:
            raise ValueError(f"unexpected candidates.csv fields: {reader.fieldnames}")
        return list(reader)


def _source_image(source: Path, row: dict[str, str]) -> Path:
    image = (source / row["image_path"]).resolve()
    if source not in image.parents:
        raise ValueError(f"candidate image escapes the source bundle: {row['image_path']}")
    expected_name = f"{row['candidate_id']}.jpg"
    if image.name != expected_name:
        raise ValueError(
            f"candidate image name mismatch for {row['candidate_id']}: {image.name}"
        )
    if not image.is_file():
        raise ValueError(f"candidate image is missing: {image}")
    return image


def main() -> int:
    args = parse_args()
    try:
        result = create_selected_bundle(args.source_bundle, args.output_dir, args.selection)
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise SystemExit(f"selection failed: {exc}") from exc
    print(f"Selected images: {result['selected_count']}")
    print(f"Output directory: {result['output_dir']}")
    print(f"CVAT ZIP: {result['archive_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
