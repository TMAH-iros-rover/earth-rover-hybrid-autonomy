from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from research.training.datasets.frodobots_2k_manifest import (
    DatasetFormatError,
    HlsTimeline,
    RIDE_PATTERN,
    normalize_timestamp,
    parse_front_hls_playlist,
)
from research.training.traversability_expansion import difference_hash, hamming_distance


CAMERA_UID = "1000"
CANDIDATE_FIELDS = (
    "candidate_id",
    "dataset",
    "ride_id",
    "camera_uid",
    "timestamp_sec",
    "playlist_path",
    "image_path",
)


@dataclass(frozen=True)
class FrontFrame:
    frame_id: int
    timestamp: float


@dataclass(frozen=True)
class RideSource:
    dataset: str
    dataset_root: Path
    ride_id: str
    ride_dir: Path
    playlist_path: Path
    timeline: HlsTimeline
    frames: tuple[FrontFrame, ...]


@dataclass(frozen=True)
class ExistingCandidates:
    ride_ids: frozenset[str]
    image_hashes: tuple[int, ...]


def load_existing_candidates(
    metadata_paths: list[str | Path],
    report_paths: list[str | Path],
) -> ExistingCandidates:
    ride_ids: set[str] = set()
    image_hashes: list[int] = []
    for value in metadata_paths:
        path = Path(value).expanduser().resolve()
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows or "ride_id" not in rows[0]:
            raise ValueError(f"metadata has no ride_id rows: {path}")
        for row in rows:
            ride_id = row.get("ride_id", "").strip()
            if not ride_id:
                raise ValueError(f"metadata contains an empty ride_id: {path}")
            ride_ids.add(ride_id)
            image_value = row.get("image_path", "").strip()
            if image_value:
                image_path = (path.parent / image_value).resolve()
                if not image_path.is_file():
                    raise ValueError(f"existing image is missing: {image_path}")
                image_hashes.append(difference_hash(image_path))
    for value in report_paths:
        path = Path(value).expanduser().resolve()
        report = json.loads(path.read_text(encoding="utf-8"))
        distribution = report.get("ride_distribution", {})
        if not isinstance(distribution, dict):
            raise ValueError(f"report ride_distribution is not an object: {path}")
        ride_ids.update(str(ride_id) for ride_id in distribution)
    return ExistingCandidates(frozenset(ride_ids), tuple(image_hashes))


def discover_front_rides(dataset_roots: list[str | Path]) -> tuple[list[RideSource], dict[str, int]]:
    rides: list[RideSource] = []
    stats = {
        "ride_directories": 0,
        "missing_front_timestamp_csv": 0,
        "invalid_front_playlist_count": 0,
        "invalid_front_timeline": 0,
        "no_usable_front_frames": 0,
        "duplicate_ride_id": 0,
    }
    seen_dataset_names: set[str] = set()
    seen_ride_ids: set[str] = set()
    for value in dataset_roots:
        root = Path(value).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"dataset root does not exist: {root}")
        dataset_name = root.name
        if dataset_name in seen_dataset_names:
            raise ValueError(f"dataset names must be unique: {dataset_name}")
        seen_dataset_names.add(dataset_name)
        for ride_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            match = RIDE_PATTERN.match(ride_dir.name)
            if match is None:
                continue
            stats["ride_directories"] += 1
            ride_id = match.group(1)
            if ride_id in seen_ride_ids:
                stats["duplicate_ride_id"] += 1
                continue
            timestamp_path = ride_dir / f"front_camera_timestamps_{ride_id}.csv"
            if not timestamp_path.is_file():
                stats["missing_front_timestamp_csv"] += 1
                continue
            playlists = front_video_playlists(ride_dir / "recordings")
            if len(playlists) != 1:
                stats["invalid_front_playlist_count"] += 1
                continue
            try:
                timeline = parse_front_hls_playlist(playlists[0], root)
                frames = load_front_frames(timestamp_path, timeline)
            except (DatasetFormatError, OSError):
                stats["invalid_front_timeline"] += 1
                continue
            if not frames:
                stats["no_usable_front_frames"] += 1
                continue
            seen_ride_ids.add(ride_id)
            rides.append(
                RideSource(
                    dataset=dataset_name,
                    dataset_root=root,
                    ride_id=ride_id,
                    ride_dir=ride_dir,
                    playlist_path=playlists[0],
                    timeline=timeline,
                    frames=frames,
                )
            )
    return rides, stats


def front_video_playlists(recordings_dir: Path) -> list[Path]:
    if not recordings_dir.is_dir():
        return []
    return sorted(
        path
        for path in recordings_dir.glob("*uid_s_1000*video.m3u8")
        if "uid_s_1001" not in path.name
    )


def load_front_frames(path: Path, timeline: HlsTimeline) -> tuple[FrontFrame, ...]:
    frames: list[FrontFrame] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not {"frame_id", "timestamp"}.issubset(reader.fieldnames or []):
            raise DatasetFormatError(f"invalid front timestamp columns: {path}")
        for row in reader:
            try:
                frame_id = int(row["frame_id"])
                timestamp = normalize_timestamp(row["timestamp"], "seconds")
            except (DatasetFormatError, TypeError, ValueError):
                continue
            if frame_id >= 0 and timeline.find_segment(timestamp) is not None:
                frames.append(FrontFrame(frame_id, timestamp))
    frames.sort(key=lambda frame: (frame.timestamp, frame.frame_id))
    return tuple(frames)


def frame_options(
    ride: RideSource,
    edge_margin_seconds: float,
    minimum_pair_separation_seconds: float,
    seed: int,
) -> tuple[FrontFrame, ...]:
    if edge_margin_seconds < 0 or minimum_pair_separation_seconds <= 0:
        raise ValueError("edge margin must be non-negative and pair separation must be positive")
    start = ride.frames[0].timestamp + edge_margin_seconds
    end = ride.frames[-1].timestamp - edge_margin_seconds
    eligible = [frame for frame in ride.frames if start <= frame.timestamp <= end]
    if not eligible:
        return ()
    primary_index = _stable_index(seed, ride.dataset, ride.ride_id, "primary", len(eligible))
    primary = eligible[primary_index]
    secondary_pool = [
        frame
        for frame in eligible
        if abs(frame.timestamp - primary.timestamp) >= minimum_pair_separation_seconds - 1e-9
    ]
    if not secondary_pool:
        return (primary,)
    secondary_index = _stable_index(seed, ride.dataset, ride.ride_id, "secondary", len(secondary_pool))
    return (primary, secondary_pool[secondary_index])


def deterministic_ride_order(rides: list[RideSource], seed: int) -> list[RideSource]:
    return sorted(
        rides,
        key=lambda ride: hashlib.sha256(
            f"{seed}:{ride.dataset}:{ride.ride_id}".encode()
        ).hexdigest(),
    )


def is_hash_distinct(value: int, existing: list[int] | tuple[int, ...], threshold: int) -> bool:
    if threshold < 0 or threshold > 64:
        raise ValueError("hash distance threshold must be between 0 and 64")
    return all(hamming_distance(value, other) > threshold for other in existing)


def minimum_hash_distance(values: list[int]) -> int | None:
    return min(
        (
            hamming_distance(left, right)
            for index, left in enumerate(values)
            for right in values[index + 1 :]
        ),
        default=None,
    )


def write_numbered_contact_sheets(
    entries: list[dict[str, str]],
    bundle_root: Path,
    sheet_size: int = 25,
) -> list[str]:
    if sheet_size <= 0:
        raise ValueError("sheet size must be positive")
    output_dir = bundle_root / "contact_sheets"
    output_dir.mkdir(parents=True, exist_ok=False)
    written: list[str] = []
    for sheet_index, offset in enumerate(range(0, len(entries), sheet_size), start=1):
        group = entries[offset : offset + sheet_size]
        canvas = np.full((5 * 174, 5 * 256, 3), 245, dtype=np.uint8)
        for position, entry in enumerate(group):
            image = cv2.imread(str(bundle_root / entry["image_path"]), cv2.IMREAD_COLOR)
            if image is None:
                raise OSError(f"cannot read contact-sheet image: {entry['image_path']}")
            tile = _fit_bgr(image, 256, 144)
            row, column = divmod(position, 5)
            y, x = row * 174, column * 256
            canvas[y : y + 144, x : x + 256] = tile
            label = f"{offset + position + 1:03d} {entry['candidate_id']}"
            cv2.putText(
                canvas,
                label,
                (x + 5, y + 164),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (20, 20, 20),
                1,
                cv2.LINE_AA,
            )
        relative = f"contact_sheets/contact_sheet_{sheet_index:03d}.jpg"
        if not cv2.imwrite(str(bundle_root / relative), canvas):
            raise OSError(f"cannot write contact sheet: {relative}")
        written.append(relative)
    return written


def _stable_index(seed: int, dataset: str, ride_id: str, role: str, length: int) -> int:
    digest = hashlib.sha256(f"{seed}:{dataset}:{ride_id}:{role}".encode()).hexdigest()
    return int(digest[:16], 16) % length


def _fit_bgr(image: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = min(width / image.shape[1], height / image.shape[0])
    resized_width = max(1, round(image.shape[1] * scale))
    resized_height = max(1, round(image.shape[0] * scale))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    x = (width - resized_width) // 2
    y = (height - resized_height) // 2
    canvas[y : y + resized_height, x : x + resized_width] = resized
    return canvas
