from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Literal

from research.training.datasets.action_labels import ACTION_NAMES, classify_action


CONTROL_COLUMNS = ("linear", "angular", "rpm_1", "rpm_2", "rpm_3", "rpm_4", "timestamp")
FRONT_COLUMNS = ("frame_id", "timestamp")
MANIFEST_COLUMNS = (
    "ride_id",
    "front_playlist_ref",
    "front_segment_ref",
    "front_frame_id",
    "front_timestamp",
    "matched_control_timestamp",
    "control_delta_ms",
    "linear",
    "angular",
    "action_class",
    "timeline_section_id",
)
RIDE_PATTERN = re.compile(r"^ride_(\d+)_")
SEGMENT_TIMESTAMP_PATTERN = re.compile(r"_(\d{17})\.ts$")


class DatasetFormatError(ValueError):
    pass


@dataclass(frozen=True)
class ControlRecord:
    timestamp: float
    linear: float
    angular: float
    rpms: tuple[float, float, float, float]

    @property
    def payload(self) -> tuple[float, ...]:
        return (self.linear, self.angular, *self.rpms)


@dataclass(frozen=True)
class ControlEvent:
    timestamp: float
    control: ControlRecord | None
    invalid_reason: str | None = None


@dataclass(frozen=True)
class ControlSection:
    section_id: int
    events: tuple[ControlEvent, ...]
    timestamps: tuple[float, ...]
    start: float
    end: float


@dataclass(frozen=True)
class ControlTimeline:
    sections: tuple[ControlSection, ...]
    stats: dict[str, int]


@dataclass(frozen=True)
class ControlMatch:
    control: ControlRecord
    delta_ms: float
    section_id: int


@dataclass(frozen=True)
class HlsSegment:
    reference: str
    start: float
    end: float
    section_id: int


@dataclass(frozen=True)
class HlsTimeline:
    playlist_reference: str
    segments: tuple[HlsSegment, ...]
    starts: tuple[float, ...]
    stats: dict[str, int]

    def find_segment(self, timestamp: float) -> HlsSegment | None:
        index = bisect_right(self.starts, timestamp) - 1
        if index < 0:
            return None
        segment = self.segments[index]
        if segment.start <= timestamp < segment.end:
            return segment
        return None


def normalize_timestamp(value: object, unit: Literal["seconds", "milliseconds"]) -> float:
    try:
        timestamp = float(value)
    except (TypeError, ValueError) as exc:
        raise DatasetFormatError(f"invalid timestamp: {value!r}") from exc
    if not math.isfinite(timestamp):
        raise DatasetFormatError(f"non-finite timestamp: {value!r}")
    if unit == "milliseconds":
        return timestamp / 1000.0
    if unit == "seconds":
        return timestamp
    raise DatasetFormatError(f"unsupported timestamp unit: {unit}")


def parse_hls_segment_timestamp(reference: str | Path) -> float:
    match = SEGMENT_TIMESTAMP_PATTERN.search(Path(reference).name)
    if match is None:
        raise DatasetFormatError(f"segment timestamp missing: {reference}")
    return _parse_segment_timestamp(match.group(1))


def load_control_timeline(path: str | Path) -> ControlTimeline:
    path = Path(path)
    parsed_sections: list[list[tuple[float, ControlRecord | None]]] = [[]]
    rows_by_timestamp: dict[float, list[tuple[float, ...] | None]] = defaultdict(list)
    stats = Counter(
        {
            "control_rows": 0,
            "malformed_control_rows": 0,
            "timestamp_reversals": 0,
            "identical_duplicate_groups": 0,
            "identical_duplicate_rows_removed": 0,
            "conflicting_duplicate_groups": 0,
            "conflicting_duplicate_rows": 0,
        }
    )
    previous_timestamp: float | None = None

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        _require_columns(reader.fieldnames, CONTROL_COLUMNS, path)
        for row in reader:
            stats["control_rows"] += 1
            try:
                timestamp = normalize_timestamp(row["timestamp"], "seconds")
            except DatasetFormatError:
                stats["malformed_control_rows"] += 1
                continue

            if previous_timestamp is not None and timestamp < previous_timestamp:
                stats["timestamp_reversals"] += 1
                parsed_sections.append([])
            previous_timestamp = timestamp

            try:
                values = tuple(float(row[name]) for name in CONTROL_COLUMNS[:-1])
                if not all(math.isfinite(value) for value in values):
                    raise ValueError("non-finite control value")
                record = ControlRecord(
                    timestamp=timestamp,
                    linear=values[0],
                    angular=values[1],
                    rpms=(values[2], values[3], values[4], values[5]),
                )
                payload: tuple[float, ...] | None = record.payload
            except (TypeError, ValueError):
                stats["malformed_control_rows"] += 1
                record = None
                payload = None

            parsed_sections[-1].append((timestamp, record))
            rows_by_timestamp[timestamp].append(payload)

    if stats["control_rows"] == 0:
        raise DatasetFormatError(f"no control rows: {path}")
    if not any(section for section in parsed_sections):
        raise DatasetFormatError(f"no usable control timestamps: {path}")

    duplicate_reasons: dict[float, str | None] = {}
    for timestamp, payloads in rows_by_timestamp.items():
        if len(payloads) == 1:
            duplicate_reasons[timestamp] = "malformed_control_row" if payloads[0] is None else None
            continue
        unique_payloads = set(payloads)
        if None not in unique_payloads and len(unique_payloads) == 1:
            stats["identical_duplicate_groups"] += 1
            stats["identical_duplicate_rows_removed"] += len(payloads) - 1
            duplicate_reasons[timestamp] = None
        else:
            stats["conflicting_duplicate_groups"] += 1
            stats["conflicting_duplicate_rows"] += len(payloads)
            duplicate_reasons[timestamp] = "conflicting_control_duplicate"

    sections: list[ControlSection] = []
    for section_id, raw_events in enumerate(parsed_sections):
        if not raw_events:
            continue
        events_by_timestamp: dict[float, ControlEvent] = {}
        for timestamp, record in raw_events:
            reason = duplicate_reasons[timestamp]
            if timestamp not in events_by_timestamp or reason is not None:
                events_by_timestamp[timestamp] = ControlEvent(timestamp, record if reason is None else None, reason)
        events = tuple(events_by_timestamp[timestamp] for timestamp in sorted(events_by_timestamp))
        timestamps = tuple(event.timestamp for event in events)
        sections.append(
            ControlSection(
                section_id=section_id,
                events=events,
                timestamps=timestamps,
                start=timestamps[0],
                end=timestamps[-1],
            )
        )

    return ControlTimeline(tuple(sections), dict(stats))


def match_nearest_control(
    timeline: ControlTimeline,
    front_timestamp: float,
    tolerance_sec: float,
) -> tuple[ControlMatch | None, str | None]:
    if not math.isfinite(tolerance_sec) or tolerance_sec < 0:
        raise ValueError("tolerance_sec must be finite and non-negative")
    sections = _candidate_control_sections(timeline.sections, front_timestamp, tolerance_sec)
    if len(sections) > 1:
        return None, "ambiguous_control_section"
    if not sections:
        return None, "no_control_in_ride"

    section = sections[0]
    index = bisect_left(section.timestamps, front_timestamp)
    candidate_indexes = [item for item in (index - 1, index) if 0 <= item < len(section.events)]
    event = min(
        (section.events[item] for item in candidate_indexes),
        key=lambda item: (abs(item.timestamp - front_timestamp), item.timestamp),
    )
    delta_sec = abs(event.timestamp - front_timestamp)
    if delta_sec > tolerance_sec + 1e-9:
        return None, "control_delta_exceeds_tolerance"
    if event.invalid_reason is not None:
        return None, event.invalid_reason
    if event.control is None:
        return None, "malformed_control_row"
    return ControlMatch(event.control, delta_sec * 1000.0, section.section_id), None


def parse_front_hls_playlist(
    path: str | Path,
    dataset_root: str | Path,
    discontinuity_tolerance_sec: float = 0.1,
) -> HlsTimeline:
    path = Path(path)
    dataset_root = Path(dataset_root)
    if not math.isfinite(discontinuity_tolerance_sec) or discontinuity_tolerance_sec < 0:
        raise ValueError("discontinuity_tolerance_sec must be finite and non-negative")
    raw_segments: list[tuple[str, float, float, bool]] = []
    pending_duration: float | None = None
    discontinuity_before_next = False
    explicit_discontinuities = 0
    previous_start: float | None = None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line == "#EXT-X-DISCONTINUITY":
            discontinuity_before_next = True
            explicit_discontinuities += 1
            continue
        if line.startswith("#EXTINF:"):
            try:
                pending_duration = float(line.split(":", 1)[1].rstrip(","))
            except ValueError as exc:
                raise DatasetFormatError(f"invalid EXTINF in {path}: {line}") from exc
            continue
        if line.startswith("#") or not line.endswith(".ts"):
            continue
        if pending_duration is None or not math.isfinite(pending_duration) or pending_duration <= 0:
            raise DatasetFormatError(f"segment without valid EXTINF in {path}: {line}")
        try:
            start = parse_hls_segment_timestamp(line)
        except DatasetFormatError as exc:
            raise DatasetFormatError(f"segment timestamp missing in {path}: {line}") from exc
        if previous_start is not None and start < previous_start:
            raise DatasetFormatError(f"non-monotonic HLS segment timestamps in {path.name}")
        segment_path = path.parent / line
        if not segment_path.is_file():
            raise DatasetFormatError(f"missing HLS segment referenced by {path.name}: {line}")
        reference = _relative_posix(segment_path, dataset_root)
        raw_segments.append((reference, start, pending_duration, discontinuity_before_next))
        previous_start = start
        pending_duration = None
        discontinuity_before_next = False

    if not raw_segments:
        raise DatasetFormatError(f"no HLS segments: {path}")

    sections = 0
    inferred_discontinuities = 0
    segments: list[HlsSegment] = []
    previous_end: float | None = None
    for reference, start, duration, explicit_boundary in raw_segments:
        inferred_boundary = previous_end is not None and abs(start - previous_end) > discontinuity_tolerance_sec
        if inferred_boundary:
            inferred_discontinuities += 1
        if segments and (explicit_boundary or inferred_boundary):
            sections += 1
        segment = HlsSegment(reference, start, start + duration, sections)
        segments.append(segment)
        previous_end = segment.end

    playlist_reference = _relative_posix(path, dataset_root)
    stats = {
        "hls_segments": len(segments),
        "hls_sections": sections + 1,
        "explicit_discontinuities": explicit_discontinuities,
        "inferred_discontinuities": inferred_discontinuities,
    }
    return HlsTimeline(playlist_reference, tuple(segments), tuple(item.start for item in segments), stats)


def build_manifest(
    dataset_root: str | Path,
    output_dir: str | Path,
    max_rides: int | None = None,
    tolerance_ms: float = 100.0,
) -> dict[str, object]:
    dataset_root = Path(dataset_root).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if not dataset_root.is_dir():
        raise DatasetFormatError(f"dataset root does not exist: {dataset_root}")
    if output_dir == dataset_root or dataset_root in output_dir.parents:
        raise ValueError("output directory must be outside the immutable raw dataset root")
    if max_rides is not None and max_rides <= 0:
        raise ValueError("max_rides must be positive")
    if not math.isfinite(tolerance_ms) or tolerance_ms < 0:
        raise ValueError("tolerance_ms must be finite and non-negative")

    ride_dirs = sorted(path for path in dataset_root.iterdir() if path.is_dir() and RIDE_PATTERN.match(path.name))
    if max_rides is not None:
        ride_dirs = ride_dirs[:max_rides]
    if not ride_dirs:
        raise DatasetFormatError(f"no ride directories found: {dataset_root}")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.csv"
    temporary_manifest = output_dir / ".manifest.csv.tmp"
    report_path = output_dir / "alignment_report.json"
    rejection_reasons: Counter[str] = Counter()
    action_distribution: Counter[str] = Counter()
    aggregate_control_stats: Counter[str] = Counter()
    aggregate_hls_stats: Counter[str] = Counter()
    ride_errors: list[dict[str, str]] = []
    control_deltas: list[float] = []
    total_front_frames = 0
    valid_samples = 0

    try:
        with temporary_manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
            writer.writeheader()
            for ride_dir in ride_dirs:
                ride_id = _ride_id(ride_dir)
                front_path = ride_dir / f"front_camera_timestamps_{ride_id}.csv"
                control_path = ride_dir / f"control_data_{ride_id}.csv"
                playlists = sorted((ride_dir / "recordings").glob("*uid_s_1000*video.m3u8"))

                try:
                    if not front_path.is_file():
                        raise DatasetFormatError(f"missing front timestamp CSV: {front_path.name}")
                    if not control_path.is_file():
                        raise DatasetFormatError(f"missing control CSV: {control_path.name}")
                    if len(playlists) != 1:
                        raise DatasetFormatError(f"expected one front playlist, found {len(playlists)}")
                    controls = load_control_timeline(control_path)
                    hls = parse_front_hls_playlist(playlists[0], dataset_root)
                    aggregate_control_stats.update(controls.stats)
                    aggregate_hls_stats.update(hls.stats)
                except (DatasetFormatError, OSError) as exc:
                    ride_errors.append({"ride_id": ride_id, "error": str(exc)})
                    rows = _count_csv_data_rows(front_path)
                    total_front_frames += rows
                    rejection_reasons["invalid_ride_input"] += rows
                    continue

                for frame_id, front_timestamp, row_reason in _read_front_rows(front_path):
                    total_front_frames += 1
                    if row_reason is not None:
                        rejection_reasons[row_reason] += 1
                        continue
                    if frame_id is None or front_timestamp is None:
                        rejection_reasons["malformed_front_row"] += 1
                        continue

                    front_segment = hls.find_segment(front_timestamp)
                    if front_segment is None:
                        rejection_reasons["front_outside_hls_coverage"] += 1
                        continue
                    match, match_reason = match_nearest_control(
                        controls,
                        front_timestamp,
                        tolerance_sec=tolerance_ms / 1000.0,
                    )
                    if match_reason is not None or match is None:
                        rejection_reasons[match_reason or "no_control_in_ride"] += 1
                        continue
                    control_segment = hls.find_segment(match.control.timestamp)
                    if control_segment is None:
                        rejection_reasons["control_outside_hls_coverage"] += 1
                        continue
                    if control_segment.section_id != front_segment.section_id:
                        rejection_reasons["hls_section_mismatch"] += 1
                        continue

                    action_class = classify_action(match.control.linear, match.control.angular)
                    writer.writerow(
                        {
                            "ride_id": ride_id,
                            "front_playlist_ref": hls.playlist_reference,
                            "front_segment_ref": front_segment.reference,
                            "front_frame_id": frame_id,
                            "front_timestamp": _format_float(front_timestamp, 6),
                            "matched_control_timestamp": _format_float(match.control.timestamp, 6),
                            "control_delta_ms": _format_float(match.delta_ms, 3),
                            "linear": _format_float(match.control.linear, 12),
                            "angular": _format_float(match.control.angular, 12),
                            "action_class": action_class,
                            "timeline_section_id": front_segment.section_id,
                        }
                    )
                    valid_samples += 1
                    control_deltas.append(match.delta_ms)
                    action_distribution[action_class] += 1
        os.replace(temporary_manifest, manifest_path)
    finally:
        if temporary_manifest.exists():
            temporary_manifest.unlink()

    rejected_samples = sum(rejection_reasons.values())
    report: dict[str, object] = {
        "processed_ride_count": len(ride_dirs),
        "ride_ids": [_ride_id(path) for path in ride_dirs],
        "total_front_frame_count": total_front_frames,
        "valid_sample_count": valid_samples,
        "rejected_sample_count": rejected_samples,
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
        "action_class_distribution": {name: action_distribution[name] for name in ACTION_NAMES},
        "control_delta_ms": {
            "p50": _rounded(_percentile(control_deltas, 50), 3),
            "p95": _rounded(_percentile(control_deltas, 95), 3),
            "max": _rounded(max(control_deltas), 3) if control_deltas else None,
        },
        "control_quality": dict(sorted(aggregate_control_stats.items())),
        "hls_quality": dict(sorted(aggregate_hls_stats.items())),
        "ride_errors": ride_errors,
        "control_tolerance_ms": tolerance_ms,
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
    }
    if total_front_frames != valid_samples + rejected_samples:
        raise RuntimeError("front-frame accounting invariant failed")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _candidate_control_sections(
    sections: tuple[ControlSection, ...],
    timestamp: float,
    tolerance_sec: float,
) -> list[ControlSection]:
    containing = [
        section
        for section in sections
        if section.start - tolerance_sec <= timestamp <= section.end + tolerance_sec
    ]
    if containing:
        return containing
    if not sections:
        return []

    distances = [
        min(abs(timestamp - section.start), abs(timestamp - section.end))
        for section in sections
    ]
    nearest_distance = min(distances)
    return [section for section, distance in zip(sections, distances) if abs(distance - nearest_distance) <= 1e-12]


def _read_front_rows(path: Path) -> Iterable[tuple[int | None, float | None, str | None]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        _require_columns(reader.fieldnames, FRONT_COLUMNS, path)
        for row in reader:
            try:
                frame_id = int(row["frame_id"])
                timestamp = normalize_timestamp(row["timestamp"], "seconds")
                if frame_id < 0:
                    raise ValueError("negative frame ID")
            except (DatasetFormatError, TypeError, ValueError):
                yield None, None, "malformed_front_row"
                continue
            yield frame_id, timestamp, None


def _require_columns(fieldnames: list[str] | None, required: Iterable[str], path: Path) -> None:
    available = set(fieldnames or [])
    missing = [name for name in required if name not in available]
    if missing:
        raise DatasetFormatError(f"missing columns in {path}: {', '.join(missing)}")


def _count_csv_data_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open(newline="", encoding="utf-8") as handle:
        return max(sum(1 for _ in csv.reader(handle)) - 1, 0)


def _ride_id(ride_dir: Path) -> str:
    match = RIDE_PATTERN.match(ride_dir.name)
    if match is None:
        raise DatasetFormatError(f"invalid ride directory name: {ride_dir.name}")
    return match.group(1)


def _parse_segment_timestamp(value: str) -> float:
    try:
        parsed = datetime.strptime(value, "%Y%m%d%H%M%S%f").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise DatasetFormatError(f"invalid segment timestamp: {value}") from exc
    return parsed.timestamp()


def _relative_posix(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise DatasetFormatError(f"path is outside dataset root: {path}") from exc


def _format_float(value: float, digits: int) -> str:
    return f"{value:.{digits}f}".rstrip("0").rstrip(".")


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rounded(value: float | None, digits: int) -> float | None:
    return round(value, digits) if value is not None else None
