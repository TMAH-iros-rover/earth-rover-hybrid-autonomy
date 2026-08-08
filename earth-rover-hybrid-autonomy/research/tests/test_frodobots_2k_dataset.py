import csv
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from research.training.datasets.frodobots_2k_dataset import (
    FrameDecodeError,
    FrodoBotsActionDataset,
    HlsFrameDecoder,
    ManifestFormatError,
    ManifestSample,
    image_to_tensor,
    load_manifest,
)
from research.training.datasets.frodobots_2k_manifest import MANIFEST_COLUMNS, parse_hls_segment_timestamp
from research.training.verify_frodobots_2k_hls import _candidate_indices


SEGMENT_REF = (
    "ride_1_20240101000000/recordings/"
    "sample_ride_1__uid_s_1000__uid_e_video_20240101000000000.ts"
)


def sample_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "ride_id": "1",
        "front_playlist_ref": "ride_1_20240101000000/recordings/front.m3u8",
        "front_segment_ref": SEGMENT_REF,
        "front_frame_id": 1,
        "front_timestamp": 1704067200.05,
        "matched_control_timestamp": 1704067200.06,
        "control_delta_ms": 10.0,
        "linear": 0.2,
        "angular": 0.0,
        "action_class": "FORWARD",
        "timeline_section_id": 0,
    }
    row.update(overrides)
    return row


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


class StubDecoder:
    def __init__(self, frame: np.ndarray | None = None) -> None:
        self.frame = frame if frame is not None else np.full((12, 20, 3), 128, dtype=np.uint8)
        self.calls: list[ManifestSample] = []

    def decode(self, sample: ManifestSample) -> np.ndarray:
        self.calls.append(sample)
        return self.frame.copy()


def test_parse_hls_segment_timestamp() -> None:
    assert parse_hls_segment_timestamp(SEGMENT_REF) == pytest.approx(1704067200.0)


def test_manifest_loader_rejects_missing_columns(tmp_path: Path) -> None:
    path = tmp_path / "manifest.csv"
    path.write_text("ride_id\n1\n", encoding="utf-8")

    with pytest.raises(ManifestFormatError, match="missing manifest columns"):
        load_manifest(path)


def test_manifest_loader_rejects_empty_manifest(tmp_path: Path) -> None:
    path = tmp_path / "manifest.csv"
    write_manifest(path, [])

    with pytest.raises(ManifestFormatError, match="manifest has no samples"):
        load_manifest(path)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"front_timestamp": "nan"}, "non-finite"),
        ({"action_class": "UNKNOWN"}, "unknown action class"),
        ({"control_delta_ms": -1}, "negative control delta"),
    ],
)
def test_manifest_loader_rejects_malformed_rows(
    tmp_path: Path,
    override: dict[str, object],
    message: str,
) -> None:
    path = tmp_path / "manifest.csv"
    write_manifest(path, [sample_row(**override)])

    with pytest.raises(ManifestFormatError, match=message):
        load_manifest(path)


def test_dataset_returns_tensor_label_ride_and_metadata(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, [sample_row()])
    decoder = StubDecoder()
    dataset = FrodoBotsActionDataset(tmp_path, manifest, decoder=decoder)

    item = dataset[0]

    assert item["image"].shape == (3, 224, 224)
    assert item["image"].dtype == torch.float32
    assert item["target"].dtype == torch.long
    assert item["target"].item() == 1
    assert item["action_class"] == "FORWARD"
    assert item["ride_id"] == "1"
    assert item["metadata"]["control_delta_ms"] == pytest.approx(10.0)
    assert item["metadata"]["front_playlist_ref"].endswith("front.m3u8")
    assert decoder.calls[0].front_segment_ref == SEGMENT_REF


def test_repeated_sample_access_and_dataloader_are_deterministic(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, [sample_row(front_frame_id=index) for index in range(4)])
    dataset = FrodoBotsActionDataset(tmp_path, manifest, decoder=StubDecoder())

    first = dataset[0]["image"]
    repeated = dataset[0]["image"]
    batch = next(iter(DataLoader(dataset, batch_size=4, shuffle=False)))

    assert torch.equal(first, repeated)
    assert tuple(batch["image"].shape) == (4, 3, 224, 224)
    assert tuple(batch["target"].shape) == (4,)


def test_visualization_candidates_are_balanced_across_available_actions(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    rows = [sample_row(front_frame_id=index, action_class="FORWARD") for index in range(10)]
    rows += [sample_row(front_frame_id=10, action_class="LEFT")]
    write_manifest(manifest, rows)
    samples = load_manifest(manifest)

    indices = _candidate_indices(samples, 4)

    assert [samples[index].action_class for index in indices[:2]] == ["FORWARD", "LEFT"]


def test_image_to_tensor_rejects_invalid_shape() -> None:
    with pytest.raises(FrameDecodeError, match="HxWx3"):
        image_to_tensor(np.zeros((10, 10), dtype=np.uint8), 224)


def test_decoder_seeks_to_manifest_timestamp(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    segment = tmp_path / SEGMENT_REF
    segment.parent.mkdir(parents=True)
    segment.touch()
    observed: dict[str, object] = {}

    class FakeCapture:
        def __init__(self, path: str) -> None:
            observed["path"] = path

        def isOpened(self) -> bool:
            return True

        def set(self, property_id: int, value: float) -> bool:
            observed["seek"] = (property_id, value)
            return True

        def read(self) -> tuple[bool, np.ndarray]:
            frame = np.zeros((8, 12, 3), dtype=np.uint8)
            frame[:, :] = (10, 20, 30)
            return True, frame

        def release(self) -> None:
            observed["released"] = True

    monkeypatch.setattr("research.training.datasets.frodobots_2k_dataset.cv2.VideoCapture", FakeCapture)
    sample = load_sample(sample_row())

    frame = HlsFrameDecoder(tmp_path).decode(sample)

    assert frame.shape == (8, 12, 3)
    assert observed["path"] == str(segment)
    assert observed["seek"][1] == pytest.approx(50.0)
    assert observed["released"] is True
    assert frame[0, 0].tolist() == [30, 20, 10]


def test_decoder_rejects_missing_and_escaping_segments(tmp_path: Path) -> None:
    decoder = HlsFrameDecoder(tmp_path)

    with pytest.raises(FrameDecodeError, match="does not exist"):
        decoder.decode(load_sample(sample_row()))
    with pytest.raises(FrameDecodeError, match="escapes dataset root"):
        decoder.decode(load_sample(sample_row(front_segment_ref="../20240101000000000.ts")))


def test_decoder_reports_unreadable_segment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    segment = tmp_path / SEGMENT_REF
    segment.parent.mkdir(parents=True)
    segment.touch()

    class ClosedCapture:
        def __init__(self, path: str) -> None:
            pass

        def isOpened(self) -> bool:
            return False

        def release(self) -> None:
            pass

    monkeypatch.setattr("research.training.datasets.frodobots_2k_dataset.cv2.VideoCapture", ClosedCapture)

    with pytest.raises(FrameDecodeError, match="cannot open HLS segment"):
        HlsFrameDecoder(tmp_path).decode(load_sample(sample_row()))


def test_decoder_reports_seek_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    segment = tmp_path / SEGMENT_REF
    segment.parent.mkdir(parents=True)
    segment.touch()

    class UnseekableCapture:
        def __init__(self, path: str) -> None:
            pass

        def isOpened(self) -> bool:
            return True

        def set(self, property_id: int, value: float) -> bool:
            return False

        def release(self) -> None:
            pass

    monkeypatch.setattr("research.training.datasets.frodobots_2k_dataset.cv2.VideoCapture", UnseekableCapture)

    with pytest.raises(FrameDecodeError, match="cannot seek"):
        HlsFrameDecoder(tmp_path).decode(load_sample(sample_row()))


def test_decoder_reports_read_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    segment = tmp_path / SEGMENT_REF
    segment.parent.mkdir(parents=True)
    segment.touch()

    class UnreadableCapture:
        def __init__(self, path: str) -> None:
            pass

        def isOpened(self) -> bool:
            return True

        def set(self, property_id: int, value: float) -> bool:
            return True

        def read(self) -> tuple[bool, None]:
            return False, None

        def release(self) -> None:
            pass

    monkeypatch.setattr("research.training.datasets.frodobots_2k_dataset.cv2.VideoCapture", UnreadableCapture)

    with pytest.raises(FrameDecodeError, match="cannot decode frame"):
        HlsFrameDecoder(tmp_path).decode(load_sample(sample_row()))


def load_sample(row: dict[str, object]) -> ManifestSample:
    return ManifestSample(
        ride_id=str(row["ride_id"]),
        front_playlist_ref=str(row["front_playlist_ref"]),
        front_segment_ref=str(row["front_segment_ref"]),
        front_frame_id=int(row["front_frame_id"]),
        front_timestamp=float(row["front_timestamp"]),
        matched_control_timestamp=float(row["matched_control_timestamp"]),
        control_delta_ms=float(row["control_delta_ms"]),
        linear=float(row["linear"]),
        angular=float(row["angular"]),
        action_class=str(row["action_class"]),
        timeline_section_id=int(row["timeline_section_id"]),
    )
