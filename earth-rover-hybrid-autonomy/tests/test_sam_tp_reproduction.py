from __future__ import annotations

import json
import importlib
from pathlib import Path

import numpy as np
import pytest
import yaml

import training.sam_tp_reproduction as sam_tp_reproduction
from research.training.inspect_sam_tp_compatibility import compare_architecture_configs
from research.training.run_sam_tp_video_review import (
    compose_sam_tp_panels,
    parse_args as parse_video_review_args,
    process_dataset,
    resolve_dataset_roots,
)
from training.sam_tp_reproduction import (
    OFFICIAL_BRANCH,
    OFFICIAL_CHECKPOINT,
    OFFICIAL_COMMIT,
    OFFICIAL_MODEL_CONFIG,
    OFFICIAL_REPOSITORY,
    OFFICIAL_TRAINING_CONFIG,
    SamTpPrediction,
    SamTpPredictor,
    compare_state_dicts,
    environment_report,
    load_reproduction_config,
    resolve_upstream_paths,
    score_to_heatmap,
    sigmoid_logits,
    validate_rgb_image,
    write_json,
)
from research.training.traversability_video_review_v2 import ReviewFrame, ReviewSegment
from research.training.write_sam_tp_reproduction_report import (
    aggregate_reports,
    markdown_report,
)


def frozen_config() -> dict:
    return {
        "upstream": {
            "repository_url": OFFICIAL_REPOSITORY,
            "branch": OFFICIAL_BRANCH,
            "commit": OFFICIAL_COMMIT,
            "model_config": OFFICIAL_MODEL_CONFIG,
            "training_config": OFFICIAL_TRAINING_CONFIG,
            "checkpoint": OFFICIAL_CHECKPOINT,
        },
        "runtime": {"device": "cuda", "seed": 17, "score_conversion": "sigmoid"},
        "benchmark": {"warmup_iterations": 1, "measured_iterations": 2},
        "video_review": {},
    }


def test_config_and_path_resolution_are_frozen_and_override_checkpoint(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(frozen_config()), encoding="utf-8")
    override = tmp_path / "checkpoint.pt"

    loaded = load_reproduction_config(path)
    resolved = resolve_upstream_paths(tmp_path / "upstream", loaded, override)

    assert resolved["model_config"] == (
        tmp_path / "upstream" / OFFICIAL_MODEL_CONFIG
    ).resolve()
    assert resolved["checkpoint"] == override.resolve()


def test_config_rejects_unreviewed_upstream_commit(tmp_path: Path) -> None:
    config = frozen_config()
    config["upstream"]["commit"] = "unreviewed"
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValueError, match="frozen official provenance"):
        load_reproduction_config(path)


@pytest.mark.parametrize(
    "image,error",
    [
        (np.zeros((10, 10), dtype=np.uint8), "HxWx3"),
        (np.zeros((10, 10, 3), dtype=np.float32), "uint8"),
        (np.zeros((0, 10, 3), dtype=np.uint8), "non-zero"),
    ],
)
def test_rgb_input_validation(image: np.ndarray, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        validate_rgb_image(image)


def test_sigmoid_score_has_expected_range_and_rejects_nonfinite() -> None:
    logits = np.array([[-100.0, 0.0, 100.0]], dtype=np.float32)
    score = sigmoid_logits(logits)

    assert np.all((0.0 <= score) & (score <= 1.0))
    assert score[0, 0] < 0.01
    assert score[0, 1] == pytest.approx(0.5)
    assert score[0, 2] > 0.99
    with pytest.raises(ValueError, match="NaN or Inf"):
        sigmoid_logits(np.array([[np.nan]], dtype=np.float32))


class FakeSamTp:
    def __init__(self) -> None:
        self.calls = 0

    def run_sam2_inference(self, image_rgb: np.ndarray) -> dict[str, np.ndarray]:
        self.calls += 1
        logits = np.zeros(image_rgb.shape[:2], dtype=np.float32)
        heatmap = np.zeros_like(image_rgb)
        return {"logits": logits, "heatmap": heatmap}


def test_predictor_loads_model_once_and_validates_output(tmp_path: Path) -> None:
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    config = tmp_path / "model.yaml"
    checkpoint = tmp_path / "checkpoint.pt"
    config.write_text("model: {}\n", encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint")
    model = FakeSamTp()
    loads = []

    def loader(model_config: str, checkpoint_path: str) -> FakeSamTp:
        loads.append((model_config, checkpoint_path))
        return model

    predictor = SamTpPredictor(
        upstream,
        config,
        checkpoint,
        device="test",
        loader=loader,
    )
    image = np.zeros((12, 20, 3), dtype=np.uint8)

    first = predictor.predict(image)
    second = predictor.predict(image)

    assert len(loads) == 1
    assert predictor.load_count == 1
    assert model.calls == 2
    assert first.output_shape == (12, 20)
    assert np.all(first.traversability_score == 0.5)
    assert second.device == "test"


def test_predictor_rejects_nan_and_wrong_output_shape(tmp_path: Path) -> None:
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    config = tmp_path / "model.yaml"
    checkpoint = tmp_path / "checkpoint.pt"
    config.touch()
    checkpoint.touch()

    class BadModel:
        def run_sam2_inference(self, image_rgb: np.ndarray) -> dict[str, np.ndarray]:
            return {
                "logits": np.full((2, 2), np.nan, dtype=np.float32),
                "heatmap": np.zeros((2, 2, 3), dtype=np.uint8),
            }

    predictor = SamTpPredictor(
        upstream,
        config,
        checkpoint,
        device="test",
        loader=lambda *_: BadModel(),
    )
    with pytest.raises(ValueError, match="differs from input|NaN"):
        predictor.predict(np.zeros((4, 4, 3), dtype=np.uint8))


def test_optional_dependency_error_is_actionable(tmp_path: Path, monkeypatch) -> None:
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    config = tmp_path / "model.yaml"
    checkpoint = tmp_path / "checkpoint.pt"
    config.touch()
    checkpoint.touch()
    predictor = SamTpPredictor(upstream, config, checkpoint)

    def fail_import(name: str):
        raise ImportError(name)

    monkeypatch.setattr(importlib, "import_module", fail_import)
    with pytest.raises(RuntimeError, match="independent SAM-TP environment"):
        predictor.load()


def test_state_dict_comparison_reports_every_mismatch() -> None:
    class TensorLike:
        def __init__(self, shape: tuple[int, ...]) -> None:
            self.shape = shape

    report = compare_state_dicts(
        {"same": TensorLike((2, 3)), "missing": TensorLike((1,))},
        {"same": TensorLike((3, 2)), "unexpected": TensorLike((1,))},
    )

    assert not report["compatible"]
    assert report["missing_keys"] == ["missing"]
    assert report["unexpected_keys"] == ["unexpected"]
    assert report["shape_mismatches"][0]["key"] == "same"


def test_training_and_inference_architecture_comparison() -> None:
    model = {
        "image_encoder": {
            "trunk": {
                "embed_dim": 96,
                "num_heads": 1,
                "stages": [1, 2, 7, 2],
                "global_att_blocks": [5, 7, 9],
            },
            "neck": {"d_model": 256, "backbone_channel_list": [768, 384, 192, 96]},
        },
        "num_maskmem": 0,
        "use_high_res_features_in_sam": True,
        "image_size": 1024,
    }
    training_model = json.loads(json.dumps(model))
    training_model["image_encoder"]["trunk"]["drop_path_rate"] = 0.1
    result = compare_architecture_configs(
        {"model": model},
        {"trainer": {"model": training_model}, "scratch": {"resolution": 1024}},
    )

    assert result["compatible"]
    training_model["image_encoder"]["trunk"]["embed_dim"] = 112
    assert not compare_architecture_configs(
        {"model": model},
        {"trainer": {"model": training_model}, "scratch": {"resolution": 1024}},
    )["compatible"]


def test_video_render_has_three_aspect_preserved_panels() -> None:
    image = np.full((90, 160, 3), 120, dtype=np.uint8)
    score = np.linspace(0, 1, 160, dtype=np.float32)[None, :].repeat(90, axis=0)

    rendered = compose_sam_tp_panels(
        image,
        score,
        {
            "dataset": "output_rides_0",
            "ride_id": "7",
            "frame_id": 10,
            "timestamp": 1000.5,
            "checkpoint_version": "abc",
            "latency_ms": 12.0,
            "measured_fps": 8.0,
        },
        320,
    )

    assert rendered.shape == (262, 960, 3)
    heatmap = score_to_heatmap(score)
    assert heatmap.shape == image.shape
    assert not np.array_equal(heatmap[:, 0], heatmap[:, -1])


def test_sam_tp_video_review_accepts_additional_numeric_datasets(
    tmp_path: Path,
) -> None:
    args = parse_video_review_args(
        [
            "--reproduction-config",
            "config.yaml",
            "--upstream-root",
            "upstream",
            "--compatibility-report",
            "compatibility.json",
            "--output-dir",
            "output",
            "--datasets",
            "9",
            "20",
            "--dataset-root",
            f"9={tmp_path / 'rides_9'}",
            "--dataset-root",
            f"20={tmp_path / 'rides_20'}",
        ]
    )
    roots = resolve_dataset_roots(
        tuple(args.datasets),
        {
            "0": args.dataset_root_0,
            "1": args.dataset_root_1,
            "2": args.dataset_root_2,
        },
        args.dataset_root,
    )

    assert args.datasets == ["9", "20"]
    assert roots["9"] == (tmp_path / "rides_9").resolve()
    assert roots["20"] == (tmp_path / "rides_20").resolve()


@pytest.mark.parametrize(
    "override,error",
    [
        ("9", "INDEX=PATH"),
        ("name=/tmp/data", "numeric"),
        ("9=/tmp/first", "duplicate"),
    ],
)
def test_sam_tp_dataset_root_override_rejects_invalid_values(
    override: str,
    error: str,
) -> None:
    overrides = [override]
    if error == "duplicate":
        overrides.append("9=/tmp/second")

    with pytest.raises(ValueError, match=error):
        resolve_dataset_roots(("9",), {}, overrides)


def test_mock_video_processing_records_frames_latency_and_no_control(
    tmp_path: Path,
    capsys,
) -> None:
    frame = ReviewFrame(
        dataset="output_rides_0",
        ride_id="7",
        frame_id=10,
        timestamp=1000.5,
        playlist_reference="ride/recordings/front.m3u8",
        segment_reference="ride/recordings/front.ts",
        timeline_section_id=0,
    )
    segment = ReviewSegment(
        dataset=frame.dataset,
        ride_id=frame.ride_id,
        start_timestamp=frame.timestamp,
        end_timestamp=frame.timestamp + 0.1,
        requested_duration_seconds=0.1,
        frames=(frame,),
    )

    class Decoder:
        def decode(self, _: ReviewFrame) -> np.ndarray:
            return np.full((36, 64, 3), 100, dtype=np.uint8)

    class Predictor:
        def predict(self, image: np.ndarray) -> SamTpPrediction:
            logits = np.zeros(image.shape[:2], dtype=np.float32)
            return SamTpPrediction(
                raw_logits=logits,
                traversability_score=np.full(image.shape[:2], 0.5, dtype=np.float32),
                heatmap=np.zeros_like(image),
                input_shape=image.shape,
                output_shape=image.shape[:2],
                inference_time_ms=3.0,
                device="test",
            )

    class Writer:
        def __init__(self, path: Path, fps: float, frame_size: tuple[int, int]) -> None:
            self.path = path
            self.frame_size = frame_size
            self.frames = 0

        def write(self, rendered: np.ndarray) -> None:
            assert rendered.shape[1::-1] == self.frame_size
            self.frames += 1

        def close(self) -> dict[str, object]:
            return {"codec": "mock", "frame_count": self.frames}

    report = process_dataset(
        tmp_path / "dataset",
        [segment],
        [],
        Decoder(),
        Predictor(),
        tmp_path / "temporary",
        10.0,
        160,
        "abc",
        writer_factory=Writer,
        reported_output_dir=tmp_path / "final",
    )

    rows = (tmp_path / "temporary/frame_statistics.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    assert report["success"]
    assert report["processed_frame_count"] == 1
    assert report["model_only_latency_ms"]["p50"] == 3.0
    assert report["sdk_or_live_rover_commands_sent"] is False
    assert report["output_file_path"] == str(tmp_path / "final/sam_tp_review.mp4")
    assert len(rows) == 1
    output = capsys.readouterr().out
    assert "dataset=dataset requested_frames=1 segments=1" in output
    assert "progress=1/1 processed=1 failed=0" in output


def test_phase1_video_processing_adds_geometry_without_camera_projection(
    tmp_path: Path,
) -> None:
    frame = ReviewFrame(
        dataset="output_rides_0",
        ride_id="7",
        frame_id=10,
        timestamp=1000.5,
        playlist_reference="ride/recordings/front.m3u8",
        segment_reference="ride/recordings/front.ts",
        timeline_section_id=0,
    )
    segment = ReviewSegment(
        dataset=frame.dataset,
        ride_id=frame.ride_id,
        start_timestamp=frame.timestamp,
        end_timestamp=frame.timestamp + 0.1,
        requested_duration_seconds=0.1,
        frames=(frame,),
    )

    class Decoder:
        def decode(self, _: ReviewFrame) -> np.ndarray:
            return np.full((36, 64, 3), 100, dtype=np.uint8)

    class Predictor:
        def predict(self, image: np.ndarray) -> SamTpPrediction:
            score = np.full(image.shape[:2], 0.5, dtype=np.float32)
            return SamTpPrediction(
                raw_logits=np.zeros_like(score),
                traversability_score=score,
                heatmap=np.zeros_like(image),
                input_shape=image.shape,
                output_shape=score.shape,
                inference_time_ms=3.0,
                device="test",
            )

    class Writer:
        def __init__(self, path: Path, fps: float, frame_size: tuple[int, int]) -> None:
            assert frame_size[0] == 640
            self.frames = 0

        def write(self, _: np.ndarray) -> None:
            self.frames += 1

        def close(self) -> dict[str, object]:
            return {"codec": "mock", "frame_count": self.frames}

    from earth_rover.planning.trajectory_sampler import (
        DEFAULT_CURVATURES,
        ConstantCurvatureTrajectorySampler,
    )
    from training.sam_tp_phase1_review import SamTpPhase1FrameProcessor

    predictor = Predictor()
    trajectories = ConstantCurvatureTrajectorySampler(
        DEFAULT_CURVATURES,
        2.0,
        0.1,
        0.4,
        0.1,
    ).sample()
    report = process_dataset(
        tmp_path / "dataset",
        [segment],
        [],
        Decoder(),
        predictor,
        tmp_path / "output",
        10.0,
        160,
        "abc",
        phase1_processor=SamTpPhase1FrameProcessor(
            predictor,
            trajectories,
            "abc",
        ),
        writer_factory=Writer,
    )

    assert report["success"]
    assert report["phase1_trajectory_geometry"] is True
    assert report["camera_projection_applied"] is False


def test_manifest_serialization_is_stable(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    write_json(path, {"z": 1, "a": [2, 3]})

    assert path.read_text(encoding="utf-8").startswith('{\n  "a"')
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": [2, 3], "z": 1}


def test_environment_report_converts_version_subclasses_to_plain_strings(
    monkeypatch,
) -> None:
    class VersionString(str):
        pass

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def get_device_name(_: int) -> VersionString:
            return VersionString("GPU")

    class FakeTorch:
        __version__ = VersionString("2.7.1+cu118")
        version = type("Version", (), {"cuda": VersionString("11.8")})()
        cuda = FakeCuda()

    class FakeTorchvision:
        __version__ = VersionString("0.22.1+cu118")

    modules = {"torch": FakeTorch(), "torchvision": FakeTorchvision()}
    monkeypatch.setattr(
        sam_tp_reproduction.importlib,
        "import_module",
        lambda name: modules[name],
    )
    monkeypatch.setattr(sam_tp_reproduction, "_command_output", lambda _: None)
    monkeypatch.setattr(sam_tp_reproduction, "_nvidia_cuda_version", lambda: None)

    report = environment_report()

    assert type(report["torch"]) is str
    assert type(report["torchvision"]) is str
    assert type(report["torch_cuda_runtime"]) is str
    assert type(report["gpu_name"]) is str
    yaml.safe_dump(report)


def test_aggregate_report_requires_every_reproduction_gate() -> None:
    compatibility = {
        "success": True,
        "checkpoint": {"filename": "checkpoint_2.pt", "sha256": "abc", "size_bytes": 3},
    }
    smoke = {
        "success": True,
        "upstream": {"commit": OFFICIAL_COMMIT},
        "source_image": "rgb.png",
        "input_resolution": [10, 20],
        "output_resolution": [10, 20],
        "raw_logits": {},
        "traversability_score": {},
        "model_load_time_ms": 10.0,
        "first_inference_latency_ms": 4.0,
        "warm_inference_latency_ms": {"p50": 3.0, "p95": 3.5},
        "throughput_fps": 300.0,
        "peak_vram_bytes": {"allocated": 1, "reserved": 2},
        "environment": {},
        "dependency_setup_method": "conda",
    }
    video = {
        "success": True,
        "settings": {},
        "datasets": {
            "0": {
                "processed_frame_count": 10,
                "skipped_frame_count": 0,
                "effective_pipeline_fps": 5.0,
            }
        },
        "peak_vram_bytes": {"allocated": 2, "reserved": 3},
    }

    report = aggregate_reports(compatibility, smoke, video)

    assert report["status"] == "PASS"
    assert report["random_initialization_used"] is False
    assert "Status: **PASS**" in markdown_report(report)
    video["success"] = False
    assert aggregate_reports(compatibility, smoke, video)["status"] == "FAIL"
