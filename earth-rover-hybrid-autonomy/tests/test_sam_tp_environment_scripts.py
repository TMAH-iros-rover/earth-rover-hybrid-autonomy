import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_setup_falls_back_to_an_ignored_independent_venv() -> None:
    script = (ROOT / "scripts/setup_sam_tp_reproduction.sh").read_text(encoding="utf-8")

    assert 'ENV_BACKEND="${ENV_BACKEND:-auto}"' in script
    assert 'ENV_BACKEND="venv"' in script
    assert 'external/venvs/$ENV_NAME' in script
    assert '"$PYTHON_BOOTSTRAP" -m venv "$VENV_PATH"' in script
    assert "env_python -m pip --version" in script
    assert "env_python -m ensurepip --upgrade" in script
    assert "sudo apt install -y python3.10-venv" in script
    assert 'pip install -r "$UPSTREAM_ROOT/requirements.txt"' in script
    assert 'TRANSFORMERS_VERSION="5.4.0"' in script
    assert 'pip install "transformers==$TRANSFORMERS_VERSION"' in script
    assert "transformers.__version__ == '$TRANSFORMERS_VERSION'" in script
    assert "conda is required" not in script


def test_runner_auto_detects_conda_or_venv_without_system_install() -> None:
    script = (ROOT / "research/scripts/run_sam_tp_reproduction.sh").read_text(encoding="utf-8")

    assert 'ENV_BACKEND="${ENV_BACKEND:-auto}"' in script
    assert 'elif [[ -x "$VENV_PATH/bin/python" ]]' in script
    assert '"$VENV_PATH/bin/python" "$@"' in script
    assert 'export SAM_TP_ENVIRONMENT_BACKEND="$ENV_BACKEND"' in script


def test_trajectory_primitive_runner_is_checkpoint_free_and_read_only() -> None:
    script = (ROOT / "research/scripts/run_sam_tp_trajectory_primitives.sh").read_text(
        encoding="utf-8"
    )

    assert "tests/test_sam_tp_adapter.py" in script
    assert "tests/test_trajectory_sampler.py" in script
    assert "ConstantCurvatureTrajectorySampler" in script
    assert "no CUDA or rover command" in script
    assert "checkpoint_2.pt" not in script
    assert ".send_control(" not in script


def test_phase1_video_runner_is_offline_and_uses_geometry_flag() -> None:
    script = (ROOT / "research/scripts/run_sam_tp_phase1_video_review.sh").read_text(
        encoding="utf-8"
    )

    assert "--phase1-trajectories" in script
    assert "run_sam_tp_video_review.py" in script
    assert "test_sam_tp_sdk_shadow.py" in script
    assert ".send_control(" not in script
    assert ".start_mission(" not in script


def test_shadow_runner_defaults_to_hf_backend_checkpoint_with_rollback_override() -> None:
    script = (ROOT / "scripts/run_sam_tp_sdk_shadow.sh").read_text(encoding="utf-8")
    config = (ROOT / "configs/default.yaml").read_text(encoding="utf-8")

    assert 'CONFIG="${CONFIG:-$PROJECT_ROOT/configs/default.yaml}"' in script
    assert (
        'MISSION_CONFIG="${MISSION_CONFIG-$PROJECT_ROOT/configs/mission1_live.yaml}"'
        in script
    )
    assert 'MODEL_CONFIG="${MODEL_CONFIG:-}"' in script
    assert 'CHECKPOINT="${CHECKPOINT:-}"' in script
    assert 'EXPECTED_CHECKPOINT_SHA256="${EXPECTED_CHECKPOINT_SHA256:-}"' in script
    assert "checkpoints/best_sam_tp.pt\"" in config
    assert (
        "99f0efab7b40d532e31909a77f1e9c4d1a0be3a534b96c7d769850cc046b951d" in config
    )
    assert '# checkpoint: "sam2_logs/' in config
    assert "checkpoints/checkpoint_2.pt\"" in config
    assert '# expected_checkpoint_sha256: "2607fd6049d37f17' in config


def test_live_launchers_default_to_calibrated_profile() -> None:
    controller = (ROOT / "scripts/run_mission1_autonomy.py").read_text(encoding="utf-8")
    shadow = (ROOT / "scripts/run_sam_tp_sdk_shadow.sh").read_text(encoding="utf-8")
    live = (ROOT / "scripts/run_mission1_live.sh").read_text(encoding="utf-8")

    assert 'ROOT / "configs/mission1_live.yaml"' in controller
    assert "$PROJECT_ROOT/configs/mission1_live.yaml" in shadow
    assert 'run_mission1_autonomy.sh" --enable-live-control' in live


def test_video_review_cli_resolves_src_outside_repository(tmp_path: Path) -> None:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "research/training/run_sam_tp_video_review.py"),
            "--help",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--phase1-trajectories" in result.stdout
