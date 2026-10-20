"""Unit tests for tools/check_env.py preflight environment diagnostics."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from tools import check_env
from tools.check_env import (
    check_calibration_config,
    check_environment,
    check_gemini_credentials,
    check_python_and_imports,
    check_real_camera,
    check_real_homography,
    check_real_lerobot,
    check_real_motor_calibration,
    check_real_serial_port,
    check_server_port,
    check_sim_ik,
    check_sim_render,
    check_sim_scene,
)


def test_check_python_and_imports_ok() -> None:
    """Core package import check should pass in active virtual environment."""
    status, msg = check_python_and_imports()
    assert status == "OK"
    assert "imported" in msg


def test_check_gemini_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Gemini credentials should report OK when set and FAIL when none."""
    mock_settings = MagicMock()
    mock_settings.credentials_mode = "api_key"
    monkeypatch.setattr("robot.settings.get_settings", lambda: mock_settings)

    status, msg = check_gemini_credentials()
    assert status == "OK"
    assert "api_key" in msg

    mock_settings.credentials_mode = "none"
    status, msg = check_gemini_credentials()
    assert status == "FAIL"
    assert "not configured" in msg


def test_check_calibration_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Calibration config check should report OK for valid config and FAIL for missing/corrupt."""
    mock_settings = MagicMock()
    missing_file = tmp_path / "nonexistent.json"
    mock_settings.calibration_config_path = missing_file
    monkeypatch.setattr("robot.settings.get_settings", lambda: mock_settings)

    # Missing file
    status, msg = check_calibration_config()
    assert status == "FAIL"
    assert "not found" in msg

    # Invalid JSON
    corrupt_file = tmp_path / "corrupt.json"
    corrupt_file.write_text("{broken", encoding="utf-8")
    mock_settings.calibration_config_path = corrupt_file
    status, msg = check_calibration_config()
    assert status == "FAIL"
    assert "not valid JSON" in msg

    # Missing keys
    incomplete_file = tmp_path / "incomplete.json"
    incomplete_file.write_text(json.dumps({"marker_ids": [0, 1, 2, 3]}), encoding="utf-8")
    mock_settings.calibration_config_path = incomplete_file
    status, msg = check_calibration_config()
    assert status == "FAIL"
    assert "missing 'marker_ids' or 'workspace_corners_mm'" in msg

    # Valid
    valid_file = tmp_path / "valid.json"
    valid_file.write_text(
        json.dumps({"marker_ids": [0, 1, 2, 3], "workspace_corners_mm": {}}),
        encoding="utf-8",
    )
    mock_settings.calibration_config_path = valid_file
    status, msg = check_calibration_config()
    assert status == "OK"
    assert "valid" in msg


def test_check_server_port(monkeypatch: pytest.MonkeyPatch) -> None:
    """Port check should report OK when port is free and FAIL when occupied."""
    mock_settings = MagicMock()
    mock_settings.server_port = 7860
    monkeypatch.setattr("robot.settings.get_settings", lambda: mock_settings)

    # Free port
    status, msg = check_server_port()
    assert status in ("OK", "FAIL")

    # Mock occupied port
    def _mock_bind_fail(*args, **kwargs):
        raise OSError("Address already in use")

    monkeypatch.setattr("socket.socket.bind", _mock_bind_fail)
    status, msg = check_server_port()
    assert status == "FAIL"
    assert "already in use" in msg


def test_check_sim_scene(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sim scene check should report OK when scene loads and FAIL on exception."""
    mock_robot = MagicMock()
    mock_robot.model.nq = 6
    mock_robot.actuator_ids = [0, 1, 2, 3, 4, 5]
    monkeypatch.setattr("simulation.sim_robot.SO101Robot", lambda: mock_robot)

    status, msg = check_sim_scene()
    assert status == "OK"
    assert "6 DOFs" in msg
    mock_robot.close.assert_called_once()


def test_check_sim_ik(monkeypatch: pytest.MonkeyPatch) -> None:
    """IK check should report OK on solution and FAIL on divergence."""
    mock_ik = MagicMock()
    mock_ik.solve_ik.return_value = np.zeros(6)
    monkeypatch.setattr("robot.cartesian_ik.CartesianIK", lambda: mock_ik)

    status, msg = check_sim_ik()
    assert status == "OK"
    assert "IK solved" in msg

    mock_ik.solve_ik.return_value = None
    status, msg = check_sim_ik()
    assert status == "FAIL"
    assert "failed to resolve" in msg


def test_check_sim_render(monkeypatch: pytest.MonkeyPatch) -> None:
    """Offscreen render check should report OK when frame returned and FAIL on bad frame."""
    mock_robot = MagicMock()
    mock_robot.render_rgb_frame.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
    monkeypatch.setattr("simulation.sim_robot.SO101Robot", lambda: mock_robot)

    status, msg = check_sim_render()
    assert status == "OK"
    assert "640x480" in msg
    mock_robot.close.assert_called_once()

    mock_robot.render_rgb_frame.return_value = None
    status, msg = check_sim_render()
    assert status == "FAIL"


def test_check_real_lerobot(monkeypatch: pytest.MonkeyPatch) -> None:
    """LeRobot check should report OK when available and FAIL when unavailable."""
    monkeypatch.setattr("robot.real_robot.LEROBOT_AVAILABLE", True)
    status, msg = check_real_lerobot()
    assert status == "OK"

    monkeypatch.setattr("robot.real_robot.LEROBOT_AVAILABLE", False)
    status, msg = check_real_lerobot()
    assert status == "FAIL"
    assert "uv sync --extra hw" in msg


def test_check_real_serial_port(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Serial port check should report OK when port rw, and FAIL when missing or not rw."""
    mock_settings = MagicMock()
    missing_port = tmp_path / "tty_missing"
    mock_settings.serial_port = str(missing_port)
    monkeypatch.setattr("robot.settings.get_settings", lambda: mock_settings)

    status, msg = check_real_serial_port()
    assert status == "FAIL"
    assert "not found" in msg

    existing_port = tmp_path / "tty_existing"
    existing_port.write_text("port")
    mock_settings.serial_port = str(existing_port)

    # RW access
    monkeypatch.setattr("os.access", lambda path, mode: True)
    status, msg = check_real_serial_port()
    assert status == "OK"

    # Lacks RW access
    monkeypatch.setattr("os.access", lambda path, mode: False)
    status, msg = check_real_serial_port()
    assert status == "FAIL"
    assert "lacks read/write permissions" in msg
    assert "chmod 666" in msg


def test_check_real_motor_calibration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Motor calibration should report OK if any candidate exists, FAIL otherwise."""
    mock_settings = MagicMock()
    c1 = tmp_path / "candidate1.json"
    c2 = tmp_path / "candidate2.json"
    mock_settings.motor_calibration_candidates = (c1, c2)
    mock_settings.follower_robot_id = "test_robot"
    monkeypatch.setattr("robot.settings.get_settings", lambda: mock_settings)

    status, msg = check_real_motor_calibration()
    assert status == "FAIL"
    assert "No motor calibration file found" in msg
    assert "lerobot-calibrate" in msg

    c2.write_text("{}", encoding="utf-8")
    status, msg = check_real_motor_calibration()
    assert status == "OK"
    assert str(c2) in msg


def test_check_real_camera(monkeypatch: pytest.MonkeyPatch) -> None:
    """Camera check should report OK when capture succeeds, FAIL on open or read failure."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.return_value = 0.0
    mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
    monkeypatch.setattr("vision.camera._open_capture", lambda idx, attempts=1: mock_cap)
    monkeypatch.setattr("vision.camera.get_default_camera_index", lambda: 0)

    status, msg = check_real_camera()
    assert status == "OK"
    assert "640x480" in msg
    assert "autofocus=" in msg
    mock_cap.release.assert_called_once()

    # Open failure
    mock_cap.isOpened.return_value = False
    mock_cap.release.reset_mock()
    status, msg = check_real_camera()
    assert status == "FAIL"
    assert "Could not open camera" in msg
    mock_cap.release.assert_called_once()


def test_check_real_homography_warns_when_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """In real mode, a missing homography file gives WARN, not FAIL."""
    mock_settings = MagicMock()
    missing_file = tmp_path / "missing_homography.npz"
    mock_settings.calibration_save_path = missing_file
    monkeypatch.setattr("robot.settings.get_settings", lambda: mock_settings)

    status, msg = check_real_homography()
    assert status == "WARN"
    assert "not found" in msg
    assert "Calibrate from the UI panel" in msg

    existing_file = tmp_path / "existing_homography.npz"
    existing_file.write_text("data")
    mock_settings.calibration_save_path = existing_file

    status, msg = check_real_homography()
    assert status == "OK"


def test_check_environment_exit_code_zero_on_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """When all checks pass (or warn), check_environment returns exit code 0."""
    monkeypatch.setattr(check_env, "check_python_and_imports", lambda: ("OK", "ok"))
    monkeypatch.setattr(check_env, "check_gemini_credentials", lambda: ("OK", "ok"))
    monkeypatch.setattr(check_env, "check_calibration_config", lambda: ("OK", "ok"))
    monkeypatch.setattr(check_env, "check_server_port", lambda: ("OK", "ok"))
    monkeypatch.setattr(check_env, "check_sim_scene", lambda: ("OK", "ok"))
    monkeypatch.setattr(check_env, "check_sim_ik", lambda: ("OK", "ok"))
    monkeypatch.setattr(check_env, "check_sim_render", lambda: ("OK", "ok"))

    exit_code = check_environment(mode="sim")
    assert exit_code == 0


def test_check_environment_exit_code_one_on_any_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """When any check fails, check_environment returns exit code 1."""
    monkeypatch.setattr(check_env, "check_python_and_imports", lambda: ("OK", "ok"))
    monkeypatch.setattr(check_env, "check_gemini_credentials", lambda: ("FAIL", "fail"))
    monkeypatch.setattr(check_env, "check_calibration_config", lambda: ("OK", "ok"))
    monkeypatch.setattr(check_env, "check_server_port", lambda: ("OK", "ok"))
    monkeypatch.setattr(check_env, "check_sim_scene", lambda: ("OK", "ok"))
    monkeypatch.setattr(check_env, "check_sim_ik", lambda: ("OK", "ok"))
    monkeypatch.setattr(check_env, "check_sim_render", lambda: ("OK", "ok"))

    exit_code = check_environment(mode="sim")
    assert exit_code == 1


def test_check_environment_real_missing_homography_is_warn_not_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """In real mode, missing homography (WARN) does not cause exit code 1 if other checks pass."""
    monkeypatch.setattr(check_env, "check_python_and_imports", lambda: ("OK", "ok"))
    monkeypatch.setattr(check_env, "check_gemini_credentials", lambda: ("OK", "ok"))
    monkeypatch.setattr(check_env, "check_calibration_config", lambda: ("OK", "ok"))
    monkeypatch.setattr(check_env, "check_server_port", lambda: ("OK", "ok"))
    monkeypatch.setattr(check_env, "check_real_lerobot", lambda: ("OK", "ok"))
    monkeypatch.setattr(check_env, "check_real_serial_port", lambda: ("OK", "ok"))
    monkeypatch.setattr(check_env, "check_real_motor_calibration", lambda: ("OK", "ok"))
    monkeypatch.setattr(check_env, "check_real_camera", lambda: ("OK", "ok"))
    monkeypatch.setattr(
        check_env, "check_real_homography", lambda: ("WARN", "homography not found")
    )

    exit_code = check_environment(mode="real")
    assert exit_code == 0


def test_check_environment_never_commands_arm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Preflight check must never command physical or simulated robot motion."""

    class MotionGuardRobot:
        def __init__(self, *args, **kwargs) -> None:
            self.model = MagicMock(nq=6)
            self.actuator_ids = [0, 1, 2, 3, 4, 5]

        def move_to_pose(self, *args, **kwargs) -> None:
            raise AssertionError("Forbidden: move_to_pose called during preflight!")

        def step(self, *args, **kwargs) -> None:
            raise AssertionError("Forbidden: step called during preflight!")

        def set_joint_positions(self, *args, **kwargs) -> None:
            raise AssertionError("Forbidden: set_joint_positions called during preflight!")

        def render_rgb_frame(self, width: int = 640, height: int = 480) -> np.ndarray:
            return np.zeros((height, width, 3), dtype=np.uint8)

        def close(self) -> None:
            pass

    monkeypatch.setattr("simulation.sim_robot.SO101Robot", MotionGuardRobot)
    monkeypatch.setattr(check_env, "check_python_and_imports", lambda: ("OK", "ok"))
    monkeypatch.setattr(check_env, "check_gemini_credentials", lambda: ("OK", "ok"))
    monkeypatch.setattr(check_env, "check_calibration_config", lambda: ("OK", "ok"))
    monkeypatch.setattr(check_env, "check_server_port", lambda: ("OK", "ok"))
    monkeypatch.setattr(check_env, "check_sim_ik", lambda: ("OK", "ok"))

    # Sim mode run
    exit_code_sim = check_environment(mode="sim")
    assert exit_code_sim == 0

    # Real mode run - RealSO101Robot should never even be called
    def _forbidden_real_robot(*args, **kwargs):
        raise AssertionError("Forbidden: RealSO101Robot instantiated during preflight!")

    monkeypatch.setattr("robot.real_robot.RealSO101Robot", _forbidden_real_robot)
    monkeypatch.setattr(check_env, "check_real_lerobot", lambda: ("OK", "ok"))
    monkeypatch.setattr(check_env, "check_real_serial_port", lambda: ("OK", "ok"))
    monkeypatch.setattr(check_env, "check_real_motor_calibration", lambda: ("OK", "ok"))
    monkeypatch.setattr(check_env, "check_real_camera", lambda: ("OK", "ok"))
    monkeypatch.setattr(check_env, "check_real_homography", lambda: ("OK", "ok"))

    exit_code_real = check_environment(mode="real")
    assert exit_code_real == 0
