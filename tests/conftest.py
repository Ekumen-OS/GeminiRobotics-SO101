"""Pytest fixtures, shared mock objects, and environment configuration."""

import os
import sys
import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import robot.calibration
import simulation.sim_calibration
from robot.interface import RobotInterface
from robot.logger import setup_logging
from robot.settings import Settings, set_settings
from simulation.sim_calibration import compute_sim_calibration

setup_logging()

if sys.platform.startswith("linux") and "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"

# Isolate default calibration artifact paths so tests never write into config/
_TEST_CALIB_DIR = tempfile.TemporaryDirectory()
_ISOLATED_SIM_PATH = Path(_TEST_CALIB_DIR.name) / "workspace_homography_sim.npz"
_ISOLATED_REAL_PATH = Path(_TEST_CALIB_DIR.name) / "workspace_homography.npz"

_isolated_settings = Settings.from_env(
    calibration_save_path=_ISOLATED_REAL_PATH,
    calibration_sim_save_path=_ISOLATED_SIM_PATH,
)
set_settings(_isolated_settings)

# Pre-populate isolated calibration matrices so calibrators initialize cleanly
compute_sim_calibration(save_path=_ISOLATED_SIM_PATH)
compute_sim_calibration(save_path=_ISOLATED_REAL_PATH)

simulation.sim_calibration.DEFAULT_SIM_SAVE_PATH = _ISOLATED_SIM_PATH
robot.calibration.DEFAULT_SIM_SAVE_PATH = _ISOLATED_SIM_PATH
robot.calibration.DEFAULT_SAVE_PATH = _ISOLATED_REAL_PATH


@pytest.fixture(autouse=True)
def fast_sim_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bypass real-time sleep in headless simulation during automated testing.

    Ensures MuJoCo physics steps execute with 100% mathematical fidelity without
    stalling automated test runner threads on wall-clock time.sleep.
    """
    monkeypatch.setattr("simulation.sim_robot.time.sleep", lambda _: None)


class MockRobot(RobotInterface):
    """Reusable deterministic mock robot for unit and integration testing."""

    def __init__(self, qpos: np.ndarray | None = None) -> None:
        """Initialize mock robot with optional joint configuration.

        Args:
            qpos: Optional initial 6-DOF joint positions in radians.
        """
        self.qpos = (
            np.asarray(qpos, dtype=np.float32)
            if qpos is not None
            else np.zeros(6, dtype=np.float32)
        )
        self.move_history: list[tuple[np.ndarray, float]] = []
        self.move_to_pose_called = False
        self.set_joint_positions_called = False
        self.last_move_target: np.ndarray | None = None
        self.last_move_duration: float | None = None
        self._stop_requested = False

    def set_joint_positions(self, qpos: np.ndarray) -> None:
        """Set joint positions immediately.

        Args:
            qpos: Target 6-DOF joint angles in radians.
        """
        self.qpos = np.asarray(qpos, dtype=np.float32)
        self.set_joint_positions_called = True

    def get_joint_positions(self) -> np.ndarray:
        """Get current 6-DOF joint positions in radians.

        Returns:
            Copy of current joint positions array.
        """
        return self.qpos.copy()

    def move_to_pose(
        self, target_qpos: np.ndarray, duration_sec: float = 1.0, dt: float = 0.002
    ) -> None:
        """Simulate moving to target pose over duration.

        Args:
            target_qpos: Desired joint angles.
            duration_sec: Simulated move duration in seconds.
            dt: Timestep in seconds.
        """
        self.qpos = np.asarray(target_qpos, dtype=np.float32)
        self.move_to_pose_called = True
        self.last_move_target = self.qpos.copy()
        self.last_move_duration = duration_sec
        self.move_history.append((self.qpos.copy(), duration_sec))

    def get_end_effector_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Return dummy end-effector Cartesian position and orientation.

        Returns:
            Tuple of position [X, Y, Z] in meters and 3x3 rotation matrix.
        """
        pos = np.array([0.18, 0.0, 0.05], dtype=np.float64)
        rot = np.eye(3, dtype=np.float64)
        return pos, rot

    def interrupt(self) -> None:
        """Flag that a stop was requested."""
        self._stop_requested = True

    def close(self) -> None:
        """Clean up mock resources."""
        self.interrupt()


class DummyToolCall:
    """Mock Gemini Live API ToolCall / FunctionCall object."""

    def __init__(
        self,
        name: str,
        args: dict[str, Any] | None = None,
        call_id: str = "test_call_001",
    ) -> None:
        """Initialize mock tool call.

        Args:
            name: Tool / function name.
            args: Arguments dictionary.
            call_id: Unique call identifier string.
        """
        self.name = name
        self.args = args if args is not None else {}
        self.id = call_id


@pytest.fixture
def mock_robot() -> MockRobot:
    """Provide a fresh MockRobot instance for testing.

    Returns:
        MockRobot instance.
    """
    return MockRobot()


@pytest.fixture
def real_homography_path(tmp_path: Path) -> Path:
    """Build synthetic real-mode homography in tmp_path from workspace corner geometry.

    Args:
        tmp_path: Temporary directory fixture provided by pytest.

    Returns:
        Path to generated real-mode homography .npz file.
    """
    save_path = tmp_path / "workspace_homography.npz"
    compute_sim_calibration(save_path=save_path)
    return save_path


@pytest.fixture
def sim_homography_path(tmp_path: Path) -> Path:
    """Build synthetic sim-mode homography in tmp_path from workspace corner geometry.

    Args:
        tmp_path: Temporary directory fixture provided by pytest.

    Returns:
        Path to generated sim-mode homography .npz file.
    """
    save_path = tmp_path / "workspace_homography_sim.npz"
    compute_sim_calibration(save_path=save_path)
    return save_path


@pytest.fixture
def neutral_frame() -> np.ndarray:
    """Provide a neutral 480x640 RGB tabletop frame.

    Returns:
        RGB uint8 numpy array of shape (480, 640, 3).
    """
    return np.full((480, 640, 3), 180, dtype=np.uint8)


@pytest.fixture
def red_cube_frame() -> np.ndarray:
    """Provide a neutral 480x640 RGB tabletop frame with a synthetic red block.

    Returns:
        RGB uint8 numpy array of shape (480, 640, 3) with red cube at [200:240, 300:340].
    """
    frame = np.full((480, 640, 3), 180, dtype=np.uint8)
    frame[200:240, 300:340] = [220, 20, 20]
    return frame


@pytest.fixture
def smooth_gradient_frame() -> np.ndarray:
    """Provide a 480x640 RGB gradient canvas for deterministic image processing.

    Returns:
        Pristine RGB gradient array.
    """
    y, x = np.mgrid[0:480, 0:640]
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    canvas[:, :, 0] = (x / 640.0 * 200).astype(np.uint8)
    canvas[:, :, 1] = (y / 480.0 * 200).astype(np.uint8)
    canvas[:, :, 2] = 100
    return canvas


@pytest.fixture
def clean_stream_state() -> Generator[None, None, None]:
    """Reset stream overlays, model targets, and frame buffers before and after each test."""
    import ui.stream as stream_mod
    from ui.state import get_active_robot, set_active_robot

    orig_robot = get_active_robot()
    stream_mod.set_target_overlay_enabled(True)
    stream_mod.set_calibration_overlay(False)
    stream_mod.set_bounding_boxes_overlay(False)
    stream_mod.set_grid_overlay(False)
    stream_mod.set_crosshair_overlay(False)
    stream_mod.clear_model_targets()
    stream_mod._SIM_RUNTIME.stop()

    yield

    stream_mod.set_target_overlay_enabled(True)
    stream_mod.set_calibration_overlay(False)
    stream_mod.set_bounding_boxes_overlay(False)
    stream_mod.set_grid_overlay(False)
    stream_mod.set_crosshair_overlay(False)
    stream_mod.clear_model_targets()
    set_active_robot(orig_robot)
    stream_mod._SIM_RUNTIME.stop()
