"""Pytest fixtures, shared mock objects, and environment configuration."""

import os
import sys
import tempfile
import time
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from robot.contracts import GripperReading
from robot.interface import RobotInterface
from robot.logger import setup_logging
from robot.settings import Settings, get_settings, set_settings
from simulation.sim_calibration import compute_sim_calibration

setup_logging()

_real_sleep = time.sleep

GRIPPER_OPEN_RAD = float(np.radians(75.0))

if sys.platform.startswith("linux") and "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"

# Isolate default calibration artifact paths so tests never write into config/
_TEST_CALIB_DIR = tempfile.TemporaryDirectory()
_ISOLATED_SIM_PATH = Path(_TEST_CALIB_DIR.name) / "workspace_homography_sim.npz"
_ISOLATED_REAL_PATH = Path(_TEST_CALIB_DIR.name) / "workspace_homography.npz"

_isolated_settings = Settings.from_env(
    calibration_save_path=_ISOLATED_REAL_PATH,
    calibration_sim_save_path=_ISOLATED_SIM_PATH,
    gripper_settle_s=0.0,
    vision_settle_s=0.0,
)
set_settings(_isolated_settings)

# Pre-populate isolated calibration matrices so calibrators initialize cleanly.
# Note: the "real" homography written here is a sim matrix, used only so that
# real-mode code paths find a calibrated file.
compute_sim_calibration(save_path=_ISOLATED_SIM_PATH)
compute_sim_calibration(save_path=_ISOLATED_REAL_PATH)


@pytest.fixture
def restore_settings() -> Any:
    """Preserve global settings instance across tests."""
    original = get_settings()
    try:
        yield
    finally:
        set_settings(original)


class _FastSimTime:
    """Wrapper that bypasses simulation time.sleep without altering global time module."""

    def __getattr__(self, name: str) -> Any:
        return getattr(time, name)

    @staticmethod
    def sleep(_: float) -> None:
        pass


_fast_sim_time = _FastSimTime()


@pytest.fixture(autouse=True)
def no_wall_clock_sleep(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Bypass real-time sleep during automated testing unless real_sleep is requested.

    Ensures tests do not stall automated runner threads on wall-clock time.sleep
    during simulation steps, real-robot interpolation/convergence, or tool settling.
    """
    monkeypatch.setattr("simulation.sim_robot.time", _fast_sim_time)
    if "real_sleep" in request.fixturenames:
        return
    monkeypatch.setattr("robot.real_robot.time.sleep", lambda _: None)
    monkeypatch.setattr("agent.tools.time.sleep", lambda _: None)


@pytest.fixture
def real_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Opt-out fixture that preserves real time.sleep for timing tests."""
    monkeypatch.setattr("robot.real_robot.time.sleep", _real_sleep)
    monkeypatch.setattr("agent.tools.time.sleep", _real_sleep)


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
        self.gripper_reading: GripperReading | None = None
        self.step_hook: Callable[[int], None] | None = None
        self.sleep_during_move: bool = False

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
        self,
        target_qpos: np.ndarray,
        duration_sec: float = 1.0,
        dt: float = 0.002,
        cancel_epoch: int | None = None,
    ) -> None:
        """Simulate moving to target pose over duration.

        Args:
            target_qpos: Desired joint angles.
            duration_sec: Simulated move duration in seconds.
            dt: Timestep in seconds.
            cancel_epoch: Interrupt epoch captured before a multi-segment motion.

        Raises:
            HardwareError: If motion is interrupted.
        """
        from robot.errors import HardwareError

        if cancel_epoch is not None and cancel_epoch != self.interrupt_epoch:
            raise HardwareError("Motion interrupted", robot_moved=False)
        self._stop_requested = False
        target_qpos = np.asarray(target_qpos, dtype=np.float32)
        start_qpos = self.qpos.copy()
        num_steps = max(1, int(duration_sec / dt))
        steps_written = 0
        try:
            for step_idx in range(1, num_steps + 1):
                if self._stop_requested:
                    raise HardwareError("Motion interrupted", robot_moved=(steps_written > 0))
                t = step_idx / num_steps
                alpha = 0.5 * (1.0 - np.cos(np.pi * t))
                self.qpos = (1.0 - alpha) * start_qpos + alpha * target_qpos
                steps_written += 1
                if getattr(self, "step_hook", None) is not None:
                    self.step_hook(step_idx)
                if getattr(self, "sleep_during_move", False):
                    time.sleep(dt)
                if self._stop_requested:
                    raise HardwareError("Motion interrupted", robot_moved=(steps_written > 0))

            self.move_to_pose_called = True
            self.last_move_target = self.qpos.copy()
            self.last_move_duration = duration_sec
            self.move_history.append((self.qpos.copy(), duration_sec))
        finally:
            self._stop_requested = False

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
        self.interrupt_epoch += 1
        self._stop_requested = True

    def close(self) -> None:
        """Clean up mock resources."""
        self.interrupt()

    def read_gripper_state(self) -> GripperReading:
        """Return scripted gripper reading if set, else delegate to base."""
        if self.gripper_reading is not None:
            return self.gripper_reading
        return super().read_gripper_state()


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
def homography_path(tmp_path: Path) -> Path:
    """Build synthetic homography in tmp_path from workspace corner geometry.

    Args:
        tmp_path: Temporary directory fixture provided by pytest.

    Returns:
        Path to generated homography .npz file.
    """
    save_path = tmp_path / "workspace_homography.npz"
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
def clean_stream_state() -> Generator[None, None, None]:
    """Reset stream overlays, model targets, and frame buffers before and after each test."""
    from robot.registry import get_active_robot, set_active_robot
    from vision.overlays import set_calibration_overlay, set_target_overlay_enabled
    from vision.sim_runtime import stop_sim_runtime
    from vision.targets import clear_model_targets

    orig_robot = get_active_robot()
    set_target_overlay_enabled(True)
    set_calibration_overlay(False)
    clear_model_targets()
    stop_sim_runtime()

    yield

    set_target_overlay_enabled(True)
    set_calibration_overlay(False)
    clear_model_targets()
    set_active_robot(orig_robot)
    stop_sim_runtime()
