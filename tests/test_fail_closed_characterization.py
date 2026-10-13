"""Characterization tests pinning TODAY's fail-open behaviour before flipping to fail-closed.

These tests document the un-hardened failure paths that Phase 3c will close:
1. Hardware driver write/read failures are swallowed and report success / commanded pose.
2. Calibration loading partial commit leaves is_calibrated True on corrupt/singular file.
3. Perception visual alignment refinement failure is silently swallowed.
4. Real-mode get_latest_raw_jpeg serves a synthetic placeholder pattern when no frame exists.
"""

import threading
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

import vision.frames as frames_mod
from robot.calibration import ArucoWorkspaceCalibrator
from robot.real_robot import RealSO101Robot
from robot.registry import set_active_robot
from tests.conftest import MockRobot
from vision.frames import _global_frame_lock, get_latest_raw_jpeg


def _make_mock_real_robot(fake_bus: MagicMock | None = None) -> RealSO101Robot:
    """Create a mock RealSO101Robot with initialized bus and calibration."""
    robot = RealSO101Robot.__new__(RealSO101Robot)
    robot.port = "/dev/mock_tty"
    robot.connected = True
    robot._bus_lock = threading.Lock()
    robot.bus = fake_bus if fake_bus is not None else MagicMock()
    robot.calibration = {"joint": MagicMock()}
    robot._last_qpos = np.zeros(6, dtype=np.float32)
    robot._stop_requested = False
    robot.convergence_timeout_s = 0.0  # fake buses do not move; no wait for convergence
    return robot


def test_driver_write_failure_raises_and_stops_flipped() -> None:
    """FLIPPED: bus sync_write failure raises HardwareError, stops sweep, preserves _last_qpos."""
    from robot.constants import JOINT_NAMES
    from robot.errors import HardwareError

    fake_bus = MagicMock()
    fake_bus.sync_read.return_value = {name: 0.0 for name in JOINT_NAMES}
    fake_bus.sync_write.side_effect = ConnectionError("Bus connection dropped")
    robot = _make_mock_real_robot(fake_bus)
    initial_qpos = robot._last_qpos.copy()

    target_qpos = np.ones(6, dtype=np.float32) * 0.1
    # Flipped: move_to_pose raises HardwareError and stops interpolation on first failure
    with pytest.raises(HardwareError, match="Failed to write joint positions"):
        robot.move_to_pose(target_qpos, duration_sec=0.04, dt=0.02)

    # Sweep stopped on first failure: exactly 1 sync_write call
    assert fake_bus.sync_write.call_count == 1
    # _last_qpos was NOT updated because the write failed
    assert np.allclose(robot._last_qpos, initial_qpos)


def test_driver_read_failure_raises_hardware_error_flipped() -> None:
    """FLIPPED: bus sync_read failure raises HardwareError instead of returning commanded pose."""
    from robot.errors import HardwareError

    fake_bus = MagicMock()
    fake_bus.sync_read.side_effect = RuntimeError("Bus read I/O error")
    robot = _make_mock_real_robot(fake_bus)
    robot._last_qpos = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32)

    # Flipped: get_joint_positions raises HardwareError and never returns cached commanded pose
    with pytest.raises(HardwareError, match="Failed to read present positions"):
        robot.get_joint_positions()


def test_driver_disconnected_operations_raise_hardware_error() -> None:
    """Verify disconnected RealSO101Robot raises HardwareError on both read and write."""
    from robot.errors import HardwareError

    robot = _make_mock_real_robot(MagicMock())
    robot.connected = False

    with pytest.raises(HardwareError, match="robot is not connected"):
        robot.get_joint_positions()

    with pytest.raises(HardwareError, match="robot is not connected"):
        robot.set_joint_positions(np.zeros(6, dtype=np.float32))


def test_driver_disconnect_sets_connected_false_in_finally() -> None:
    """Verify disconnect guarantees connected=False even when bus.disconnect raises."""
    fake_bus = MagicMock()
    fake_bus.disconnect.side_effect = RuntimeError("Teardown bus error")
    robot = _make_mock_real_robot(fake_bus)
    assert robot.connected is True

    robot.disconnect()
    assert robot.connected is False


def test_calibration_load_atomic_commit_fails_closed_flipped(tmp_path) -> None:
    """FLIPPED: singular homography leaves is_calibrated=False and state untouched."""
    from robot.errors import CalibrationUnavailable

    calib_file = tmp_path / "singular_calib.npz"
    # Singular matrix (all zeros) has no inverse
    np.savez(calib_file, homography=np.zeros((3, 3), dtype=np.float32))

    calib = ArucoWorkspaceCalibrator(
        config_path="/tmp/non_existent.json", save_path=tmp_path / "dummy.npz"
    )
    assert not calib.is_calibrated

    # Flipped: load_calibration returns False and leaves is_calibrated False
    success = calib.load_calibration(calib_file)
    assert not success
    assert calib.is_calibrated is False
    assert calib.homography_matrix is None

    with pytest.raises(CalibrationUnavailable, match="Workspace is not calibrated"):
        calib.require_calibrated()


def test_calibration_load_corrupt_npz_leaves_uncalibrated(tmp_path) -> None:
    """Verify loading truncated/garbage .npz leaves calibrator uncalibrated."""
    corrupt_file = tmp_path / "corrupt.npz"
    corrupt_file.write_bytes(b"NOT_A_VALID_NUMPY_FILE_GARBAGE_HEADER")

    calib = ArucoWorkspaceCalibrator(
        config_path="/tmp/non_existent.json", save_path=tmp_path / "dummy.npz"
    )
    assert not calib.is_calibrated
    success = calib.load_calibration(corrupt_file)
    assert not success
    assert calib.is_calibrated is False


def test_calibration_update_homography_atomic_leaves_previous_matrix() -> None:
    """Verify failed update_homography leaves existing valid matrix intact."""
    calib = ArucoWorkspaceCalibrator(config_path="/tmp/non_existent.json")
    # Manually configure a valid calibration state
    valid_h = np.eye(3, dtype=np.float32)
    calib.homography_matrix = valid_h.copy()
    calib.inverse_homography = valid_h.copy()
    calib.reprojection_error = 0.5
    assert calib.is_calibrated

    # Supply degenerate markers (collinear/identical corners -> singular)
    detected = {
        1: np.array([[0, 0], [0, 0], [0, 0], [0, 0]], dtype=np.float32),
        2: np.array([[0, 0], [0, 0], [0, 0], [0, 0]], dtype=np.float32),
        3: np.array([[0, 0], [0, 0], [0, 0], [0, 0]], dtype=np.float32),
        0: np.array([[0, 0], [0, 0], [0, 0], [0, 0]], dtype=np.float32),
    }

    with patch("cv2.findHomography", return_value=(np.zeros((3, 3), dtype=np.float32), None)):
        success = calib.update_homography(detected_markers=detected)

    assert not success
    # Previous matrix preserved untouched
    assert np.allclose(calib.homography_matrix, valid_h)
    assert np.allclose(calib.inverse_homography, valid_h)
    assert calib.reprojection_error == 0.5


def test_perception_refinement_raises_perception_unavailable_flipped() -> None:
    """FLIPPED: a detector crash during target refinement raises PerceptionUnavailable.

    Refinement now happens once, in the tool (``_refine_point``); ``compute_visual_alignment``
    no longer re-detects the target.
    """
    from agent.tools import _refine_point
    from robot.errors import PerceptionUnavailable

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    ok, enc = cv2.imencode(".jpg", frame)
    assert ok
    with patch(
        "agent.tools.refine_point_to_nearest_object",
        side_effect=RuntimeError("Salient detector crash"),
    ):
        with pytest.raises(PerceptionUnavailable, match="Visual refinement failed") as exc:
            _refine_point([500.0, 500.0], lambda: enc.tobytes())
    assert exc.value.robot_moved is False


def test_real_mode_placeholder_frame_fails_closed_flipped() -> None:
    """FLIPPED: get_latest_raw_jpeg returns None in real mode if buffer empty."""
    with _global_frame_lock:
        frames_mod._global_latest_raw_frame = None
        frames_mod._global_latest_annotated_frame = None
        frames_mod._global_latest_frame = None

    robot = MockRobot()  # cannot render (real-like)
    set_active_robot(robot)

    jpeg = get_latest_raw_jpeg()
    assert jpeg is None


def test_driver_write_failure_on_first_step_sets_robot_moved_false() -> None:
    """Verify bus write failure on first call raises HardwareError with robot_moved=False."""
    from robot.constants import JOINT_NAMES
    from robot.errors import HardwareError

    fake_bus = MagicMock()
    fake_bus.sync_read.return_value = {name: 0.0 for name in JOINT_NAMES}
    fake_bus.sync_write.side_effect = RuntimeError("Bus write glitch on step 1")
    robot = _make_mock_real_robot(fake_bus)

    target_qpos = np.ones(6, dtype=np.float32) * 0.2
    with pytest.raises(HardwareError, match="Failed to write joint positions") as exc_info:
        robot.move_to_pose(target_qpos, duration_sec=0.1, dt=0.02)

    assert exc_info.value.robot_moved is False
    assert fake_bus.sync_write.call_count == 1


def test_driver_write_failure_on_third_step_sets_robot_moved_true() -> None:
    """Verify bus write failure on third call raises HardwareError with robot_moved=True."""
    from robot.constants import JOINT_NAMES
    from robot.errors import HardwareError

    fake_bus = MagicMock()
    fake_bus.sync_read.return_value = {name: 0.0 for name in JOINT_NAMES}
    # First two writes succeed, third write fails
    fake_bus.sync_write.side_effect = [None, None, RuntimeError("Bus write glitch on step 3")]
    robot = _make_mock_real_robot(fake_bus)

    target_qpos = np.ones(6, dtype=np.float32) * 0.2
    with pytest.raises(HardwareError, match="Failed to write joint positions") as exc_info:
        robot.move_to_pose(target_qpos, duration_sec=0.1, dt=0.02)

    assert exc_info.value.robot_moved is True
    assert fake_bus.sync_write.call_count == 3


def test_idle_interrupt_does_not_cancel_subsequent_motion_real() -> None:
    """Verify calling interrupt() while real robot is idle does not abort next move_to_pose."""
    from robot.constants import JOINT_NAMES

    fake_bus = MagicMock()
    fake_bus.sync_read.return_value = {name: 0.0 for name in JOINT_NAMES}
    robot = _make_mock_real_robot(fake_bus)

    # Pre-set interrupt flag while idle
    robot.interrupt()
    assert robot._stop_requested is True

    target_qpos = np.ones(6, dtype=np.float32) * 0.2
    robot.move_to_pose(target_qpos, duration_sec=0.04, dt=0.02)

    # Must clear flag and run to completion (2 steps written)
    assert fake_bus.sync_write.call_count == 2
    assert robot._stop_requested is False


def test_idle_interrupt_does_not_cancel_subsequent_motion_sim() -> None:
    """Verify calling interrupt() while sim robot is idle does not abort next move_to_pose."""
    from simulation.sim_robot import SO101Robot

    robot = SO101Robot()
    robot.interrupt()
    assert robot._stop_requested is True

    target_qpos = np.ones(6, dtype=np.float32) * 0.1
    robot.move_to_pose(target_qpos, duration_sec=0.04, dt=0.02)

    assert robot._stop_requested is False
    current_qpos = robot.get_joint_positions()
    assert np.allclose(current_qpos[:5], target_qpos[:5], atol=0.05)


def test_idle_interrupt_does_not_cancel_subsequent_motion_mock() -> None:
    """Verify calling interrupt() while mock robot is idle does not abort next move_to_pose."""
    robot = MockRobot()
    robot.interrupt()
    assert robot._stop_requested is True

    target_qpos = np.ones(6, dtype=np.float32) * 0.3
    robot.move_to_pose(target_qpos, duration_sec=0.04, dt=0.02)

    assert robot._stop_requested is False
    assert robot.move_to_pose_called is True
    assert np.allclose(robot.qpos, target_qpos)


def test_mid_motion_interrupt_from_thread_raises_hardware_error() -> None:
    """Verify mid-motion thread interrupt raises HardwareError with robot_moved=True."""
    from robot.constants import JOINT_NAMES
    from robot.errors import HardwareError

    fake_bus = MagicMock()
    fake_bus.sync_read.return_value = {name: 0.0 for name in JOINT_NAMES}
    step_reached = threading.Event()
    interrupt_done = threading.Event()

    def sync_write_hook(*args, **kwargs) -> None:
        step_reached.set()
        interrupt_done.wait(timeout=1.0)

    fake_bus.sync_write.side_effect = sync_write_hook
    robot = _make_mock_real_robot(fake_bus)

    def trigger_interrupt() -> None:
        step_reached.wait(timeout=1.0)
        robot.interrupt()
        interrupt_done.set()

    t = threading.Thread(target=trigger_interrupt)
    t.start()

    target_qpos = np.ones(6, dtype=np.float32) * 0.5
    with pytest.raises(HardwareError, match="Motion interrupted") as exc_info:
        robot.move_to_pose(target_qpos, duration_sec=0.2, dt=0.01)
    t.join()

    assert exc_info.value.robot_moved is True
    assert robot._stop_requested is False


@pytest.mark.asyncio
async def test_mid_motion_interrupt_through_execute_tool_call_yields_hardware_error() -> None:
    """Verify execute_tool_call under mid-motion interrupt returns hardware error payload."""
    from agent.tools import execute_tool_call
    from robot.constants import JOINT_NAMES
    from tests.conftest import DummyToolCall

    fake_bus = MagicMock()
    fake_bus.sync_read.return_value = {name: 0.0 for name in JOINT_NAMES}
    step_reached = threading.Event()
    interrupt_done = threading.Event()

    def sync_write_hook(*args, **kwargs) -> None:
        step_reached.set()
        interrupt_done.wait(timeout=1.0)

    fake_bus.sync_write.side_effect = sync_write_hook
    robot = _make_mock_real_robot(fake_bus)

    def trigger_interrupt() -> None:
        step_reached.wait(timeout=1.0)
        robot.interrupt()
        interrupt_done.set()

    t = threading.Thread(target=trigger_interrupt)
    t.start()

    call = DummyToolCall(
        "move_to_pose",
        {"qpos": [10.0, 10.0, 10.0, 10.0, 10.0, 10.0], "duration_sec": 0.2},
    )
    result = await execute_tool_call(call, robot=robot)
    t.join()

    assert result.get("error_type") == "hardware"
    assert result.get("robot_moved") is True
    assert "Motion interrupted" in result.get("error", "")
    assert "status" not in result or result["status"] != "success"
