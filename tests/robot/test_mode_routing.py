"""Tests for explicit real/sim mode routing of tool calls and calibrator selection.

Regression context: ``RobotInterface`` defines ``render_rgb_frame`` (raising), so the old
``hasattr(robot, "render_rgb_frame")`` duck-typing classified every robot, physical arms
included, as a simulation and loaded the simulation homography in real mode.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from agent.tools import execute_tool_call, resolve_robot_mode
from robot.interface import RobotInterface
from robot.real_robot import RealSO101Robot
from simulation.sim_robot import SO101Robot
from tests.conftest import DummyToolCall, MockRobot


class _SimLikeRobot(MockRobot):
    """Mock that declares itself a simulation without overriding any rendering method."""

    mode = "sim"


def test_robot_classes_declare_their_mode() -> None:
    """Physical and simulated controllers carry an explicit mode attribute."""
    assert RobotInterface.mode == "real"
    assert RealSO101Robot.mode == "real"
    assert SO101Robot.mode == "sim"


def test_resolve_robot_mode_explicit_argument_wins() -> None:
    """An explicit mode overrides whatever the robot declares."""
    assert resolve_robot_mode(_SimLikeRobot(), mode="real") == "real"
    assert resolve_robot_mode(MockRobot(), mode="sim") == "sim"


def test_resolve_robot_mode_reads_robot_attribute_not_render_method() -> None:
    """A robot that inherits (but does not override) render_rgb_frame is still real."""
    physical_like = MockRobot()
    assert hasattr(physical_like, "render_rgb_frame")  # the trap the old code fell into
    assert resolve_robot_mode(physical_like) == "real"
    assert resolve_robot_mode(_SimLikeRobot()) == "sim"
    assert resolve_robot_mode(None) == "real"
    assert resolve_robot_mode(MockRobot(), mode="bogus") == "real"


def _calibrator_mock() -> MagicMock:
    calib = MagicMock()
    calib.is_calibrated = True
    calib.gemini_to_robot_xyz.return_value = np.array([0.18, 0.05, 0.05])
    return calib


@pytest.mark.asyncio
async def test_physical_robot_spatial_move_loads_real_homography() -> None:
    """Regression: a physical robot must not be routed through the sim calibration file."""
    from robot.settings import get_settings

    robot = MockRobot()
    calib = _calibrator_mock()
    call = DummyToolCall(
        "move_to_spatial_point",
        {"point": [500.0, 500.0], "z_height_cm": 5.0},
    )
    with patch("vision.calibrator.get_calibrator", return_value=calib):
        res = await execute_tool_call(call, robot=robot)

    assert res.get("status") == "success"
    real_path = get_settings().calibration_save_path
    sim_path = get_settings().calibration_sim_save_path
    calib.load_calibration.assert_called_once_with(real_path)
    assert calib.mode == "real"
    assert sim_path not in [c.args[0] for c in calib.load_calibration.call_args_list]


@pytest.mark.asyncio
async def test_sim_robot_spatial_move_loads_sim_homography() -> None:
    """A robot declaring mode='sim' selects the simulation calibration file."""
    from robot.settings import get_settings

    calib = _calibrator_mock()
    call = DummyToolCall(
        "move_to_spatial_point",
        {"point": [500.0, 500.0], "z_height_cm": 5.0},
    )
    with patch("vision.calibrator.get_calibrator", return_value=calib):
        res = await execute_tool_call(call, robot=_SimLikeRobot())

    assert res.get("status") == "success"
    sim_path = get_settings().calibration_sim_save_path
    calib.load_calibration.assert_called_once_with(sim_path)
    assert calib.mode == "sim"


@pytest.mark.asyncio
async def test_controller_mode_argument_overrides_robot_mode() -> None:
    """The controller's mode (what the user launched) wins over the robot's own attribute."""
    from robot.settings import get_settings

    calib = _calibrator_mock()
    call = DummyToolCall(
        "move_to_spatial_point",
        {"point": [500.0, 500.0], "z_height_cm": 5.0},
    )
    with patch("vision.calibrator.get_calibrator", return_value=calib):
        res = await execute_tool_call(call, robot=MockRobot(), mode="sim")

    assert res.get("status") == "success"
    sim_path = get_settings().calibration_sim_save_path
    calib.load_calibration.assert_called_once_with(sim_path)


def test_main_parser_accepts_only_real_and_sim() -> None:
    """The removed 'twin' mode is rejected at the command line."""
    from main import _build_parser

    parser = _build_parser()
    assert parser.parse_args(["--mode", "sim"]).mode == "sim"
    assert parser.parse_args(["--mode", "real"]).mode == "real"
    with pytest.raises(SystemExit):
        parser.parse_args(["--mode", "twin"])


def test_get_ready_calibrator_sim_with_no_cached_file(
    tmp_path: pytest.TempPathFactory, restore_settings: None
) -> None:
    """Sim mode with no cached file runs init_sim_workspace_calibration and returns mode=='sim'."""
    from pathlib import Path

    from robot.calibration import ArucoWorkspaceCalibrator
    from robot.settings import Settings, set_settings
    from vision.calibrator import get_ready_calibrator

    sim_npz = Path(str(tmp_path)) / "nonexistent_sim.npz"
    set_settings(Settings.from_env(calibration_sim_save_path=sim_npz))
    calib = ArucoWorkspaceCalibrator(mode="real", save_path=sim_npz)

    def fake_init_sim(calibrator: ArucoWorkspaceCalibrator) -> ArucoWorkspaceCalibrator:
        h = np.eye(3, dtype=np.float64)
        calibrator.set_geometry(h, h, 0.0)
        calibrator.mode = "sim"
        return calibrator

    with (
        patch("vision.calibrator.get_calibrator", return_value=calib),
        patch(
            "simulation.sim_calibration.init_sim_workspace_calibration",
            side_effect=fake_init_sim,
        ) as init_mock,
    ):
        ready = get_ready_calibrator("sim")

    init_mock.assert_called_once_with(calibrator=calib)
    assert ready is calib
    assert ready.is_calibrated
    assert ready.mode == "sim"


def test_get_ready_calibrator_real_with_valid_npz(
    homography_path: pytest.TempPathFactory, restore_settings: None
) -> None:
    """Real mode with a valid .npz file loads it and returns mode=='real'."""
    from pathlib import Path

    from robot.calibration import ArucoWorkspaceCalibrator
    from robot.settings import Settings, set_settings
    from vision.calibrator import get_ready_calibrator

    real_npz = Path(str(homography_path))
    set_settings(Settings.from_env(calibration_save_path=real_npz))
    calib = ArucoWorkspaceCalibrator(mode="sim", save_path=real_npz)

    with patch("vision.calibrator.get_calibrator", return_value=calib):
        ready = get_ready_calibrator("real")

    assert ready is calib
    assert ready.is_calibrated
    assert ready.mode == "real"


def test_get_ready_calibrator_real_with_no_npz_and_uncalibrated_raises(
    tmp_path: pytest.TempPathFactory, restore_settings: None
) -> None:
    """Real mode with no .npz file and an uncalibrated singleton raises CalibrationUnavailable."""
    from pathlib import Path

    from robot.calibration import ArucoWorkspaceCalibrator
    from robot.errors import CalibrationUnavailable
    from robot.settings import Settings, set_settings
    from vision.calibrator import get_ready_calibrator

    missing_npz = Path(str(tmp_path)) / "missing_real.npz"
    set_settings(Settings.from_env(calibration_save_path=missing_npz))
    calib = ArucoWorkspaceCalibrator(mode="real", save_path=missing_npz)

    with patch("vision.calibrator.get_calibrator", return_value=calib):
        with pytest.raises(CalibrationUnavailable):
            get_ready_calibrator("real")


def test_get_ready_calibrator_real_with_no_npz_clears_stale_sim_singleton_and_raises(
    tmp_path: pytest.TempPathFactory, restore_settings: None
) -> None:
    """Real mode with no .npz file and singleton in mode=='sim' clears sim geometry and raises."""
    from pathlib import Path

    from robot.calibration import ArucoWorkspaceCalibrator
    from robot.errors import CalibrationUnavailable
    from robot.settings import Settings, set_settings
    from vision.calibrator import get_ready_calibrator

    missing_npz = Path(str(tmp_path)) / "missing_real.npz"
    set_settings(Settings.from_env(calibration_save_path=missing_npz))
    calib = ArucoWorkspaceCalibrator(mode="sim", save_path=missing_npz)
    h = np.eye(3, dtype=np.float64)
    calib.set_geometry(h, h, 0.0)
    assert calib.is_calibrated
    assert calib.mode == "sim"

    with patch("vision.calibrator.get_calibrator", return_value=calib):
        with pytest.raises(CalibrationUnavailable):
            get_ready_calibrator("real")

    assert not calib.is_calibrated
    assert calib.mode == "real"


def test_image_point_to_xyz_mode_routing() -> None:
    """Validate active robot mode context routing and table offsets in image_point_to_xyz."""
    from agent.tools import image_point_to_xyz
    from robot.registry import set_active_robot
    from robot.settings import get_settings

    pt = [760.0, 484.0]
    xyz_sim = image_point_to_xyz(pt, hover_z=0.03, mode="sim")
    xyz_real = image_point_to_xyz(pt, hover_z=0.03, mode="real")
    s = get_settings()
    table_x = s.get_table_x_mm("real")
    table_y = s.get_table_y_mm("real")
    table_z = s.get_table_z_mm("real")
    expected_real = xyz_sim + np.array([table_x / 1000.0, table_y / 1000.0, table_z / 1000.0])
    assert 0.12 <= xyz_sim[1] <= 0.17
    assert 0.12 + (table_y / 1000.0) <= xyz_real[1] <= 0.17 + (table_y / 1000.0)
    np.testing.assert_allclose(xyz_real, expected_real, atol=2e-3)

    # Dynamic resolution from active robot in simulation
    sim_robot = _SimLikeRobot()
    try:
        set_active_robot(sim_robot)
        xyz_auto = image_point_to_xyz(pt, hover_z=0.03)
        assert 0.12 <= xyz_auto[1] <= 0.17
        np.testing.assert_allclose(xyz_auto, xyz_sim, atol=1e-4)
    finally:
        set_active_robot(None)
