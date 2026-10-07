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
    from robot.calibration import DEFAULT_SAVE_PATH, DEFAULT_SIM_SAVE_PATH

    robot = MockRobot()
    calib = _calibrator_mock()
    call = DummyToolCall(
        "move_to_spatial_point",
        {"point": [500.0, 500.0], "z_height_cm": 5.0, "approach": "top_down"},
    )
    with patch("agent.tools.get_calibrator", return_value=calib):
        res = await execute_tool_call(call, robot=robot)

    assert res.get("status") == "success"
    calib.load_calibration.assert_called_once_with(DEFAULT_SAVE_PATH)
    assert calib.mode == "real"
    assert DEFAULT_SIM_SAVE_PATH not in [c.args[0] for c in calib.load_calibration.call_args_list]


@pytest.mark.asyncio
async def test_sim_robot_spatial_move_loads_sim_homography() -> None:
    """A robot declaring mode='sim' selects the simulation calibration file."""
    from robot.calibration import DEFAULT_SIM_SAVE_PATH

    calib = _calibrator_mock()
    call = DummyToolCall(
        "move_to_spatial_point",
        {"point": [500.0, 500.0], "z_height_cm": 5.0, "approach": "top_down"},
    )
    with patch("agent.tools.get_calibrator", return_value=calib):
        res = await execute_tool_call(call, robot=_SimLikeRobot())

    assert res.get("status") == "success"
    calib.load_calibration.assert_called_once_with(DEFAULT_SIM_SAVE_PATH)
    assert calib.mode == "sim"


@pytest.mark.asyncio
async def test_controller_mode_argument_overrides_robot_mode() -> None:
    """The controller's mode (what the user launched) wins over the robot's own attribute."""
    from robot.calibration import DEFAULT_SIM_SAVE_PATH

    calib = _calibrator_mock()
    call = DummyToolCall(
        "move_to_spatial_point",
        {"point": [500.0, 500.0], "z_height_cm": 5.0, "approach": "top_down"},
    )
    with patch("agent.tools.get_calibrator", return_value=calib):
        res = await execute_tool_call(call, robot=MockRobot(), mode="sim")

    assert res.get("status") == "success"
    calib.load_calibration.assert_called_once_with(DEFAULT_SIM_SAVE_PATH)


def test_main_parser_accepts_only_real_and_sim() -> None:
    """The removed 'twin' mode is rejected at the command line."""
    from main import _build_parser

    parser = _build_parser()
    assert parser.parse_args(["--mode", "sim"]).mode == "sim"
    assert parser.parse_args(["--mode", "real"]).mode == "real"
    with pytest.raises(SystemExit):
        parser.parse_args(["--mode", "twin"])
