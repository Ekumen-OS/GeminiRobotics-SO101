"""Tests for physical grasp sensing, classification, and fail-closed guardrails."""

import threading
from unittest.mock import MagicMock

import numpy as np
import pytest

from agent.tools import execute_tool_call
from robot.cartesian_ik import CartesianIK
from robot.constants import GRIPPER_CLOSE_DEG
from robot.contracts import GripperReading
from robot.grasp import classify_grasp
from robot.real_robot import RealSO101Robot
from robot.settings import Settings
from simulation.sim_calibration import compute_sim_calibration
from simulation.sim_robot import SO101Robot
from tests.conftest import DummyToolCall, MockRobot
from vision import get_calibrator


@pytest.mark.asyncio
async def test_close_gripper_measured_empty_flipped() -> None:
    """Flipped Task 0 Characterization: verify close_gripper returns measured empty state.

    Previously unmeasured gripper_state='closed' now flips to gripper_state='closed_empty'
    with grasp classification grasped=False, reason='closed_on_air'.
    """
    robot = MockRobot()
    call = DummyToolCall("close_gripper", {})
    res = await execute_tool_call(call, robot=robot)

    # Flipped assertions: measured physical grasp detection
    assert res.get("status") == "success"
    assert res.get("gripper_state") == "closed_empty"
    assert "grasp" in res
    assert res["grasp"]["grasped"] is False
    assert res["grasp"]["reason"] == "closed_on_air"
    assert "Gripper closed on NOTHING" in res.get("visual_hint", "")


# ==============================================================================
# Task 5 Unit & Integration Coverage
# ==============================================================================


@pytest.mark.asyncio
async def test_mock_robot_scriptable_gripper_closed_on_air() -> None:
    """Validate close_gripper payload when sensor reading indicates closed on air."""
    robot = MockRobot()
    robot.gripper_reading = {
        "aperture_deg": -75.0,
        "load": None,
        "source": "commanded",
    }
    call = DummyToolCall("close_gripper", {})
    res = await execute_tool_call(call, robot=robot)

    assert res["status"] == "success"
    assert res["gripper_state"] == "closed_empty"
    assert res["grasp"]["grasped"] is False
    assert res["grasp"]["reason"] == "closed_on_air"
    assert "Gripper closed on NOTHING" in res["visual_hint"]
    assert "Grasp missed. Open the gripper" in res["visual_hint"]


@pytest.mark.asyncio
async def test_mock_robot_scriptable_gripper_closed_on_object() -> None:
    """Validate close_gripper payload when sensor reading indicates object held."""
    robot = MockRobot()
    robot.gripper_reading = {
        "aperture_deg": -65.0,
        "load": 2.5,
        "source": "measured",
    }
    call = DummyToolCall("close_gripper", {})
    res = await execute_tool_call(call, robot=robot)

    assert res["status"] == "success"
    assert res["gripper_state"] == "closed"
    assert res["grasp"]["grasped"] is True
    assert res["grasp"]["reason"] == "aperture"
    assert "Gripper stalled at -65.0° — an object is between the jaws" in res["visual_hint"]
    assert "Lift to 5 cm to verify the grasp before transporting" in res["visual_hint"]


@pytest.mark.asyncio
async def test_mock_robot_scriptable_gripper_sensor_unavailable() -> None:
    """Validate close_gripper payload when sensor reading is unavailable."""
    robot = MockRobot()
    robot.gripper_reading = {
        "aperture_deg": -75.0,
        "load": None,
        "source": "unavailable",
    }
    call = DummyToolCall("close_gripper", {})
    res = await execute_tool_call(call, robot=robot)

    assert res["status"] == "success"
    assert res["gripper_state"] == "closed_unverified"
    assert res["grasp"]["grasped"] is None
    assert res["grasp"]["reason"] == "sensor_unavailable"
    assert "Gripper sensor unavailable; grasp NOT verified" in res["visual_hint"]


@pytest.mark.asyncio
async def test_open_gripper_reports_measured_aperture() -> None:
    """Validate open_gripper reports measured aperture in visual hint without asserting."""
    robot = MockRobot()
    robot.gripper_reading = {
        "aperture_deg": 74.8,
        "load": None,
        "source": "measured",
    }
    call = DummyToolCall("open_gripper", {})
    res = await execute_tool_call(call, robot=robot)

    assert res["status"] == "success"
    assert res["gripper_state"] == "open"
    assert res["visual_hint"] == "Gripper opened (aperture 74.8°)."


def _make_mock_real_robot(fake_bus: MagicMock | None = None) -> RealSO101Robot:
    """Create a disconnected or mocked RealSO101Robot instance for hermetic unit testing."""
    robot = RealSO101Robot.__new__(RealSO101Robot)
    robot.connected = True
    robot._bus_lock = threading.Lock()
    robot.bus = fake_bus if fake_bus is not None else MagicMock()
    robot._last_qpos = np.zeros(6, dtype=np.float32)
    return robot


def test_real_robot_read_gripper_state_success() -> None:
    """Test RealSO101Robot.read_gripper_state returns measured position and load."""
    fake_bus = MagicMock()
    fake_bus.sync_read.return_value = {"gripper": -64.2}
    fake_bus.read.return_value = 115.0

    robot = _make_mock_real_robot(fake_bus)
    reading = robot.read_gripper_state()

    assert reading["source"] == "measured"
    assert np.isclose(reading["aperture_deg"], -64.2)
    assert reading["load"] == 115.0
    fake_bus.sync_read.assert_called_once_with("Present_Position", normalize=True)
    fake_bus.read.assert_called_once_with("Present_Load", "gripper", normalize=False)


def test_real_robot_read_gripper_state_load_read_failure_fallback() -> None:
    """Test RealSO101Robot.read_gripper_state succeeds with load=None if load read fails."""
    fake_bus = MagicMock()
    fake_bus.sync_read.return_value = {"gripper": -64.2}
    fake_bus.read.side_effect = RuntimeError("Load register read timeout")

    robot = _make_mock_real_robot(fake_bus)
    reading = robot.read_gripper_state()

    assert reading["source"] == "measured"
    assert np.isclose(reading["aperture_deg"], -64.2)
    assert reading["load"] is None


def test_real_robot_read_gripper_state_position_read_failure() -> None:
    """Test RealSO101Robot.read_gripper_state returns source='unavailable' on read error.

    In accordance with architectural invariant: never report _last_qpos as measured.
    """
    fake_bus = MagicMock()
    fake_bus.sync_read.side_effect = RuntimeError("Bus read I/O error")

    robot = _make_mock_real_robot(fake_bus)
    reading = robot.read_gripper_state()

    assert reading["source"] == "unavailable"
    assert reading["load"] is None


def test_real_robot_read_gripper_state_disconnected() -> None:
    """Test RealSO101Robot.read_gripper_state returns source='unavailable' when disconnected."""
    robot = _make_mock_real_robot()
    robot.connected = False
    robot.bus = None

    reading = robot.read_gripper_state()
    assert reading["source"] == "unavailable"
    assert reading["load"] is None


def test_classify_grasp_boundary_thresholds() -> None:
    """Test classify_grasp boundary values around GRIPPER_CLOSE_DEG + grasp_empty_margin_deg."""
    # Threshold = -75.0 + 6.0 = -69.0
    threshold = GRIPPER_CLOSE_DEG + 6.0

    # Exactly at threshold -> closed_on_air
    reading_at: GripperReading = {"aperture_deg": threshold, "load": None, "source": "measured"}
    assert classify_grasp(reading_at)["grasped"] is False
    assert classify_grasp(reading_at)["reason"] == "closed_on_air"

    # Below threshold -> closed_on_air
    reading_below: GripperReading = {
        "aperture_deg": threshold - 0.1,
        "load": None,
        "source": "measured",
    }
    assert classify_grasp(reading_below)["grasped"] is False
    assert classify_grasp(reading_below)["reason"] == "closed_on_air"

    # Margin + 0.1 -> aperture stall detected (grasped)
    reading_above: GripperReading = {
        "aperture_deg": threshold + 0.1,
        "load": None,
        "source": "measured",
    }
    assert classify_grasp(reading_above)["grasped"] is True
    assert classify_grasp(reading_above)["reason"] == "aperture"


def test_classify_grasp_load_threshold() -> None:
    """Test classify_grasp when GRASP_LOAD_MIN is configured."""
    settings = Settings.from_env(grasp_load_min=20.0)

    # Above aperture threshold, but load is None -> no_load
    reading_no_load: GripperReading = {
        "aperture_deg": -65.0,
        "load": None,
        "source": "measured",
    }
    res_no_load = classify_grasp(reading_no_load, settings=settings)
    assert res_no_load["grasped"] is False
    assert res_no_load["reason"] == "no_load"

    # Above aperture threshold, load below minimum -> no_load
    reading_under_load: GripperReading = {
        "aperture_deg": -65.0,
        "load": 15.0,
        "source": "measured",
    }
    res_under = classify_grasp(reading_under_load, settings=settings)
    assert res_under["grasped"] is False
    assert res_under["reason"] == "no_load"

    # Above aperture threshold, load meets minimum -> grasped
    reading_with_load: GripperReading = {
        "aperture_deg": -65.0,
        "load": 20.0,
        "source": "measured",
    }
    res_with = classify_grasp(reading_with_load, settings=settings)
    assert res_with["grasped"] is True
    assert res_with["reason"] == "aperture"


def test_classify_grasp_threshold_anchors_to_calibrated_empty_aperture() -> None:
    """An arm whose servo closes at -60 deg must not report every empty close as a grasp."""
    reading: GripperReading = {"aperture_deg": -60.0, "load": None, "source": "measured"}

    # Default anchor (GRIPPER_CLOSE_DEG = -75): -60 is above -69, so it looks like a grasp.
    default_settings = Settings(grasp_empty_margin_deg=6.0)
    assert classify_grasp(reading, default_settings)["grasped"] is True

    # Calibrated anchor: threshold becomes -60 + 6 = -54, so the same reading is empty.
    calibrated = Settings(gripper_empty_aperture_deg=-60.0, grasp_empty_margin_deg=6.0)
    res = classify_grasp(reading, calibrated)
    assert res["grasped"] is False
    assert res["reason"] == "closed_on_air"
    assert res["closed_on_air_threshold_deg"] == pytest.approx(-54.0)

    # Env wiring
    import os

    os.environ["GRIPPER_EMPTY_APERTURE_DEG"] = "-61.5"
    try:
        assert Settings.from_env().gripper_empty_aperture_deg == pytest.approx(-61.5)
    finally:
        del os.environ["GRIPPER_EMPTY_APERTURE_DEG"]


def test_classify_grasp_sensor_unavailable() -> None:
    """Test classify_grasp returns grasped=None when source is unavailable."""
    reading: GripperReading = {
        "aperture_deg": -50.0,
        "load": 100.0,
        "source": "unavailable",
    }
    res = classify_grasp(reading)
    assert res["grasped"] is None
    assert res["reason"] == "sensor_unavailable"


@pytest.mark.asyncio
async def test_sim_robot_empty_vs_cube_jaws() -> None:
    """Sim override verification: empty jaws vs cube between jaws in headless MuJoCo."""
    calib = get_calibrator(mode="sim")
    if not calib.is_calibrated:
        compute_sim_calibration(calibrator=calib)

    robot = SO101Robot()
    robot.reset("HOME")

    # 1. Close gripper on empty jaws
    close_call = DummyToolCall("close_gripper", {})
    res_empty = await execute_tool_call(call=close_call, robot=robot)
    assert res_empty["status"] == "success"
    assert res_empty["gripper_state"] == "closed_empty"
    assert res_empty["grasp"]["grasped"] is False
    assert res_empty["grasp"]["reason"] == "closed_on_air"
    aperture_empty = res_empty["grasp"]["aperture_deg"]

    # 2. Open gripper, position jaws surrounding red cube, and close
    open_call = DummyToolCall("open_gripper", {})
    await execute_tool_call(call=open_call, robot=robot)

    t_id = robot.model.body("target").id
    target_pos = robot.data.xpos[t_id].copy()

    ik = CartesianIK()
    hover_pos = np.array([target_pos[0], target_pos[1], 0.06])
    hover_q = ik.solve_ik(hover_pos, initial_qpos=robot.get_joint_positions(), approach="top_down")
    hover_q[5] = np.radians(75.0)
    robot.move_to_pose(hover_q, duration_sec=1.0)

    # The grasp site is the pad midpoint: z=0 puts the pads beside the cube at table level
    # (the old z=-0.03 pushed the open jaw into the table).
    descend_pos = np.array([target_pos[0], target_pos[1], 0.0])
    descend_q = ik.solve_ik(descend_pos, initial_qpos=hover_q, approach="top_down")
    descend_q[5] = np.radians(75.0)
    robot.move_to_pose(descend_q, duration_sec=1.0)

    res_cube = await execute_tool_call(call=close_call, robot=robot)
    assert res_cube["status"] == "success"
    assert res_cube["gripper_state"] == "closed"
    assert res_cube["grasp"]["grasped"] is True
    assert res_cube["grasp"]["reason"] == "aperture"
    aperture_cube = res_cube["grasp"]["aperture_deg"]

    # Confirm cube stalls gripper significantly above empty jaw closure
    assert aperture_cube > aperture_empty + 3.0
