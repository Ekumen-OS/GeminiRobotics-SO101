"""Unit tests for robot tool declarations and execution adapters."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from agent.tools import (
    _BLOCKING_TOOLS,
    GRIPPER_CLOSE_DEG,
    GRIPPER_OPEN_DEG,
    ROBOT_TOOL_DECLARATIONS,
    execute_tool_call,
    execute_tool_call_sync,
    image_point_to_xyz,
    validate_and_clamp_joints,
)
from robot.constants import JOINT_LIMITS, JOINT_NAMES, PRESET_POSES
from robot.interface import RobotInterface
from robot.settings import Settings, set_settings
from tests.conftest import DummyToolCall, MockRobot


def test_tool_declarations_structure() -> None:
    """Verify all tool declarations follow the schema and omit deleted tools."""
    tool_names = [d["name"] for d in ROBOT_TOOL_DECLARATIONS]

    # Required primitives present
    expected_tools = {
        "move_to_pose",
        "set_joint_positions",
        "get_joint_positions",
        "open_gripper",
        "close_gripper",
        "move_to_preset",
        "get_end_effector_pose",
        "move_to_spatial_point",
    }
    assert expected_tools.issubset(set(tool_names))

    # SPAV tool removed
    assert "execute_spav_pick_and_place" not in tool_names

    # Check schema completeness
    for decl in ROBOT_TOOL_DECLARATIONS:
        assert "name" in decl
        assert "description" in decl
        assert "parameters" in decl
        assert decl["parameters"]["type"] == "OBJECT"

    # Blocking tools check
    for name in _BLOCKING_TOOLS:
        matching = [d for d in ROBOT_TOOL_DECLARATIONS if d["name"] == name]
        assert len(matching) == 1
        assert matching[0].get("behavior") == "BLOCKING"


def test_move_to_spatial_point_schema() -> None:
    """Verify move_to_spatial_point declaration includes z_height_cm parameter."""
    matching = [d for d in ROBOT_TOOL_DECLARATIONS if d["name"] == "move_to_spatial_point"]
    assert len(matching) == 1
    props = matching[0]["parameters"]["properties"]
    assert "point" in props
    assert "z_height_cm" in props
    assert "approach" in props
    assert "approach" in matching[0]["parameters"]["required"]


def test_validate_and_clamp_joints() -> None:
    """Test joint angle clamping and warning generation."""
    # Joint limits: shoulder_pan is [-110, 110]
    pan_min, pan_max = JOINT_LIMITS["shoulder_pan"]
    q_in = [pan_max + 20.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    clamped, warnings = validate_and_clamp_joints(q_in)
    assert clamped[0] == pan_max
    assert len(warnings) == 1
    assert "shoulder_pan" in warnings[0]

    # Valid angles within bounds
    q_valid = [0.0, -30.0, 45.0, -20.0, 10.0, 0.0]
    clamped_valid, warnings_valid = validate_and_clamp_joints(q_valid)
    assert clamped_valid == q_valid
    assert len(warnings_valid) == 0


@pytest.mark.asyncio
async def test_execute_tool_robot_none() -> None:
    """Test execute_tool_call returns error when robot is None."""
    call = DummyToolCall("move_to_preset", {"pose_name": "HOME"})
    res = await execute_tool_call(call, robot=None)
    assert "error" in res


@pytest.mark.asyncio
async def test_execute_tool_move_to_pose() -> None:
    """Test move_to_pose dispatches correctly in radians."""
    robot = MockRobot()
    q_deg = [10.0, 20.0, -30.0, 40.0, -50.0, 60.0]
    call = DummyToolCall("move_to_pose", {"qpos": q_deg, "duration_sec": 2.0})

    res = await execute_tool_call(call, robot=robot)
    assert res.get("status") == "success"
    assert robot.move_to_pose_called
    assert robot.last_move_duration == 2.0
    np.testing.assert_allclose(robot.get_joint_positions(), np.radians(q_deg), atol=1e-4)

    # Invalid joint length
    bad_call = DummyToolCall("move_to_pose", {"qpos": [10.0, 20.0]})
    bad_res = await execute_tool_call(bad_call, robot=robot)
    assert "error" in bad_res


@pytest.mark.asyncio
async def test_execute_tool_set_joint_positions() -> None:
    """Test set_joint_positions snaps to target."""
    robot = MockRobot()
    q_deg = [-10.0, 15.0, -25.0, 35.0, -45.0, 55.0]
    call = DummyToolCall("set_joint_positions", {"qpos": q_deg})

    res = await execute_tool_call(call, robot=robot)
    assert res.get("status") == "success"
    assert robot.set_joint_positions_called
    np.testing.assert_allclose(robot.get_joint_positions(), np.radians(q_deg), atol=1e-4)


@pytest.mark.asyncio
async def test_execute_tool_get_joint_positions() -> None:
    """Test get_joint_positions returns degrees."""
    q_rad = np.radians([15.0, -30.0, 45.0, -60.0, 75.0, -90.0])
    robot = MockRobot(qpos=q_rad)
    call = DummyToolCall("get_joint_positions")

    res = await execute_tool_call(call, robot=robot)
    assert "qpos_deg" in res
    np.testing.assert_allclose(res["qpos_deg"], [15.0, -30.0, 45.0, -60.0, 75.0, -90.0], atol=1e-4)


@pytest.mark.asyncio
async def test_execute_tool_gripper() -> None:
    """Test open_gripper and close_gripper."""
    robot = MockRobot()

    # Open gripper
    open_call = DummyToolCall("open_gripper")
    open_res = await execute_tool_call(open_call, robot=robot)
    assert open_res.get("status") == "success"
    assert "visual_hint" in open_res
    assert "Gripper opened" in open_res["visual_hint"]
    assert np.isclose(robot.get_joint_positions()[5], np.radians(GRIPPER_OPEN_DEG), atol=1e-3)

    # Close gripper
    close_call = DummyToolCall("close_gripper")
    close_res = await execute_tool_call(close_call, robot=robot)
    assert close_res.get("status") == "success"
    assert "visual_hint" in close_res
    assert "Gripper closed" in close_res["visual_hint"]
    assert np.isclose(robot.get_joint_positions()[5], np.radians(GRIPPER_CLOSE_DEG), atol=1e-3)


@pytest.mark.asyncio
async def test_execute_tool_move_to_preset() -> None:
    """Test move_to_preset handles valid and invalid poses."""
    robot = MockRobot()

    call = DummyToolCall("move_to_preset", {"pose_name": "HOME"})
    res = await execute_tool_call(call, robot=robot)
    assert res.get("status") == "success"
    assert "visual_hint" in res
    assert "Arm is at HOME pose" in res["visual_hint"]
    np.testing.assert_allclose(robot.get_joint_positions(), PRESET_POSES["HOME"], atol=1e-4)

    bad_call = DummyToolCall("move_to_preset", {"pose_name": "NONEXISTENT"})
    bad_res = await execute_tool_call(bad_call, robot=robot)
    assert "error" in bad_res


@pytest.mark.asyncio
async def test_execute_tool_get_end_effector_pose() -> None:
    """Test get_end_effector_pose when robot supports it."""
    robot = MockRobot()
    call = DummyToolCall("get_end_effector_pose")
    res = await execute_tool_call(call, robot=robot)
    assert "position" in res
    assert "rotation_matrix" in res

    # Base RobotInterface without method
    plain_robot = MagicMock(spec=RobotInterface)
    del plain_robot.get_end_effector_pose
    res_err = await execute_tool_call(call, robot=plain_robot)
    assert "error" in res_err
    assert res_err.get("error_type") == "argument"
    plain_robot.interrupt.assert_not_called()


@pytest.mark.asyncio
async def test_execute_tool_move_to_spatial_point() -> None:
    """Test move_to_spatial_point with z_height_cm parameter and IK resolution."""
    robot = MockRobot(qpos=PRESET_POSES["HOME"].copy())

    # Mock calibrator
    mock_calibrator = MagicMock()
    mock_calibrator.is_calibrated = True
    mock_calibrator.gemini_to_robot_xyz.return_value = np.array([0.18, 0.05, 0.05])

    with patch("agent.tools.get_calibrator", return_value=mock_calibrator):
        call = DummyToolCall(
            "move_to_spatial_point",
            {"point": [500.0, 500.0], "z_height_cm": 5.0, "approach": "top_down"},
        )
        res = await execute_tool_call(call, robot=robot)
        assert res.get("status") == "success"
        assert res.get("approach") == "top_down"
        assert robot.move_to_pose_called
        assert "visual_hint" in res
        assert "Visual Verification Loop" in res["visual_hint"]
        mock_calibrator.gemini_to_robot_xyz.assert_called_once_with(
            [500.0, 500.0], img_shape=(480, 640), z_m=0.05
        )

    # Missing approach returns error
    missing_approach_call = DummyToolCall(
        "move_to_spatial_point", {"point": [500.0, 500.0], "z_height_cm": 5.0}
    )
    missing_res = await execute_tool_call(missing_approach_call, robot=robot)
    assert "error" in missing_res
    assert "requires 'approach' parameter" in missing_res["error"]

    # Bad point format
    bad_call = DummyToolCall("move_to_spatial_point", {"point": [500.0], "approach": "top_down"})
    bad_res = await execute_tool_call(bad_call, robot=robot)
    assert "error" in bad_res


@pytest.mark.asyncio
async def test_execute_tool_move_to_spatial_uncalibrated() -> None:
    """Test move_to_spatial_point raises error when uncalibrated."""
    robot = MockRobot()
    mock_calibrator = MagicMock()
    mock_calibrator.is_calibrated = False

    with patch("agent.tools.get_calibrator", return_value=mock_calibrator):
        call = DummyToolCall("move_to_spatial_point", {"point": [500.0, 500.0]})
        res = await execute_tool_call(call, robot=robot)
        assert "error" in res


def test_image_point_to_xyz_uncalibrated() -> None:
    """Test image_point_to_xyz raises RuntimeError when uncalibrated."""
    with patch("agent.tools.get_calibrator", return_value=None):
        with pytest.raises(RuntimeError, match="Workspace is not calibrated"):
            image_point_to_xyz([500.0, 500.0])


@pytest.mark.asyncio
async def test_execute_tool_unknown() -> None:
    """Test calling an unknown tool returns error."""
    robot = MockRobot()
    call = DummyToolCall("nonexistent_tool", {})
    res = await execute_tool_call(call, robot=robot)
    assert "error" in res
    assert "Unknown tool" in res["error"]


def test_execute_tool_call_sync() -> None:
    """Test execute_tool_call_sync helper wrapper."""
    robot = MockRobot()
    call = DummyToolCall("get_joint_positions")
    res = execute_tool_call_sync(call, robot=robot)
    assert "qpos_deg" in res


@pytest.mark.asyncio
async def test_execute_tool_send_message() -> None:
    """Test send_message tool execution."""
    call = DummyToolCall("send_message", {"message": "Hello from robot!"})
    res = await execute_tool_call(call, robot=None)
    assert res["status"] == "success"
    assert res["message"] == "Hello from robot!"


@pytest.mark.asyncio
async def test_tools_visual_hints_enrichment(restore_settings: None) -> None:
    """Test all tool returns contain the expected enriched visual_hint strings."""
    set_settings(Settings.from_env(descent_gate="off"))
    robot = MockRobot(qpos=PRESET_POSES["HOME"].copy())

    mock_calibrator = MagicMock()
    mock_calibrator.is_calibrated = True
    mock_calibrator.gemini_to_robot_xyz.return_value = np.array([0.18, 0.05, 0.05])

    with patch("agent.tools.get_calibrator", return_value=mock_calibrator):
        # Hover height >= 3.0
        hover_call = DummyToolCall(
            "move_to_spatial_point",
            {"point": [500.0, 500.0], "z_height_cm": 5.0, "approach": "top_down"},
        )
        hover_res = await execute_tool_call(hover_call, robot=robot)
        assert "visual_hint" in hover_res
        hint = hover_res["visual_hint"]
        assert "hover position" in hint
        assert "Visual Verification Loop" in hint
        assert "alignment not measured" in hint

        # Contact height < 3.0
        contact_call = DummyToolCall(
            "move_to_spatial_point",
            {"point": [500.0, 500.0], "z_height_cm": 0.0, "approach": "top_down"},
        )
        contact_res = await execute_tool_call(contact_call, robot=robot)
        assert "visual_hint" in contact_res
        assert "contact height" in contact_res["visual_hint"]

    # Open gripper
    open_res = await execute_tool_call(DummyToolCall("open_gripper"), robot=robot)
    assert "visual_hint" in open_res
    assert "Gripper opened" in open_res["visual_hint"]

    # Close gripper
    close_res = await execute_tool_call(DummyToolCall("close_gripper"), robot=robot)
    assert "visual_hint" in close_res
    assert "Gripper closed" in close_res["visual_hint"]

    # Preset HOME
    home_res = await execute_tool_call(
        DummyToolCall("move_to_preset", {"pose_name": "HOME"}), robot=robot
    )
    assert "visual_hint" in home_res
    assert "Arm is at HOME pose with clear workspace view" in home_res["visual_hint"]


@pytest.mark.asyncio
async def test_tools_logging_and_stderr(capsys: pytest.CaptureFixture) -> None:
    """Test tool execution logging to stdout and error/warning routing to stderr."""
    import re

    robot = MockRobot()
    mock_calibrator = MagicMock()
    mock_calibrator.is_calibrated = True
    mock_calibrator.gemini_to_robot_xyz.return_value = np.array([0.18, 0.05, 0.05])

    # 1. Successful move_to_spatial_point logs to stdout with ISO timestamp
    with patch("agent.tools.get_calibrator", return_value=mock_calibrator):
        call = DummyToolCall(
            "move_to_spatial_point",
            {"point": [600.0, 400.0], "z_height_cm": 3.0, "approach": "top_down"},
        )
        await execute_tool_call(call, robot=robot)

    captured = capsys.readouterr()
    assert "[Tools] [EXEC] move_to_spatial_point: image [600, 400]" in captured.out
    timestamp_pattern = re.compile(
        r"\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}\] \[Tools\] \[EXEC\]"
    )
    assert timestamp_pattern.search(captured.out) is not None

    # 2. Clamping warning logs to stderr
    q_exceeded = [JOINT_LIMITS["shoulder_pan"][1] + 15.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    validate_and_clamp_joints(q_exceeded)
    captured_clamp = capsys.readouterr()
    assert "[Tools] [WARN] shoulder_pan was clamped" in captured_clamp.err
    warn_timestamp_pattern = re.compile(
        r"\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}\] \[Tools\] \[WARN\]"
    )
    assert warn_timestamp_pattern.search(captured_clamp.err) is not None

    # 3. Tool error logs to stderr
    err_call = DummyToolCall("move_to_pose", {"qpos": [0.0, 0.0]})  # invalid joint count
    await execute_tool_call(err_call, robot=robot)
    captured_err = capsys.readouterr()
    assert "[Tools] [ERROR] move_to_pose expects 6 joint angles" in captured_err.err


@pytest.mark.asyncio
async def test_move_to_spatial_point_custom_label() -> None:
    """Verify that custom label parameter is preserved on the visual overlay."""
    from vision.targets import clear_model_targets, get_model_targets

    clear_model_targets()
    robot = MockRobot()
    mock_calib = MagicMock()
    mock_calib.is_calibrated = True
    mock_calib.gemini_to_robot_xyz.return_value = np.array([0.2, 0.0, 0.05])

    with patch("agent.tools.get_calibrator", return_value=mock_calib):
        call = DummyToolCall(
            "move_to_spatial_point",
            {
                "point": [762, 483],
                "z_height_cm": 5.0,
                "approach": "top_down",
                "label": "Red Cube Pick",
            },
        )
        resp = await execute_tool_call(call, robot=robot)
        assert resp["status"] == "success"

        # Check visual target overlay
        targets = get_model_targets()
        assert len(targets) >= 1
        active = targets[-1]
        assert "[RED CUBE PICK]" in active["label"]
        assert "(z=5.0cm, top_down)" in active["label"]
        assert "🎯" not in active["label"]
        clear_model_targets()


@pytest.mark.asyncio
async def test_execute_tool_hardware_error_calls_interrupt_and_returns_honestly() -> None:
    """Verify HardwareError from driver calls robot.interrupt() and returns honest error payload."""
    from tests.robot.test_fail_closed_characterization import _make_mock_real_robot

    fake_bus = MagicMock()
    fake_bus.sync_read.return_value = {name: 0.0 for name in JOINT_NAMES}
    # Step 1 write succeeds, step 2 write fails -> driver sets robot_moved = True
    fake_bus.sync_write.side_effect = [None, RuntimeError("Bus write failed mid-motion")]
    robot = _make_mock_real_robot(fake_bus)
    robot.interrupt = MagicMock(wraps=robot.interrupt)

    call = DummyToolCall("move_to_preset", {"pose_name": "HOME"})
    res = await execute_tool_call(call, robot=robot)

    robot.interrupt.assert_called_once()
    assert res.get("error_type") == "hardware"
    assert res.get("robot_moved") is True
    assert "Failed to write joint positions" in res.get("error", "")


@pytest.mark.asyncio
@pytest.mark.usefixtures("real_sleep")
async def test_close_gripper_honors_nonzero_settle_setting() -> None:
    """Verify close_gripper executes time.sleep(gripper_settle_s) when setting is non-zero."""
    import dataclasses

    from robot.settings import get_settings

    robot = MockRobot()
    call = DummyToolCall("close_gripper")

    custom_settings = dataclasses.replace(get_settings(), gripper_settle_s=0.45)
    with patch("agent.tools.get_settings", return_value=custom_settings):
        with patch("agent.tools.time.sleep") as mock_sleep:
            res = await execute_tool_call(call, robot=robot)

    assert res.get("status") == "success"
    mock_sleep.assert_called_with(0.45)


@pytest.mark.asyncio
async def test_execute_tool_taxonomic_error_responses() -> None:
    """Verify tool errors map cleanly to argument, calibration, perception, and internal types."""
    from robot.errors import CalibrationUnavailable

    robot = MockRobot()

    # 1. Argument error
    bad_arg_call = DummyToolCall("move_to_pose", {"qpos": [1.0, 2.0], "duration_sec": 1.0})
    arg_res = await execute_tool_call(bad_arg_call, robot=robot)
    assert arg_res.get("error_type") == "argument"
    assert arg_res.get("robot_moved") is False

    # 2. Calibration error
    mock_calib = MagicMock()
    mock_calib.require_calibrated.side_effect = CalibrationUnavailable("Workspace uncalibrated")
    with patch("agent.tools.get_calibrator", return_value=mock_calib):
        spatial_call = DummyToolCall(
            "move_to_spatial_point",
            {"point": [500.0, 500.0], "approach": "top_down"},
        )
        calib_res = await execute_tool_call(spatial_call, robot=robot)
        assert calib_res.get("error_type") == "calibration"
        assert calib_res.get("robot_moved") is False

    # 3. Perception error (frame provider returns None in spatial move)
    valid_calib = MagicMock()
    valid_calib.is_calibrated = True
    valid_calib.gemini_to_robot_xyz.return_value = np.array([0.2, 0.0, 0.05])
    with patch("agent.tools.get_calibrator", return_value=valid_calib):
        spatial_call = DummyToolCall(
            "move_to_spatial_point",
            {"point": [500.0, 500.0], "approach": "top_down"},
        )
        percep_res = await execute_tool_call(
            spatial_call, robot=robot, get_latest_jpeg_fn=lambda: None
        )
        assert percep_res.get("error_type") == "perception"
        assert percep_res.get("robot_moved") is False

    # 4. Internal error (unexpected exception)
    def crash(*args, **kwargs):
        raise TypeError("Unexpected type error in driver")

    robot.get_joint_positions = crash
    get_qpos_call = DummyToolCall("get_joint_positions", {})
    internal_res = await execute_tool_call(get_qpos_call, robot=robot)
    assert internal_res.get("error_type") == "internal"
    assert internal_res.get("robot_moved") is False
