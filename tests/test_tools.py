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
from robot.constants import JOINT_LIMITS, PRESET_POSES
from robot.interface import RobotInterface


class MockRobot(RobotInterface):
    """Mock robot for testing tool dispatching."""

    def __init__(self, qpos: np.ndarray | None = None) -> None:
        """Initialize mock robot with joint configuration."""
        self.qpos = qpos if qpos is not None else np.zeros(6, dtype=np.float32)
        self.move_to_pose_called = False
        self.last_move_target = None
        self.last_move_duration = None
        self.set_joint_positions_called = False

    def set_joint_positions(self, qpos: np.ndarray) -> None:
        """Set joint positions."""
        self.qpos = np.asarray(qpos, dtype=np.float32)
        self.set_joint_positions_called = True

    def get_joint_positions(self) -> np.ndarray:
        """Get current joint positions."""
        return self.qpos.copy()

    def move_to_pose(
        self, target_qpos: np.ndarray, duration_sec: float = 1.0, dt: float = 0.002
    ) -> None:
        """Simulate moving to pose."""
        self.qpos = np.asarray(target_qpos, dtype=np.float32)
        self.move_to_pose_called = True
        self.last_move_target = self.qpos.copy()
        self.last_move_duration = duration_sec

    def get_end_effector_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Return dummy end effector pose."""
        pos = np.array([0.15, 0.0, 0.05], dtype=np.float64)
        rot = np.eye(3, dtype=np.float64)
        return pos, rot


class DummyToolCall:
    """Simulated Live API FunctionCall object."""

    def __init__(self, name: str, args: dict | None = None) -> None:
        """Initialize dummy tool call."""
        self.name = name
        self.args = args or {}
        self.id = "mock_call_id"


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
    assert np.isclose(robot.get_joint_positions()[5], np.radians(GRIPPER_OPEN_DEG), atol=1e-3)

    # Close gripper
    close_call = DummyToolCall("close_gripper")
    close_res = await execute_tool_call(close_call, robot=robot)
    assert close_res.get("status") == "success"
    assert np.isclose(robot.get_joint_positions()[5], np.radians(GRIPPER_CLOSE_DEG), atol=1e-3)


@pytest.mark.asyncio
async def test_execute_tool_move_to_preset() -> None:
    """Test move_to_preset handles valid and invalid poses."""
    robot = MockRobot()

    call = DummyToolCall("move_to_preset", {"pose_name": "HOME"})
    res = await execute_tool_call(call, robot=robot)
    assert res.get("status") == "success"
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


@pytest.mark.asyncio
async def test_execute_tool_move_to_spatial_point() -> None:
    """Test move_to_spatial_point with z_height_cm parameter and IK resolution."""
    robot = MockRobot(qpos=PRESET_POSES["HOME"].copy())

    # Mock calibrator
    mock_calibrator = MagicMock()
    mock_calibrator.is_calibrated = True
    mock_calibrator.gemini_to_robot_xyz.return_value = np.array([0.18, 0.05, 0.05])

    with patch("ui.stream.get_calibrator", return_value=mock_calibrator):
        call = DummyToolCall("move_to_spatial_point", {"point": [500.0, 500.0], "z_height_cm": 5.0})
        res = await execute_tool_call(call, robot=robot)
        assert res.get("status") == "success"
        assert robot.move_to_pose_called
        mock_calibrator.gemini_to_robot_xyz.assert_called_once_with(
            [500.0, 500.0], img_shape=(480, 640), z_m=0.05
        )

    # Bad point format
    bad_call = DummyToolCall("move_to_spatial_point", {"point": [500.0]})
    bad_res = await execute_tool_call(bad_call, robot=robot)
    assert "error" in bad_res


@pytest.mark.asyncio
async def test_execute_tool_move_to_spatial_uncalibrated() -> None:
    """Test move_to_spatial_point raises error when uncalibrated."""
    robot = MockRobot()
    mock_calibrator = MagicMock()
    mock_calibrator.is_calibrated = False

    with patch("ui.stream.get_calibrator", return_value=mock_calibrator):
        call = DummyToolCall("move_to_spatial_point", {"point": [500.0, 500.0]})
        res = await execute_tool_call(call, robot=robot)
        assert "error" in res


def test_image_point_to_xyz_uncalibrated() -> None:
    """Test image_point_to_xyz raises RuntimeError when uncalibrated."""
    with patch("ui.stream.get_calibrator", return_value=None):
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
