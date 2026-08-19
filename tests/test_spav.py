"""Unit tests for the SPAV (Sense-Plan-Act-Verify) state machine and evaluative feedback loop."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from agent.spav import SPAVStateMachine, _clean_json_text
from agent.tools import execute_tool_call
from robot.calibration import ArucoWorkspaceCalibrator
from robot.interface import RobotInterface
from robot.trajectory import TrajectoryPlanner


class MockRobot(RobotInterface):
    """Mock robot tracking joint position commands."""

    def __init__(self) -> None:
        """Initialize mock robot with default HOME pose."""
        self._qpos = np.zeros(6, dtype=np.float32)
        self.history: list[np.ndarray] = []

    def set_joint_positions(self, qpos: np.ndarray) -> None:
        """Update mock joint positions."""
        self._qpos = np.asarray(qpos, dtype=np.float32)
        self.history.append(self._qpos.copy())

    def get_joint_positions(self) -> np.ndarray:
        """Get mock joint positions."""
        return self._qpos.copy()

    def move_to_pose(
        self,
        target_qpos: np.ndarray,
        duration_sec: float = 1.0,
        dt: float = 0.002,
    ) -> None:
        """Move directly to target pose."""
        self._qpos = np.asarray(target_qpos, dtype=np.float32)
        self.history.append(self._qpos.copy())


def test_clean_json_text() -> None:
    """Test cleaning json string from markdown formatting."""
    raw_markdown = '```json\n{\n  "pick_point": [100, 200]\n}\n```'
    assert _clean_json_text(raw_markdown) == '{\n  "pick_point": [100, 200]\n}'

    raw_plain = '{"pick_point": [100, 200]}'
    assert _clean_json_text(raw_plain) == '{"pick_point": [100, 200]}'


@pytest.mark.asyncio
async def test_spav_success_path() -> None:
    """Test full successful SPAV execution loop in a single attempt."""
    robot = MockRobot()
    calibrator = ArucoWorkspaceCalibrator()
    planner = TrajectoryPlanner(robot=robot)

    # Mock GenAI client responses
    mock_client = MagicMock()
    # Response 1: Sense & Plan
    resp_sense = MagicMock()
    resp_sense.text = (
        "```json\n{\n"
        '  "pick_label": "red apple",\n'
        '  "pick_point": [450, 350],\n'
        '  "place_label": "fruit basket",\n'
        '  "place_point": [200, 750],\n'
        '  "plan_description": "pick apple and place in basket"\n'
        "}\n```"
    )
    # Response 2: Visual verification (Success)
    resp_verify = MagicMock()
    resp_verify.text = (
        "{\n"
        '  "grasp_success": true,\n'
        '  "confidence": 0.95,\n'
        '  "failure_reason": null,\n'
        '  "visual_offset_delta": null\n'
        "}"
    )

    mock_client.models.generate_content.side_effect = [resp_sense, resp_verify]

    spav = SPAVStateMachine(
        robot=robot,
        calibrator=calibrator,
        trajectory_planner=planner,
        max_retries=3,
        genai_client=mock_client,
    )

    dummy_jpeg = b"\xff\xd8\xff\xe0\x00\x10JFIF"
    status_messages: list[str] = []

    res = await spav.execute_spav(
        instruction="Pick up the red apple and put it in the fruit basket",
        get_latest_jpeg_fn=lambda: dummy_jpeg,
        status_callback=status_messages.append,
    )

    assert res.success is True
    assert res.pick_label == "red apple"
    assert res.place_label == "fruit basket"
    assert res.attempts == 1
    assert "Successfully" in res.final_message
    assert any("State 1: Sense & Plan" in msg for msg in status_messages)
    assert any("State 2: Act" in msg for msg in status_messages)
    assert any("State 3: Visual Verification" in msg for msg in status_messages)
    assert any("State 4: Decision" in msg for msg in status_messages)


@pytest.mark.asyncio
async def test_spav_retry_and_self_correction() -> None:
    """Test SPAV retry mechanism applying visual delta offset on initial grasp failure."""
    robot = MockRobot()
    calibrator = ArucoWorkspaceCalibrator()
    planner = TrajectoryPlanner(robot=robot)

    mock_client = MagicMock()
    # Response 1: Sense & Plan
    resp_sense = MagicMock()
    resp_sense.text = (
        "{\n"
        '  "pick_label": "blue bolt",\n'
        '  "pick_point": [500, 500],\n'
        '  "place_label": "parts bin",\n'
        '  "place_point": [150, 800],\n'
        '  "plan_description": "pick bolt and drop in bin"\n'
        "}"
    )
    # Response 2: Verification 1 (Failed, offset provided)
    resp_verify_1 = MagicMock()
    resp_verify_1.text = (
        "{\n"
        '  "grasp_success": false,\n'
        '  "confidence": 0.85,\n'
        '  "failure_reason": "Gripper missed object by 15 units",\n'
        '  "visual_offset_delta": [15, -20]\n'
        "}"
    )
    # Response 3: Verification 2 (Succeeded on retry)
    resp_verify_2 = MagicMock()
    resp_verify_2.text = (
        "{\n"
        '  "grasp_success": true,\n'
        '  "confidence": 0.98,\n'
        '  "failure_reason": null,\n'
        '  "visual_offset_delta": null\n'
        "}"
    )

    mock_client.models.generate_content.side_effect = [
        resp_sense,
        resp_verify_1,
        resp_verify_2,
    ]

    spav = SPAVStateMachine(
        robot=robot,
        calibrator=calibrator,
        trajectory_planner=planner,
        max_retries=3,
        genai_client=mock_client,
    )

    dummy_jpeg = b"\xff\xd8\xff\xe0\x00\x10JFIF"
    status_messages: list[str] = []

    res = await spav.execute_spav(
        instruction="Sort the blue bolt",
        get_latest_jpeg_fn=lambda: dummy_jpeg,
        status_callback=status_messages.append,
    )

    assert res.success is True
    assert res.attempts == 2
    assert res.pick_point == [515.0, 480.0]  # 500+15, 500-20
    assert any("Adjusted pick coord" in msg for msg in status_messages)


@pytest.mark.asyncio
async def test_spav_max_retries_exhaustion() -> None:
    """Test SPAV loop safely terminates and parks robot after reaching max retry limit."""
    robot = MockRobot()
    calibrator = ArucoWorkspaceCalibrator()
    planner = TrajectoryPlanner(robot=robot)

    mock_client = MagicMock()
    resp_sense = MagicMock()
    resp_sense.text = (
        "{\n"
        '  "pick_label": "slippery washer",\n'
        '  "pick_point": [300, 300],\n'
        '  "place_label": null,\n'
        '  "place_point": null,\n'
        '  "plan_description": "pick washer"\n'
        "}"
    )
    resp_fail = MagicMock()
    resp_fail.text = (
        "{\n"
        '  "grasp_success": false,\n'
        '  "confidence": 0.90,\n'
        '  "failure_reason": "Slipped out of jaws",\n'
        '  "visual_offset_delta": [0, 0]\n'
        "}"
    )

    # 1 sense + 2 failed verifications for max_retries=2
    mock_client.models.generate_content.side_effect = [
        resp_sense,
        resp_fail,
        resp_fail,
    ]

    spav = SPAVStateMachine(
        robot=robot,
        calibrator=calibrator,
        trajectory_planner=planner,
        max_retries=2,
        genai_client=mock_client,
    )

    dummy_jpeg = b"\xff\xd8\xff\xe0\x00\x10JFIF"
    status_messages: list[str] = []

    res = await spav.execute_spav(
        instruction="Pick the washer",
        get_latest_jpeg_fn=lambda: dummy_jpeg,
        status_callback=status_messages.append,
    )

    assert res.success is False
    assert res.attempts == 2
    assert "Maximum retry limit" in "\n".join(status_messages)


@pytest.mark.asyncio
async def test_tool_spav_dispatch() -> None:
    """Test execute_tool_call correctly handles execute_spav_pick_and_place."""
    robot = MockRobot()
    mock_call = MagicMock()
    mock_call.name = "execute_spav_pick_and_place"
    mock_call.args = {"instruction": "Move the red block"}

    dummy_jpeg = b"\xff\xd8\xff\xe0\x00\x10JFIF"
    status_logs: list[str] = []

    # Mock SPAV execution within execute_tool_call
    from unittest.mock import AsyncMock, patch

    with patch("agent.spav.SPAVStateMachine.execute_spav", new_callable=AsyncMock) as mock_exec:
        from agent.spav import SPAVResult

        mock_exec.return_value = SPAVResult(
            success=True,
            pick_label="red block",
            place_label="bin",
            attempts=1,
            final_message="Done successfully",
        )

        resp = await execute_tool_call(
            call=mock_call,
            robot=robot,
            get_latest_jpeg_fn=lambda: dummy_jpeg,
            status_callback=status_logs.append,
        )

        assert resp["status"] == "success"
        assert resp["pick_label"] == "red block"
        assert resp["attempts"] == 1


@pytest.mark.asyncio
async def test_spav_uncalibrated_blocked() -> None:
    """Test SPAV execution is immediately blocked if workspace is uncalibrated."""
    robot = MockRobot()
    calibrator = ArucoWorkspaceCalibrator()
    calibrator.homography_matrix = None  # Force uncalibrated state

    spav = SPAVStateMachine(robot=robot, calibrator=calibrator)
    dummy_jpeg = b"\xff\xd8\xff\xe0\x00\x10JFIF"
    status_messages: list[str] = []

    res = await spav.execute_spav(
        instruction="Pick the block",
        get_latest_jpeg_fn=lambda: dummy_jpeg,
        status_callback=status_messages.append,
    )

    assert res.success is False
    assert "Workspace Uncalibrated" in res.final_message
    assert any("Workspace Uncalibrated" in msg for msg in status_messages)
