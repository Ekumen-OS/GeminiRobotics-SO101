"""Unit tests for VLAController orchestration and streaming loop."""

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
from google.genai import types

from agent.vla_controller import VLAController
from robot.interface import RobotInterface


class MockRobot(RobotInterface):
    """Mock robot for testing VLAController."""

    def __init__(self) -> None:
        """Initialize mock robot."""
        self.qpos = np.zeros(6, dtype=np.float32)
        self._stop_requested = False

    def set_joint_positions(self, qpos: np.ndarray) -> None:
        """Set joint positions."""
        self.qpos = np.asarray(qpos, dtype=np.float32)

    def get_joint_positions(self) -> np.ndarray:
        """Get current joint positions."""
        return self.qpos.copy()

    def interrupt(self) -> None:
        """Set stop requested flag."""
        self._stop_requested = True


def test_vla_controller_init() -> None:
    """Test VLAController initializes with correct prompt and history."""
    robot = MockRobot()
    controller = VLAController(mode="sim", robot=robot)

    history = controller.get_history()
    assert len(history) == 1
    assert history[0]["role"] == "system"
    prompt = history[0]["content"]

    # Verify system prompt components
    assert "Gemini Robotics Assistant" in prompt
    assert "'SIM' mode" in prompt
    assert "move_to_spatial_point" in prompt
    assert "z_height_cm" in prompt
    assert "Pick-and-Place Orchestration Pattern" in prompt
    assert "Closed-Loop Feedback & Verification Rules" in prompt
    assert "execute_spav_pick_and_place" not in prompt


def test_vla_controller_history_management() -> None:
    """Test get_history and clear_history preserve system prompt."""
    controller = VLAController(mode="sim")
    controller.history.append({"role": "user", "content": "hello"})
    controller.history.append({"role": "assistant", "content": "hi"})
    assert len(controller.get_history()) == 3

    controller.clear_history()
    history = controller.get_history()
    assert len(history) == 1
    assert history[0]["role"] == "system"


def test_vla_controller_api_unavailable() -> None:
    """Test fallback behavior when Gemini API is unavailable."""
    controller = VLAController(mode="sim")
    controller._api_available = False

    chunks = list(controller.process_instruction_stream("Pick up the red cube"))
    assert len(chunks) == 1
    assert "Gemini API unavailable" in chunks[0]


def test_vla_controller_calibration_gate() -> None:
    """Test calibration gate blocks physical/twin execution when uncalibrated."""
    controller_real = VLAController(mode="real")
    controller_real._api_available = True

    mock_calibrator = MagicMock()
    mock_calibrator.is_calibrated = False

    with patch("ui.stream.get_calibrator", return_value=mock_calibrator):
        chunks = list(controller_real.process_instruction_stream("Pick up the cube"))
        assert len(chunks) == 1
        assert "Action Blocked — Workspace Not Calibrated" in chunks[0]


def test_vla_controller_empty_instruction() -> None:
    """Test empty instruction yields nothing and does not modify history."""
    controller = VLAController(mode="sim")
    initial_len = len(controller.get_history())

    chunks = list(controller.process_instruction_stream("   "))
    assert len(chunks) == 0
    assert len(controller.get_history()) == initial_len


def test_vla_controller_interrupt() -> None:
    """Test interrupt halts robot and cancels ongoing task."""
    robot = MockRobot()
    controller = VLAController(mode="sim", robot=robot)

    mock_task = MagicMock()
    mock_task.done.return_value = False
    controller._current_async_task = mock_task

    controller.interrupt()
    assert robot._stop_requested is True
    mock_task.cancel.assert_called_once()


def test_vla_controller_process_instruction_sync() -> None:
    """Test process_instruction synchronous convenience wrapper."""
    controller = VLAController(mode="sim")
    controller._api_available = False

    res = controller.process_instruction("test instruction")
    assert "Gemini API unavailable" in res


def test_vla_controller_stream_processing_mocked() -> None:
    """Test end-to-end streaming processing with mocked live session."""
    robot = MockRobot()
    controller = VLAController(mode="sim", robot=robot)
    controller._api_available = True

    mock_session = MagicMock()
    mock_session.is_connected = True
    mock_session.connect = AsyncMock()
    mock_session.send_instruction = AsyncMock()
    mock_session.send_tool_response = AsyncMock()

    # Generate mock live server messages
    async def mock_receive():
        # Text part
        yield types.LiveServerMessage(
            server_content=types.LiveServerContent(
                model_turn=types.Content(parts=[types.Part.from_text(text="Locating red cube...")])
            )
        )
        # Tool call part
        yield types.LiveServerMessage(
            tool_call=types.LiveServerToolCall(
                function_calls=[
                    types.FunctionCall(
                        name="move_to_preset",
                        args={"pose_name": "HOME"},
                        id="call_1",
                    )
                ]
            )
        )
        # Turn complete
        yield types.LiveServerMessage(server_content=types.LiveServerContent(turn_complete=True))

    mock_session.receive = mock_receive
    controller._session = mock_session

    chunks = list(controller.process_instruction_stream("Move home"))
    full_output = "".join(chunks)

    assert "Locating red cube..." in full_output
    assert "move_to_preset" in full_output
    assert "completed" in full_output
