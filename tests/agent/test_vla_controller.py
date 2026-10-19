"""Unit tests for VLAController orchestration and streaming loop."""

import re
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from google.genai import types

from agent.vla_controller import VLAController, load_system_prompt
from robot.errors import CalibrationUnavailable
from tests.conftest import MockRobot
from tests.harness.live_session import (
    fake_live_session,
    text_message,
    thought_message,
    tool_call_message,
    turn_complete_message,
)


def test_vla_controller_init() -> None:
    """Test VLAController initializes and load_system_prompt renders expected sections."""
    robot = MockRobot()
    controller = VLAController(mode="sim", robot=robot)
    assert controller.mode == "sim"
    prompt = load_system_prompt("sim")

    # Verify system prompt components
    assert "Gemini Robotics Assistant" in prompt
    assert "'SIM' mode" in prompt
    assert "Move -> Look -> Correct" in prompt
    assert "Conversational Interaction & Narration Rules" in prompt
    assert "move_to_spatial_point" in prompt
    assert "z_height_cm" in prompt
    assert "Step-by-Step Pick-and-Place Workflow with Visual Verification" in prompt
    assert "Clear Occlusion" in prompt
    assert "measured_offset" in prompt
    assert "execute_spav_pick_and_place" not in prompt


def test_vla_controller_history_management() -> None:
    """Test clear_history disconnects active session on the background event loop."""
    import asyncio

    controller = VLAController(mode="sim")
    mock_session = MagicMock()
    mock_session.disconnect = AsyncMock()
    controller._session = mock_session

    controller.clear_history()
    asyncio.run_coroutine_threadsafe(asyncio.sleep(0), controller._loop).result(timeout=1.0)
    mock_session.disconnect.assert_called_once()


def test_vla_controller_api_unavailable() -> None:
    """Test fallback behavior when Gemini API is unavailable."""
    controller = VLAController(mode="sim")
    controller._api_available = False

    chunks = list(controller.process_instruction_stream("Pick up the red cube"))
    assert len(chunks) == 1
    assert "Gemini API unavailable" in chunks[0]


def test_vla_controller_calibration_gate() -> None:
    """Test calibration gate blocks physical execution when uncalibrated."""
    controller_real = VLAController(mode="real")
    controller_real._api_available = True

    with patch(
        "agent.vla_controller.get_ready_calibrator",
        side_effect=CalibrationUnavailable("Workspace is not calibrated"),
    ):
        chunks = list(controller_real.process_instruction_stream("Pick up the cube"))
        assert len(chunks) == 1
        assert "Action Blocked — Workspace Not Calibrated" in chunks[0]


def test_vla_controller_empty_instruction() -> None:
    """Test empty instruction yields nothing."""
    controller = VLAController(mode="sim")

    chunks = list(controller.process_instruction_stream("   "))
    assert len(chunks) == 0


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


def test_vla_controller_send_message_tool() -> None:
    """Test send_message tool call translates directly into conversational text."""
    controller = VLAController(mode="sim")
    controller._api_available = True

    messages = [
        tool_call_message(
            "send_message",
            {"message": "I see the red cube and will pick it up."},
            "call_msg",
        ),
        turn_complete_message(),
    ]
    mock_session = fake_live_session(messages)
    controller._session = mock_session

    chunks = list(controller.process_instruction_stream("Pick up cube"))
    full_output = "".join(chunks)

    assert "I see the red cube and will pick it up." in full_output
    assert "Executing `send_message`" not in full_output
    mock_session.send_tool_response.assert_called_once()


def test_vla_controller_thought_and_narration_streaming() -> None:
    """Test that model thoughts and narration stream properly with formatted markdown."""
    controller = VLAController(mode="sim")
    controller._api_available = True

    messages = [
        thought_message("I see the red cube on the table at [650, 420]."),
        text_message("I will pick up the red cube and move it to the target."),
        turn_complete_message(),
    ]
    controller._session = fake_live_session(messages)

    chunks = list(controller.process_instruction_stream("Pick up the red cube"))
    full_output = "".join(chunks)

    # Check thought formatting
    assert "> 💭 *" in full_output
    assert "I see the red cube on the table at [650, 420]." in full_output
    # Check narration
    assert "I will pick up the red cube and move it to the target." in full_output


def test_vla_controller_conversational_dialogue() -> None:
    """Test conversational response when user asks a general question without tool calls."""
    controller = VLAController(mode="sim")
    controller._api_available = True

    messages = [
        text_message(
            "Hello! I see a calibrated tabletop with 4 ArUco corner fiducials and a red cube."
        ),
        turn_complete_message(),
    ]
    mock_session = fake_live_session(messages)
    controller._session = mock_session

    chunks = list(controller.process_instruction_stream("Hello! What do you see?"))
    full_output = "".join(chunks)

    assert "Hello! I see a calibrated tabletop" in full_output
    assert mock_session.send_instruction.called


def test_vla_controller_move_look_correct_system_prompt() -> None:
    """Test system prompt rigorously mandates the 3-step Move -> Look -> Correct loop."""
    prompt = load_system_prompt("sim")

    # Rule 1 (MOVE) assertions
    assert "1. **MOVE (Single Primitive Action)**" in prompt
    assert "Execute exactly ONE physical action per turn" in prompt
    assert "NEVER batch multiple physical commands together" in prompt

    # Rule 2 (LOOK) assertions: the controller measures, the model reads the measurement
    assert "2. **LOOK (Mandatory Visual Observation Pause)**" in prompt
    assert "The controller, not you, measures gripper alignment" in prompt
    assert "`perception.measured_offset`" in prompt
    assert "NEVER estimate the gripper-to-target offset from the image" in prompt
    assert "NEVER offset a point yourself" in prompt
    assert "[dy, dx]" not in prompt  # no more image-space offsets for the model to compute

    # Rule 3 (CORRECT) assertions
    assert "3. **CORRECT (Alignment Verification or Adjustment)**" in prompt
    assert "If `perception.is_aligned` is `false` (offset more than 4 mm)" in prompt
    assert "If `perception.is_aligned` is `true` (within 4 mm)" in prompt
    assert "re-issue the SAME hover call" in prompt
    assert "hover height before attempting to descend" in prompt
    assert "Descending to grasp" in prompt
    assert "Only proceed to descend when `is_aligned` is `true`" in prompt
    assert "z_height_cm=3.0" in prompt


def test_terminal_logging_timestamps_and_formatting(capsys: pytest.CaptureFixture) -> None:
    """Test that terminal logging outputs ISO timestamps and standardized prefixes."""
    controller = VLAController(mode="sim")
    controller._api_available = True

    messages = [
        types.LiveServerMessage(
            server_content=types.LiveServerContent(
                model_turn=types.Content(
                    parts=[
                        types.Part(text="Observing target red cube at [650, 420].", thought=True),
                        types.Part.from_text(text="Moving to hover above target."),
                    ]
                )
            )
        ),
        turn_complete_message(),
    ]
    controller._session = fake_live_session(messages)

    list(controller.process_instruction_stream("Pick up cube"))
    captured = capsys.readouterr()

    # Pattern for timestamp prefix: [YYYY-MM-DD HH:MM:SS.mmm]
    timestamp_pattern = re.compile(
        r"\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}\] \[(?:VLAController|LiveSession|Tools)\] "
        r"\[(?:DECISION|REASONING|INFO|TOOL_CALL|TOOL_RESULT|STREAM|WARN|ERROR)\]"
    )

    matches = timestamp_pattern.findall(captured.out)
    assert len(matches) > 0, f"No timestamped log entries found in stdout:\n{captured.out}"
    assert "[DECISION] 🚀 Received user instruction" in captured.out
    assert "[REASONING] 💭 Observing target red cube" in captured.out
    assert "[DECISION] 📝 Model text: Moving to hover" in captured.out


def test_terminal_logging_tool_call_payload_and_result(capsys: pytest.CaptureFixture) -> None:
    """Test that tool calls log payload/result to stdout and never leak raw dicts into chat."""
    robot = MockRobot()
    controller = VLAController(mode="sim", robot=robot)
    controller._api_available = True

    messages = [
        tool_call_message("move_to_preset", {"pose_name": "HOME"}, "call_test_123"),
        turn_complete_message(),
    ]
    controller._session = fake_live_session(messages)

    chunks = list(controller.process_instruction_stream("Reset home"))
    chat_output = "".join(chunks)
    captured = capsys.readouterr()

    # Check tool call dispatch logging with full payload
    assert "[TOOL_CALL] 🔧 Dispatching 'move_to_preset' [ID: call_test_123]" in captured.out
    assert "'pose_name': 'HOME'" in captured.out

    # Check tool result logging with full payload
    assert "[TOOL_RESULT] ✅ Tool 'move_to_preset' completed [ID: call_test_123]" in captured.out
    assert "'status': 'success'" in captured.out
    assert "'pose_name': 'HOME'" in captured.out

    # Chat output must NOT contain raw Python object addresses or unformatted dicts
    assert "object at 0x" not in chat_output
    assert "'executed_qpos_deg': [" not in chat_output


def test_terminal_logging_error_routing_stderr(capsys: pytest.CaptureFixture) -> None:
    """Test that diagnostic errors and failed tool results route to stderr with timestamps."""
    robot = MockRobot()
    controller = VLAController(mode="sim", robot=robot)
    controller._api_available = True

    messages = [
        tool_call_message("move_to_preset", {"pose_name": "INVALID_POSE"}, "call_err_456"),
        turn_complete_message(),
    ]
    controller._session = fake_live_session(messages)

    list(controller.process_instruction_stream("Do invalid move"))
    captured = capsys.readouterr()

    # Stderr should contain the error log from Tools and VLAController
    assert "[Tools] [ERROR] Unknown preset pose: INVALID_POSE" in captured.err
    assert "[VLAController] [TOOL_RESULT] ⚠️ Tool 'move_to_preset' completed" in captured.err
    assert "INVALID_POSE" in captured.err

    # Check stderr timestamp formatting
    timestamp_pattern = re.compile(
        r"\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}\] \[(?:VLAController|Tools)\] "
        r"\[(?:TOOL_RESULT|ERROR)\]"
    )
    assert timestamp_pattern.search(captured.err) is not None


def test_vla_controller_visual_hint_in_tool_response() -> None:
    """Test that tool responses sent back to the model contain enriched visual hints."""
    robot = MockRobot()
    controller = VLAController(mode="sim", robot=robot)
    controller._api_available = True

    messages = [
        tool_call_message("open_gripper", {}, "call_open_hint"),
        turn_complete_message(),
    ]
    mock_session = fake_live_session(messages)
    controller._session = mock_session

    list(controller.process_instruction_stream("Open gripper"))

    mock_session.send_tool_response.assert_called_once()
    call_args = mock_session.send_tool_response.call_args
    called_responses = (
        call_args[1].get("function_responses")
        if call_args[1] and "function_responses" in call_args[1]
        else call_args[0][0]
    )
    assert len(called_responses) == 1
    resp_dict = called_responses[0].response
    assert "visual_hint" in resp_dict
    assert "Gripper opened (aperture" in resp_dict["visual_hint"]
