"""Unit tests for VLAController orchestration and streaming loop."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from google.genai import types

from agent.vla_controller import VLAController
from tests.conftest import MockRobot


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
    assert "Move -> Look -> Correct" in prompt
    assert "Conversational Interaction & Narration Rules" in prompt
    assert "move_to_spatial_point" in prompt
    assert "z_height_cm" in prompt
    assert "Step-by-Step Pick-and-Place Workflow with Visual Verification" in prompt
    assert "Clear Occlusion" in prompt
    assert "Relative Position Error" in prompt
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
    mock_session.send_frame = AsyncMock()
    mock_session.send_text = AsyncMock()

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

    fake_jpeg = b"\xff\xd8\xff\xe0mock_jpeg"
    chunks = list(
        controller.process_instruction_stream("Move home", get_latest_jpeg_fn=lambda: fake_jpeg)
    )
    full_output = "".join(chunks)

    assert "Locating red cube..." in full_output
    assert "move_to_preset" in full_output
    assert "completed" in full_output
    assert mock_session.send_frame.called


def test_vla_controller_post_action_frame_delivery() -> None:
    """Test that a post-action observation frame is delivered after a blocking tool call."""
    robot = MockRobot()
    controller = VLAController(mode="sim", robot=robot)
    controller._api_available = True

    mock_session = MagicMock()
    mock_session.is_connected = True
    mock_session.connect = AsyncMock()
    mock_session.send_instruction = AsyncMock()
    mock_session.send_tool_response = AsyncMock()
    mock_session.send_frame = AsyncMock()
    mock_session.send_text = AsyncMock()

    async def mock_receive():
        # Tool call
        yield types.LiveServerMessage(
            tool_call=types.LiveServerToolCall(
                function_calls=[
                    types.FunctionCall(
                        name="open_gripper",
                        args={},
                        id="call_open",
                    )
                ]
            )
        )
        yield types.LiveServerMessage(server_content=types.LiveServerContent(turn_complete=True))

    mock_session.receive = mock_receive
    controller._session = mock_session

    mock_frame = b"\xff\xd8\xff\xe0post_action_jpeg"
    get_frame_mock = MagicMock(return_value=mock_frame)

    list(
        controller.process_instruction_stream("Open the gripper", get_latest_jpeg_fn=get_frame_mock)
    )

    # Verify send_frame was called with mock_frame
    mock_session.send_frame.assert_called_with(mock_frame)
    mock_session.send_tool_response.assert_called_once()


def test_vla_controller_send_message_tool() -> None:
    """Test send_message tool call translates directly into conversational text."""
    controller = VLAController(mode="sim")
    controller._api_available = True

    mock_session = MagicMock()
    mock_session.is_connected = True
    mock_session.connect = AsyncMock()
    mock_session.send_instruction = AsyncMock()
    mock_session.send_tool_response = AsyncMock()
    mock_session.send_frame = AsyncMock()

    async def mock_receive():
        yield types.LiveServerMessage(
            tool_call=types.LiveServerToolCall(
                function_calls=[
                    types.FunctionCall(
                        name="send_message",
                        args={"message": "I see the red cube and will pick it up."},
                        id="call_msg",
                    )
                ]
            )
        )
        yield types.LiveServerMessage(server_content=types.LiveServerContent(turn_complete=True))

    mock_session.receive = mock_receive
    controller._session = mock_session

    chunks = list(controller.process_instruction_stream("Pick up cube"))
    full_output = "".join(chunks)

    assert "I see the red cube and will pick it up." in full_output
    assert "Executing `send_message`" not in full_output
    mock_session.send_tool_response.assert_called_once()


@pytest.mark.asyncio
async def test_vla_controller_background_streamer() -> None:
    """Test background frame streamer sends frames and respects tool execution state."""
    import asyncio

    controller = VLAController(mode="sim")
    mock_session = MagicMock()
    mock_session.send_frame = AsyncMock()
    controller._session = mock_session

    mock_frame = b"mock_bg_frame"
    get_frame = MagicMock(return_value=mock_frame)

    stop_event = asyncio.Event()
    tool_executing_event = asyncio.Event()

    task = asyncio.create_task(
        controller._stream_frames_in_background(get_frame, stop_event, tool_executing_event)
    )

    # Allow one background tick
    await asyncio.sleep(0.05)
    assert mock_session.send_frame.called

    # When tool is executing, send_frame should not be called by background task
    mock_session.send_frame.reset_mock()
    tool_executing_event.set()
    await asyncio.sleep(0.05)
    assert not mock_session.send_frame.called

    # Stop task
    stop_event.set()
    await task


def test_vla_controller_thought_and_narration_streaming() -> None:
    """Test that model thoughts and narration stream properly with formatted markdown."""
    controller = VLAController(mode="sim")
    controller._api_available = True

    mock_session = MagicMock()
    mock_session.is_connected = True
    mock_session.connect = AsyncMock()
    mock_session.send_instruction = AsyncMock()
    mock_session.send_tool_response = AsyncMock()
    mock_session.send_frame = AsyncMock()
    mock_session.send_text = AsyncMock()

    async def mock_receive():
        # Thought part
        yield types.LiveServerMessage(
            server_content=types.LiveServerContent(
                model_turn=types.Content(
                    parts=[
                        types.Part(
                            text="I see the red cube on the table at [650, 420].", thought=True
                        )
                    ]
                )
            )
        )
        # Narration text part
        yield types.LiveServerMessage(
            server_content=types.LiveServerContent(
                model_turn=types.Content(
                    parts=[
                        types.Part.from_text(
                            text="I will pick up the red cube and move it to the target."
                        )
                    ]
                )
            )
        )
        yield types.LiveServerMessage(server_content=types.LiveServerContent(turn_complete=True))

    mock_session.receive = mock_receive
    controller._session = mock_session

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

    mock_session = MagicMock()
    mock_session.is_connected = True
    mock_session.connect = AsyncMock()
    mock_session.send_instruction = AsyncMock()
    mock_session.send_frame = AsyncMock()

    async def mock_receive():
        yield types.LiveServerMessage(
            server_content=types.LiveServerContent(
                model_turn=types.Content(
                    parts=[
                        types.Part.from_text(
                            text=(
                                "Hello! I see a calibrated tabletop with 4 ArUco corner "
                                "fiducials and a red cube."
                            )
                        )
                    ]
                )
            )
        )
        yield types.LiveServerMessage(server_content=types.LiveServerContent(turn_complete=True))

    mock_session.receive = mock_receive
    controller._session = mock_session

    chunks = list(controller.process_instruction_stream("Hello! What do you see?"))
    full_output = "".join(chunks)

    assert "Hello! I see a calibrated tabletop" in full_output
    assert mock_session.send_instruction.called
    assert len(controller.get_history()) == 3


def test_vla_controller_move_look_correct_system_prompt() -> None:
    """Test system prompt rigorously mandates the 3-step Move -> Look -> Correct loop."""
    controller = VLAController(mode="sim")
    prompt = controller.get_history()[0]["content"]

    # Rule 1 (MOVE) assertions
    assert "1. **MOVE (Single Primitive Action)**" in prompt
    assert "Execute exactly ONE physical action per turn" in prompt
    assert "NEVER batch multiple physical commands together" in prompt

    # Rule 2 (LOOK) assertions
    assert "2. **LOOK (Mandatory Visual Observation Pause)**" in prompt
    assert "Gripper Position" in prompt
    assert "Target Position" in prompt
    assert "Relative Position Error" in prompt
    assert "[dy, dx] = [target_y - gripper_y, target_x - gripper_x]" in prompt

    # Rule 3 (CORRECT) assertions
    assert "3. **CORRECT (Alignment Verification or Adjustment)**" in prompt
    assert "If Misaligned (|dy| > 25 or |dx| > 25)" in prompt
    assert "If Aligned (|dy| <= 25 and |dx| <= 25)" in prompt
    assert "hover height before attempting to descend" in prompt


def test_terminal_logging_timestamps_and_formatting(capsys: pytest.CaptureFixture) -> None:
    """Test that terminal logging outputs ISO timestamps and standardized prefixes."""
    import re

    controller = VLAController(mode="sim")
    controller._api_available = True

    mock_session = MagicMock()
    mock_session.is_connected = True
    mock_session.connect = AsyncMock()
    mock_session.send_instruction = AsyncMock()
    mock_session.send_frame = AsyncMock()

    async def mock_receive():
        yield types.LiveServerMessage(
            server_content=types.LiveServerContent(
                model_turn=types.Content(
                    parts=[
                        types.Part(text="Observing target red cube at [650, 420].", thought=True),
                        types.Part.from_text(text="Moving to hover above target."),
                    ]
                )
            )
        )
        yield types.LiveServerMessage(server_content=types.LiveServerContent(turn_complete=True))

    mock_session.receive = mock_receive
    controller._session = mock_session

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
    """Test that tool calls log both arguments payload and execution results to stdout."""
    robot = MockRobot()
    controller = VLAController(mode="sim", robot=robot)
    controller._api_available = True

    mock_session = MagicMock()
    mock_session.is_connected = True
    mock_session.connect = AsyncMock()
    mock_session.send_instruction = AsyncMock()
    mock_session.send_tool_response = AsyncMock()
    mock_session.send_frame = AsyncMock()

    async def mock_receive():
        yield types.LiveServerMessage(
            tool_call=types.LiveServerToolCall(
                function_calls=[
                    types.FunctionCall(
                        name="move_to_preset",
                        args={"pose_name": "HOME"},
                        id="call_test_123",
                    )
                ]
            )
        )
        yield types.LiveServerMessage(server_content=types.LiveServerContent(turn_complete=True))

    mock_session.receive = mock_receive
    controller._session = mock_session

    list(controller.process_instruction_stream("Reset home"))
    captured = capsys.readouterr()

    # Check tool call dispatch logging with full payload
    assert "[TOOL_CALL] 🔧 Dispatching 'move_to_preset' [ID: call_test_123]" in captured.out
    assert "'pose_name': 'HOME'" in captured.out

    # Check tool result logging with full payload
    assert "[TOOL_RESULT] ✅ Tool 'move_to_preset' completed [ID: call_test_123]" in captured.out
    assert "'status': 'success'" in captured.out
    assert "'pose_name': 'HOME'" in captured.out


def test_terminal_logging_error_routing_stderr(capsys: pytest.CaptureFixture) -> None:
    """Test that diagnostic errors and failed tool results route to stderr with timestamps."""
    import re

    robot = MockRobot()
    controller = VLAController(mode="sim", robot=robot)
    controller._api_available = True

    mock_session = MagicMock()
    mock_session.is_connected = True
    mock_session.connect = AsyncMock()
    mock_session.send_instruction = AsyncMock()
    mock_session.send_tool_response = AsyncMock()
    mock_session.send_frame = AsyncMock()

    async def mock_receive():
        yield types.LiveServerMessage(
            tool_call=types.LiveServerToolCall(
                function_calls=[
                    types.FunctionCall(
                        name="move_to_preset",
                        args={"pose_name": "INVALID_POSE"},
                        id="call_err_456",
                    )
                ]
            )
        )
        yield types.LiveServerMessage(server_content=types.LiveServerContent(turn_complete=True))

    mock_session.receive = mock_receive
    controller._session = mock_session

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

    mock_session = MagicMock()
    mock_session.is_connected = True
    mock_session.connect = AsyncMock()
    mock_session.send_instruction = AsyncMock()
    mock_session.send_tool_response = AsyncMock()
    mock_session.send_frame = AsyncMock()

    async def mock_receive():
        yield types.LiveServerMessage(
            tool_call=types.LiveServerToolCall(
                function_calls=[
                    types.FunctionCall(
                        name="open_gripper",
                        args={},
                        id="call_open_hint",
                    )
                ]
            )
        )
        yield types.LiveServerMessage(server_content=types.LiveServerContent(turn_complete=True))

    mock_session.receive = mock_receive
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
    assert "Gripper opened. Inspect camera frame" in resp_dict["visual_hint"]
