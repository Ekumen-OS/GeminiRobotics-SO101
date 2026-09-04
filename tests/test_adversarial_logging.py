"""Adversarial stress-test suite for Verbose Diagnostic Terminal Logging (Requirement 3).

This suite tests:
1. Strict ISO-8601 millisecond timestamp and [MODULE] [TAG] prefix formatting.
2. Full argument payloads (call.args) and result dictionaries (result_dict) in stdout/stderr.
3. Strict stderr routing for IK errors, joint clamping warnings, tool errors, and API exceptions.
4. Clean separation between terminal diagnostic traces and Chatbot UI Markdown history.
"""

import re
from unittest.mock import AsyncMock, MagicMock, patch

import gradio as gr
import numpy as np
import pytest
from google.genai import types

import agent.live_session as live_session_mod
import agent.tools as tools_mod
import agent.vla_controller as vla_mod
from agent.live_session import GeminiLiveSession
from agent.tools import (
    execute_tool_call,
    validate_and_clamp_joints,
)
from agent.vla_controller import VLAController
from robot.constants import JOINT_LIMITS
from robot.interface import RobotInterface
from ui.components.chat import build_chat_interface, wire_chat_events

# Strict regex matching: [YYYY-MM-DD HH:MM:SS.mmm] [MODULE] [TAG] message
LOG_REGEX = re.compile(
    r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})\] \[(\w+)\] \[([\w:]+)\] (.*)$"
)


class MockRobot(RobotInterface):
    """Mock robot for adversarial logging tests."""

    def __init__(self, qpos: np.ndarray | None = None) -> None:
        """Initialize mock robot."""
        self.qpos = qpos if qpos is not None else np.zeros(6, dtype=np.float32)
        self.move_to_pose_called = False
        self._stop_requested = False

    def set_joint_positions(self, qpos: np.ndarray) -> None:
        """Set joint positions."""
        self.qpos = np.asarray(qpos, dtype=np.float32)

    def get_joint_positions(self) -> np.ndarray:
        """Get joint positions."""
        return self.qpos.copy()

    def move_to_pose(
        self, target_qpos: np.ndarray, duration_sec: float = 1.0, dt: float = 0.002
    ) -> None:
        """Simulate moving to pose."""
        self.qpos = np.asarray(target_qpos, dtype=np.float32)
        self.move_to_pose_called = True

    def get_end_effector_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Return end effector pose."""
        return np.array([0.18, 0.0, 0.05]), np.eye(3)


class DummyCall:
    """Helper mock for Live API ToolCall."""

    def __init__(self, name: str, args: dict | None = None, call_id: str = "call_test_id") -> None:
        """Initialize dummy call."""
        self.name = name
        self.args = args or {}
        self.id = call_id


# ==============================================================================
# 1. Format and Timestamp Conformance Tests
# ==============================================================================


@pytest.mark.parametrize(
    "mod,mod_name",
    [
        (vla_mod, "VLAController"),
        (live_session_mod, "LiveSession"),
        (tools_mod, "Tools"),
    ],
)
@pytest.mark.parametrize(
    "tag,expected_stream",
    [
        ("INFO", "stdout"),
        ("DECISION", "stdout"),
        ("REASONING", "stdout"),
        ("TOOL_CALL", "stdout"),
        ("TOOL_RESULT", "stdout"),
        ("STREAM", "stdout"),
        ("EXEC", "stdout"),
        ("WARN", "stderr"),
        ("ERROR", "stderr"),
    ],
)
def test_log_format_and_stream_conformance(
    capsys: pytest.CaptureFixture, mod, mod_name: str, tag: str, expected_stream: str
) -> None:
    """Verify that _log across all modules produces exact ISO-8601 timestamped format."""
    test_msg = f"Test message for {mod_name} with tag {tag}"
    mod._log(mod_name, tag, test_msg)

    captured = capsys.readouterr()
    output = captured.err if expected_stream == "stderr" else captured.out
    other_output = captured.out if expected_stream == "stderr" else captured.err

    assert other_output == "", f"Unexpected output in opposite stream: {other_output}"
    assert output.strip() != "", "Expected output stream is empty"

    lines = output.strip().split("\n")
    assert len(lines) == 1

    match = LOG_REGEX.match(lines[0])
    assert match is not None, f"Log output does not match required format: {lines[0]}"

    timestamp_str, module_logged, tag_logged, msg_logged = match.groups()

    # Check timestamp components
    assert len(timestamp_str.split(".")[1]) == 3, "Millisecond precision must be exactly 3 digits"
    assert module_logged == mod_name
    assert tag_logged == tag
    assert msg_logged == test_msg


# ==============================================================================
# 2. Tool Dispatch and Result Payload Logging (Stdout)
# ==============================================================================


def test_tool_dispatch_and_result_payload_logging(capsys: pytest.CaptureFixture) -> None:
    """Verify full argument and result payload logging during successful tool execution."""
    robot = MockRobot()
    controller = VLAController(mode="sim", robot=robot)
    controller._api_available = True

    mock_session = MagicMock()
    mock_session.is_connected = True
    mock_session.connect = AsyncMock()
    mock_session.send_instruction = AsyncMock()
    mock_session.send_tool_response = AsyncMock()
    mock_session.send_frame = AsyncMock()

    # Multi-tool message sequence
    async def mock_receive():
        yield types.LiveServerMessage(
            tool_call=types.LiveServerToolCall(
                function_calls=[
                    types.FunctionCall(
                        name="move_to_pose",
                        args={
                            "qpos": [10.0, 20.0, -30.0, 40.0, -50.0, 60.0],
                            "duration_sec": 1.5,
                        },
                        id="call_pose_99",
                    ),
                    types.FunctionCall(
                        name="send_message",
                        args={"message": "Moved to intermediate posture successfully."},
                        id="call_msg_100",
                    ),
                ]
            )
        )
        yield types.LiveServerMessage(server_content=types.LiveServerContent(turn_complete=True))

    mock_session.receive = mock_receive
    controller._session = mock_session

    # Clear any initialization logs (e.g. sim calibration)
    capsys.readouterr()

    list(controller.process_instruction_stream("Move to designated posture"))
    captured = capsys.readouterr()

    # Verify move_to_pose tool dispatch log
    assert "[VLAController] [TOOL_CALL] 🔧 Dispatching 'move_to_pose'" in captured.out
    assert "[ID: call_pose_99]" in captured.out
    assert "'qpos': [10.0, 20.0, -30.0, 40.0, -50.0, 60.0]" in captured.out
    assert "'duration_sec': 1.5" in captured.out

    # Verify move_to_pose tool result log
    assert "[VLAController] [TOOL_RESULT] ✅ Tool 'move_to_pose' completed" in captured.out
    assert "[ID: call_pose_99]" in captured.out
    assert "'status': 'success'" in captured.out
    assert "'executed_qpos_deg': [10.0, 20.0, -30.0, 40.0, -50.0, 60.0]" in captured.out

    # Verify send_message tool dispatch & result
    assert "[VLAController] [TOOL_CALL] 🔧 Dispatching 'send_message'" in captured.out
    assert "[ID: call_msg_100]" in captured.out
    assert "Moved to intermediate posture successfully." in captured.out
    assert "[VLAController] [TOOL_RESULT] ✅ Tool 'send_message' completed" in captured.out
    assert "[ID: call_msg_100]" in captured.out

    # All stdout lines from agent execution must conform to LOG_REGEX
    for line in captured.out.strip().split("\n"):
        if line.strip():
            assert LOG_REGEX.match(line) is not None, f"Non-conforming stdout line: {line}"


# ==============================================================================
# 3. Diagnostic Error Routing (Stderr)
# ==============================================================================


def test_tool_failure_routes_strictly_to_stderr(capsys: pytest.CaptureFixture) -> None:
    """Verify tool execution failures, unknown tools, and invalid schemas route to stderr."""
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
                        name="move_to_pose",
                        args={"qpos": [10.0, 20.0]},  # Invalid length
                        id="call_fail_01",
                    ),
                    types.FunctionCall(
                        name="nonexistent_gripper_action",
                        args={"param": 123},
                        id="call_fail_02",
                    ),
                ]
            )
        )
        yield types.LiveServerMessage(server_content=types.LiveServerContent(turn_complete=True))

    mock_session.receive = mock_receive
    controller._session = mock_session

    capsys.readouterr()
    list(controller.process_instruction_stream("Run failing actions"))
    captured = capsys.readouterr()

    # Tool dispatch logs are still logged to stdout
    assert "[VLAController] [TOOL_CALL] 🔧 Dispatching 'move_to_pose'" in captured.out
    assert "[ID: call_fail_01]" in captured.out
    assert "[VLAController] [TOOL_CALL] 🔧 Dispatching 'nonexistent_gripper_action'" in captured.out
    assert "[ID: call_fail_02]" in captured.out

    # Error diagnostics and tool failure result must route to STDERR
    assert "[Tools] [ERROR] move_to_pose expects 6 joint angles, got 2" in captured.err
    assert "[VLAController] [TOOL_RESULT] ⚠️ Tool 'move_to_pose' completed" in captured.err
    assert "[ID: call_fail_01]" in captured.err
    assert "[Tools] [ERROR] Unknown tool called: nonexistent_gripper_action" in captured.err
    assert (
        "[VLAController] [TOOL_RESULT] ⚠️ Tool 'nonexistent_gripper_action' completed"
        in captured.err
    )
    assert "[ID: call_fail_02]" in captured.err

    # Verify all stderr lines match format
    for line in captured.err.strip().split("\n"):
        if line.strip():
            assert LOG_REGEX.match(line) is not None, f"Non-conforming stderr line: {line}"


def test_joint_clamping_warning_routes_to_stderr(capsys: pytest.CaptureFixture) -> None:
    """Verify joint clamping generates standard [WARN] messages to stderr."""
    pan_max = JOINT_LIMITS["shoulder_pan"][1]
    lift_min = JOINT_LIMITS["shoulder_lift"][0]

    qpos_exceeded = [pan_max + 30.0, lift_min - 25.0, 0.0, 0.0, 0.0, 0.0]
    clamped, warnings = validate_and_clamp_joints(qpos_exceeded)

    assert clamped[0] == pan_max
    assert clamped[1] == lift_min
    assert len(warnings) == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    stderr_lines = [line for line in captured.err.strip().split("\n") if line.strip()]
    assert len(stderr_lines) == 2

    for line in stderr_lines:
        match = LOG_REGEX.match(line)
        assert match is not None
        _, mod, tag, msg = match.groups()
        assert mod == "Tools"
        assert tag == "WARN"
        assert "was clamped from" in msg


@pytest.mark.asyncio
async def test_ik_workspace_uncalibrated_error_routes_to_stderr(
    capsys: pytest.CaptureFixture,
) -> None:
    """Verify uncalibrated spatial movement throws and logs [ERROR] to stderr."""
    robot = MockRobot()
    mock_calibrator = MagicMock()
    mock_calibrator.is_calibrated = False

    with patch("ui.stream.get_calibrator", return_value=mock_calibrator):
        call = DummyCall(
            "move_to_spatial_point",
            {"point": [500.0, 500.0], "z_height_cm": 3.0, "approach": "top_down"},
        )
        res = await execute_tool_call(call, robot=robot)

    assert "error" in res
    captured = capsys.readouterr()
    assert "[Tools] [ERROR]" in captured.err
    assert "Workspace is not calibrated" in captured.err

    for line in captured.err.strip().split("\n"):
        if line.strip():
            assert LOG_REGEX.match(line) is not None


# ==============================================================================
# 4. LiveSession Lifecycle Logging
# ==============================================================================


@pytest.mark.asyncio
async def test_live_session_lifecycle_logging(capsys: pytest.CaptureFixture) -> None:
    """Verify LiveSession connection, disconnection, instruction, and streaming logs."""
    session = GeminiLiveSession(
        model="gemini-robotics-er-2-streaming-preview",
        tools=[],
        system_instruction="System prompt",
    )

    mock_client = MagicMock()
    mock_aio_live = MagicMock()
    mock_ws = AsyncMock()
    mock_cm = AsyncMock()
    mock_cm.__aenter__.return_value = mock_ws
    mock_cm.__aexit__.return_value = None
    mock_aio_live.connect.return_value = mock_cm
    mock_client.aio.live = mock_aio_live
    session.client = mock_client

    # 1. Connect
    await session.connect()
    cap_connect = capsys.readouterr()
    assert "[LiveSession] [INFO] Connecting to gemini-robotics-er-2" in cap_connect.out
    assert "[LiveSession] [INFO] Connected successfully." in cap_connect.out

    # 2. Send instruction with frame
    fake_frame = b"\xff\xd8\xff\xe0mock_frame_bytes"
    await session.send_instruction(text="Pick red cube", frame_jpeg=fake_frame)
    cap_send = capsys.readouterr()
    expected_stream_msg = (
        f"[LiveSession] [STREAM] 📸 Sent initial observation frame ({len(fake_frame)} bytes)"
    )
    assert expected_stream_msg in cap_send.out
    assert "[LiveSession] [INFO] Sending text: 'Pick red cube'" in cap_send.out

    # 3. Send video frame
    await session.send_frame(fake_frame)
    cap_frame = capsys.readouterr()
    assert f"[LiveSession] [STREAM] 📸 Sent video frame ({len(fake_frame)} bytes)" in cap_frame.out

    # 4. Send tool response
    resp = types.FunctionResponse(name="move_to_preset", id="id1", response={"status": "success"})
    await session.send_tool_response([resp])
    cap_tool = capsys.readouterr()
    assert "[LiveSession] [INFO] Sending tool responses:" in cap_tool.out

    # 5. Disconnect
    await session.disconnect()
    cap_disc = capsys.readouterr()
    assert "[LiveSession] [INFO] Disconnecting..." in cap_disc.out
    assert "[LiveSession] [INFO] Disconnected." in cap_disc.out


# ==============================================================================
# 5. Clean Isolation: Terminal Logs vs Chatbot UI Markdown History
# ==============================================================================


def test_chatbot_ui_history_remains_clean_of_diagnostic_logs(
    capsys: pytest.CaptureFixture,
) -> None:
    """Verify that Chatbot Markdown history contains ZERO terminal logs, timestamps, or raw JSON."""
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
        # 1. Thought reasoning
        yield types.LiveServerMessage(
            server_content=types.LiveServerContent(
                model_turn=types.Content(
                    parts=[
                        types.Part(
                            text="Analyzing scene. Target red cube at [650, 420].",
                            thought=True,
                        )
                    ]
                )
            )
        )
        # 2. Conversational narration
        yield types.LiveServerMessage(
            server_content=types.LiveServerContent(
                model_turn=types.Content(
                    parts=[
                        types.Part.from_text(
                            text="I will move the arm to HOME pose first to ensure full visibility."
                        )
                    ]
                )
            )
        )
        # 3. Tool call execution
        yield types.LiveServerMessage(
            tool_call=types.LiveServerToolCall(
                function_calls=[
                    types.FunctionCall(
                        name="move_to_preset",
                        args={"pose_name": "HOME"},
                        id="call_clean_check_1",
                    )
                ]
            )
        )
        # 4. Post-tool narration
        yield types.LiveServerMessage(
            server_content=types.LiveServerContent(
                model_turn=types.Content(
                    parts=[
                        types.Part.from_text(
                            text="Arm is at HOME. Now descending to hover above the target."
                        )
                    ]
                )
            )
        )
        yield types.LiveServerMessage(server_content=types.LiveServerContent(turn_complete=True))

    mock_session.receive = mock_receive
    controller._session = mock_session

    with gr.Blocks():
        prompt_input, chatbot, status_banner, exec_btn, home_btn, clear_btn, rand_btn = (
            build_chat_interface(controller=controller, api_available=True)
        )
        sliders = [gr.Slider(value=0.0) for _ in range(6)]
        handlers = wire_chat_events(
            controller,
            prompt_input,
            chatbot,
            status_banner,
            exec_btn,
            home_btn,
            clear_btn,
            sliders,
            rand_btn=rand_btn,
        )

    process_fn = handlers["process_instruction"]
    initial_ui_history = [
        {"role": "assistant", "content": "🤖 **System Active**: Ready in `SIM` mode."}
    ]

    capsys.readouterr()
    with patch(
        "ui.components.chat.get_latest_raw_jpeg",
        return_value=b"\xff\xd8\xff\xe0mock_clean_frame",
    ):
        all_ui_yields = list(process_fn("Pick up the red cube", initial_ui_history))
    assert len(all_ui_yields) > 1

    # Extract final Chatbot history
    final_ui_history = all_ui_yields[-1][1]

    # Verify UI history structure
    assert len(final_ui_history) == 3
    assert final_ui_history[0]["role"] == "assistant"
    assert final_ui_history[1]["role"] == "user"
    assert final_ui_history[1]["content"] == "Pick up the red cube"

    assistant_message = final_ui_history[2]["content"]

    # Verify Assistant message contains expected user-facing elements
    assert "> 💭 *Analyzing scene. Target red cube at [650, 420].*" in assistant_message
    assert "I will move the arm to HOME pose" in assistant_message
    assert "⚙️ Executing `move_to_preset`..." in assistant_message
    assert "✅ `move_to_preset` completed." in assistant_message
    assert "Arm is at HOME. Now descending to hover" in assistant_message

    # Adversarial verification: Chatbot markdown MUST NOT contain any terminal log artifacts
    assert "[" not in assistant_message or not re.search(
        r"\[\d{4}-\d{2}-\d{2}", assistant_message
    ), "Found ISO timestamp in Chatbot UI history!"
    assert "[VLAController]" not in assistant_message
    assert "[LiveSession]" not in assistant_message
    assert "[Tools]" not in assistant_message
    assert "[TOOL_CALL]" not in assistant_message
    assert "[TOOL_RESULT]" not in assistant_message
    assert "[DECISION]" not in assistant_message
    assert "[REASONING]" not in assistant_message
    assert "[STREAM]" not in assistant_message
    assert "call_clean_check_1" not in assistant_message
    assert "'pose_name': 'HOME'" not in assistant_message

    # Verify terminal received the full diagnostic logs
    captured = capsys.readouterr()
    assert (
        "[VLAController] [TOOL_CALL] 🔧 Dispatching 'move_to_preset' [ID: call_clean_check_1]"
        in captured.out
    )
    assert (
        "[VLAController] [TOOL_RESULT] ✅ Tool 'move_to_preset' completed [ID: call_clean_check_1]"
        in captured.out
    )
