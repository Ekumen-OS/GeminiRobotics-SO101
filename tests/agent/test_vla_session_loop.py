"""Tests for the VLA Live API streaming session loop, tool batching, and frame delivery."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from google.genai import types

from agent.vla_controller import VLAController
from robot.settings import Settings, set_settings
from tests.conftest import MockRobot, _real_sleep
from tests.harness.live_session import (
    fake_live_session,
    text_message,
    tool_call_message,
    turn_complete_message,
)


def test_send_tool_response_single_call_batching() -> None:
    """Verify send_tool_response is called once per tool-call message with all responses."""
    robot = MockRobot()
    controller = VLAController(mode="sim", robot=robot)
    controller._api_available = True

    tool_msg = types.LiveServerMessage(
        tool_call=types.LiveServerToolCall(
            function_calls=[
                types.FunctionCall(name="get_joint_positions", args={}, id="call_1"),
                types.FunctionCall(name="open_gripper", args={}, id="call_2"),
            ]
        )
    )

    messages = [tool_msg, turn_complete_message()]
    mock_session = fake_live_session(messages)
    controller._session = mock_session

    list(controller.process_instruction_stream("Check joints and open"))

    assert mock_session.send_tool_response.call_count == 1
    args, _ = mock_session.send_tool_response.call_args
    responses = args[0]
    assert len(responses) == 2
    assert responses[0].id == "call_1"
    assert responses[0].name == "get_joint_positions"
    assert responses[1].id == "call_2"
    assert responses[1].name == "open_gripper"


def test_post_action_frame_is_attached_to_function_response_parts() -> None:
    """Verify post-action observation frame is attached inside the FunctionResponse parts."""
    robot = MockRobot()
    controller = VLAController(mode="sim", robot=robot)
    controller._api_available = True

    post_frame = b"\xff\xd8\xff\xe0post_action_jpeg"
    messages = [
        tool_call_message("open_gripper", {}, "call_open"),
        turn_complete_message(),
    ]
    mock_session = fake_live_session(messages)
    controller._session = mock_session

    list(
        controller.process_instruction_stream("Open gripper", get_latest_jpeg_fn=lambda: post_frame)
    )

    mock_session.send_tool_response.assert_called_once()
    call_args = mock_session.send_tool_response.call_args[0][0]
    assert len(call_args) == 1
    assert call_args[0].parts is not None
    assert len(call_args[0].parts) == 1
    assert call_args[0].parts[0].inline_data.data == post_frame
    assert call_args[0].parts[0].inline_data.mime_type == "image/jpeg"


def test_tool_call_message_with_model_turn_text() -> None:
    """Verify message carrying both model_turn text and tool_call handles both correctly."""
    robot = MockRobot()
    controller = VLAController(mode="sim", robot=robot)
    controller._api_available = True

    combined_msg = types.LiveServerMessage(
        server_content=types.LiveServerContent(
            model_turn=types.Content(parts=[types.Part.from_text(text="Pondering before action.")])
        ),
        tool_call=types.LiveServerToolCall(
            function_calls=[
                types.FunctionCall(name="move_to_preset", args={"pose_name": "HOME"}, id="call_c"),
            ]
        ),
    )

    messages = [combined_msg, turn_complete_message()]
    mock_session = fake_live_session(messages)
    controller._session = mock_session

    fake_jpeg = b"\xff\xd8\xff\xe0mock_jpeg"
    chunks = list(
        controller.process_instruction_stream(
            "Think and inspect", get_latest_jpeg_fn=lambda: fake_jpeg
        )
    )
    full_text = "".join(chunks)

    assert "Pondering before action." in full_text
    assert "move_to_preset" in full_text
    assert "completed" in full_text
    mock_session.send_tool_response.assert_called_once()
    call_args = mock_session.send_tool_response.call_args[0][0]
    assert len(call_args) == 1
    assert call_args[0].parts is not None
    assert call_args[0].parts[0].inline_data.data == fake_jpeg


def test_mid_turn_complete_with_tool_call_continues_loop() -> None:
    """Verify sub-turn turn_complete with tool_call continues loop for next model turn."""
    robot = MockRobot()
    controller = VLAController(mode="sim", robot=robot)
    controller._api_available = True

    subturn_tool_msg = types.LiveServerMessage(
        server_content=types.LiveServerContent(turn_complete=True),
        tool_call=types.LiveServerToolCall(
            function_calls=[
                types.FunctionCall(name="get_joint_positions", args={}, id="call_sub"),
            ]
        ),
    )
    final_text_msg = text_message("Subturn complete handled cleanly.")
    final_turn_complete = turn_complete_message()

    messages = [subturn_tool_msg, final_text_msg, final_turn_complete]
    mock_session = fake_live_session(messages)
    controller._session = mock_session

    chunks = list(controller.process_instruction_stream("Inspect joints"))
    full_text = chunks[-1] if chunks else ""

    assert "Subturn complete handled cleanly." in full_text
    mock_session.send_tool_response.assert_called_once()


def test_no_in_tool_frame_streaming_during_blocking_tool() -> None:
    """Verify frames are NOT streamed during blocking tools; frame is in tool response."""
    robot = MockRobot()
    controller = VLAController(mode="sim", robot=robot)
    controller._api_available = True

    tool_msg = tool_call_message("open_gripper", {}, "call_slow")
    messages = [tool_msg, turn_complete_message()]
    mock_session = fake_live_session(messages)
    controller._session = mock_session

    post_frame = b"\xff\xd8fake_post_frame"
    list(
        controller.process_instruction_stream("Open slowly", get_latest_jpeg_fn=lambda: post_frame)
    )

    # In-tool frame streaming is removed: send_tool_response delivers the post-action frame
    assert mock_session.send_tool_response.call_count == 1
    call_args = mock_session.send_tool_response.call_args[0][0]
    assert call_args[0].parts is not None
    assert call_args[0].parts[0].inline_data.data == post_frame


@pytest.mark.usefixtures("real_sleep")
def test_frame_streamer_does_not_delay_tool_response(restore_settings: None) -> None:
    """Verify blocking tool taking 0.25s produces tool response within 0.5s of tool returning."""
    import time

    set_settings(Settings.from_env(vision_settle_s=0.05))

    robot = MockRobot()
    tool_return_time: float = 0.0
    tool_response_time: float = 0.0

    def quick_blocking_move(*args, **kwargs):
        nonlocal tool_return_time
        _real_sleep(0.25)
        tool_return_time = time.monotonic()

    robot.move_to_pose = quick_blocking_move

    controller = VLAController(mode="sim", robot=robot)
    controller._api_available = True

    tool_msg = tool_call_message("open_gripper", {}, "call_quick")
    messages = [tool_msg, turn_complete_message()]
    mock_session = fake_live_session(messages)

    orig_send_tool_response = mock_session.send_tool_response

    async def tracking_send_tool_response(responses):
        nonlocal tool_response_time
        tool_response_time = time.monotonic()
        return await orig_send_tool_response(responses)

    mock_session.send_tool_response = AsyncMock(side_effect=tracking_send_tool_response)
    controller._session = mock_session

    list(
        controller.process_instruction_stream(
            "Open quickly", get_latest_jpeg_fn=lambda: b"\xff\xd8quick_frame"
        )
    )

    assert mock_session.send_tool_response.call_count == 1
    assert tool_return_time > 0.0
    assert tool_response_time > 0.0
    elapsed_after_return = tool_response_time - tool_return_time
    assert elapsed_after_return < 0.5, (
        f"Tool response took {elapsed_after_return:.3f}s after tool return (must be < 0.5s)"
    )


def test_post_action_frame_delivery_failure_marks_function_response_missing() -> None:
    """Verify post-action frame failure marks post_action_frame='missing' in payload."""
    robot = MockRobot()
    controller = VLAController(mode="sim", robot=robot)
    controller._api_available = True

    tool_msg = tool_call_message("open_gripper", {}, "call_open")
    messages = [tool_msg, turn_complete_message()]
    mock_session = fake_live_session(messages)
    controller._session = mock_session

    calls = 0

    def frame_provider():
        nonlocal calls
        calls += 1
        if calls == 1:
            return b"\xff\xd8initial_frame"
        return None  # Post-action frame is unavailable

    chunks = list(
        controller.process_instruction_stream("Open gripper", get_latest_jpeg_fn=frame_provider)
    )
    full_output = "".join(chunks)

    assert "⚠️" in full_output
    assert mock_session.send_tool_response.call_count == 1
    call_args = mock_session.send_tool_response.call_args[0][0]
    resp_dict = call_args[0].response
    assert resp_dict.get("post_action_frame") == "missing"
    assert call_args[0].parts is None


@pytest.mark.asyncio
async def test_live_session_send_instruction_ordered_turn_with_frame() -> None:
    """Verify send_instruction sends image first in send_client_content, no send_realtime_input."""
    from unittest.mock import patch

    from agent.live_session import GeminiLiveSession

    with patch("agent.live_session.create_genai_client"):
        session = GeminiLiveSession(model="test-model", tools=[], system_instruction="test")
    mock_underlying = AsyncMock()
    session._session = mock_underlying

    fake_jpeg = b"\xff\xd8\xff\xe0jpeg_bytes"
    await session.send_instruction(text="Pick cube", frame_jpeg=fake_jpeg)

    mock_underlying.send_client_content.assert_called_once()
    assert not mock_underlying.send_realtime_input.called

    call_kwargs = mock_underlying.send_client_content.call_args.kwargs
    assert call_kwargs.get("turn_complete") is True
    content = call_kwargs.get("turns")
    assert isinstance(content, types.Content)
    assert content.role == "user"
    assert len(content.parts) == 2
    # Image part is first
    assert content.parts[0].inline_data is not None
    assert content.parts[0].inline_data.data == fake_jpeg
    assert content.parts[0].inline_data.mime_type == "image/jpeg"
    # Text part is second
    assert content.parts[1].text == "Pick cube"


@pytest.mark.asyncio
async def test_live_session_send_instruction_without_frame() -> None:
    """Verify send_instruction without frame sends text-only Content, no send_realtime_input."""
    from unittest.mock import patch

    from agent.live_session import GeminiLiveSession

    with patch("agent.live_session.create_genai_client"):
        session = GeminiLiveSession(model="test-model", tools=[], system_instruction="test")
    mock_underlying = AsyncMock()
    session._session = mock_underlying

    await session.send_instruction(text="Where is cube?", frame_jpeg=None)

    mock_underlying.send_client_content.assert_called_once()
    assert not mock_underlying.send_realtime_input.called

    call_kwargs = mock_underlying.send_client_content.call_args.kwargs
    assert call_kwargs.get("turn_complete") is True
    content = call_kwargs.get("turns")
    assert isinstance(content, types.Content)
    assert content.role == "user"
    assert len(content.parts) == 1
    assert content.parts[0].text == "Where is cube?"


def test_loop_failure_calls_robot_interrupt() -> None:
    """Verify that a fatal loop failure invokes robot.interrupt()."""
    robot = MockRobot()
    robot.interrupt = MagicMock()
    controller = VLAController(mode="sim", robot=robot)
    controller._api_available = True

    mock_session = fake_live_session([])

    async def crashing_receive():
        raise RuntimeError("Fatal connection tear mid-turn")
        yield  # make it a generator

    mock_session.receive = crashing_receive
    controller._session = mock_session

    list(controller.process_instruction_stream("Crash turn"))

    robot.interrupt.assert_called_once()


class _FakeWS:
    """Fake WebSocket client connection recording sent messages."""

    def __init__(self) -> None:
        self.sent_messages: list[str] = []

    async def send(self, msg: str) -> None:
        self.sent_messages.append(msg)


class _FakeAsyncSession:
    """Fake AsyncSession supporting both wire _ws and SDK send_tool_response."""

    def __init__(self, has_ws: bool = True) -> None:
        self._ws: _FakeWS | None = _FakeWS() if has_ws else None
        self.sdk_send_tool_response = AsyncMock()

    async def send_tool_response(self, function_responses: Any) -> None:
        await self.sdk_send_tool_response(function_responses=function_responses)


@pytest.mark.asyncio
async def test_send_tool_response_wire_serialization() -> None:
    """Verify responses with parts use wire path with camelCase JSON and base64 data."""
    import base64
    import json
    from unittest.mock import patch

    from agent.live_session import GeminiLiveSession

    fake_session = _FakeAsyncSession(has_ws=True)
    with patch("agent.live_session.create_genai_client"):
        live_session = GeminiLiveSession(model="test-model", tools=[], system_instruction="test")
    live_session._session = fake_session

    jpeg_bytes = b"\xff\xd8\xff\xe0raw_image_data"
    fr = types.FunctionResponse(
        id="call_test_123",
        name="move_to_spatial_point",
        response={"status": "success", "reached": True},
        parts=[
            types.FunctionResponsePart(
                inline_data=types.FunctionResponseBlob(
                    mime_type="image/jpeg",
                    data=jpeg_bytes,
                )
            )
        ],
    )

    await live_session.send_tool_response([fr])

    # Assert wire send was used, not SDK method
    assert not fake_session.sdk_send_tool_response.called
    assert fake_session._ws is not None
    assert len(fake_session._ws.sent_messages) == 1

    wire_msg = fake_session._ws.sent_messages[0]
    payload = json.loads(wire_msg)

    # 1. Valid JSON under tool_response.functionResponses[0]
    assert "tool_response" in payload
    assert "functionResponses" in payload["tool_response"]
    first_resp = payload["tool_response"]["functionResponses"][0]

    # 2. id, name, and response are preserved
    assert first_resp["id"] == "call_test_123"
    assert first_resp["name"] == "move_to_spatial_point"
    assert first_resp["response"] == {"status": "success", "reached": True}

    # 3. parts[0].inlineData.mimeType == "image/jpeg"
    assert "parts" in first_resp
    assert len(first_resp["parts"]) == 1
    part = first_resp["parts"][0]
    assert "inlineData" in part
    assert part["inlineData"]["mimeType"] == "image/jpeg"

    # 4. base64-decoding data returns the exact JPEG bytes
    encoded_data = part["inlineData"]["data"]
    assert base64.b64decode(encoded_data) == jpeg_bytes


@pytest.mark.asyncio
async def test_send_tool_response_without_parts_uses_sdk_method() -> None:
    """Verify responses without parts bypass wire serialization and use SDK send_tool_response."""
    from unittest.mock import patch

    from agent.live_session import GeminiLiveSession

    fake_session = _FakeAsyncSession(has_ws=True)
    with patch("agent.live_session.create_genai_client"):
        live_session = GeminiLiveSession(model="test-model", tools=[], system_instruction="test")
    live_session._session = fake_session

    fr = types.FunctionResponse(
        id="call_no_parts",
        name="open_gripper",
        response={"status": "success"},
    )

    await live_session.send_tool_response([fr])

    # Uses SDK method
    fake_session.sdk_send_tool_response.assert_called_once()
    assert fake_session._ws is not None
    assert len(fake_session._ws.sent_messages) == 0


@pytest.mark.asyncio
async def test_send_tool_response_missing_ws_raises_session_error() -> None:
    """Verify sending response with parts when session._ws is missing raises SessionError."""
    from unittest.mock import patch

    from agent.live_session import GeminiLiveSession, SessionError

    fake_session = _FakeAsyncSession(has_ws=False)
    with patch("agent.live_session.create_genai_client"):
        live_session = GeminiLiveSession(model="test-model", tools=[], system_instruction="test")
    live_session._session = fake_session

    fr = types.FunctionResponse(
        id="call_parts_err",
        name="move_to_spatial_point",
        response={"status": "success"},
        parts=[
            types.FunctionResponsePart(
                inline_data=types.FunctionResponseBlob(
                    mime_type="image/jpeg",
                    data=b"dummy",
                )
            )
        ],
    )

    with pytest.raises(SessionError, match="session._ws is not available"):
        await live_session.send_tool_response([fr])


def test_serialize_function_responses_rejects_unsupported_parts() -> None:
    """Verify _serialize_function_responses rejects unsupported parts like file_data."""
    from agent.live_session import _serialize_function_responses

    fr = types.FunctionResponse(
        id="call_bad",
        name="test",
        response={},
        parts=[
            types.FunctionResponsePart(
                file_data=types.FunctionResponseFileData(
                    file_uri="gs://bucket/file",
                    mime_type="image/jpeg",
                )
            )
        ],
    )
    with pytest.raises(ValueError, match="file_data parts in tool responses are not supported"):
        _serialize_function_responses([fr])
