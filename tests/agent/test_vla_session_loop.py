"""Tests for the VLA Live API streaming session loop, tool batching, and frame delivery."""

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


def test_post_action_frame_is_last_send_frame() -> None:
    """Verify post-action observation frame is sent as the final send_frame call."""
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

    assert mock_session.send_frame.called
    last_call_args = mock_session.send_frame.call_args[0]
    assert last_call_args[0] == post_frame
    mock_session.send_tool_response.assert_called_once()


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
    assert mock_session.send_frame.called
    mock_session.send_tool_response.assert_called_once()


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


@pytest.mark.slow
def test_streaming_frames_during_blocking_tool_and_single_tool_response() -> None:
    """Verify frames are streamed during blocking tool and send_tool_response called once."""
    import time

    robot = MockRobot()
    tool_return_time: float = 0.0
    send_frame_times: list[float] = []
    tool_response_time: float = 0.0

    # Make move_to_pose block for 1.2s to verify asyncio.to_thread concurrency
    def slow_move(*args, **kwargs):
        nonlocal tool_return_time
        _real_sleep(1.2)
        tool_return_time = time.monotonic()

    robot.move_to_pose = slow_move

    controller = VLAController(mode="sim", robot=robot)
    controller._api_available = True

    tool_msg = tool_call_message("open_gripper", {}, "call_slow")
    messages = [tool_msg, turn_complete_message()]
    mock_session = fake_live_session(messages)

    orig_send_frame = mock_session.send_frame

    async def tracking_send_frame(frame_bytes):
        send_frame_times.append(time.monotonic())
        return await orig_send_frame(frame_bytes)

    orig_send_tool_response = mock_session.send_tool_response

    async def tracking_send_tool_response(responses):
        nonlocal tool_response_time
        tool_response_time = time.monotonic()
        return await orig_send_tool_response(responses)

    mock_session.send_frame = AsyncMock(side_effect=tracking_send_frame)
    mock_session.send_tool_response = AsyncMock(side_effect=tracking_send_tool_response)
    controller._session = mock_session

    jpeg_counter = 0

    def frame_provider():
        nonlocal jpeg_counter
        jpeg_counter += 1
        return f"\xff\xd8frame_{jpeg_counter}".encode("latin1")

    list(controller.process_instruction_stream("Open slowly", get_latest_jpeg_fn=frame_provider))

    # A frame was sent during motion (1 fps interval during 1.2s) + post-action frame
    assert mock_session.send_frame.call_count >= 2
    # send_tool_response called exactly once
    assert mock_session.send_tool_response.call_count == 1
    # Post-action frame is last
    last_frame_sent = mock_session.send_frame.call_args[0][0]
    assert last_frame_sent.startswith(b"\xff\xd8frame_")

    # Assert that at least one send_frame call happened before the tool handler returned
    assert any(t < tool_return_time for t in send_frame_times)
    # Assert that send_tool_response call happens after the last send_frame
    assert tool_response_time >= max(send_frame_times)


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

    async def fail_send_frame(frame_bytes):
        raise ConnectionError("Network dropped during frame injection")

    mock_session.send_frame = fail_send_frame
    controller._session = mock_session

    post_frame = b"\xff\xd8fake_post_frame"
    chunks = list(
        controller.process_instruction_stream("Open gripper", get_latest_jpeg_fn=lambda: post_frame)
    )
    full_output = "".join(chunks)

    assert "⚠️" in full_output
    assert mock_session.send_tool_response.call_count == 1
    call_args = mock_session.send_tool_response.call_args[0][0]
    resp_dict = call_args[0].response
    assert resp_dict.get("post_action_frame") == "missing"


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
