"""Regression tests for VLAController live session loop using the fake session fixture."""

from google.genai import types

from agent.vla_controller import VLAController
from tests.conftest import (
    MockRobot,
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
                types.FunctionCall(name="send_message", args={"message": "All set"}, id="call_2"),
            ]
        )
    )

    messages = [tool_msg, turn_complete_message()]
    mock_session = fake_live_session(messages)
    controller._session = mock_session

    list(controller.process_instruction_stream("Check joints and speak"))

    assert mock_session.send_tool_response.call_count == 1
    args, _ = mock_session.send_tool_response.call_args
    responses = args[0]
    assert len(responses) == 2
    assert responses[0].id == "call_1"
    assert responses[0].name == "get_joint_positions"
    assert responses[1].id == "call_2"
    assert responses[1].name == "send_message"


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


def test_send_message_yields_no_executing_status() -> None:
    """Verify send_message does not emit an 'Executing' status update."""
    robot = MockRobot()
    controller = VLAController(mode="sim", robot=robot)
    controller._api_available = True

    messages = [
        tool_call_message("send_message", {"message": "Hello user"}, "call_msg"),
        turn_complete_message(),
    ]
    controller._session = fake_live_session(messages)

    chunks = list(controller.process_instruction_stream("Greet"))
    full_text = chunks[-1] if chunks else ""

    assert "Executing `send_message`" not in full_text
    assert "Hello user" in full_text


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
                types.FunctionCall(name="get_joint_positions", args={}, id="call_c"),
            ]
        ),
    )

    messages = [combined_msg, turn_complete_message()]
    mock_session = fake_live_session(messages)
    controller._session = mock_session

    chunks = list(controller.process_instruction_stream("Think and inspect"))
    full_text = chunks[-1] if chunks else ""

    assert "Pondering before action." in full_text
    assert "get_joint_positions" in full_text
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


def test_clear_history_keeps_system_prompt() -> None:
    """Verify clear_history preserves system prompt while discarding subsequent history."""
    controller = VLAController(mode="sim")
    controller.history.append({"role": "user", "content": "hello"})
    controller.history.append({"role": "assistant", "content": "world"})
    assert len(controller.get_history()) == 3

    controller.clear_history()
    history = controller.get_history()
    assert len(history) == 1
    assert history[0]["role"] == "system"
    assert "Gemini Robotics Assistant" in history[0]["content"]
