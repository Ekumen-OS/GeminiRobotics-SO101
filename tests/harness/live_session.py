"""Shared mock Live API session and message builders for agent tests."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

from google.genai import types


def text_message(text: str) -> types.LiveServerMessage:
    """Build a LiveServerMessage containing regular text from the model.

    Args:
        text: Text string for model turn.

    Returns:
        LiveServerMessage wrapping the text part.
    """
    return types.LiveServerMessage(
        server_content=types.LiveServerContent(
            model_turn=types.Content(parts=[types.Part.from_text(text=text)])
        )
    )


def thought_message(text: str) -> types.LiveServerMessage:
    """Build a LiveServerMessage containing model thought/reasoning text.

    Args:
        text: Reasoning thought text string.

    Returns:
        LiveServerMessage wrapping the thought part.
    """
    return types.LiveServerMessage(
        server_content=types.LiveServerContent(
            model_turn=types.Content(parts=[types.Part(thought=True, text=text)])
        )
    )


def tool_call_message(
    name: str,
    args: dict[str, Any] | None = None,
    call_id: str = "call_1",
) -> types.LiveServerMessage:
    """Build a LiveServerMessage containing a function tool call.

    Args:
        name: Name of tool function called.
        args: Optional arguments dictionary.
        call_id: Unique call identifier.

    Returns:
        LiveServerMessage wrapping the tool call.
    """
    return types.LiveServerMessage(
        tool_call=types.LiveServerToolCall(
            function_calls=[
                types.FunctionCall(
                    name=name,
                    args=args or {},
                    id=call_id,
                )
            ]
        )
    )


def turn_complete_message() -> types.LiveServerMessage:
    """Build a LiveServerMessage indicating turn completion.

    Returns:
        LiveServerMessage with turn_complete=True.
    """
    return types.LiveServerMessage(server_content=types.LiveServerContent(turn_complete=True))


def fake_live_session(messages: list[Any] | None = None) -> MagicMock:
    """Create a mock Live API session satisfying the session protocol.

    Args:
        messages: Optional list of LiveServerMessage objects or Exceptions to yield from receive().

    Returns:
        MagicMock configuring is_connected and AsyncMock methods.
    """
    mock_session = MagicMock()
    mock_session.is_connected = True
    mock_session.connect = AsyncMock()
    mock_session.send_instruction = AsyncMock()
    mock_session.send_tool_response = AsyncMock()
    mock_session.disconnect = AsyncMock()

    msg_list = list(messages) if messages is not None else []

    async def mock_receive():
        for msg in msg_list:
            if isinstance(msg, Exception):
                raise msg
            yield msg

    mock_session.receive = mock_receive
    return mock_session
