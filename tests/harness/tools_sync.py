"""Synchronous test wrapper around agent.tools.execute_tool_call."""

import asyncio
from collections.abc import Callable
from typing import Any

from agent.contracts import GripperToolResponse, MoveToSpatialPointResponse
from agent.guardrails import GuardrailState
from agent.tools import execute_tool_call
from robot.interface import RobotInterface


def execute_tool_call_sync(
    call: Any,
    robot: RobotInterface,
    get_latest_jpeg_fn: Callable[[], bytes | None] | None = None,
    mode: str | None = None,
    guardrail_state: GuardrailState | None = None,
    get_fresh_jpeg_fn: Callable[[float, float], bytes | None] | None = None,
) -> dict[str, Any] | GripperToolResponse | MoveToSpatialPointResponse:
    """Synchronous test wrapper around ``execute_tool_call``.

    Args:
        call: Gemini FunctionCall object.
        robot: Active RobotInterface instance.
        get_latest_jpeg_fn: Optional callback returning raw JPEG bytes.
        mode: Optional explicit mode ("real" or "sim").
        guardrail_state: Optional per-controller closed-loop guardrail state.
        get_fresh_jpeg_fn: Optional ``(t_after, timeout_s) -> bytes | None`` frame callback.

    Returns:
        Dictionary of execution results.
    """
    return asyncio.run(
        execute_tool_call(
            call,
            robot,
            get_latest_jpeg_fn=get_latest_jpeg_fn,
            mode=mode,
            guardrail_state=guardrail_state,
            get_fresh_jpeg_fn=get_fresh_jpeg_fn,
        )
    )
