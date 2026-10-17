"""Controller for managing Vision-Language-Action (VLA) agent interactions."""

import asyncio
import concurrent.futures
import json
import logging
import re
import threading
from collections.abc import Callable, Generator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from google.genai import errors as genai_errors
from google.genai import types

from agent.guardrails import GuardrailState
from agent.live_session import GeminiLiveSession, SessionError
from agent.response_assembler import ResponseAssembler
from agent.tools import (
    ROBOT_TOOL_DECLARATIONS,
    TOOL_REGISTRY,
    execute_tool_call,
    render_prompt_tool_list,
)
from robot.errors import CalibrationUnavailable, HardwareError
from robot.grasp import classify_grasp
from robot.interface import RobotInterface
from robot.logger import log_with_tag
from robot.settings import get_settings
from vision.calibrator import get_ready_calibrator
from vision.targets import set_model_targets

logger = logging.getLogger(__name__)


# Failures an ER2 pointing query may raise; anything else is a bug and must surface.
ER2_QUERY_ERRORS = (
    genai_errors.APIError,
    asyncio.TimeoutError,
    json.JSONDecodeError,
    ValueError,
    KeyError,
    OSError,
)


def load_system_prompt(mode: str, tolerance: int | float | None = None) -> str:
    """Load and render the VLA system instruction prompt.

    Args:
        mode: Operational mode ('real' or 'sim').
        tolerance: Alignment tolerance in millimetres. Defaults to settings value.

    Returns:
        Rendered markdown prompt string.
    """
    if tolerance is None:
        tolerance = get_settings().alignment_tolerance_mm
    sim_note = (
        "- In SIM mode, the workspace is pre-calibrated using a fixed virtual camera "
        "matching the physical camera view. ArUco fiducial markers (IDs 0-3) and workspace "
        "boundary guides are visible on the workspace floor at the 4 calibration corners.\n"
        if mode == "sim"
        else ""
    )
    prompt_path = Path(__file__).parent / "prompts" / "vla_system.md"
    template = prompt_path.read_text(encoding="utf-8").rstrip("\n")
    return template.format(
        mode_upper=mode.upper(),
        tolerance=f"{tolerance:g}",
        sim_workspace_note=sim_note,
        tool_list=render_prompt_tool_list(),
    )


def parse_targets_from_text(text: str) -> list[dict]:
    """Extract 2D spatial points and bounding boxes with associated labels from text.

    Args:
        text: Input string from model speech or response containing spatial coordinates.

    Returns:
        List of target dictionaries suitable for set_model_targets().
    """
    if not text:
        return []

    targets: list[dict] = []
    # Pattern to find all bracketed coordinate sequences: [123, 456] or [123, 456, 789, 101]
    coord_blocks = list(re.finditer(r"\[\s*(\d{1,4}(?:\s*,\s*\d{1,4}){1,3})\s*\]", text))

    for match in coord_blocks:
        nums = [float(n.strip()) for n in match.group(1).split(",") if n.strip()]
        if not all(0 <= n <= 1000 for n in nums):
            continue

        start_idx = match.start()
        context_before = text[max(0, start_idx - 50) : start_idx].strip()
        label = None
        label_match = re.search(
            r"([A-Za-z0-9_\-\s]{2,25})\s*(?:at|:|\sis\sat|=)?\s*$", context_before
        )
        if label_match:
            candidate = label_match.group(1).strip(" -*#:\t\n")
            if (
                candidate
                and len(candidate) > 1
                and candidate.lower()
                not in (
                    "coordinates",
                    "point",
                    "box",
                    "at",
                    "target",
                    "value",
                    "pos",
                )
            ):
                label = candidate.upper()

        if len(nums) == 4:
            ymin, xmin, ymax, xmax = nums
            cy, cx = (ymin + ymax) / 2.0, (xmin + xmax) / 2.0
            tag_label = label if label else f"OBJ #{len(targets) + 1} [{int(cy)}, {int(cx)}]"
            targets.append(
                {
                    "point": (cy, cx),
                    "box_2d": [ymin, xmin, ymax, xmax],
                    "label": tag_label,
                }
            )
        elif len(nums) == 2:
            y_val, x_val = nums
            tag_label = (
                label if label else f"SPOTTED #{len(targets) + 1} [{int(y_val)}, {int(x_val)}]"
            )
            targets.append(
                {
                    "point": (y_val, x_val),
                    "box_2d": None,
                    "label": tag_label,
                }
            )

    return targets


def parse_er2_pointing_reply(text: str) -> dict[str, list[float]] | None:
    """Parse JSON or text reply for gripper_fingertips and target coordinates.

    Args:
        text: Response string from pointing query.

    Returns:
        Dictionary mapping coordinate names to [y, x] lists, or None.
    """
    if not text:
        return None
    try:
        match = re.search(r"\{.*?\}", text, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            gripper = data.get("gripper_fingertips")
            target = data.get("target")
            res: dict[str, list[float]] = {}
            if isinstance(gripper, (list, tuple)) and len(gripper) == 2:
                res["gripper_fingertips"] = [float(gripper[0]), float(gripper[1])]
            if isinstance(target, (list, tuple)) and len(target) == 2:
                res["target"] = [float(target[0]), float(target[1])]
            if "gripper_fingertips" in res:
                return res
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    targets = parse_targets_from_text(text)
    if targets:
        return {
            "gripper_fingertips": [
                float(targets[0]["point"][0]),
                float(targets[0]["point"][1]),
            ]
        }
    return None


@dataclass
class _TurnContext:
    """Context holding queues and callbacks for an instruction processing turn."""

    get_latest_jpeg_fn: Callable[[], bytes | None] | None
    get_fresh_jpeg_fn: Callable[[float, float], bytes | None] | None
    response_queue: asyncio.Queue
    tool_executing_event: asyncio.Event


class VLAController:
    """Core controller for managing interaction with gemini-robotics-er.

    This class handles state management, the async event loop for the WebSocket session,
    and interaction with the VLA model API, abstracting it from the frontend UI.
    """

    def __init__(self, mode: str = "sim", robot: RobotInterface | None = None) -> None:
        """Initialize the VLA Controller.

        Args:
            mode: The operational mode of the robot ('real' or 'sim').
            robot: The initialized robot instance to execute commands against.
        """
        self.mode = mode
        self.robot = robot
        system_prompt = load_system_prompt(mode=self.mode)

        self._api_available = True
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

        self._session: GeminiLiveSession | None = None
        self._active_instruction_lock = threading.Lock()
        self._async_instruction_lock = asyncio.Lock()
        self._current_async_task: asyncio.Future | None = None
        self.guardrail_state: GuardrailState = GuardrailState()

        # We try to initialize the session to test credentials, but don't connect yet.
        try:
            self._session = GeminiLiveSession(
                model=get_settings().gemini_model_id,
                tools=ROBOT_TOOL_DECLARATIONS,
                system_instruction=system_prompt,
            )
        except ConnectionError as e:
            log_with_tag(logger, "WARN", f"⚠️ {e}")
            self._api_available = False

    async def _query_er2_locator_async(self, frame_jpeg: bytes) -> dict[str, list[float]] | None:
        """Ask Gemini model for structured gripper & target coordinates via pointing query.

        Args:
            frame_jpeg: Raw camera frame JPEG bytes.

        Returns:
            Dictionary with parsed coordinates or None.
        """
        client = getattr(self._session, "client", None)
        if client is None:
            try:
                from agent.live_session import create_genai_client

                client = create_genai_client()
            except ConnectionError as e:
                log_with_tag(logger, "WARN", f"Cannot obtain genai client for ER2 locator: {e}")
                return None

        prompt_text = (
            'Return JSON {"gripper_fingertips": [y, x], "target": [y, x]} '
            "in 0–1000 image coordinates, nothing else"
        )
        try:
            image_part = types.Part.from_bytes(data=frame_jpeg, mime_type="image/jpeg")
            resp = await asyncio.wait_for(
                client.aio.models.generate_content(
                    model=get_settings().er2_locator_model,
                    contents=[image_part, prompt_text],
                ),
                timeout=5.0,
            )
            if hasattr(resp, "text") and isinstance(resp.text, str):
                reply_text = resp.text
            elif isinstance(resp, str):
                reply_text = resp
            else:
                reply_text = str(getattr(resp, "text", resp))
            return parse_er2_pointing_reply(reply_text)
        except ER2_QUERY_ERRORS as e:
            log_with_tag(logger, "WARN", f"Failed to query ER2 pointing locator: {e}")
            return None

    def _begin_guardrail_instruction(self) -> None:
        """Reset per-instruction guardrail state, deriving the phase from the real gripper.

        The state object is mutated in place (its epoch invalidates writes from handler threads
        still running for the previous instruction). The phase is derived from the last gripper
        target this session commanded plus the gripper sensor: HOME's partly closed gripper is
        not a grasp, but a gripper commanded closed that still reads stalled is still holding.
        """
        state = self.guardrail_state
        grasped: bool | None = None
        if self.robot is not None and state.last_commanded_gripper == "closed":
            try:
                grasped = classify_grasp(self.robot.read_gripper_state(), get_settings())["grasped"]
            except HardwareError as e:
                log_with_tag(logger, "WARN", f"Gripper read failed at instruction start: {e}")
        state.begin_instruction(state.derive_phase(grasped, state.last_commanded_gripper))

    def _run_loop(self) -> None:
        """Run the background asyncio event loop."""
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    @property
    def is_api_available(self) -> bool:
        """Check if the Gemini API is configured and reachable."""
        return self._api_available

    async def _ensure_connected(self) -> None:
        """Lazy connection with auto-reconnect and exponential backoff."""
        if not self._session:
            return

        if self._session.is_connected:
            return

        backoff = [0.5, 1.0, 2.0]
        for attempt, delay in enumerate(backoff):
            try:
                await self._session.connect()
                return
            except (SessionError, OSError, TimeoutError, ConnectionError) as e:
                log_with_tag(
                    logger,
                    "ERROR",
                    f"Connection attempt {attempt + 1} failed: {e}",
                )
                await asyncio.sleep(delay)

        raise SessionError("Failed to connect to Gemini Live API after 3 attempts.")

    async def _stream_frames_during_execution(
        self, get_latest_jpeg_fn, tool_executing_event: asyncio.Event
    ) -> None:
        """Stream pristine raw frames at ~1 FPS while a BLOCKING tool is executing."""
        if not self._session or not get_latest_jpeg_fn:
            return

        while tool_executing_event.is_set():
            jpeg = get_latest_jpeg_fn()
            if jpeg:
                try:
                    await self._session.send_frame(jpeg)
                except (SessionError, OSError) as e:
                    log_with_tag(logger, "ERROR", f"Frame stream error: {e}")
                    break
            # Wait up to 1.0 s in <= 0.1 s slices while tool_executing_event is set
            for _ in range(10):
                if not tool_executing_event.is_set():
                    break
                await asyncio.sleep(0.1)

    async def _handle_model_turn(self, parts: list[Any], ctx: _TurnContext) -> None:
        """Process model turn parts for reasoning thoughts and narration text.

        Args:
            parts: Sequence of content parts emitted by the model.
            ctx: Active turn context.
        """
        for part in parts:
            is_thought = getattr(part, "thought", False)
            if is_thought:
                log_with_tag(logger, "REASONING", f"💭 {part.text}")
                if part.text:
                    await ctx.response_queue.put({"type": "thought", "content": part.text})
            elif part.text:
                log_with_tag(logger, "DECISION", f"📝 Model text: {part.text}")
                await ctx.response_queue.put({"type": "text", "content": part.text})

    async def _execute_tool(self, call: Any, ctx: _TurnContext) -> types.FunctionResponse:
        """Execute a single function call, emit UI updates, and return response payload.

        Args:
            call: Function call object from Gemini.
            ctx: Active turn context.

        Returns:
            FunctionResponse wrapping the tool result.
        """
        log_with_tag(
            logger,
            "TOOL_CALL",
            f"🔧 Dispatching '{call.name}' [ID: {call.id}] | Payload: {call.args}",
        )

        if call.name == "send_message":
            msg_text = call.args.get("message", "") if hasattr(call, "args") else ""
            log_with_tag(logger, "DECISION", f"💬 Model speech: {msg_text}")
            await ctx.response_queue.put({"type": "text", "content": msg_text})

            parsed_targets = parse_targets_from_text(msg_text)
            if parsed_targets:
                try:
                    set_model_targets(parsed_targets)
                except (TypeError, KeyError, ValueError) as e:
                    log_with_tag(logger, "WARN", f"Failed to set model targets: {e}")

            log_with_tag(
                logger,
                "TOOL_RESULT",
                f"✅ Tool 'send_message' completed [ID: {call.id}] | "
                "Result: {'status': 'success'}",
            )
            return types.FunctionResponse(
                name=call.name,
                id=call.id,
                response={"status": "success"},
            )

        await ctx.response_queue.put({"type": "status", "content": f"⚙️ Executing `{call.name}`..."})

        spec = TOOL_REGISTRY.get(call.name)
        is_blocking = spec.blocking if spec is not None else False

        frame_task = None
        if is_blocking:
            ctx.tool_executing_event.set()
            frame_task = asyncio.create_task(
                self._stream_frames_during_execution(
                    ctx.get_latest_jpeg_fn, ctx.tool_executing_event
                )
            )

        def query_er2(jpeg: bytes) -> dict[str, list[float]] | None:
            try:
                fut = asyncio.run_coroutine_threadsafe(
                    self._query_er2_locator_async(jpeg), self._loop
                )
                return fut.result(timeout=6.0)
            except (concurrent.futures.TimeoutError, RuntimeError) as e:
                log_with_tag(logger, "WARN", f"ER2 locator query failed: {e}")
                return None

        result_dict = await execute_tool_call(
            call=call,
            robot=self.robot,
            get_latest_jpeg_fn=ctx.get_latest_jpeg_fn,
            mode=self.mode,
            guardrail_state=self.guardrail_state,
            query_er2_locator_fn=query_er2,
            get_fresh_jpeg_fn=ctx.get_fresh_jpeg_fn,
        )

        post_frame_missing = False
        if is_blocking:
            ctx.tool_executing_event.clear()
            if frame_task:
                await frame_task

            # Post-action observation frame for the model (measurement waits are in the tools)
            settle_s = get_settings().vision_settle_s
            if settle_s > 0:
                await asyncio.sleep(settle_s)

            if ctx.get_latest_jpeg_fn:
                post_frame = ctx.get_latest_jpeg_fn()
            else:
                post_frame = None

            if post_frame:
                try:
                    log_with_tag(
                        logger,
                        "STREAM",
                        f"📸 Injected post-action observation frame ({len(post_frame)} bytes)",
                    )
                    if self._session:
                        await self._session.send_frame(post_frame)
                except (SessionError, OSError) as e:
                    log_with_tag(logger, "ERROR", f"Post-action frame error: {e}")
                    post_frame_missing = True
                    await ctx.response_queue.put(
                        {"type": "status", "content": f"⚠️ Failed to deliver post-action frame: {e}"}
                    )
            elif ctx.get_latest_jpeg_fn is not None:
                post_frame_missing = True
                await ctx.response_queue.put(
                    {"type": "status", "content": "⚠️ Post-action frame unavailable"}
                )

        if post_frame_missing and isinstance(result_dict, dict):
            result_dict["post_action_frame"] = "missing"

        if self.guardrail_state.escalated:
            await ctx.response_queue.put(
                {
                    "type": "status",
                    "content": f"🛑 {self.guardrail_state.escalation_message}",
                }
            )

        has_error = "error" in result_dict
        status_emoji = "⚠️" if (has_error or post_frame_missing) else "✅"
        log_with_tag(
            logger,
            "TOOL_RESULT",
            f"{status_emoji} Tool '{call.name}' completed [ID: {call.id}] | Result: {result_dict}",
            level=logging.ERROR if has_error else logging.INFO,
        )

        if has_error:
            err_msg = result_dict.get("error", "Unknown error")
            err_type = result_dict.get("error_type")
            err_type_str = f" ({err_type})" if err_type else ""
            robot_moved_str = (
                " (Arm may have moved partially)" if result_dict.get("robot_moved") else ""
            )
            await ctx.response_queue.put(
                {
                    "type": "status",
                    "content": f"⚠️ `{call.name}` failed: {err_msg}{err_type_str}{robot_moved_str}",
                }
            )
        else:
            await ctx.response_queue.put(
                {"type": "status", "content": f"✅ `{call.name}` completed."}
            )

        return types.FunctionResponse(name=call.name, id=call.id, response=result_dict)

    async def _handle_tool_call(self, tool_call: Any, ctx: _TurnContext) -> None:
        """Handle tool call message by executing each function call and sending responses.

        Args:
            tool_call: Tool call payload containing function_calls.
            ctx: Active turn context.
        """
        function_responses = []
        for call in tool_call.function_calls:
            resp = await self._execute_tool(call, ctx)
            function_responses.append(resp)
        if function_responses and self._session:
            await self._session.send_tool_response(function_responses)

    async def _handle_server_message(self, message: Any, ctx: _TurnContext) -> bool:
        """Handle a single LiveServerMessage from Gemini.

        Args:
            message: Inbound LiveServerMessage from Gemini Live session.
            ctx: Active turn context.

        Returns:
            True if turn is fully complete, False to keep receiving messages.
        """
        if not isinstance(message, types.LiveServerMessage):
            return False

        log_with_tag(
            logger,
            "INFO",
            f"📩 Live Message: "
            f"server_content={message.server_content is not None}, "
            f"tool_call={message.tool_call is not None}",
        )

        if message.server_content and message.server_content.model_turn:
            await self._handle_model_turn(message.server_content.model_turn.parts, ctx)

        if message.tool_call:
            await self._handle_tool_call(message.tool_call, ctx)

        if message.server_content and message.server_content.turn_complete:
            if not message.tool_call:
                log_with_tag(logger, "DECISION", "🏁 Full turn complete received.")
                return True
            else:
                log_with_tag(
                    logger,
                    "INFO",
                    "ℹ️ Sub-turn complete on tool_call message; "
                    "waiting for model response to tool output.",
                )

        return False

    async def _process_instruction_async(
        self,
        instruction: str,
        get_latest_jpeg_fn,
        response_queue: asyncio.Queue,
        get_fresh_jpeg_fn=None,
    ) -> None:
        """Async core orchestrator for processing an instruction and running the loop.

        Args:
            instruction: Prompt text from user.
            get_latest_jpeg_fn: Optional frame provider.
            response_queue: Async queue receiving progressive updates.
            get_fresh_jpeg_fn: Optional ``(t_after, timeout_s) -> JPEG | None`` frame provider
                for measurements that must follow a motion.
        """
        if not self._session:
            await response_queue.put({"type": "error", "content": "API unavailable"})
            await response_queue.put(None)
            return

        try:
            await self._ensure_connected()
        except (SessionError, ConnectionError) as e:
            if self.robot is not None:
                self.robot.interrupt()
            await response_queue.put({"type": "error", "content": str(e)})
            await response_queue.put(None)
            return

        log_with_tag(logger, "DECISION", f"🚀 Received user instruction: {instruction!r}")

        async with self._async_instruction_lock:
            initial_frame = get_latest_jpeg_fn() if get_latest_jpeg_fn else None

            ctx = _TurnContext(
                get_latest_jpeg_fn=get_latest_jpeg_fn,
                get_fresh_jpeg_fn=get_fresh_jpeg_fn,
                response_queue=response_queue,
                tool_executing_event=asyncio.Event(),
            )

            try:
                await self._session.send_instruction(text=instruction, frame_jpeg=initial_frame)
                async for message in self._session.receive():
                    if await self._handle_server_message(message, ctx):
                        break
            except asyncio.CancelledError:
                log_with_tag(logger, "WARN", "Instruction task cancelled.")
                if self.robot is not None:
                    self.robot.interrupt()
                raise
            except Exception as e:
                log_with_tag(logger, "ERROR", f"Error during session loop: {e}")
                if self.robot is not None:
                    self.robot.interrupt()
                await response_queue.put({"type": "error", "content": f"Session error: {e}"})
                try:
                    await self._session.disconnect()
                except Exception as de:
                    log_with_tag(logger, "DEBUG", f"Error during session disconnect: {de}")
            finally:
                await response_queue.put(None)

    def process_instruction_stream(
        self, instruction: str, get_latest_jpeg_fn=None, get_fresh_jpeg_fn=None
    ) -> Generator[str, None, None]:
        """Process a user instruction and yield streaming text/status updates.

        Args:
            instruction: The text instruction from the user.
            get_latest_jpeg_fn: Optional callable returning pristine raw camera frame as JPEG bytes.
            get_fresh_jpeg_fn: Optional callable ``(t_after, timeout_s)`` returning a raw frame
                captured after ``t_after`` (``time.monotonic()``), used for post-motion
                measurements.

        Yields:
            Progressive response string for Gradio chatbot streaming.
        """
        if not instruction.strip():
            return

        self._begin_guardrail_instruction()

        if not self._api_available:
            msg = (
                "⚠️ Gemini API unavailable. Falling back to manual control. Check your credentials."
            )
            yield msg
            return

        # In real mode, block execution if workspace calibration is missing
        if self.mode == "real":
            try:
                get_ready_calibrator("real")
            except CalibrationUnavailable:
                block_msg = (
                    "⚠️ **Action Blocked — Workspace Not Calibrated**\n\n"
                    "The physical workspace has not been calibrated. The robot cannot safely "
                    "execute motions without coordinate calibration.\n\n"
                    "👉 **Please Calibrate the Workspace:**\n"
                    "1. Ensure all 4 ArUco markers are visible in the camera stream.\n"
                    "2. Open the **🎯 ArUco Workspace Calibration** panel on the left.\n"
                    "3. Click **'💾 Save Calibration'** once the 4 corners are detected."
                )
                yield block_msg
                return

        # Cancel any previous task if it's still running
        if self._current_async_task and not self._current_async_task.done():
            self._current_async_task.cancel()

        # Queue to pass responses from async loop to this generator
        response_queue = asyncio.Queue()

        # Schedule the async task in the background loop
        self._current_async_task = asyncio.run_coroutine_threadsafe(
            self._process_instruction_async(
                instruction, get_latest_jpeg_fn, response_queue, get_fresh_jpeg_fn
            ),
            self._loop,
        )

        assembler = ResponseAssembler()

        # Synchronously consume the queue
        while True:
            # wait for item in thread-safe way by running a future on the loop
            future = asyncio.run_coroutine_threadsafe(response_queue.get(), self._loop)
            item = future.result()

            if item is None:
                break

            yield assembler.feed(item)

    def interrupt(self) -> None:
        """Interrupt any ongoing execution and halt the robot."""
        self.guardrail_state.on_interrupt()
        if self.robot is not None:
            self.robot.interrupt()

        if self._current_async_task and not self._current_async_task.done():
            self._current_async_task.cancel()

    def clear_history(self) -> None:
        """Reset the active Gemini Live session."""
        if self._session:
            try:
                asyncio.run_coroutine_threadsafe(self._session.disconnect(), self._loop)
            except Exception as e:
                log_with_tag(
                    logger, "DEBUG", f"Error disconnecting session during clear_history: {e}"
                )
