"""Controller for managing Vision-Language-Action (VLA) agent interactions."""

import asyncio
import logging
import re
import sys
import threading
from pathlib import Path
from typing import Any, Generator

from google.genai import types

from agent.live_session import GeminiLiveSession
from agent.tools import ROBOT_TOOL_DECLARATIONS, execute_tool_call
from robot.interface import RobotInterface
from robot.logger import log_with_tag
from robot.settings import get_settings
from vision import get_calibrator, set_model_targets

logger = logging.getLogger(__name__)


def _get_gemini_model() -> str:
    """Return configured Gemini robotics model ID."""
    return get_settings().gemini_model_id


def __getattr__(name: str) -> Any:
    """Resolve dynamic constants."""
    if name == "GEMINI_ROBOTICS_MODEL":
        return _get_gemini_model()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def load_system_prompt(mode: str, tolerance: int | float | None = None) -> str:
    """Load and render the VLA system instruction prompt.

    Args:
        mode: Operational mode ('real' or 'sim').
        tolerance: Pixel alignment tolerance. Defaults to settings value.

    Returns:
        Rendered markdown prompt string.
    """
    if tolerance is None:
        tolerance = get_settings().alignment_tolerance_px
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
        tolerance=tolerance,
        sim_workspace_note=sim_note,
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
        self.history: list[dict[str, str]] = []
        system_prompt = load_system_prompt(mode=self.mode)
        self.history.append({"role": "system", "content": system_prompt})

        self._api_available = True
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

        if self.mode == "sim":
            from simulation.sim_calibration import init_sim_workspace_calibration

            calib = get_calibrator(mode="sim")
            init_sim_workspace_calibration(calibrator=calib)

        self._session: GeminiLiveSession | None = None
        self._active_instruction_lock = threading.Lock()
        self._async_instruction_lock = asyncio.Lock()
        self._current_async_task: asyncio.Future | None = None

        # We try to initialize the session to test credentials, but don't connect yet.
        try:
            self._session = GeminiLiveSession(
                model=_get_gemini_model(),
                tools=ROBOT_TOOL_DECLARATIONS,
                system_instruction=system_prompt,
            )
        except ConnectionError as e:
            log_with_tag(logger, "WARN", f"⚠️ {e}")
            self._api_available = False

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
            except Exception as e:
                log_with_tag(
                    logger,
                    "ERROR",
                    f"Connection attempt {attempt + 1} failed: {e}",
                )
                await asyncio.sleep(delay)

        raise ConnectionError("Failed to connect to Gemini Live API after 3 attempts.")

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
                except Exception as e:
                    log_with_tag(logger, "ERROR", f"Frame stream error: {e}")
                    break
            await asyncio.sleep(1.0)

    async def _stream_frames_in_background(
        self,
        get_latest_jpeg_fn,
        stop_event: asyncio.Event,
        tool_executing_event: asyncio.Event,
    ) -> None:
        """Continuously stream frames at ~1 FPS between tool calls while model is reasoning.

        Skips sending when a BLOCKING tool is executing since _stream_frames_during_execution
        handles tool execution periods.

        Args:
            get_latest_jpeg_fn: Callable returning latest pristine raw camera frame as JPEG bytes.
            stop_event: Event signaling the background streamer to terminate.
            tool_executing_event: Event indicating whether a BLOCKING tool is currently executing.
        """
        if not self._session or not get_latest_jpeg_fn:
            return

        while not stop_event.is_set():
            if not tool_executing_event.is_set():
                jpeg = get_latest_jpeg_fn()
                if jpeg:
                    try:
                        await self._session.send_frame(jpeg)
                    except Exception as e:
                        log_with_tag(logger, "ERROR", f"Background frame stream error: {e}")
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=3.0)
            except asyncio.TimeoutError:
                pass

    async def _process_instruction_async(
        self, instruction: str, get_latest_jpeg_fn, response_queue: asyncio.Queue
    ) -> None:
        """Async core for processing an instruction and running the tool-call loop."""
        if not self._session:
            await response_queue.put({"type": "error", "content": "API unavailable"})
            await response_queue.put(None)
            return

        try:
            await self._ensure_connected()
        except ConnectionError as e:
            await response_queue.put({"type": "error", "content": str(e)})
            await response_queue.put(None)
            return

        log_with_tag(logger, "DECISION", f"🚀 Received user instruction: {instruction!r}")

        async with self._async_instruction_lock:
            # Inject initial snapshot
            initial_frame = get_latest_jpeg_fn() if get_latest_jpeg_fn else None

            bg_stream_stop_event = asyncio.Event()
            tool_executing_event = asyncio.Event()
            bg_stream_task = None

            try:
                await self._session.send_instruction(text=instruction, frame_jpeg=initial_frame)

                # Disable background streamer to prevent overwhelming the Live API
                # bg_streamer_task = asyncio.create_task(
                #     self._stream_frames_in_background(
                #         get_latest_jpeg_fn, bg_stream_stop_event, tool_executing_event
                #     )
                # )

                async for message in self._session.receive():
                    if isinstance(message, types.LiveServerMessage):
                        log_with_tag(
                            logger,
                            "INFO",
                            f"📩 Live Message: "
                            f"server_content={message.server_content is not None}, "
                            f"tool_call={message.tool_call is not None}",
                        )
                        if message.server_content and message.server_content.model_turn:
                            for part in message.server_content.model_turn.parts:
                                is_thought = getattr(part, "thought", False)
                                if is_thought:
                                    log_with_tag(logger, "REASONING", f"💭 {part.text}")
                                    if part.text:
                                        await response_queue.put(
                                            {"type": "thought", "content": part.text}
                                        )
                                elif part.text:
                                    log_with_tag(logger, "DECISION", f"📝 Model text: {part.text}")
                                    await response_queue.put({"type": "text", "content": part.text})
                                elif part.executable_code:
                                    pass  # not handling executable code in this scope

                        if message.tool_call:
                            # Handle tool calls
                            function_responses = []
                            for call in message.tool_call.function_calls:
                                log_with_tag(
                                    logger,
                                    "TOOL_CALL",
                                    f"🔧 Dispatching '{call.name}' [ID: {call.id}] | "
                                    f"Payload: {call.args}",
                                )

                                if call.name == "send_message":
                                    msg_text = (
                                        call.args.get("message", "")
                                        if hasattr(call, "args")
                                        else ""
                                    )
                                    log_with_tag(
                                        logger,
                                        "DECISION",
                                        f"💬 Model speech: {msg_text}",
                                    )
                                    await response_queue.put({"type": "text", "content": msg_text})

                                    # If message mentions coordinates, preview targets overlay
                                    parsed_targets = parse_targets_from_text(msg_text)
                                    if parsed_targets:
                                        try:
                                            set_model_targets(parsed_targets)
                                        except Exception:
                                            pass

                                    function_responses.append(
                                        types.FunctionResponse(
                                            name=call.name,
                                            id=call.id,
                                            response={"status": "success"},
                                        )
                                    )
                                    log_with_tag(
                                        logger,
                                        "TOOL_RESULT",
                                        f"✅ Tool 'send_message' completed [ID: {call.id}] | "
                                        "Result: {'status': 'success'}",
                                    )
                                    continue

                                await response_queue.put(
                                    {"type": "status", "content": f"⚙️ Executing `{call.name}`..."}
                                )

                                is_blocking = any(
                                    d["name"] == call.name and d.get("behavior") == "BLOCKING"
                                    for d in ROBOT_TOOL_DECLARATIONS
                                )

                                frame_task = None
                                if is_blocking:
                                    tool_executing_event.set()
                                    # Start frame streaming task
                                    frame_task = asyncio.create_task(
                                        self._stream_frames_during_execution(
                                            get_latest_jpeg_fn, tool_executing_event
                                        )
                                    )

                                def tool_status_callback(status_msg: str) -> None:
                                    asyncio.run_coroutine_threadsafe(
                                        response_queue.put(
                                            {"type": "status", "content": status_msg}
                                        ),
                                        self._loop,
                                    )

                                # Execute tool
                                result_dict = await execute_tool_call(
                                    call=call,
                                    robot=self.robot,
                                    get_latest_jpeg_fn=get_latest_jpeg_fn,
                                    status_callback=tool_status_callback,
                                    mode=self.mode,
                                )

                                if is_blocking:
                                    tool_executing_event.clear()
                                    if frame_task:
                                        await frame_task

                                    # Send ONE fresh post-action frame before the tool response
                                    # This guarantees the image is in context when the model resumes
                                    await asyncio.sleep(0.3)
                                    post_frame = (
                                        get_latest_jpeg_fn() if get_latest_jpeg_fn else None
                                    )
                                    if post_frame:
                                        try:
                                            log_with_tag(
                                                logger,
                                                "STREAM",
                                                "📸 Injected post-action observation frame "
                                                f"({len(post_frame)} bytes)",
                                            )
                                            await self._session.send_frame(post_frame)
                                        except Exception as e:
                                            log_with_tag(
                                                logger,
                                                "ERROR",
                                                f"Post-action frame error: {e}",
                                            )

                                function_responses.append(
                                    types.FunctionResponse(
                                        name=call.name, id=call.id, response=result_dict
                                    )
                                )

                                has_error = "error" in result_dict
                                status_emoji = "⚠️" if has_error else "✅"
                                log_with_tag(
                                    logger,
                                    "TOOL_RESULT",
                                    f"{status_emoji} Tool '{call.name}' completed [ID: {call.id}] "
                                    f"| Result: {result_dict}",
                                    level=logging.ERROR if has_error else logging.INFO,
                                    stream=sys.stderr if has_error else sys.stdout,
                                )

                                if has_error:
                                    err_msg = result_dict["error"]
                                    await response_queue.put(
                                        {
                                            "type": "status",
                                            "content": f"⚠️ `{call.name}` error: {err_msg}",
                                        }
                                    )
                                else:
                                    await response_queue.put(
                                        {
                                            "type": "status",
                                            "content": f"✅ `{call.name}` completed.",
                                        }
                                    )

                            if function_responses:
                                await self._session.send_tool_response(function_responses)

                        if message.server_content and message.server_content.turn_complete:
                            if not message.tool_call:
                                log_with_tag(
                                    logger,
                                    "DECISION",
                                    "🏁 Full turn complete received.",
                                )
                                break
                            else:
                                log_with_tag(
                                    logger,
                                    "INFO",
                                    "ℹ️ Sub-turn complete on tool_call message; "
                                    "waiting for model response to tool output.",
                                )

            except asyncio.CancelledError:
                log_with_tag(logger, "WARN", "Instruction task cancelled.")
                raise
            except Exception as e:
                log_with_tag(logger, "ERROR", f"Error during session loop: {e}")
                await response_queue.put({"type": "error", "content": f"Session error: {e}"})
                # Try to disconnect so it can reconnect cleanly
                try:
                    await self._session.disconnect()
                except Exception:
                    pass
            finally:
                bg_stream_stop_event.set()
                if bg_stream_task and not bg_stream_task.done():
                    bg_stream_task.cancel()
                    try:
                        await bg_stream_task
                    except asyncio.CancelledError:
                        pass
                await response_queue.put(None)  # Signal completion

    def process_instruction_stream(
        self, instruction: str, get_latest_jpeg_fn=None
    ) -> Generator[str, None, None]:
        """Process a user instruction and yield streaming text/status updates.

        Args:
            instruction: The text instruction from the user.
            get_latest_jpeg_fn: Optional callable returning pristine raw camera frame as JPEG bytes.

        Yields:
            Progressive response string for Gradio chatbot streaming.
        """
        if not instruction.strip():
            return

        self.history.append({"role": "user", "content": instruction})

        if not self._api_available:
            msg = (
                "⚠️ Gemini API unavailable. Falling back to manual control. Check your credentials."
            )
            self.history.append({"role": "assistant", "content": msg})
            yield msg
            return

        # In real mode, block execution if workspace calibration is missing
        if self.mode == "real":
            calib = get_calibrator()
            if calib is None or not calib.is_calibrated:
                block_msg = (
                    "⚠️ **Action Blocked — Workspace Not Calibrated**\n\n"
                    "The physical workspace has not been calibrated. The robot cannot safely "
                    "execute motions without coordinate calibration.\n\n"
                    "👉 **Please Calibrate the Workspace:**\n"
                    "1. Ensure all 4 ArUco markers are visible in the camera stream.\n"
                    "2. Open the **🎯 ArUco Workspace Calibration** panel on the left.\n"
                    "3. Click **'💾 Save Calibration'** once the 4 corners are detected."
                )
                self.history.append({"role": "assistant", "content": block_msg})
                yield block_msg
                return

        # Cancel any previous task if it's still running
        if self._current_async_task and not self._current_async_task.done():
            self._current_async_task.cancel()

        # Queue to pass responses from async loop to this generator
        response_queue = asyncio.Queue()

        # Schedule the async task in the background loop
        self._current_async_task = asyncio.run_coroutine_threadsafe(
            self._process_instruction_async(instruction, get_latest_jpeg_fn, response_queue),
            self._loop,
        )

        full_response = ""
        in_thought_block = False

        # Synchronously consume the queue
        while True:
            # wait for item in thread-safe way by running a future on the loop
            future = asyncio.run_coroutine_threadsafe(response_queue.get(), self._loop)
            item = future.result()

            if item is None:
                if in_thought_block:
                    full_response += "*\n\n"
                    in_thought_block = False
                break

            if item["type"] == "thought":
                if not in_thought_block:
                    if full_response and not full_response.endswith("\n"):
                        full_response += "\n\n"
                    full_response += "> 💭 *"
                    in_thought_block = True
                full_response += item["content"]
                yield full_response
            elif item["type"] == "text":
                if in_thought_block:
                    full_response += "*\n\n"
                    in_thought_block = False
                if (
                    full_response
                    and not full_response.endswith("\n")
                    and not full_response.endswith(" ")
                ):
                    full_response += "\n\n"
                full_response += item["content"]
                yield full_response
            elif item["type"] == "status":
                if in_thought_block:
                    full_response += "*\n\n"
                    in_thought_block = False
                if full_response and not full_response.endswith("\n"):
                    full_response += "\n"
                full_response += f"{item['content']}\n"
                yield full_response
            elif item["type"] == "error":
                if in_thought_block:
                    full_response += "*\n\n"
                    in_thought_block = False
                if full_response and not full_response.endswith("\n"):
                    full_response += "\n"
                full_response += f"\n**Error:** {item['content']}\n"
                yield full_response

        self.history.append({"role": "assistant", "content": full_response.strip()})

    def process_instruction(self, instruction: str, context_images: list | None = None) -> str:
        """Process a user instruction and return the agent's full response.

        Args:
            instruction: The text instruction from the user.
            context_images: Ignored. Used only for backward compatibility.

        Returns:
            The agent's complete response.
        """
        # Consume the entire stream and return the final string
        final_text = ""
        for chunk in self.process_instruction_stream(instruction):
            final_text = chunk
        return final_text

    def interrupt(self) -> None:
        """Interrupt any ongoing execution and halt the robot."""
        if self.robot is not None:
            if hasattr(self.robot, "stop"):
                self.robot.stop()
            elif hasattr(self.robot, "_stop_requested"):
                self.robot._stop_requested = True

        if self._current_async_task and not self._current_async_task.done():
            self._current_async_task.cancel()

    def get_history(self) -> list[dict[str, str]]:
        """Retrieve the current conversation history."""
        return self.history

    def clear_history(self) -> None:
        """Clear the conversation history, keeping only the system prompt."""
        if self.history:
            system_prompt = self.history[0]
            self.history = [system_prompt]

        # Reset session
        if self._session:
            asyncio.run_coroutine_threadsafe(self._session.disconnect(), self._loop)
