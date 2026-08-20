"""Controller for managing Vision-Language-Action (VLA) agent interactions."""

import asyncio
import re
import threading
from typing import Generator

from google.genai import types

from agent.live_session import GeminiLiveSession
from agent.tools import ROBOT_TOOL_DECLARATIONS, execute_tool_call
from robot.interface import RobotInterface

# The model to use for the live streaming session
GEMINI_ROBOTICS_MODEL = "gemini-robotics-er-2-streaming-preview"


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
            mode: The operational mode of the robot ('real', 'sim', 'twin').
            robot: The initialized robot instance to execute commands against.
        """
        self.mode = mode
        self.robot = robot
        self.history: list[dict[str, str]] = []

        system_prompt = (
            f"You are the Gemini Robotics Assistant, an intelligent, conversational, and "
            f"autonomous 6-DOF robot arm orchestrator. The system is operating in "
            f"'{self.mode.upper()}' mode.\n\n"
            "### Conversational Interaction & Narration Rules (MANDATORY):\n"
            "- You are directly communicating with a human user in an interactive chat "
            "interface.\n"
            "- ALWAYS speak to the user! Never execute tool actions in complete silence.\n"
            "- When receiving an instruction, emit 1-2 conversational sentences in natural "
            "language explaining your visual observations, reasoning, and planned actions "
            "BEFORE calling any tool.\n"
            "- After every tool execution and observation frame, explain what you observe (e.g., "
            "'The gripper closed around the red cube; lifting now.') and state your next step.\n"
            "- If a user's instruction is conversational, general, or asks a question (e.g., "
            "'What do you see on the table?', 'Can you help me?', 'Hello!'), respond naturally, "
            "describe what is visible in the camera frame, and ask clarifying questions if "
            "needed.\n"
            "- If multiple candidate objects exist or user intent is ambiguous (e.g., 'Pick up the "
            "cube' when multiple cubes are present), ask the user to clarify before taking "
            "action.\n"
            "- When all steps are finished, provide a clear, friendly completion confirmation.\n"
            "- If an action fails or you encounter difficulty after retrying, transparently "
            "explain what you observe and ask the user for guidance or assistance.\n\n"
            "### Physical Workspace & Calibration Setup:\n"
            "- The camera is mounted on the LEFT side of the workspace, viewing the tabletop "
            "at an oblique angle.\n"
            "- The SO-101 robot arm is mounted on the RIGHT side of the camera's field of view, "
            "facing and reaching into the workspace towards the LEFT.\n"
            "- Forward (+X) for the robot points towards the LEFT of the camera frame.\n"
            "- Left (+Y) for the robot points towards the BOTTOM of the camera frame.\n"
            "- The workspace is planar-calibrated using 4 ArUco corner fiducials on the table "
            "surface (Z=0 plane): ID 0 (Top-Left), ID 1 (Top-Right), ID 2 (Bottom-Right), "
            "ID 3 (Bottom-Left).\n"
            + (
                "- In SIM mode, the workspace is pre-calibrated using a fixed virtual camera "
                "matching the physical camera view. ArUco fiducial markers (IDs 0-3) and workspace "
                "boundary guides are visible on the workspace floor at the 4 calibration corners.\n"
                if self.mode == "sim"
                else ""
            )
            + "\n"
            "### Available Tools:\n"
            "- `send_message(message='...')`: Speak, narrate your observations, explain your "
            "planned steps, ask clarifying questions, or confirm task completion to the user.\n"
            "- `move_to_spatial_point(point=[y, x], z_height_cm=3.0)`: Move end-effector to a "
            "camera-space [y, x] coordinate (0-1000). Use z_height_cm=3.0 to 5.0 for hover/transit "
            "clearance, and z_height_cm=0.0 to descend to table level for grasping.\n"
            "- `open_gripper()`: Open the gripper jaws fully.\n"
            "- `close_gripper()`: Close the gripper jaws fully to grasp an object.\n"
            "- `move_to_pose(qpos=[...], duration_sec=1.0)`: Move arm joints to explicit target "
            "angles (degrees).\n"
            "- `move_to_preset(pose_name='HOME')`: Move to named preset pose ('HOME', 'MIDDLE').\n"
            "- `get_joint_positions()`: Read current joint positions in degrees.\n"
            "- `get_end_effector_pose()`: Read current 3D position and rotation matrix "
            "(sim/twin mode).\n\n"
            "### Dynamic Closed-Loop Pick-and-Place Workflow:\n"
            "When instructed to pick up, move, or sort objects, execute actions dynamically:\n"
            "**CRITICAL ANTI-BATCHING RULE**: Do NOT batch multiple physical tools into a "
            "single response turn! You MUST execute ONE physical action "
            "(e.g. `move_to_spatial_point` or `close_gripper`), then wait for the tool "
            "response and fresh camera frame before deciding the next action. "
            "If you batch actions, you are executing open-loop and will fail to adapt to errors.\n"
            "1. **Clear Occlusion & Observe**: If not already at HOME, first call "
            "`move_to_preset(pose_name='HOME')` to give camera an unobstructed view.\n"
            "2. **Locate Target in Current Frame**: ONLY after arriving at HOME, inspect latest "
            "camera image to find target's normalized [y, x] coordinate (0-1000) at its "
            "**bottom table contact center**. Call `send_message` to state your observation.\n"
            "3. **Pre-Grasp & Open**: Call `open_gripper()` then "
            "`move_to_spatial_point(point=[y, x], z_height_cm=5.0)` to hover safely above target.\n"
            "4. **Descend & Grasp**: Call `move_to_spatial_point(point=[y, x], z_height_cm=0.0)` "
            "to lower jaws around the object, then call `close_gripper()`.\n"
            "5. **Lift to Hover**: Call `move_to_spatial_point(point=[y, x], z_height_cm=5.0)`.\n"
            "6. **Visual Inspection**: Look at the fresh camera frame after lifting:\n"
            "   - **Missed Grasp**: If the object is still resting on the tabletop surface and the "
            "gripper is empty, the grasp MISSED. Call `send_message` (e.g. 'Visual check: grasp "
            "missed, object is still on the table at [y, x]. Retrying.'), call `open_gripper()`, "
            "re-target the object's visible position, and retry (up to 3 attempts).\n"
            "   - **Successful Grasp**: If the object is visibly elevated between the jaws, call "
            "`send_message` and proceed to transport.\n"
            "7. **Transport & Release**: Call `move_to_spatial_point(point=[dest_y, dest_x], "
            "z_height_cm=5.0)`, descend to `z_height_cm=1.0`, call `open_gripper()`, and retract.\n"
            "8. **Return Home & Confirm**: Call `move_to_preset(pose_name='HOME')` and confirm "
            "completion with `send_message`.\n\n"
            "### CRITICAL Spatial Grounding Rule:\n"
            "- Because 3D coordinate mapping uses a 2D-to-3D planar homography calibrated on the "
            "table surface (Z=0), ALWAYS designate coordinates [y, x] at the exact **BOTTOM "
            "CONTACT CENTER** where the object touches the tabletop.\n"
            "- DO NOT target top faces, top edges, or upper corners of 3D objects to avoid "
            "perspective parallax error."
        )
        self.history.append({"role": "system", "content": system_prompt})

        self._api_available = True
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

        if self.mode == "sim":
            from simulation.sim_calibration import init_sim_workspace_calibration
            from ui.stream import get_calibrator

            calib = get_calibrator(mode="sim")
            init_sim_workspace_calibration(calibrator=calib)

        self._session: GeminiLiveSession | None = None
        self._active_instruction_lock = threading.Lock()
        self._async_instruction_lock = asyncio.Lock()
        self._current_async_task: asyncio.Future | None = None

        # We try to initialize the session to test credentials, but don't connect yet.
        try:
            self._session = GeminiLiveSession(
                model=GEMINI_ROBOTICS_MODEL,
                tools=ROBOT_TOOL_DECLARATIONS,
                system_instruction=system_prompt,
            )
        except ConnectionError as e:
            print(f"[VLAController] ⚠️ {e}")
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
                print(f"[VLAController] Connection attempt {attempt + 1} failed: {e}")
                await asyncio.sleep(delay)

        raise ConnectionError("Failed to connect to Gemini Live API after 3 attempts.")

    async def _stream_frames_during_execution(
        self, get_latest_jpeg_fn, tool_executing_event: asyncio.Event
    ) -> None:
        """Stream frames at ~1 FPS while a BLOCKING tool is executing."""
        if not self._session or not get_latest_jpeg_fn:
            return

        while tool_executing_event.is_set():
            jpeg = get_latest_jpeg_fn()
            if jpeg:
                try:
                    await self._session.send_frame(jpeg)
                except Exception as e:
                    print(f"[VLAController] Frame stream error: {e}")
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
            get_latest_jpeg_fn: Callable returning the latest camera frame as JPEG bytes.
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
                        print(f"[VLAController] Background frame stream error: {e}")
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
                        print(
                            f"[VLAController] 📩 Live Message: "
                            f"server_content={message.server_content is not None}, "
                            f"tool_call={message.tool_call is not None}",
                            flush=True,
                        )
                        if message.server_content and message.server_content.model_turn:
                            for part in message.server_content.model_turn.parts:
                                is_thought = getattr(part, "thought", False)
                                print(
                                    f"[VLAController] 📝 Part: thought={is_thought}, "
                                    f"text={part.text!r}",
                                    flush=True,
                                )
                                if is_thought:
                                    if part.text:
                                        await response_queue.put(
                                            {"type": "thought", "content": part.text}
                                        )
                                elif part.text:
                                    await response_queue.put({"type": "text", "content": part.text})
                                elif part.executable_code:
                                    pass  # not handling executable code in this scope

                        if message.tool_call:
                            # Handle tool calls
                            function_responses = []
                            for call in message.tool_call.function_calls:
                                print(
                                    f"[VLAController] 🔧 Tool Call: {call.name} "
                                    f"with args: {call.args}",
                                    flush=True,
                                )

                                if call.name == "send_message":
                                    msg_text = (
                                        call.args.get("message", "")
                                        if hasattr(call, "args")
                                        else ""
                                    )
                                    print(
                                        f"[VLAController] 💬 Model speech: {msg_text}",
                                        flush=True,
                                    )
                                    await response_queue.put({"type": "text", "content": msg_text})

                                    # If message mentions coordinates, preview targets overlay
                                    parsed_targets = parse_targets_from_text(msg_text)
                                    if parsed_targets:
                                        try:
                                            from ui.stream import set_model_targets

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
                                            await self._session.send_frame(post_frame)
                                        except Exception as e:
                                            print(
                                                f"[VLAController] Post-action frame error: {e}",
                                                flush=True,
                                            )

                                function_responses.append(
                                    types.FunctionResponse(
                                        name=call.name, id=call.id, response=result_dict
                                    )
                                )

                                if "error" in result_dict:
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
                                print("[VLAController] 🏁 Full turn complete received.", flush=True)
                                break
                            else:
                                print(
                                    "[VLAController] ℹ️ Sub-turn complete on tool_call message; "
                                    "waiting for model response to tool output.",
                                    flush=True,
                                )

            except asyncio.CancelledError:
                print("[VLAController] Instruction task cancelled.")
                raise
            except Exception as e:
                print(f"[VLAController] Error during session loop: {e}")
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
                    # if bg_streamer_task:
                    #     await bg_streamer_task
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
            get_latest_jpeg_fn: Optional callable returning latest camera frame as JPEG bytes.

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

        # In real or twin mode, block execution if workspace calibration is missing
        if self.mode in ("real", "twin"):
            from ui.stream import get_calibrator

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
