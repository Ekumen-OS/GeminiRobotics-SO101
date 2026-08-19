"""Controller for managing Vision-Language-Action (VLA) agent interactions."""

import asyncio
import threading
from typing import Generator

from google.genai import types

from agent.live_session import GeminiLiveSession
from agent.tools import ROBOT_TOOL_DECLARATIONS, execute_tool_call
from robot.interface import RobotInterface

# The model to use for the live streaming session
GEMINI_ROBOTICS_MODEL = "gemini-robotics-er-2-streaming-preview"


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
            f"You are the Gemini Robotics Assistant controlling the physical 6-DOF SO-101 "
            f"robot arm. The system is operating in '{self.mode.upper()}' mode.\n\n"
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
            "### Pick-and-Place Orchestration Pattern:\n"
            "When instructed to pick up, move, or sort objects, act as closed-loop orchestrator\n"
            "by calling tools sequentially and observing the resulting video stream:\n"
            "1. **Locate Target**: Identify the target object in the camera image and determine "
            "the normalized [y, x] coordinate (0-1000) of its **bottom table contact center**.\n"
            "2. **Pre-Grasp Hover**: Call `move_to_spatial_point(point=[y, x], z_height_cm=5.0)` "
            "to position the open gripper safely above the object.\n"
            "3. **Ensure Gripper Open**: Call `open_gripper()` if not already open.\n"
            "4. **Descend to Grasp**: Call `move_to_spatial_point(point=[y, x], z_height_cm=0.0)` "
            "to lower jaws around the target.\n"
            "5. **Grasp**: Call `close_gripper()` to secure the object.\n"
            "6. **Retract to Hover**: Call `move_to_spatial_point(point=[y, x], z_height_cm=5.0)` "
            "to lift the object.\n"
            "7. **Visual Verification**: Observe the live video feed. Verify whether the object "
            "is held in the gripper jaws.\n"
            "   - If grasp failed (object still on table): call `open_gripper()`, adjust [y, x] "
            "coordinates to correct any visual offset, and retry from step 2 (up to 3 attempts).\n"
            "   - If grasp succeeded: proceed to transport.\n"
            "8. **Transport to Destination**: Call `move_to_spatial_point(point=[dest_y, dest_x], "
            "z_height_cm=5.0)` to position above the destination bin or target location.\n"
            "9. **Lower & Release**: Call `move_to_spatial_point(point=[dest_y, dest_x], "
            "z_height_cm=1.0)` then `open_gripper()` to deposit the object.\n"
            "10. **Retract & Home**: Call `move_to_spatial_point(point=[dest_y, dest_x], "
            "z_height_cm=5.0)` then `move_to_preset(pose_name='HOME')`.\n\n"
            "### Closed-Loop Feedback & Verification Rules:\n"
            "- After executing actions, use incoming video frames (~1 FPS) to verify state before "
            "proceeding.\n"
            "- Explain your reasoning, visual observations, and planned actions clearly.\n"
            "- If an operation does not achieve the goal, self-correct by adjusting target "
            "coordinates and retrying.\n\n"
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

            try:
                await self._session.send_instruction(text=instruction, frame_jpeg=initial_frame)

                tool_executing_event = asyncio.Event()

                async for message in self._session.receive():
                    if isinstance(message, types.LiveServerMessage):
                        if message.server_content and message.server_content.model_turn:
                            for part in message.server_content.model_turn.parts:
                                if part.text:
                                    await response_queue.put({"type": "text", "content": part.text})
                                elif part.executable_code:
                                    pass  # not handling executable code in this scope

                        if message.tool_call:
                            # Handle tool calls
                            function_responses = []
                            for call in message.tool_call.function_calls:
                                await response_queue.put(
                                    {"type": "status", "content": f"⚙️ Executing `{call.name}`..."}
                                )

                                is_blocking = any(
                                    d["name"] == call.name and d.get("behavior") == "BLOCKING"
                                    for d in ROBOT_TOOL_DECLARATIONS
                                )

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
                                    await frame_task

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
                            # Turn complete, end of processing
                            break

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

        # Synchronously consume the queue
        while True:
            # wait for item in thread-safe way by running a future on the loop
            future = asyncio.run_coroutine_threadsafe(response_queue.get(), self._loop)
            item = future.result()

            if item is None:
                break

            if item["type"] == "text":
                full_response += item["content"]
                yield full_response
            elif item["type"] == "status":
                full_response += f"\n*{item['content']}*\n"
                yield full_response
            elif item["type"] == "error":
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
