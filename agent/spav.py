"""Sense-Plan-Act-Verify (SPAV) closed-loop state machine for SO-101 robot arm.

Implements multi-turn evaluative execution:
- State 1 (Sense & Plan): Visual snapshot reasoning with gemini-robotics-er-2-preview.
- State 2 (Act): Calibrated coordinate transformation and quintic minimum-jerk motion.
- State 3 (Visual Feedback & Verification): Post-grasp hover snapshot evaluation.
- State 4 (Self-Correction Decision): Success transport or retry with visual delta offset.
"""

import asyncio
import json
import re
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from google import genai
from google.genai import types

from agent.live_session import create_genai_client
from robot.calibration import ArucoWorkspaceCalibrator
from robot.constants import PRESET_POSES
from robot.interface import RobotInterface
from robot.trajectory import GRIPPER_OPEN_RAD, TrajectoryPhase, TrajectoryPlanner

PRIMARY_SPAV_MODEL = "gemini-robotics-er-2-preview"
FALLBACK_SPAV_MODEL = "gemini-2.0-flash"


@dataclass
class SPAVResult:
    """Outcome of a SPAV execution attempt.

    Attributes:
        success: Whether the entire pick-and-place operation succeeded.
        pick_label: Identified name/label of the picked object.
        place_label: Identified name/label of the destination/bin.
        attempts: Number of attempts made.
        final_message: Human-readable summary message.
        pick_point: [y, x] normalized coordinates of final pick point.
        place_point: [y, x] normalized coordinates of place point.
    """

    success: bool
    pick_label: str = ""
    place_label: str | None = None
    attempts: int = 1
    final_message: str = ""
    pick_point: list[float] | None = None
    place_point: list[float] | None = None


def _clean_json_text(raw_text: str) -> str:
    """Strip markdown code block wrappers from raw model text."""
    text = raw_text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        return match.group(1).strip()
    return text


class SPAVStateMachine:
    """Evaluative Sense-Plan-Act-Verify closed-loop execution controller."""

    def __init__(
        self,
        robot: RobotInterface,
        calibrator: ArucoWorkspaceCalibrator | None = None,
        trajectory_planner: TrajectoryPlanner | None = None,
        max_retries: int = 3,
        genai_client: genai.Client | None = None,
    ) -> None:
        """Initialize SPAV state machine.

        Args:
            robot: Connected RobotInterface instance.
            calibrator: Optional ArucoWorkspaceCalibrator instance.
            trajectory_planner: Optional TrajectoryPlanner instance.
            max_retries: Maximum number of retry attempts for failed grasps.
            genai_client: Optional Google GenAI client instance.
        """
        self.robot = robot
        self.calibrator = calibrator if calibrator is not None else ArucoWorkspaceCalibrator()
        self.planner = (
            trajectory_planner
            if trajectory_planner is not None
            else TrajectoryPlanner(robot=self.robot)
        )
        self.max_retries = max_retries

        if genai_client is not None:
            self.client = genai_client
        else:
            try:
                self.client = create_genai_client()
            except Exception as e:
                print(f"[SPAV] Warning: Could not create GenAI client: {e}")
                self.client = None

    async def _query_gemini_vision(
        self,
        prompt: str,
        jpeg_bytes: bytes,
    ) -> str:
        """Query Gemini ER 2 model with image and prompt, falling back if needed."""
        if self.client is None:
            raise ConnectionError("Google GenAI client is not configured or authenticated.")

        models_to_try = [PRIMARY_SPAV_MODEL, FALLBACK_SPAV_MODEL]
        last_err: Exception | None = None

        for model_name in models_to_try:
            try:
                response = await asyncio.to_thread(
                    self.client.models.generate_content,
                    model=model_name,
                    contents=[
                        types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg"),
                        prompt,
                    ],
                    config=types.GenerateContentConfig(
                        thinking_config=types.ThinkingConfig(thinking_level="high")
                        if "er-2" in model_name
                        else None
                    ),
                )
                if response and response.text:
                    return response.text
            except Exception as e:
                print(f"[SPAV] Call to {model_name} failed: {e}")
                last_err = e

        raise RuntimeError(f"All SPAV model queries failed. Last error: {last_err}")

    async def execute_spav(
        self,
        instruction: str,
        get_latest_jpeg_fn: Callable[[], bytes | None],
        status_callback: Callable[[str], None] | None = None,
        stop_flag: Callable[[], bool] | None = None,
    ) -> SPAVResult:
        """Execute full closed-loop Sense-Plan-Act-Verify state machine.

        Args:
            instruction: Natural language instruction (e.g. "Pick up the green block").
            get_latest_jpeg_fn: Callable returning the latest camera frame JPEG bytes.
            status_callback: Optional callable to report real-time state transitions to UI.
            stop_flag: Optional callable returning True to halt execution.

        Returns:
            SPAVResult describing task outcome.
        """

        def emit(msg: str) -> None:
            print(f"[SPAV] {msg}")
            if status_callback is not None:
                status_callback(msg)

        emit(f"🚀 **Initiating Closed-Loop SPAV Pipeline** for: *'{instruction}'*")

        # -------------------------------------------------------------
        # CALIBRATION CHECK
        # -------------------------------------------------------------
        if not self.calibrator.is_calibrated:
            err = (
                "⚠️ Workspace Uncalibrated: SPAV pipeline requires an active ArUco workspace "
                "calibration. Please place 4 ArUco markers in view and save calibration in the UI."
            )
            emit(err)
            return SPAVResult(success=False, final_message=err)

        # -------------------------------------------------------------
        # STATE 1: SENSE & PLAN
        # -------------------------------------------------------------
        emit("🔍 **[State 1: Sense & Plan]** Capturing workspace snapshot and reasoning...")

        jpeg_snapshot = get_latest_jpeg_fn()
        if not jpeg_snapshot:
            err = "Camera frame snapshot unavailable for SPAV State 1."
            emit(f"⚠️ {err}")
            return SPAVResult(success=False, final_message=err)

        sense_prompt = (
            "You are the spatial reasoning brain of a 6-DOF SO-101 robot arm operating on "
            "a calibrated tabletop workspace.\n\n"
            "### Physical Calibration & Camera Setup:\n"
            "- The camera is mounted on the LEFT side of the workspace, viewing the tabletop "
            "at an oblique angle.\n"
            "- The SO-101 robot base is located on the RIGHT side of the image, facing and "
            "reaching into the workspace towards the LEFT.\n"
            "- Forward (+X) for the robot points towards the LEFT of the camera frame.\n"
            "- Left (+Y) for the robot points towards the BOTTOM of the camera frame.\n"
            "- The workspace is planar-calibrated using 4 ArUco corner fiducial markers on "
            "the tabletop surface (Z=0 plane):\n"
            "  * Marker ID 0: Top-Left in robot coordinates (far-left in camera view)\n"
            "  * Marker ID 1: Top-Right in robot coordinates (far-top in camera view)\n"
            "  * Marker ID 2: Bottom-Right in robot coordinates (near-top in camera view)\n"
            "  * Marker ID 3: Bottom-Left in robot coordinates (near-bottom in camera view)\n\n"
            "### CRITICAL Spatial Grounding Rule:\n"
            "- Because 3D coordinate mapping uses a 2D-to-3D planar homography calibrated "
            "on the table surface (Z=0), you MUST pinpoint the exact **BOTTOM CONTACT CENTER** "
            "where the target object touches the tabletop surface.\n"
            "- DO NOT point at the top face, top corners, or upper edges of 3D objects. "
            "Perspective parallax from the oblique side camera will shift the calculated "
            "real-world coordinates if you pick upper features instead of the table contact "
            "point.\n\n"
            f'User Instruction: "{instruction}"\n\n'
            "Analyze the image and return ONLY a valid JSON object with this exact structure:\n"
            "{\n"
            '  "pick_label": "identifying name of object to pick",\n'
            '  "pick_point": [y, x],\n'
            '  "place_label": "identifying name of target destination or container",\n'
            '  "place_point": [y, x] or null,\n'
            '  "plan_description": "concise description of the planned motion"\n'
            "}\n"
            "The points must be in [y, x] format normalized to 0-1000 where [0,0] is top-left "
            "and [1000, 1000] is bottom-right."
        )

        try:
            raw_sense = await self._query_gemini_vision(sense_prompt, jpeg_snapshot)
            cleaned_json = _clean_json_text(raw_sense)
            plan_data = json.loads(cleaned_json)
        except Exception as e:
            err = f"Failed to sense and parse spatial plan: {e}"
            emit(f"⚠️ {err}")
            return SPAVResult(success=False, final_message=err)

        pick_label = plan_data.get("pick_label", "target object")
        pick_point = plan_data.get("pick_point", [500, 500])
        place_label = plan_data.get("place_label", "destination bin")
        place_point = plan_data.get("place_point")

        p_str = f"[{pick_point[0]}, {pick_point[1]}]"
        emit(f"🎯 **[State 1 Complete]** Target: **'{pick_label}'** at {p_str}.")
        if place_point:
            pl_str = f"[{place_point[0]}, {place_point[1]}]"
            emit(f"📦 Destination: **'{place_label}'** at {pl_str}.")
        else:
            emit(f"📦 Destination: Using default calibrated bin for **'{place_label}'**.")

        # -------------------------------------------------------------
        # SPAV RETRY LOOP (States 2, 3, 4)
        # -------------------------------------------------------------
        current_pick_point = list(pick_point)
        attempts = 0

        hover_h_m = float(self.calibrator.config.get("default_hover_z_mm", 30)) / 1000.0 + 0.03
        grasp_z_m = float(self.calibrator.config.get("default_grasp_z_mm", 5)) / 1000.0

        # Destination 3D position
        if place_point:
            place_xyz = self.calibrator.gemini_to_robot_xyz(
                place_point, img_shape=(480, 640), z_m=grasp_z_m
            )
        else:
            default_bin = self.calibrator.config.get(
                "default_destination_bin_xyz_mm", {"x": 180, "y": 250, "z": 30}
            )
            place_xyz = np.array(
                [
                    float(default_bin["x"]) / 1000.0,
                    float(default_bin["y"]) / 1000.0,
                    float(default_bin["z"]) / 1000.0,
                ]
            )

        while attempts < self.max_retries:
            attempts += 1
            if stop_flag is not None and stop_flag():
                emit("🛑 Execution halted by user stop request.")
                return SPAVResult(
                    success=False,
                    pick_label=pick_label,
                    place_label=place_label,
                    attempts=attempts,
                    final_message="Interrupted by user.",
                )

            emit(f"\n🔄 **--- SPAV Attempt #{attempts} of {self.max_retries} ---**")

            # -------------------------------------------------------------
            # STATE 2: ACT (Generate & Execute Pick Trajectory)
            # -------------------------------------------------------------
            pick_xyz = self.calibrator.gemini_to_robot_xyz(
                current_pick_point, img_shape=(480, 640), z_m=grasp_z_m
            )
            xyz_display = f"[{pick_xyz[0]:.3f}, {pick_xyz[1]:.3f}, {pick_xyz[2]:.3f}]"
            emit(f"🦾 **[State 2: Act]** Trajectory calculated for XYZ {xyz_display} m.")
            emit("Executing smooth minimum-jerk motion: Hover ➔ Descend ➔ Grasp ➔ Retract...")

            current_q = self.robot.get_joint_positions()
            pick_phases = self.planner.generate_pick_and_place(
                pick_xyz=pick_xyz,
                place_xyz=None,  # Only execute pick half first for verification
                hover_height_m=hover_h_m,
                grasp_z_m=grasp_z_m,
                current_qpos=current_q,
                transit_duration=1.2,
                descend_duration=0.8,
                retract_duration=0.8,
            )

            success_motion = await asyncio.to_thread(
                self.planner.execute_trajectory,
                self.robot,
                pick_phases,
                stop_flag,
                0.02,
            )

            if not success_motion:
                emit("🛑 Motion interrupted during pick execution.")
                return SPAVResult(
                    success=False,
                    pick_label=pick_label,
                    place_label=place_label,
                    attempts=attempts,
                    final_message="Interrupted during pick execution.",
                )

            # -------------------------------------------------------------
            # STATE 3: VISUAL FEEDBACK & VERIFICATION
            # -------------------------------------------------------------
            emit("📸 **[State 3: Visual Verification]** Capturing snapshot at hover height...")
            await asyncio.sleep(0.3)  # Brief pause to let camera frame stabilize
            post_grasp_jpeg = get_latest_jpeg_fn()

            if not post_grasp_jpeg:
                emit("⚠️ Verification frame unavailable. Proceeding with optimistic decision.")
                verification_data = {"grasp_success": True, "confidence": 0.5}
            else:
                verify_prompt = (
                    "You are the visual verification module for a 6-DOF SO-101 robot arm.\n"
                    f"The robot attempted to grasp '{pick_label}' and is held at hover height.\n"
                    "- Inspect the image to determine if object is held between gripper jaws.\n"
                    "- If the object remains on the table (grasp failed), determine the spatial "
                    "offset [dy, dx] needed to center the next grasp directly onto the "
                    "object's bottom table contact center.\n\n"
                    "Return ONLY a valid JSON object with this exact structure:\n"
                    "{\n"
                    '  "grasp_success": true or false,\n'
                    '  "confidence": 0.0 to 1.0,\n'
                    '  "failure_reason": "explanation if failed, else null",\n'
                    '  "visual_offset_delta": [dy, dx] or null\n'
                    "}\n"
                    "If object is on table, visual_offset_delta is the [dy, dx] shift in\n"
                    "normalized units (-100 to +100) needed to center on the object's "
                    "bottom contact point."
                )

                try:
                    raw_verify = await self._query_gemini_vision(verify_prompt, post_grasp_jpeg)
                    cleaned_v_json = _clean_json_text(raw_verify)
                    verification_data = json.loads(cleaned_v_json)
                except Exception as e:
                    emit(f"⚠️ Verification query error ({e}). Assuming grasp succeeded.")
                    verification_data = {"grasp_success": True, "confidence": 0.6}

            grasp_ok = bool(verification_data.get("grasp_success", False))
            confidence = float(verification_data.get("confidence", 0.8))
            failure_reason = (
                verification_data.get("failure_reason") or "Object not detected in jaws"
            )
            offset_delta = verification_data.get("visual_offset_delta")

            # -------------------------------------------------------------
            # STATE 4: SELF-CORRECTION DECISION
            # -------------------------------------------------------------
            if grasp_ok:
                emit(f"✅ **[State 4: Decision]** Grasp SUCCESS (conf: {confidence:.2f})!")
                emit(f"🚚 Transporting '{pick_label}' to destination '{place_label}'...")

                # Plan transport and release
                curr_q = self.robot.get_joint_positions()
                transport_phases = self.planner.generate_pick_and_place(
                    pick_xyz=pick_xyz,
                    place_xyz=place_xyz,
                    hover_height_m=hover_h_m,
                    grasp_z_m=grasp_z_m,
                    current_qpos=curr_q,
                    transit_duration=1.2,
                    descend_duration=0.8,
                    retract_duration=0.8,
                )
                # Keep only phases E & F (transport + place)
                place_only_phases = [
                    p
                    for p in transport_phases
                    if p.name.startswith("E_") or p.name.startswith("F_")
                ]

                await asyncio.to_thread(
                    self.planner.execute_trajectory,
                    self.robot,
                    place_only_phases,
                    stop_flag,
                    0.02,
                )

                # Return safely to HOME pose
                emit("🏠 Returning arm to HOME pose...")
                home_rad = PRESET_POSES["HOME"]
                curr_q = self.robot.get_joint_positions()
                home_phase = self.planner.plan_joint_trajectory(
                    curr_q, home_rad, duration_sec=1.0, dt=0.02
                )

                await asyncio.to_thread(
                    self.planner.execute_trajectory,
                    self.robot,
                    [
                        TrajectoryPhase(
                            name="Home",
                            waypoints=home_phase,
                            phase_type="joint",
                            duration_sec=1.0,
                        )
                    ],
                    stop_flag,
                    0.02,
                )

                final_msg = (
                    f"🎉 Successfully picked up **'{pick_label}'** and placed at "
                    f"**'{place_label}'** in {attempts} attempt(s)."
                )
                emit(final_msg)
                return SPAVResult(
                    success=True,
                    pick_label=pick_label,
                    place_label=place_label,
                    attempts=attempts,
                    final_message=final_msg,
                    pick_point=current_pick_point,
                    place_point=place_point,
                )

            else:
                emit(f"❌ **[State 4: Decision]** Grasp FAILED (reason: *{failure_reason}*).")

                if attempts < self.max_retries:
                    # Open gripper for retry
                    emit("Reopening gripper and applying visual offset correction...")
                    curr_q = self.robot.get_joint_positions()
                    curr_q[5] = GRIPPER_OPEN_RAD
                    self.robot.set_joint_positions(curr_q)

                    # Apply delta offset if provided
                    if offset_delta and len(offset_delta) == 2:
                        dy, dx = float(offset_delta[0]), float(offset_delta[1])
                        # Bound delta to prevent large jumps
                        dy = max(-80.0, min(80.0, dy))
                        dx = max(-80.0, min(80.0, dx))
                        current_pick_point[0] = max(0.0, min(1000.0, current_pick_point[0] + dy))
                        current_pick_point[1] = max(0.0, min(1000.0, current_pick_point[1] + dx))
                        t_str = f"[{current_pick_point[0]:.1f}, {current_pick_point[1]:.1f}]"
                        emit(
                            f"🔧 Adjusted pick coord by [dy={dy:+.1f}, dx={dx:+.1f}] "
                            f"➔ New target: {t_str}."
                        )
                    else:
                        emit("Repeating grasp attempt at target position...")
                else:
                    emit(f"🛑 **[SPAV Aborted]** Maximum retry limit ({self.max_retries}) reached.")
                    # Park safely to HOME
                    emit("🏠 Parking arm safely to HOME pose...")
                    self.robot.move_to_pose(PRESET_POSES["HOME"], duration_sec=1.0)
                    fail_msg = (
                        f"Grasp failed after {self.max_retries} attempts for '{pick_label}'. "
                        f"Reason: {failure_reason}."
                    )
                    return SPAVResult(
                        success=False,
                        pick_label=pick_label,
                        place_label=place_label,
                        attempts=attempts,
                        final_message=fail_msg,
                        pick_point=current_pick_point,
                        place_point=place_point,
                    )

        return SPAVResult(success=False, final_message="SPAV execution completed with no action.")

    def execute_sync(
        self,
        instruction: str,
        get_latest_jpeg_fn: Callable[[], bytes | None],
        status_callback: Callable[[str], None] | None = None,
        stop_flag: Callable[[], bool] | None = None,
    ) -> SPAVResult:
        """Synchronous wrapper for execute_spav."""
        return asyncio.run(
            self.execute_spav(
                instruction=instruction,
                get_latest_jpeg_fn=get_latest_jpeg_fn,
                status_callback=status_callback,
                stop_flag=stop_flag,
            )
        )
