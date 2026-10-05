"""Robot tool declarations and execution adapters for Gemini Live API."""

import asyncio
import logging
from collections.abc import Callable
from typing import Any

import numpy as np

from agent.contracts import GripperToolResponse, MoveToSpatialPointResponse
from robot.cartesian_ik import CartesianIK
from robot.constants import JOINT_LIMITS, PRESET_POSES
from robot.contracts import VisualAlignmentResponse
from robot.interface import RobotInterface
from robot.logger import log_compat, log_with_tag
from robot.settings import get_settings

logger = logging.getLogger(__name__)

# Retain _log alias for backwards compatibility with tests importing _log from agent.tools
_log = log_compat


def build_move_spatial_response(
    target_xyz_m: list[float],
    executed_qpos_deg: list[float],
    approach: str,
    visual_hint: str,
    perception: VisualAlignmentResponse | dict[str, Any] | None = None,
    warnings: list[str] | None = None,
) -> MoveToSpatialPointResponse:
    """Build a typed response payload for move_to_spatial_point tool execution."""
    resp: MoveToSpatialPointResponse = {
        "status": "success",
        "target_xyz_m": target_xyz_m,
        "executed_qpos_deg": executed_qpos_deg,
        "approach": approach,
        "visual_hint": visual_hint,
    }
    if perception is not None:
        resp["perception"] = perception
    if warnings:
        resp["warnings"] = warnings
    return resp


def build_gripper_response(
    gripper_state: str,
    gripper_deg: float,
    executed_qpos_deg: list[float],
    visual_hint: str,
    warnings: list[str] | None = None,
) -> GripperToolResponse:
    """Build a typed response payload for open_gripper / close_gripper tool execution."""
    resp: GripperToolResponse = {
        "status": "success",
        "gripper_state": gripper_state,
        "gripper_deg": gripper_deg,
        "executed_qpos_deg": executed_qpos_deg,
        "visual_hint": visual_hint,
    }
    if warnings:
        resp["warnings"] = warnings
    return resp


# Gripper limits derived from JOINT_LIMITS in degrees
GRIPPER_OPEN_DEG = JOINT_LIMITS["gripper"][1]
GRIPPER_CLOSE_DEG = JOINT_LIMITS["gripper"][0]

ROBOT_TOOL_DECLARATIONS = [
    {
        "name": "send_message",
        "description": (
            "Speak, narrate, or deliver a conversational message to the user. "
            "Use this to explain your thoughts, describe visual observations, "
            "ask questions, or confirm task progress."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "message": {
                    "type": "STRING",
                    "description": "The natural language message to narrate to the user.",
                },
            },
            "required": ["message"],
        },
    },
    {
        "name": "move_to_pose",
        "description": "Smoothly interpolate motion to a 6-DOF joint target.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "qpos": {
                    "type": "ARRAY",
                    "description": "Target joint angles in degrees for all 6 joints.",
                    "items": {"type": "NUMBER"},
                },
                "duration_sec": {
                    "type": "NUMBER",
                    "description": "Duration of the movement in seconds.",
                },
            },
            "required": ["qpos", "duration_sec"],
        },
    },
    {
        "name": "set_joint_positions",
        "description": "Instant or snap joint position command to 6-DOF joint target.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "qpos": {
                    "type": "ARRAY",
                    "description": "Target joint angles in degrees for all 6 joints.",
                    "items": {"type": "NUMBER"},
                },
            },
            "required": ["qpos"],
        },
    },
    {
        "name": "get_joint_positions",
        "description": "Read current joint state (informational, non-blocking). Returns degrees.",
        "parameters": {
            "type": "OBJECT",
            "properties": {},
        },
    },
    {
        "name": "open_gripper",
        "description": "Convenience wrapper to open the gripper fully.",
        "parameters": {
            "type": "OBJECT",
            "properties": {},
        },
    },
    {
        "name": "close_gripper",
        "description": "Convenience wrapper to close the gripper fully.",
        "parameters": {
            "type": "OBJECT",
            "properties": {},
        },
    },
    {
        "name": "move_to_preset",
        "description": "Move the robot to a predefined preset pose (e.g., 'HOME', 'MIDDLE').",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "pose_name": {
                    "type": "STRING",
                    "description": "Name of the preset pose.",
                },
            },
            "required": ["pose_name"],
        },
    },
    {
        "name": "get_end_effector_pose",
        "description": (
            "Read current end-effector XYZ position and rotation matrix. "
            "Available in sim or twin mode."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {},
        },
    },
    {
        "name": "move_to_spatial_point",
        "description": (
            "Move the robot end-effector to a 2D point visible in the camera feed. "
            "Provide a normalized [y, x] coordinate (0-1000) from the camera image. "
            "The system maps this to 3D workspace coordinates using ArUco workspace calibration "
            "and solves inverse kinematics to reach the target. "
            "Specify z_height_cm to control vertical offset above table (0 for table contact, "
            "3-5 for hover/transit)."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "point": {
                    "type": "ARRAY",
                    "description": "Normalized [y, x] image coordinate (0-1000) of the target.",
                    "items": {"type": "NUMBER"},
                },
                "z_height_cm": {
                    "type": "NUMBER",
                    "description": (
                        "Height above the table surface in centimeters (default: 3.0). "
                        "Use 0 for grasping contact at table level, 3-5 for safe hover."
                    ),
                },
                "approach": {
                    "type": "STRING",
                    "description": (
                        "Approach mode (REQUIRED): 'top_down' (vertical grasp perpendicular "
                        "to table, MANDATORY for all tabletop pick-and-place, grasping, and "
                        "table descends) or 'horizontal' (lateral approach)."
                    ),
                    "enum": ["top_down", "horizontal"],
                },
            },
            "required": ["point", "approach"],
        },
    },
]

# Set BLOCKING behavior for motion tools
_BLOCKING_TOOLS = {
    "move_to_pose",
    "set_joint_positions",
    "open_gripper",
    "close_gripper",
    "move_to_preset",
    "move_to_spatial_point",
}
for decl in ROBOT_TOOL_DECLARATIONS:
    if decl["name"] in _BLOCKING_TOOLS:
        decl["behavior"] = "BLOCKING"

# Lazy-initialized IK solver (avoids loading MuJoCo XML until first spatial call)
_ik_solver: CartesianIK | None = None


def _get_ik_solver() -> CartesianIK:
    """Return a lazily-initialized CartesianIK solver singleton.

    Returns:
        The shared CartesianIK solver instance.
    """
    global _ik_solver
    if _ik_solver is None:
        _ik_solver = CartesianIK()
    return _ik_solver


DEFAULT_HOVER_Z = 0.03  # 3 cm above table


def image_point_to_xyz(
    point: list[float],
    hover_z: float = 0.03,
    img_shape: tuple[int, int] = (480, 640),
    mode: str | None = None,
) -> np.ndarray:
    """Transform a normalized Gemini coordinate [y, x] to 3D robot base XYZ.

    Args:
        point: Normalized image point [y, x] where 0 <= y, x <= 1000.
        hover_z: Height above the table surface in metres.
        img_shape: Frame dimensions as (height, width).
        mode: Optional operational mode ('real', 'sim', 'twin'). If None, resolved
            automatically from active robot state.

    Returns:
        A (3,) numpy array [X, Y, Z] in the robot base frame (metres).

    Raises:
        RuntimeError: If the workspace is not calibrated.
    """
    from robot.calibration import DEFAULT_SAVE_PATH, DEFAULT_SIM_SAVE_PATH
    from ui.state import get_active_robot
    from ui.stream import get_calibrator

    if mode is None:
        active_robot = get_active_robot()
        is_sim = False
        if active_robot is not None:
            mode_attr = getattr(active_robot, "mode", None)
            if mode_attr == "sim":
                is_sim = True
            elif mode_attr == "real":
                is_sim = False
            elif hasattr(active_robot, "render_rgb_frame") or hasattr(active_robot, "model"):
                is_sim = True
        active_mode = "sim" if is_sim else "real"
    else:
        active_mode = mode
        is_sim = active_mode == "sim"

    calib = get_calibrator(mode=active_mode)
    target_path = DEFAULT_SIM_SAVE_PATH if is_sim else DEFAULT_SAVE_PATH

    if calib is not None:
        if is_sim and (getattr(calib, "mode", None) != "sim" or not calib.is_calibrated):
            if target_path.exists():
                calib.load_calibration(target_path)
                calib.mode = "sim"
            else:
                from simulation.sim_calibration import init_sim_workspace_calibration

                calib = init_sim_workspace_calibration(calibrator=calib)
        elif not is_sim and (getattr(calib, "mode", None) != "real" or not calib.is_calibrated):
            if target_path.exists():
                calib.load_calibration(target_path)
                calib.mode = "real"

    if calib is None or not calib.is_calibrated:
        raise RuntimeError(
            "Workspace is not calibrated. Please ensure 4 ArUco markers are in view and "
            "save calibration under the ArUco Workspace Calibration panel before executing "
            "spatial commands."
        )

    return calib.gemini_to_robot_xyz(point, img_shape=img_shape, z_m=hover_z)


def validate_and_clamp_joints(qpos_deg: list[float]) -> tuple[list[float], list[str]]:
    """Validate joint angles against JOINT_LIMITS and clamp if necessary.

    Args:
        qpos_deg: Joint angles in degrees.

    Returns:
        A tuple of (clamped_qpos_deg, warnings).
    """
    joint_names = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    ]
    clamped_qpos_deg = []
    warnings = []
    for i, angle in enumerate(qpos_deg):
        if i < len(joint_names):
            jname = joint_names[i]
            min_val, max_val = JOINT_LIMITS[jname]
            clamped_val = min(max(angle, min_val), max_val)
            if clamped_val != angle:
                warn_msg = (
                    f"{jname} was clamped from {angle:.1f}° to {clamped_val:.1f}° (limit exceeded)"
                )
                warnings.append(warn_msg)
                log_with_tag(logger, "WARN", warn_msg)
            clamped_qpos_deg.append(clamped_val)
        else:
            clamped_qpos_deg.append(angle)
    return clamped_qpos_deg, warnings


async def execute_tool_call(
    call: Any,
    robot: RobotInterface | None,
    get_latest_jpeg_fn: Callable[[], bytes | None] | None = None,
    status_callback: Callable[[str], None] | None = None,
) -> dict[str, Any] | GripperToolResponse | MoveToSpatialPointResponse:
    """Dispatch tool call to the RobotInterface asynchronously.

    Args:
        call: The tool call object with name and args.
        robot: The initialized RobotInterface instance.
        get_latest_jpeg_fn: Optional callable returning latest camera frame JPEG bytes.
        status_callback: Optional callable for streaming status messages to UI.

    Returns:
        A dictionary containing the response result to be sent back to Gemini.
    """
    name = call.name
    args = call.args if hasattr(call, "args") else {}

    if name == "send_message":
        message_text = args.get("message", "")
        return {"status": "success", "message": message_text}

    if robot is None:
        err_msg = "RobotInterface is not initialized or injected."
        log_with_tag(logger, "ERROR", err_msg)
        return {"error": err_msg}

    try:
        from ui.state import get_active_robot, set_active_robot

        if get_active_robot() is None:
            set_active_robot(robot)
    except Exception:
        pass

    try:
        if name == "move_to_pose":
            qpos_deg = args.get("qpos", [])
            duration_sec = args.get("duration_sec", 1.0)
            if len(qpos_deg) != 6:
                err_msg = f"move_to_pose expects 6 joint angles, got {len(qpos_deg)}"
                log_with_tag(logger, "ERROR", err_msg)
                return {"error": err_msg}
            clamped_qpos, warnings = validate_and_clamp_joints(qpos_deg)
            qpos_rad = np.radians(clamped_qpos)
            robot.move_to_pose(qpos_rad, duration_sec=duration_sec)
            response = {"status": "success", "executed_qpos_deg": clamped_qpos}
            if warnings:
                response["warnings"] = warnings
            return response

        elif name == "set_joint_positions":
            qpos_deg = args.get("qpos", [])
            if len(qpos_deg) != 6:
                err_msg = f"set_joint_positions expects 6 joint angles, got {len(qpos_deg)}"
                log_with_tag(logger, "ERROR", err_msg)
                return {"error": err_msg}
            clamped_qpos, warnings = validate_and_clamp_joints(qpos_deg)
            qpos_rad = np.radians(clamped_qpos)
            robot.set_joint_positions(qpos_rad)
            response = {"status": "success", "executed_qpos_deg": clamped_qpos}
            if warnings:
                response["warnings"] = warnings
            return response

        elif name == "get_joint_positions":
            qpos_rad = robot.get_joint_positions()
            qpos_deg = [float(v) for v in np.degrees(qpos_rad)]
            return {"qpos_deg": qpos_deg}

        elif name == "open_gripper":
            qpos_rad = robot.get_joint_positions()
            qpos_deg = [float(v) for v in np.degrees(qpos_rad)]
            qpos_deg[5] = float(GRIPPER_OPEN_DEG)
            clamped_qpos, warnings = validate_and_clamp_joints(qpos_deg)
            robot.move_to_pose(np.radians(clamped_qpos), duration_sec=0.5)
            return build_gripper_response(
                gripper_state="open",
                gripper_deg=float(GRIPPER_OPEN_DEG),
                executed_qpos_deg=clamped_qpos,
                visual_hint=(
                    "Gripper opened. Inspect camera frame to visually verify object release and "
                    "clearance."
                ),
                warnings=warnings if warnings else None,
            )

        elif name == "close_gripper":
            qpos_rad = robot.get_joint_positions()
            qpos_deg = [float(v) for v in np.degrees(qpos_rad)]
            qpos_deg[5] = float(GRIPPER_CLOSE_DEG)
            clamped_qpos, warnings = validate_and_clamp_joints(qpos_deg)
            robot.move_to_pose(np.radians(clamped_qpos), duration_sec=0.5)
            return build_gripper_response(
                gripper_state="closed",
                gripper_deg=float(GRIPPER_CLOSE_DEG),
                executed_qpos_deg=clamped_qpos,
                visual_hint=(
                    "Gripper closed. Lift arm to hover (z_height_cm=5.0) and inspect camera frame "
                    "to visually confirm if object is held between jaws or still on table."
                ),
                warnings=warnings if warnings else None,
            )

        elif name == "move_to_preset":
            pose_name = args.get("pose_name", "HOME").upper()
            if pose_name not in PRESET_POSES:
                available_poses = list(PRESET_POSES.keys())
                err_msg = f"Unknown preset pose: {pose_name}. Available: {available_poses}"
                log_with_tag(logger, "ERROR", err_msg)
                return {"error": err_msg}
            target_rad = PRESET_POSES[pose_name]
            robot.move_to_pose(target_rad, duration_sec=1.0)
            target_deg = [float(v) for v in np.degrees(target_rad)]
            response = {
                "status": "success",
                "pose_name": pose_name,
                "executed_qpos_deg": target_deg,
            }
            if pose_name == "HOME":
                response["visual_hint"] = (
                    "Arm is at HOME pose with clear workspace view. "
                    "Inspect fresh camera frame to locate target object coordinates [y, x] at "
                    "bottom contact center."
                )
            return response

        elif name == "get_end_effector_pose":
            if not hasattr(robot, "get_end_effector_pose"):
                err_msg = "End effector pose is not available in real physical mode."
                log_with_tag(logger, "ERROR", err_msg)
                return {"error": err_msg}
            pos, mat = robot.get_end_effector_pose()
            return {
                "position": pos.tolist(),
                "rotation_matrix": mat.tolist(),
            }

        elif name == "move_to_spatial_point":
            point = args.get("point", [])
            if len(point) != 2:
                err_msg = f"move_to_spatial_point expects [y, x] (2 values), got {len(point)}"
                log_with_tag(logger, "ERROR", err_msg)
                return {"error": err_msg}

            approach = (
                str(args.get("approach", "")).strip().lower().replace("-", "_").replace(" ", "_")
            )
            if not approach or approach not in ("top_down", "horizontal"):
                err_msg = (
                    "move_to_spatial_point requires 'approach' parameter "
                    "('top_down' or 'horizontal')"
                )
                log_with_tag(logger, "ERROR", err_msg)
                return {"error": err_msg}

            # Visual refinement via OpenCV multi-color salient object detector
            if get_latest_jpeg_fn:
                try:
                    raw_jpeg = get_latest_jpeg_fn()
                    if raw_jpeg:
                        import cv2

                        from robot.vision_detector import refine_point_to_nearest_object

                        nparr = np.frombuffer(raw_jpeg, np.uint8)
                        frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        if frame_bgr is not None:
                            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                            refined_pt, matched_obj = refine_point_to_nearest_object(
                                point, frame_rgb
                            )
                            if matched_obj is not None and (
                                abs(refined_pt[0] - point[0]) > 5
                                or abs(refined_pt[1] - point[1]) > 5
                            ):
                                log_with_tag(
                                    logger,
                                    "EXEC",
                                    f"Auto-refined coordinate [{point[0]:.0f}, {point[1]:.0f}] "
                                    f"→ [{refined_pt[0]:.0f}, {refined_pt[1]:.0f}] "
                                    f"(detected {matched_obj['color']} object)",
                                )
                                point = refined_pt
                except Exception as e:
                    log_with_tag(logger, "WARN", f"Visual refinement skipped: {e}")

            z_height_cm = float(args.get("z_height_cm", 3.0))
            # Clamp height to reasonable safety bounds [0.0cm, 20.0cm]
            z_height_cm = max(0.0, min(20.0, z_height_cm))
            hover_z = z_height_cm / 100.0
            robot_is_sim = (
                getattr(robot, "mode", None) == "sim"
                or hasattr(robot, "render_rgb_frame")
                or hasattr(robot, "model")
            )
            target_xyz = image_point_to_xyz(
                point,
                hover_z=hover_z,
                mode="sim" if robot_is_sim else None,
            )
            log_with_tag(
                logger,
                "EXEC",
                f"move_to_spatial_point: image [{point[0]:.0f}, {point[1]:.0f}] "
                f"(z={z_height_cm:.1f}cm, approach={approach}) → XYZ "
                f"[{target_xyz[0]:.3f}, {target_xyz[1]:.3f}, {target_xyz[2]:.3f}]",
            )

            # Update live visual target overlay on camera stream
            try:
                from ui.stream import get_model_targets, set_model_targets

                user_label = args.get("label")
                if user_label and str(user_label).strip():
                    badge_name = str(user_label).strip().upper()
                elif "matched_obj" in locals() and matched_obj and "color" in matched_obj:
                    badge_name = f"{matched_obj['color'].upper()} CUBE"
                else:
                    badge_name = "TARGET"

                active_label = (
                    f"[{badge_name}] [{int(point[0])}, {int(point[1])}] "
                    f"(z={z_height_cm:.1f}cm, {approach})"
                )
                active_target: dict[str, Any] = {
                    "point": (float(point[0]), float(point[1])),
                    "label": active_label,
                    "color": (0, 225, 255),
                }
                existing = get_model_targets()
                if existing:
                    updated: list[dict[str, Any] | tuple[float, float] | list[float]] = []
                    matched = False
                    for tgt in existing:
                        if isinstance(tgt, dict) and "point" in tgt:
                            dy = abs(tgt["point"][0] - point[0])
                            dx = abs(tgt["point"][1] - point[1])
                            if dy < 35 and dx < 35 and not matched:
                                updated.append(active_target)
                                matched = True
                            else:
                                updated.append(tgt)
                        else:
                            updated.append(tgt)
                    if not matched:
                        updated.insert(0, active_target)
                    set_model_targets(updated)
                else:
                    set_model_targets([active_target])
            except Exception:
                pass

            # Use current joints as IK seed for smooth continuation
            current_rad = robot.get_joint_positions()
            ik = _get_ik_solver()
            solved_qpos = ik.solve_ik(
                target_pos=target_xyz,
                initial_qpos=current_rad,
                approach=approach,
                null_weight=1e-3,
            )
            # Retain current gripper joint angle (joint 5)
            if len(current_rad) >= 6:
                solved_qpos[5] = current_rad[5]

            solved_deg = [float(v) for v in np.degrees(solved_qpos)]
            clamped_qpos, warnings = validate_and_clamp_joints(solved_deg)
            qpos_rad = np.radians(clamped_qpos)
            robot.move_to_pose(qpos_rad, duration_sec=1.5)

            perception_res: VisualAlignmentResponse | None = None
            if get_latest_jpeg_fn:
                try:
                    import cv2

                    from robot.perception import compute_visual_alignment

                    raw_jpeg = get_latest_jpeg_fn()
                    if raw_jpeg:
                        nparr = np.frombuffer(raw_jpeg, np.uint8)
                        frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        if frame_bgr is not None:
                            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                            perception_res = compute_visual_alignment(
                                frame_rgb=frame_rgb,
                                target_point_1000=point,
                                tolerance=get_settings().alignment_tolerance_px,
                            )
                except Exception as e:
                    log_with_tag(logger, "WARN", f"Perception computation skipped: {e}")

            if z_height_cm >= 3.0:
                visual_hint = (
                    "Arm reached hover position (z>=3cm). "
                    "Visual Verification Loop: Inspect the new camera frame, identify gripper "
                    "position [y_g, x_g] vs target [y_t, x_t], state the relative error [dy, dx], "
                    "and correct or confirm alignment before descending."
                )
            else:
                visual_hint = (
                    f"Arm reached contact height (z={z_height_cm:.1f}cm). Visually verify gripper "
                    "jaws surround the target contact center before closing."
                )

            return build_move_spatial_response(
                target_xyz_m=target_xyz.tolist(),
                executed_qpos_deg=clamped_qpos,
                approach=approach,
                visual_hint=visual_hint,
                perception=perception_res,
                warnings=warnings if warnings else None,
            )

        else:
            err_msg = f"Unknown tool called: {name}"
            log_with_tag(logger, "ERROR", err_msg)
            return {"error": err_msg}

    except Exception as e:
        err_msg = f"Exception during execution of {name}: {e}"
        log_with_tag(logger, "ERROR", err_msg)
        return {"error": err_msg}


def execute_tool_call_sync(
    call: Any,
    robot: RobotInterface | None,
    get_latest_jpeg_fn: Callable[[], bytes | None] | None = None,
    status_callback: Callable[[str], None] | None = None,
) -> dict[str, Any] | GripperToolResponse | MoveToSpatialPointResponse:
    """Synchronous convenience wrapper for execute_tool_call."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(
                asyncio.run,
                execute_tool_call(
                    call=call,
                    robot=robot,
                    get_latest_jpeg_fn=get_latest_jpeg_fn,
                    status_callback=status_callback,
                ),
            ).result()
    else:
        return asyncio.run(
            execute_tool_call(
                call=call,
                robot=robot,
                get_latest_jpeg_fn=get_latest_jpeg_fn,
                status_callback=status_callback,
            )
        )
