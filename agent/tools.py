"""Robot tool declarations and execution adapters for Gemini Live API."""

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from agent.contracts import GripperToolResponse, MoveToSpatialPointResponse
from robot.cartesian_ik import CartesianIK
from robot.constants import JOINT_LIMITS, PRESET_POSES
from robot.contracts import GraspClassification, VisualAlignmentResponse
from robot.grasp import classify_grasp
from robot.interface import RobotInterface
from robot.logger import log_compat, log_with_tag
from robot.perception import compute_visual_alignment
from robot.registry import get_active_robot, set_active_robot
from robot.settings import get_settings
from robot.vision_detector import refine_point_to_nearest_object
from vision import get_calibrator, get_model_targets, set_model_targets

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
    grasp: GraspClassification | None = None,
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
    if grasp is not None:
        resp["grasp"] = grasp
    if warnings:
        resp["warnings"] = warnings
    return resp


# Gripper limits derived from JOINT_LIMITS in degrees
GRIPPER_OPEN_DEG = JOINT_LIMITS["gripper"][1]
GRIPPER_CLOSE_DEG = JOINT_LIMITS["gripper"][0]

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


def resolve_robot_mode(robot: RobotInterface | None, mode: str | None = None) -> str:
    """Return the operational mode ('real' or 'sim') a tool call should use.

    An explicit ``mode`` wins. Otherwise the robot's own ``mode`` attribute decides, and no
    robot means 'real'. Duck-typing on ``render_rgb_frame`` is deliberately avoided:
    ``RobotInterface`` defines that method, so ``hasattr`` is true for every robot, physical
    arms included, which used to route real-mode moves through the simulation homography.

    Args:
        robot: Active robot controller, or None.
        mode: Explicit mode override ('real' or 'sim').

    Returns:
        'sim' or 'real'.
    """
    if mode in ("real", "sim"):
        return mode
    return "sim" if getattr(robot, "mode", None) == "sim" else "real"


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
        mode: Optional operational mode ('real' or 'sim'). If None, resolved from the
            active robot's ``mode`` attribute (see :func:`resolve_robot_mode`).

    Returns:
        A (3,) numpy array [X, Y, Z] in the robot base frame (metres).

    Raises:
        RuntimeError: If the workspace is not calibrated.
    """
    from robot.calibration import DEFAULT_SAVE_PATH, DEFAULT_SIM_SAVE_PATH

    active_mode = resolve_robot_mode(get_active_robot(), mode)
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


@dataclass
class ToolContext:
    """Execution context provided to tool handlers."""

    robot: RobotInterface | None
    get_latest_jpeg_fn: Callable[[], bytes | None] | None = None
    status_callback: Callable[[str], None] | None = None
    mode: str | None = None


@dataclass(frozen=True)
class ToolSpec:
    """Specification of a robot tool in the registry."""

    name: str
    description: str
    parameters: dict[str, Any]
    blocking: bool
    handler: Callable[
        [ToolContext, dict[str, Any]],
        dict[str, Any] | GripperToolResponse | MoveToSpatialPointResponse,
    ]
    prompt_line: str


def _handle_send_message(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    """Execute send_message tool call.

    Args:
        ctx: Tool execution context.
        args: Tool arguments containing 'message'.

    Returns:
        Success response dictionary containing the message.
    """
    message_text = args.get("message", "")
    return {"status": "success", "message": message_text}


def _handle_move_to_pose(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    """Execute move_to_pose tool call.

    Args:
        ctx: Tool execution context.
        args: Tool arguments containing 'qpos' and 'duration_sec'.

    Returns:
        Execution status response or error dictionary.
    """
    robot = ctx.robot
    assert robot is not None
    qpos_deg = args.get("qpos", [])
    duration_sec = args.get("duration_sec", 1.0)
    if len(qpos_deg) != 6:
        err_msg = f"move_to_pose expects 6 joint angles, got {len(qpos_deg)}"
        log_with_tag(logger, "ERROR", err_msg)
        return {"error": err_msg}
    clamped_qpos, warnings = validate_and_clamp_joints(qpos_deg)
    qpos_rad = np.radians(clamped_qpos)
    robot.move_to_pose(qpos_rad, duration_sec=duration_sec)
    response: dict[str, Any] = {"status": "success", "executed_qpos_deg": clamped_qpos}
    if warnings:
        response["warnings"] = warnings
    return response


def _handle_set_joint_positions(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    """Execute set_joint_positions tool call.

    Args:
        ctx: Tool execution context.
        args: Tool arguments containing 'qpos'.

    Returns:
        Execution status response or error dictionary.
    """
    robot = ctx.robot
    assert robot is not None
    qpos_deg = args.get("qpos", [])
    if len(qpos_deg) != 6:
        err_msg = f"set_joint_positions expects 6 joint angles, got {len(qpos_deg)}"
        log_with_tag(logger, "ERROR", err_msg)
        return {"error": err_msg}
    clamped_qpos, warnings = validate_and_clamp_joints(qpos_deg)
    qpos_rad = np.radians(clamped_qpos)
    robot.set_joint_positions(qpos_rad)
    response: dict[str, Any] = {"status": "success", "executed_qpos_deg": clamped_qpos}
    if warnings:
        response["warnings"] = warnings
    return response


def _handle_get_joint_positions(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    """Execute get_joint_positions tool call.

    Args:
        ctx: Tool execution context.
        args: Tool arguments (unused).

    Returns:
        Dictionary with current joint positions in degrees.
    """
    robot = ctx.robot
    assert robot is not None
    qpos_rad = robot.get_joint_positions()
    qpos_deg = [float(v) for v in np.degrees(qpos_rad)]
    return {"qpos_deg": qpos_deg}


def _handle_open_gripper(ctx: ToolContext, args: dict[str, Any]) -> GripperToolResponse:
    """Execute open_gripper tool call.

    Args:
        ctx: Tool execution context.
        args: Tool arguments (unused).

    Returns:
        GripperToolResponse payload.
    """
    robot = ctx.robot
    assert robot is not None
    qpos_rad = robot.get_joint_positions()
    qpos_deg = [float(v) for v in np.degrees(qpos_rad)]
    qpos_deg[5] = float(GRIPPER_OPEN_DEG)
    clamped_qpos, warnings = validate_and_clamp_joints(qpos_deg)
    robot.move_to_pose(np.radians(clamped_qpos), duration_sec=0.5)
    reading = robot.read_gripper_state()
    aperture_deg = float(reading["aperture_deg"])
    return build_gripper_response(
        gripper_state="open",
        gripper_deg=aperture_deg,
        executed_qpos_deg=clamped_qpos,
        visual_hint=f"Gripper opened (aperture {aperture_deg:.1f}°).",
        warnings=warnings if warnings else None,
    )


def _handle_close_gripper(ctx: ToolContext, args: dict[str, Any]) -> GripperToolResponse:
    """Execute close_gripper tool call.

    Args:
        ctx: Tool execution context.
        args: Tool arguments (unused).

    Returns:
        GripperToolResponse payload.
    """
    robot = ctx.robot
    assert robot is not None
    qpos_rad = robot.get_joint_positions()
    qpos_deg = [float(v) for v in np.degrees(qpos_rad)]
    qpos_deg[5] = float(GRIPPER_CLOSE_DEG)
    clamped_qpos, warnings = validate_and_clamp_joints(qpos_deg)
    robot.move_to_pose(np.radians(clamped_qpos), duration_sec=0.5)

    settle_s = get_settings().gripper_settle_s
    if settle_s > 0:
        time.sleep(settle_s)

    reading = robot.read_gripper_state()
    grasp = classify_grasp(reading, get_settings())

    log_with_tag(
        logger,
        "GRASP",
        f"aperture={grasp['aperture_deg']:.1f}° load={reading['load']} "
        f"source={reading['source']} → grasped={grasp['grasped']} ({grasp['reason']})",
    )

    if grasp["grasped"] is False:
        gripper_state = "closed_empty"
        visual_hint = (
            f"Gripper closed on NOTHING (aperture {grasp['aperture_deg']:.1f}° "
            "equals the empty-jaw value). Grasp missed. Open the gripper, hover again, "
            "re-observe the target, and retry."
        )
    elif grasp["grasped"] is True:
        gripper_state = "closed"
        visual_hint = (
            f"Gripper stalled at {grasp['aperture_deg']:.1f}° — an object is between "
            "the jaws. Lift to 5 cm and visually confirm before transporting."
        )
    else:  # grasped is None
        gripper_state = "closed_unverified"
        visual_hint = (
            "Gripper sensor unavailable; grasp NOT verified. Lift and inspect before transporting."
        )

    return build_gripper_response(
        gripper_state=gripper_state,
        gripper_deg=float(grasp["aperture_deg"]),
        executed_qpos_deg=clamped_qpos,
        visual_hint=visual_hint,
        grasp=grasp,
        warnings=warnings if warnings else None,
    )


def _handle_move_to_preset(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    """Execute move_to_preset tool call.

    Args:
        ctx: Tool execution context.
        args: Tool arguments containing 'pose_name'.

    Returns:
        Execution status response or error dictionary.
    """
    robot = ctx.robot
    assert robot is not None
    pose_name = args.get("pose_name", "HOME").upper()
    if pose_name not in PRESET_POSES:
        available_poses = list(PRESET_POSES.keys())
        err_msg = f"Unknown preset pose: {pose_name}. Available: {available_poses}"
        log_with_tag(logger, "ERROR", err_msg)
        return {"error": err_msg}
    target_rad = PRESET_POSES[pose_name]
    robot.move_to_pose(target_rad, duration_sec=1.0)
    target_deg = [float(v) for v in np.degrees(target_rad)]
    response: dict[str, Any] = {
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


def _handle_get_end_effector_pose(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    """Execute get_end_effector_pose tool call.

    Args:
        ctx: Tool execution context.
        args: Tool arguments (unused).

    Returns:
        Dictionary with Cartesian position and rotation matrix.
    """
    robot = ctx.robot
    assert robot is not None
    if not hasattr(robot, "get_end_effector_pose"):
        err_msg = "End effector pose is not available in real physical mode."
        log_with_tag(logger, "ERROR", err_msg)
        return {"error": err_msg}
    pos, mat = robot.get_end_effector_pose()
    return {
        "position": pos.tolist(),
        "rotation_matrix": mat.tolist(),
    }


def _refine_point(
    point: list[float],
    get_latest_jpeg_fn: Callable[[], bytes | None] | None,
) -> tuple[list[float], dict[str, Any] | None]:
    """Refine target point via visual salient object detection if frame is available.

    Args:
        point: 2D image coordinates [y, x].
        get_latest_jpeg_fn: Optional frame provider.

    Returns:
        Tuple of (possibly refined [y, x] point, matched object dictionary or None).
    """
    matched_obj: dict[str, Any] | None = None
    if get_latest_jpeg_fn:
        try:
            raw_jpeg = get_latest_jpeg_fn()
            if raw_jpeg:
                nparr = np.frombuffer(raw_jpeg, np.uint8)
                frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame_bgr is not None:
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    refined_pt, matched_obj = refine_point_to_nearest_object(
                        point, frame_rgb, calibrator=get_calibrator()
                    )
                    if matched_obj is not None and (
                        abs(refined_pt[0] - point[0]) > 5 or abs(refined_pt[1] - point[1]) > 5
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
    return point, matched_obj


def _update_target_overlay(
    point: list[float],
    z_height_cm: float,
    approach: str,
    user_label: Any,
    matched_obj: dict[str, Any] | None,
) -> None:
    """Update live visual target overlay on camera stream.

    Args:
        point: 2D image coordinates [y, x].
        z_height_cm: Height offset above table.
        approach: Approach direction string.
        user_label: Optional label provided in args.
        matched_obj: Optional matched object from visual refinement.
    """
    try:
        if user_label and str(user_label).strip():
            badge_name = str(user_label).strip().upper()
        elif matched_obj and "color" in matched_obj:
            badge_name = f"{matched_obj['color'].upper()} CUBE"
        else:
            badge_name = "TARGET"

        active_label = (
            f"[{badge_name}] [{int(point[0])}, {int(point[1])}] (z={z_height_cm:.1f}cm, {approach})"
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


def _measure_alignment(
    point: list[float],
    get_latest_jpeg_fn: Callable[[], bytes | None] | None,
) -> VisualAlignmentResponse | None:
    """Compute visual alignment relative to target in current frame.

    Args:
        point: 2D image coordinates [y, x].
        get_latest_jpeg_fn: Optional frame provider.

    Returns:
        VisualAlignmentResponse if successfully computed, else None.
    """
    perception_res: VisualAlignmentResponse | None = None
    if get_latest_jpeg_fn:
        try:
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
                        calibrator=get_calibrator(),
                    )
        except Exception as e:
            log_with_tag(logger, "WARN", f"Perception computation skipped: {e}")
    return perception_res


def _handle_move_to_spatial_point(
    ctx: ToolContext, args: dict[str, Any]
) -> MoveToSpatialPointResponse | dict[str, Any]:
    """Execute move_to_spatial_point tool call.

    Args:
        ctx: Tool execution context.
        args: Tool arguments containing 'point', 'approach', etc.

    Returns:
        MoveToSpatialPointResponse payload or error dictionary.
    """
    robot = ctx.robot
    assert robot is not None

    point = args.get("point", [])
    if len(point) != 2:
        err_msg = f"move_to_spatial_point expects [y, x] (2 values), got {len(point)}"
        log_with_tag(logger, "ERROR", err_msg)
        return {"error": err_msg}

    approach = str(args.get("approach", "")).strip().lower().replace("-", "_").replace(" ", "_")
    if not approach or approach not in ("top_down", "horizontal"):
        err_msg = "move_to_spatial_point requires 'approach' parameter ('top_down' or 'horizontal')"
        log_with_tag(logger, "ERROR", err_msg)
        return {"error": err_msg}

    point, matched_obj = _refine_point(point, ctx.get_latest_jpeg_fn)

    z_height_cm = float(args.get("z_height_cm", 3.0))
    # Clamp height to reasonable safety bounds [0.0cm, 20.0cm]
    z_height_cm = max(0.0, min(20.0, z_height_cm))
    hover_z = z_height_cm / 100.0
    target_xyz = image_point_to_xyz(
        point,
        hover_z=hover_z,
        mode=resolve_robot_mode(robot, ctx.mode),
    )
    log_with_tag(
        logger,
        "EXEC",
        f"move_to_spatial_point: image [{point[0]:.0f}, {point[1]:.0f}] "
        f"(z={z_height_cm:.1f}cm, approach={approach}) → XYZ "
        f"[{target_xyz[0]:.3f}, {target_xyz[1]:.3f}, {target_xyz[2]:.3f}]",
    )

    _update_target_overlay(point, z_height_cm, approach, args.get("label"), matched_obj)

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

    perception_res = _measure_alignment(point, ctx.get_latest_jpeg_fn)

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


TOOL_REGISTRY: dict[str, ToolSpec] = {
    "send_message": ToolSpec(
        name="send_message",
        description=(
            "Speak, narrate, or deliver a conversational message to the user. "
            "Use this to explain your thoughts, describe visual observations, "
            "ask questions, or confirm task progress."
        ),
        parameters={
            "type": "OBJECT",
            "properties": {
                "message": {
                    "type": "STRING",
                    "description": "The natural language message to narrate to the user.",
                },
            },
            "required": ["message"],
        },
        blocking=False,
        handler=_handle_send_message,
        prompt_line=(
            "`send_message(message='...')`: "
            "Speak, narrate visual observations and relative error calculations."
        ),
    ),
    "move_to_pose": ToolSpec(
        name="move_to_pose",
        description="Smoothly interpolate motion to a 6-DOF joint target.",
        parameters={
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
        blocking=True,
        handler=_handle_move_to_pose,
        prompt_line=(
            "`move_to_pose(qpos=[...], duration_sec=1.0)`: "
            "Move arm joints to explicit target angles (degrees)."
        ),
    ),
    "set_joint_positions": ToolSpec(
        name="set_joint_positions",
        description="Instant or snap joint position command to 6-DOF joint target.",
        parameters={
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
        blocking=True,
        handler=_handle_set_joint_positions,
        prompt_line="",
    ),
    "get_joint_positions": ToolSpec(
        name="get_joint_positions",
        description="Read current joint state (informational, non-blocking). Returns degrees.",
        parameters={
            "type": "OBJECT",
            "properties": {},
        },
        blocking=False,
        handler=_handle_get_joint_positions,
        prompt_line="`get_joint_positions()`: Read current joint positions in degrees.",
    ),
    "open_gripper": ToolSpec(
        name="open_gripper",
        description="Convenience wrapper to open the gripper fully.",
        parameters={
            "type": "OBJECT",
            "properties": {},
        },
        blocking=True,
        handler=_handle_open_gripper,
        prompt_line="`open_gripper()`: Open the gripper jaws fully.",
    ),
    "close_gripper": ToolSpec(
        name="close_gripper",
        description="Convenience wrapper to close the gripper fully.",
        parameters={
            "type": "OBJECT",
            "properties": {},
        },
        blocking=True,
        handler=_handle_close_gripper,
        prompt_line="`close_gripper()`: Close the gripper jaws fully to grasp an object.",
    ),
    "move_to_preset": ToolSpec(
        name="move_to_preset",
        description="Move the robot to a predefined preset pose (e.g., 'HOME', 'MIDDLE').",
        parameters={
            "type": "OBJECT",
            "properties": {
                "pose_name": {
                    "type": "STRING",
                    "description": "Name of the preset pose.",
                },
            },
            "required": ["pose_name"],
        },
        blocking=True,
        handler=_handle_move_to_preset,
        prompt_line=(
            "`move_to_preset(pose_name='HOME')`: Move to named preset pose ('HOME', 'MIDDLE')."
        ),
    ),
    "get_end_effector_pose": ToolSpec(
        name="get_end_effector_pose",
        description=(
            "Read current end-effector XYZ position and rotation matrix. Available in sim mode."
        ),
        parameters={
            "type": "OBJECT",
            "properties": {},
        },
        blocking=False,
        handler=_handle_get_end_effector_pose,
        prompt_line=(
            "`get_end_effector_pose()`: Read current 3D position and rotation matrix (sim mode)."
        ),
    ),
    "move_to_spatial_point": ToolSpec(
        name="move_to_spatial_point",
        description=(
            "Move the robot end-effector to a 2D point visible in the camera feed. "
            "Provide a normalized [y, x] coordinate (0-1000) from the camera image. "
            "The system maps this to 3D workspace coordinates using ArUco workspace calibration "
            "and solves inverse kinematics to reach the target. "
            "Specify z_height_cm to control vertical offset above table (0 for table contact, "
            "3-5 for hover/transit)."
        ),
        parameters={
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
        blocking=True,
        handler=_handle_move_to_spatial_point,
        prompt_line=(
            "`move_to_spatial_point(point=[y, x], z_height_cm=3.0)`: "
            "Move end-effector to camera-space [y, x] coordinate (0-1000). "
            "Use z_height_cm=5.0 for hover/transit and z_height_cm=0.0 for table contact."
        ),
    ),
}

_CANONICAL_PROMPT_ORDER = [
    "send_message",
    "move_to_spatial_point",
    "open_gripper",
    "close_gripper",
    "move_to_pose",
    "move_to_preset",
    "get_joint_positions",
    "get_end_effector_pose",
]


def render_prompt_tool_list(registry: dict[str, ToolSpec] | None = None) -> str:
    """Render markdown bullet list of available tools from ToolSpec registry.

    Args:
        registry: Optional ToolSpec registry dictionary. Defaults to TOOL_REGISTRY.

    Returns:
        Formatted markdown bullet list of available tools.
    """
    if registry is None:
        registry = TOOL_REGISTRY
    lines: list[str] = []
    seen: set[str] = set()
    for name in _CANONICAL_PROMPT_ORDER:
        spec = registry.get(name)
        if spec and spec.prompt_line:
            lines.append(f"- {spec.prompt_line}")
            seen.add(name)
    for name, spec in registry.items():
        if name not in seen and spec.prompt_line:
            lines.append(f"- {spec.prompt_line}")
    return "\n".join(lines)


def _build_declarations(registry: dict[str, ToolSpec]) -> list[dict[str, Any]]:
    """Build Gemini tool declaration schemas derived from ToolSpec registry."""
    declarations = []
    for spec in registry.values():
        decl: dict[str, Any] = {
            "name": spec.name,
            "description": spec.description,
            "parameters": spec.parameters,
        }
        if spec.blocking:
            decl["behavior"] = "BLOCKING"
        declarations.append(decl)
    return declarations


ROBOT_TOOL_DECLARATIONS: list[dict[str, Any]] = _build_declarations(TOOL_REGISTRY)
_BLOCKING_TOOLS: set[str] = {spec.name for spec in TOOL_REGISTRY.values() if spec.blocking}


def register_tool(spec: ToolSpec) -> None:
    """Register a new tool specification into the global registry.

    Args:
        spec: ToolSpec instance to register.
    """
    TOOL_REGISTRY[spec.name] = spec
    ROBOT_TOOL_DECLARATIONS[:] = _build_declarations(TOOL_REGISTRY)
    _BLOCKING_TOOLS.clear()
    _BLOCKING_TOOLS.update(s.name for s in TOOL_REGISTRY.values() if s.blocking)


def unregister_tool(name: str) -> None:
    """Unregister a tool specification from the global registry.

    Args:
        name: Name of tool to remove.
    """
    TOOL_REGISTRY.pop(name, None)
    ROBOT_TOOL_DECLARATIONS[:] = _build_declarations(TOOL_REGISTRY)
    _BLOCKING_TOOLS.clear()
    _BLOCKING_TOOLS.update(s.name for s in TOOL_REGISTRY.values() if s.blocking)


async def execute_tool_call(
    call: Any,
    robot: RobotInterface | None,
    get_latest_jpeg_fn: Callable[[], bytes | None] | None = None,
    status_callback: Callable[[str], None] | None = None,
    mode: str | None = None,
) -> dict[str, Any] | GripperToolResponse | MoveToSpatialPointResponse:
    """Dispatch tool call to the RobotInterface asynchronously.

    Args:
        call: The tool call object with name and args.
        robot: The initialized RobotInterface instance.
        get_latest_jpeg_fn: Optional callable returning latest camera frame JPEG bytes.
        status_callback: Optional callable for streaming status messages to UI.
        mode: Operational mode ('real' or 'sim') selecting the workspace calibration. When
            None it is taken from ``robot.mode``.

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
        if get_active_robot() is None:
            set_active_robot(robot)
    except Exception:
        pass

    try:
        spec = TOOL_REGISTRY.get(name)
        if spec is None:
            err_msg = f"Unknown tool called: {name}"
            log_with_tag(logger, "ERROR", err_msg)
            return {"error": err_msg}

        ctx = ToolContext(
            robot=robot,
            get_latest_jpeg_fn=get_latest_jpeg_fn,
            status_callback=status_callback,
            mode=mode,
        )
        return spec.handler(ctx, args)
    except Exception as e:
        err_msg = f"Exception during execution of {name}: {e}"
        log_with_tag(logger, "ERROR", err_msg)
        return {"error": err_msg}


def execute_tool_call_sync(
    call: Any,
    robot: RobotInterface | None,
    get_latest_jpeg_fn: Callable[[], bytes | None] | None = None,
    status_callback: Callable[[str], None] | None = None,
    mode: str | None = None,
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
                    mode=mode,
                ),
            ).result()
    else:
        return asyncio.run(
            execute_tool_call(
                call=call,
                robot=robot,
                get_latest_jpeg_fn=get_latest_jpeg_fn,
                status_callback=status_callback,
                mode=mode,
            )
        )
