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
from agent.guardrails import (
    HOVER_STILL_VALID_MM,
    LOCK_MATCH_RADIUS_MM,
    MAX_GRASPABLE_WIDTH_MM,
    MAX_HOVERS_PER_LOCK,
    MIN_LIFT_HEIGHT_CM,
    GuardrailState,
    HoverRecord,
    Phase,
    TargetLock,
    decide_grasp,
)
from agent.motion import MotionSegment, plan_approach_segments, run_segments
from agent.pick_loop import (
    IMPLAUSIBLE_RESIDUAL_MM,
    hover_call,
    lift_call,
    transport_gate_message,
    vision_vote,
)
from robot.cartesian_ik import CartesianIK
from robot.constants import JOINT_LIMITS, PRESET_POSES
from robot.contracts import AlignmentMeasurement, GraspClassification
from robot.errors import (
    CalibrationUnavailable,
    HardwareError,
    PerceptionUnavailable,
    ToolArgumentError,
)
from robot.grasp import classify_grasp
from robot.interface import RobotInterface
from robot.logger import log_compat, log_with_tag
from robot.perception import compute_alignment_mm
from robot.registry import get_active_robot, set_active_robot
from robot.settings import get_settings
from robot.vision_detector import color_from_label, refine_point_to_nearest_object
from vision import get_calibrator, get_model_targets, set_model_targets
from vision.gripper_locator import locate_jaw_xy_mm

logger = logging.getLogger(__name__)

# Exceptions the OpenCV-based perception helpers raise on a bad frame or bad geometry.
PERCEPTION_ERRORS = (cv2.error, ValueError, IndexError, KeyError, TypeError, RuntimeError)

_UNCALIBRATED_MESSAGE = (
    "Workspace is not calibrated. Please ensure 4 ArUco markers are in view and "
    "save calibration under the ArUco Workspace Calibration panel before executing "
    "spatial commands."
)

# Retain _log alias for backwards compatibility with tests importing _log from agent.tools
_log = log_compat


def build_move_spatial_response(
    target_xyz_m: list[float],
    executed_qpos_deg: list[float],
    approach: str,
    visual_hint: str,
    perception: AlignmentMeasurement | dict[str, Any] | None = None,
    warnings: list[str] | None = None,
    applied_correction_mm: list[float] | None = None,
    gripper_source: str | None = None,
    grasp_verification: dict[str, Any] | None = None,
    guardrail_warning: str | None = None,
    target_xy_mm: list[float] | None = None,
    lift_note: str | None = None,
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
    if applied_correction_mm is not None:
        resp["applied_correction_mm"] = applied_correction_mm
    if target_xy_mm is not None:
        resp["target_xy_mm"] = target_xy_mm
    if gripper_source is not None:
        resp["gripper_source"] = gripper_source
    if grasp_verification is not None:
        resp["grasp_verification"] = grasp_verification
    if guardrail_warning is not None:
        resp["guardrail_warning"] = guardrail_warning
    if lift_note is not None:
        resp["lift_note"] = lift_note
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
        CalibrationUnavailable: If the workspace is not calibrated.
    """
    active_mode = resolve_robot_mode(get_active_robot(), mode)
    calib = _ready_calibrator(active_mode)

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
    guardrail_state: GuardrailState | None = None
    query_er2_locator_fn: Callable[[bytes], dict[str, list[float]] | None] | None = None
    # (t_after, timeout_s) -> JPEG of a frame captured after ``t_after`` (time.monotonic()), or
    # None if none arrived in time. Measurements wait for it so they never read a frame that was
    # taken before the arm arrived.
    get_fresh_jpeg_fn: Callable[[float, float], bytes | None] | None = None


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


def _retry_exhausted_response(state: GuardrailState | None) -> dict[str, Any]:
    """Build the refusal returned while the loop is escalated and waiting for the user."""
    message = state.escalation_message if state is not None else "Waiting for the user."
    return {
        "status": "error",
        "error": message,
        "error_type": "retry_exhausted",
        "robot_moved": False,
        "visual_hint": f"{message} Only open_gripper() is accepted until the user responds.",
    }


def _gate_motion(ctx: ToolContext, state: GuardrailState) -> dict[str, Any] | None:
    """Gate a non-spatial motion tool (pose, preset, joints) against escalation and the phase.

    Args:
        ctx: Tool execution context.
        state: Guardrail state.

    Returns:
        Error dictionary if the motion is refused, None if allowed.
    """
    if state.escalated:
        return _retry_exhausted_response(state)
    if state.phase in (Phase.HELD_UNVERIFIED, Phase.CLOSED_EMPTY):
        return _transport_gate_error(state)
    return None


def _transport_gate_error(state: GuardrailState) -> dict[str, Any]:
    """Build the transport_gate refusal: the grasp is not verified, so nothing may move."""
    err_text = transport_gate_message(state)
    return {
        "status": "error",
        "error": err_text,
        "error_type": "transport_gate",
        "robot_moved": False,
        "visual_hint": err_text,
    }


def _site_xyz(robot: RobotInterface) -> np.ndarray:
    """Return the grasp-site position (m) from forward kinematics of the reported joints."""
    return _get_ik_solver().forward_kinematics(robot.get_joint_positions())


def _after_other_motion(
    ctx: ToolContext, state: GuardrailState, epoch: int
) -> dict[str, Any] | None:
    """Bookkeeping after a pose/preset/joint move; re-read the aperture while transporting.

    Args:
        ctx: Tool execution context.
        state: Guardrail state.
        epoch: Epoch captured when the handler started.

    Returns:
        A ``grasp_verification`` dictionary if the object was dropped during the move, else None.
    """
    robot = ctx.robot
    assert robot is not None
    dropped: dict[str, Any] | None = None
    if state.phase == Phase.TRANSPORT:
        dropped = _recheck_hold(robot, state, epoch)

    def mark(s: GuardrailState) -> None:
        s.last_motion = "other"
        s.last_hover = None
        s.record_motion_success()

    state.commit(epoch, mark)
    return dropped


def _recheck_hold(
    robot: RobotInterface, state: GuardrailState, epoch: int
) -> dict[str, Any] | None:
    """Re-read the aperture during transport; a drop means the object was lost.

    Args:
        robot: Robot controller.
        state: Guardrail state (phase TRANSPORT).
        epoch: Epoch captured when the handler started.

    Returns:
        A missed ``grasp_verification`` dictionary if the jaws closed on air, else None.
    """
    grasp = classify_grasp(robot.read_gripper_state(), get_settings())
    if grasp["grasped"] is not False:
        return None
    verdict = decide_grasp(False, None)

    def lost(s: GuardrailState) -> None:
        s.phase = Phase.CLOSED_EMPTY
        s.record_miss_or_unverified("dropped_in_transport")

    state.commit(epoch, lost)
    out = verdict.to_dict()
    out["reason"] = "dropped_in_transport"
    return out


def _handle_move_to_pose(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    """Execute move_to_pose tool call.

    Args:
        ctx: Tool execution context.
        args: Tool arguments containing 'qpos' and 'duration_sec'.

    Returns:
        Execution status response or error dictionary.
    """
    state = ctx.guardrail_state or GuardrailState()
    epoch = state.epoch
    refused = _gate_motion(ctx, state)
    if refused is not None:
        return refused

    robot = ctx.robot
    assert robot is not None
    qpos_deg = args.get("qpos", [])
    duration_sec = args.get("duration_sec", 1.0)
    if len(qpos_deg) != 6:
        err_msg = f"move_to_pose expects 6 joint angles, got {len(qpos_deg)}"
        log_with_tag(logger, "ERROR", err_msg)
        raise ToolArgumentError(err_msg, robot_moved=False)
    clamped_qpos, warnings = validate_and_clamp_joints(qpos_deg)
    qpos_rad = np.radians(clamped_qpos)
    robot.move_to_pose(qpos_rad, duration_sec=duration_sec)
    dropped = _after_other_motion(ctx, state, epoch)
    response: dict[str, Any] = {"status": "success", "executed_qpos_deg": clamped_qpos}
    if warnings:
        response["warnings"] = warnings
    if dropped is not None:
        response["grasp_verification"] = dropped
    return response


def _handle_set_joint_positions(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    """Execute set_joint_positions tool call.

    Args:
        ctx: Tool execution context.
        args: Tool arguments containing 'qpos'.

    Returns:
        Execution status response or error dictionary.
    """
    state = ctx.guardrail_state or GuardrailState()
    epoch = state.epoch
    refused = _gate_motion(ctx, state)
    if refused is not None:
        return refused

    robot = ctx.robot
    assert robot is not None
    qpos_deg = args.get("qpos", [])
    if len(qpos_deg) != 6:
        err_msg = f"set_joint_positions expects 6 joint angles, got {len(qpos_deg)}"
        log_with_tag(logger, "ERROR", err_msg)
        raise ToolArgumentError(err_msg, robot_moved=False)
    clamped_qpos, warnings = validate_and_clamp_joints(qpos_deg)
    qpos_rad = np.radians(clamped_qpos)
    robot.set_joint_positions(qpos_rad)
    dropped = _after_other_motion(ctx, state, epoch)
    response: dict[str, Any] = {"status": "success", "executed_qpos_deg": clamped_qpos}
    if warnings:
        response["warnings"] = warnings
    if dropped is not None:
        response["grasp_verification"] = dropped
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


def _handle_open_gripper(
    ctx: ToolContext, args: dict[str, Any]
) -> GripperToolResponse | dict[str, Any]:
    """Execute open_gripper tool call (always allowed, even when the loop is escalated).

    Args:
        ctx: Tool execution context.
        args: Tool arguments (unused).

    Returns:
        GripperToolResponse payload.
    """
    state = ctx.guardrail_state or GuardrailState()
    epoch = state.epoch
    robot = ctx.robot
    assert robot is not None
    qpos_rad = robot.get_joint_positions()
    qpos_deg = [float(v) for v in np.degrees(qpos_rad)]
    qpos_deg[5] = float(GRIPPER_OPEN_DEG)
    clamped_qpos, warnings = validate_and_clamp_joints(qpos_deg)
    robot.move_to_pose(np.radians(clamped_qpos), duration_sec=0.5)

    state.commit(epoch, lambda s: s.on_opened())

    reading = robot.read_gripper_state()
    aperture_deg = float(reading["aperture_deg"])
    return build_gripper_response(
        gripper_state="open",
        gripper_deg=aperture_deg,
        executed_qpos_deg=clamped_qpos,
        visual_hint=f"Gripper opened (aperture {aperture_deg:.1f}°).",
        warnings=warnings if warnings else None,
    )


def _handle_close_gripper(
    ctx: ToolContext, args: dict[str, Any]
) -> GripperToolResponse | dict[str, Any]:
    """Execute close_gripper tool call.

    Args:
        ctx: Tool execution context.
        args: Tool arguments (unused).

    Returns:
        GripperToolResponse payload.
    """
    state = ctx.guardrail_state or GuardrailState()
    epoch = state.epoch
    if state.escalated:
        return _retry_exhausted_response(state)

    robot = ctx.robot
    assert robot is not None

    site = _site_xyz(robot)
    jaw_xy_mm = site[:2] * 1000.0
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
            "the jaws. Lift to 5 cm to verify the grasp before transporting."
        )
    else:  # grasped is None
        gripper_state = "closed_unverified"
        visual_hint = (
            "Gripper sensor unavailable; grasp NOT verified. Lift to verify before transporting."
        )

    def closed(s: GuardrailState) -> None:
        phase = Phase.CLOSED_EMPTY if grasp["grasped"] is False else Phase.HELD_UNVERIFIED
        if s.target is None:
            # Closing with no lock: the grasp location is wherever the jaw actually is (FK).
            s.new_lock(
                color=None,
                centre_xy_mm=(float(jaw_xy_mm[0]), float(jaw_xy_mm[1])),
                centre_1000=[500.0, 500.0],
                bbox_px=None,
                width_mm=None,
            )
        s.on_closed(
            phase,
            (float(jaw_xy_mm[0]), float(jaw_xy_mm[1])),
            grasp_z_mm=float(site[2] * 1000.0),
        )
        if grasp["grasped"] is False:
            s.record_miss_or_unverified("closed_on_air")
        s.record_motion_success()

    state.commit(epoch, closed)

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
    state = ctx.guardrail_state or GuardrailState()
    epoch = state.epoch
    refused = _gate_motion(ctx, state)
    if refused is not None:
        return refused

    robot = ctx.robot
    assert robot is not None
    pose_name = args.get("pose_name", "HOME").upper()
    if pose_name not in PRESET_POSES:
        available_poses = list(PRESET_POSES.keys())
        err_msg = f"Unknown preset pose: {pose_name}. Available: {available_poses}"
        log_with_tag(logger, "ERROR", err_msg)
        raise ToolArgumentError(err_msg, robot_moved=False)
    target_rad = PRESET_POSES[pose_name]
    robot.move_to_pose(target_rad, duration_sec=1.0)
    dropped = _after_other_motion(ctx, state, epoch)
    target_deg = [float(v) for v in np.degrees(target_rad)]
    response: dict[str, Any] = {
        "status": "success",
        "pose_name": pose_name,
        "executed_qpos_deg": target_deg,
    }
    if dropped is not None:
        response["grasp_verification"] = dropped
    if pose_name == "HOME":
        response["visual_hint"] = (
            "Arm is at HOME pose with clear workspace view. "
            "Inspect fresh camera frame to locate the target object."
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
        raise ToolArgumentError(err_msg, robot_moved=False)
    pos, mat = robot.get_end_effector_pose()
    return {
        "position": pos.tolist(),
        "rotation_matrix": mat.tolist(),
    }


def _decode_frame(raw_jpeg: bytes | None, purpose: str, robot_moved: bool) -> np.ndarray:
    """Decode a camera JPEG to an RGB array.

    Args:
        raw_jpeg: JPEG bytes from the frame provider (None if no frame is available).
        purpose: What the frame is for, for the error message.
        robot_moved: Whether the arm has already moved in this call.

    Returns:
        RGB frame.

    Raises:
        PerceptionUnavailable: If the frame is missing or cannot be decoded.
    """
    if raw_jpeg is None:
        raise PerceptionUnavailable(
            f"Camera frame unavailable for {purpose}", robot_moved=robot_moved
        )
    frame_bgr = cv2.imdecode(np.frombuffer(raw_jpeg, np.uint8), cv2.IMREAD_COLOR)
    if frame_bgr is None:
        raise PerceptionUnavailable(
            f"Failed to decode camera frame for {purpose}", robot_moved=robot_moved
        )
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def _refine_point(
    point: list[float],
    get_latest_jpeg_fn: Callable[[], bytes | None] | None,
    label: Any = None,
) -> tuple[list[float], dict[str, Any] | None]:
    """Snap a requested point to the footprint centre of the object it points at.

    Args:
        point: 2D image coordinates [y, x].
        get_latest_jpeg_fn: Optional frame provider.
        label: Optional model label; a colour word in it restricts the match to that colour.

    Returns:
        Tuple of (possibly refined [y, x] point, matched object dictionary or None). The object
        dictionary carries ``centre_xy_mm`` / ``width_mm`` when the table homography is known.

    Raises:
        PerceptionUnavailable: If frame provider was supplied but returned None,
            decoding failed, or visual refinement failed.
    """
    matched_obj: dict[str, Any] | None = None
    if get_latest_jpeg_fn is not None:
        frame_rgb = _decode_frame(get_latest_jpeg_fn(), "target refinement", robot_moved=False)
        try:
            refined_pt, matched_obj = refine_point_to_nearest_object(
                point,
                frame_rgb,
                target_color=color_from_label(label),
                calibrator=get_calibrator(),
            )
            if matched_obj is not None and (
                abs(refined_pt[0] - point[0]) > 5 or abs(refined_pt[1] - point[1]) > 5
            ):
                log_with_tag(
                    logger,
                    "EXEC",
                    f"Auto-refined coordinate [{point[0]:.0f}, {point[1]:.0f}] "
                    f"→ [{refined_pt[0]:.0f}, {refined_pt[1]:.0f}] "
                    f"(footprint centre of {matched_obj['color']} object)",
                )
                point = refined_pt
        except PERCEPTION_ERRORS as e:
            raise PerceptionUnavailable(f"Visual refinement failed: {e}", robot_moved=False) from e
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
    except (TypeError, KeyError, IndexError, ValueError) as e:
        log_with_tag(logger, "WARN", f"Failed to update target overlay: {e}")


@dataclass
class JawReading:
    """Where the jaw was located in a frame.

    Attributes:
        xy_mm: Grasp-site XY in robot mm, or None if it could not be located.
        source: 'aruco', 'kinematic' or 'none'.
        er2_point: Report-only ER2 pointing result ([y, x] normalised), if queried.
    """

    xy_mm: list[float] | None
    source: str
    er2_point: list[float] | None = None


def _locate_jaw(
    ctx: ToolContext, calib: Any, frame_rgb: np.ndarray, raw_jpeg: bytes, mode: str
) -> JawReading:
    """Locate the jaw in the frame: jaw tag (parallax removed) first, FK as the fallback.

    ER2 pointing is report-only: it is attached to the payload but never decides alignment.

    Args:
        ctx: Tool context (robot, ER2 query).
        calib: Workspace calibrator.
        frame_rgb: Measurement frame.
        raw_jpeg: The same frame as JPEG bytes (for the ER2 query).
        mode: 'real' or 'sim'.

    Returns:
        JawReading.
    """
    er2_point: list[float] | None = None
    if mode == "real" and get_settings().get_gripper_locator_er2() and ctx.query_er2_locator_fn:
        try:
            er2_dict = ctx.query_er2_locator_fn(raw_jpeg)
            if er2_dict and "gripper_fingertips" in er2_dict:
                er2_point = [float(v) for v in er2_dict["gripper_fingertips"]]
        except (TimeoutError, OSError, ValueError, KeyError, TypeError) as e:
            log_with_tag(logger, "WARN", f"ER2 pointing query failed: {e}")

    if ctx.robot is not None and calib is not None:
        snapshot = calib.geometry_snapshot()
        located = locate_jaw_xy_mm(
            frame_rgb,
            snapshot,
            ctx.robot.get_joint_positions(),
            tag_offset_mm=get_settings().gripper_tag_local_offset_mm,
            ik=_get_ik_solver(),
        )
        if located is not None:
            return JawReading([located.xy_mm[0], located.xy_mm[1]], "aruco", er2_point)
    if ctx.robot is not None:
        site = _site_xyz(ctx.robot)
        return JawReading(
            [float(site[0] * 1000.0), float(site[1] * 1000.0)], "kinematic", er2_point
        )
    return JawReading(None, "none", er2_point)


# Longest wait (s) for a camera frame newer than the end of a motion.
FRESH_FRAME_TIMEOUT_S = 0.5


def _fresh_jpeg(ctx: ToolContext, t_motion_end: float | None) -> bytes | None:
    """Return a frame captured after the motion ended, or None if none arrived in time.

    Args:
        ctx: Tool context (fresh and latest frame providers).
        t_motion_end: ``time.monotonic()`` when the arm finished moving (None: no wait).

    Returns:
        JPEG bytes or None.
    """
    if ctx.get_fresh_jpeg_fn is not None and t_motion_end is not None:
        return ctx.get_fresh_jpeg_fn(t_motion_end, FRESH_FRAME_TIMEOUT_S)
    if ctx.get_latest_jpeg_fn is not None:
        return ctx.get_latest_jpeg_fn()
    return None


def _measure_alignment(
    lock: TargetLock,
    ctx: ToolContext,
    calib: Any,
    mode: str,
    t_motion_end: float | None = None,
) -> tuple[AlignmentMeasurement | None, JawReading | None, np.ndarray | None]:
    """Measure the jaw against the locked target in the table plane.

    The frame must have been captured after the motion ended (``t_motion_end``); if no such
    frame arrives in time the measurement is unavailable (the motion itself did not fail).

    Args:
        lock: The active target lock.
        ctx: Tool context (frame providers, robot).
        calib: Workspace calibrator.
        mode: 'real' or 'sim'.
        t_motion_end: ``time.monotonic()`` when the arm finished moving.

    Returns:
        Tuple of (alignment payload, jaw reading, frame); all None when there is no frame
        provider or no fresh frame arrived.

    Raises:
        PerceptionUnavailable: If a frame arrived but cannot be decoded or processed.
    """
    if ctx.get_latest_jpeg_fn is None and ctx.get_fresh_jpeg_fn is None:
        return None, None, None
    raw_jpeg = _fresh_jpeg(ctx, t_motion_end)
    if raw_jpeg is None:
        return None, None, None
    frame_rgb = _decode_frame(raw_jpeg, "visual verification", robot_moved=True)
    try:
        jaw = _locate_jaw(ctx, calib, frame_rgb, raw_jpeg, mode)
        payload = compute_alignment_mm(lock.centre_xy_mm, jaw.xy_mm, jaw.source)
    except PerceptionUnavailable as e:
        e.robot_moved = True
        raise
    except PERCEPTION_ERRORS as e:
        raise PerceptionUnavailable(f"Perception computation failed: {e}", robot_moved=True) from e
    if jaw.er2_point is not None:
        payload["er2_gripper_point_1000"] = jaw.er2_point
    return payload, jaw, frame_rgb


def _ready_calibrator(mode: str) -> Any:
    """Return the calibrator for ``mode`` with its homography loaded (same routing as xyz)."""
    from robot.calibration import DEFAULT_SAVE_PATH, DEFAULT_SIM_SAVE_PATH

    calib = get_calibrator(mode=mode)
    is_sim = mode == "sim"
    target_path = DEFAULT_SIM_SAVE_PATH if is_sim else DEFAULT_SAVE_PATH
    if calib is not None:
        if is_sim and (getattr(calib, "mode", None) != "sim" or not calib.is_calibrated):
            if target_path.exists():
                if calib.load_calibration(target_path):
                    calib.mode = "sim"
            else:
                from simulation.sim_calibration import init_sim_workspace_calibration

                calib = init_sim_workspace_calibration(calibrator=calib)
        elif not is_sim and (getattr(calib, "mode", None) != "real" or not calib.is_calibrated):
            if target_path.exists():
                if calib.load_calibration(target_path):
                    calib.mode = "real"
    if calib is None:
        raise CalibrationUnavailable(_UNCALIBRATED_MESSAGE)
    if hasattr(calib, "require_calibrated"):
        calib.require_calibrated()
    elif getattr(calib, "is_calibrated", None) is False:
        raise CalibrationUnavailable(_UNCALIBRATED_MESSAGE)
    return calib


def _table_xy_mm(calib: Any, point_1000: list[float]) -> tuple[float, float]:
    """Map a normalised image point to robot-frame table XY (mm) through the homography."""
    px, py = calib.gemini_to_pixel(point_1000)
    xy = calib.pixel_to_robot_mm(px, py)
    return float(xy[0]), float(xy[1])


def _resolve_lock(
    ctx: ToolContext,
    state: GuardrailState,
    epoch: int,
    point: list[float],
    label: Any,
    calib: Any,
    frame_rgb: np.ndarray,
) -> TargetLock | dict[str, Any] | None:
    """Match the request to the active lock, re-acquire a stale one, or lock a new object.

    A request whose table point is within 40 mm of the lock centre IS the lock and is never
    re-refined. When further away, if the detected footprint is within 40 mm of the active lock
    (with matching colour), the active lock is preserved. Otherwise the object under the point
    is locked as a new target.

    Args:
        ctx: Tool context.
        state: Guardrail state.
        epoch: Epoch captured when the handler started.
        point: Requested [y, x].
        label: Optional model label (colour word).
        calib: Workspace calibrator.
        frame_rgb: The frame captured before the move (the lock never uses a later one).

    Returns:
        The TargetLock, an error dictionary (``target_ambiguous``), or None when there is no
        object under the point (a raw, un-locked move).

    Raises:
        PerceptionUnavailable: If the frame is missing or detection fails.
    """
    request_xy = _table_xy_mm(calib, point)
    lock = state.target
    if lock is not None and lock.distance_to_mm(request_xy) <= LOCK_MATCH_RADIUS_MM:
        if not lock.stale:
            return lock
        # Missed and re-opened: refresh the footprint near the old centre, keep c_mm/counters.
        refreshed = _detect_footprint(frame_rgb, lock.centre_1000, lock.color, calib)
        if refreshed is not None and (
            lock.distance_to_mm(refreshed["centre_xy_mm"]) <= LOCK_MATCH_RADIUS_MM
        ):
            state.commit(
                epoch,
                lambda s: s.reacquire(
                    lock,
                    (refreshed["centre_xy_mm"][0], refreshed["centre_xy_mm"][1]),
                    refreshed["centre_1000"],
                    refreshed["bbox_px"],
                    refreshed["width_mm"],
                    yaw_deg=refreshed.get("yaw_deg"),
                ),
            )
            return lock
        # The object is gone from where it was: fall through to a fresh lock.

    found = _detect_footprint(frame_rgb, point, color_from_label(label), calib)
    if found is None:
        return None

    if (
        lock is not None
        and lock.distance_to_mm(found["centre_xy_mm"]) <= LOCK_MATCH_RADIUS_MM
        and (lock.color is None or found["color"] is None or lock.color == found["color"])
    ):
        if lock.stale:
            state.commit(
                epoch,
                lambda s: s.reacquire(
                    lock,
                    (found["centre_xy_mm"][0], found["centre_xy_mm"][1]),
                    found["centre_1000"],
                    found["bbox_px"],
                    found["width_mm"],
                    yaw_deg=found.get("yaw_deg"),
                ),
            )
        return lock

    if found["width_mm"] is not None and found["width_mm"] > MAX_GRASPABLE_WIDTH_MM:
        text = (
            f"Target ambiguous: the {found['color']} blob is {found['width_mm']:.0f} mm wide "
            "(wider than one graspable object, likely two touching). Point at one object and "
            "separate them."
        )
        return {
            "status": "error",
            "error": text,
            "error_type": "perception",
            "robot_moved": False,
            "visual_hint": text,
            "reason": "target_ambiguous",
        }
    created: list[TargetLock] = []
    state.commit(
        epoch,
        lambda s: created.append(
            s.new_lock(
                color=found["color"],
                centre_xy_mm=(found["centre_xy_mm"][0], found["centre_xy_mm"][1]),
                centre_1000=found["centre_1000"],
                bbox_px=found["bbox_px"],
                width_mm=found["width_mm"],
                yaw_deg=found.get("yaw_deg"),
            )
        ),
    )
    return created[0] if created else None


def _detect_footprint(
    frame_rgb: np.ndarray, point: list[float], color: str | None, calib: Any
) -> dict[str, Any] | None:
    """Detect the object near ``point`` in ``frame_rgb`` and return its footprint, or None."""
    try:
        _, obj = refine_point_to_nearest_object(
            point, frame_rgb, target_color=color, calibrator=calib
        )
    except PERCEPTION_ERRORS as e:
        raise PerceptionUnavailable(f"Visual refinement failed: {e}", robot_moved=False) from e
    if obj is None or "centre_xy_mm" not in obj:
        return None
    return {
        "color": obj["color"],
        "centre_xy_mm": obj["centre_xy_mm"],
        "centre_1000": obj["centre_1000"],
        "bbox_px": obj["bbox_px"],
        "width_mm": obj.get("width_mm"),
        "yaw_deg": obj.get("yaw_deg"),
    }


def _refusal(
    err_text: str, error_type: str, hint: str | None = None, reason: str | None = None
) -> dict[str, Any]:
    """Build a motion-refusal response (the arm did not move)."""
    resp: dict[str, Any] = {
        "status": "error",
        "error": err_text,
        "error_type": error_type,
        "robot_moved": False,
        "visual_hint": hint or err_text,
    }
    if reason is not None:
        resp["reason"] = reason
    return resp


def _check_descent(
    ctx: ToolContext,
    state: GuardrailState,
    epoch: int,
    lock: TargetLock | None,
    point: list[float],
    mode: str,
) -> tuple[dict[str, Any] | None, str | None]:
    """Gate a descent (z < 3 cm) on a measured, aligned hover at the same lock.

    Args:
        ctx: Tool context.
        state: Guardrail state.
        epoch: Epoch captured when the handler started.
        lock: The matched lock, or None for a raw move.
        point: Requested [y, x].
        mode: 'real' or 'sim'.

    Returns:
        Tuple of (refusal dictionary or None, warning text or None).
    """
    policy = get_settings().get_descent_gate(mode)
    if policy == "off":
        return None, None

    hover_text = hover_call(lock, point)
    hover = state.last_hover
    robot = ctx.robot
    assert robot is not None

    if lock is None:
        text = (
            "No object detected under the point; hover over the object first: "
            f"{hover_call(None, point)}."
        )
        return _refusal(text, "perception", text, reason="no_lock"), None

    if hover is None or state.last_motion != "hover":
        text = f"No measured hover yet; {hover_call(lock)}."
        return _refusal(text, "perception", text, reason="no_hover"), None

    if hover.lock_id != lock.lock_id:
        text = f"The last hover was over a different object; {hover_call(lock)}."
        return _refusal(text, "perception", text, reason="other_lock"), None

    site_xy = _site_xyz(robot)[:2] * 1000.0
    dist_mm = float(
        np.hypot(site_xy[0] - hover.commanded_xy_mm[0], site_xy[1] - hover.commanded_xy_mm[1])
    )
    if dist_mm > HOVER_STILL_VALID_MM:
        text = (
            f"Arm moved {dist_mm:.1f} mm from the hover position (limit is "
            f"{HOVER_STILL_VALID_MM:.1f} mm); {hover_call(lock)}."
        )
        return _refusal(text, "perception", text, reason="arm_moved"), None

    if hover.is_aligned is None:
        text = "alignment unobservable (no jaw tag): the jaw could not be located in the image"
        if policy == "block":
            state.commit(epoch, lambda s: s.escalate("alignment_unobservable"))
            msg = (
                f"Descent blocked by descent_gate: {text}; DESCENT_GATE=block. "
                "Waiting for the user."
            )
            return _refusal(msg, "descent_gate", msg), None
        return None, f"Descending with {text}, per DESCENT_GATE=warn."

    if hover.is_aligned is False:
        residual = hover.residual_mm or (0.0, 0.0)
        correction = lock.c_mm if lock is not None else (0.0, 0.0)
        hint = (
            f"Re-issue the hover {hover_text}; the controller applies the measured correction "
            f"([{correction[0]:.1f}, {correction[1]:.1f}] mm) automatically. Do not offset the "
            "point yourself."
        )
        if policy == "block":
            msg = (
                "Descent blocked by descent_gate: the gripper is misaligned with the target by "
                f"[dx={residual[0]:.1f}, dy={residual[1]:.1f}] mm. {hint}"
            )
            return _refusal(msg, "descent_gate", msg), None
        return None, "Gripper was misaligned at hover; descending anyway per DESCENT_GATE=warn."
    return None, None


def _move_site(
    robot: RobotInterface,
    target_xyz: np.ndarray,
    approach: str,
) -> tuple[list[float], list[str]]:
    """Move the grasp site to ``target_xyz`` through safe segments; return clamped joints.

    Args:
        robot: Robot controller.
        target_xyz: Target site position (m).
        approach: IK approach mode.

    Returns:
        Tuple of (executed joint angles in degrees, clamp warnings).
    """
    segments = plan_approach_segments(
        _get_ik_solver(), robot.get_joint_positions(), target_xyz, approach=approach
    )
    warnings: list[str] = []
    clamped_segments: list[MotionSegment] = []
    clamped_qpos: list[float] = []
    for i, seg in enumerate(segments):
        clamped, warns = validate_and_clamp_joints([float(v) for v in np.degrees(seg.qpos_rad)])
        if i == len(segments) - 1:
            clamped_qpos = clamped
            warnings = warns
        clamped_segments.append(
            MotionSegment(np.radians(clamped), seg.duration_sec, seg.site_xyz_m)
        )
    run_segments(robot, clamped_segments)
    return clamped_qpos, warnings


def _handle_move_to_spatial_point(
    ctx: ToolContext, args: dict[str, Any]
) -> MoveToSpatialPointResponse | dict[str, Any]:
    """Execute move_to_spatial_point tool call through the pick-loop state machine.

    APPROACH: hover/descent at a locked target with the descent gate and the bias corrector.
    HELD_UNVERIFIED / CLOSED_EMPTY: only the lift (straight up, verifies the grasp) is allowed.
    TRANSPORT: free motion to a raw destination; the aperture is re-read after every move.

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
        raise ToolArgumentError(err_msg, robot_moved=False)

    approach = str(args.get("approach", "")).strip().lower().replace("-", "_").replace(" ", "_")
    if not approach or approach not in ("top_down", "horizontal"):
        err_msg = "move_to_spatial_point requires 'approach' parameter ('top_down' or 'horizontal')"
        log_with_tag(logger, "ERROR", err_msg)
        raise ToolArgumentError(err_msg, robot_moved=False)

    z_height_cm = float(args.get("z_height_cm", 3.0))
    # Clamp height to reasonable safety bounds [0.0cm, 20.0cm]
    z_height_cm = max(0.0, min(20.0, z_height_cm))
    point = [float(point[0]), float(point[1])]

    state = ctx.guardrail_state or GuardrailState()
    epoch = state.epoch
    mode = resolve_robot_mode(robot, ctx.mode)

    if state.escalated:
        return _retry_exhausted_response(state)

    if state.phase in (Phase.HELD_UNVERIFIED, Phase.CLOSED_EMPTY):
        if z_height_cm < 3.0:
            return _transport_gate_error(state)
        return _lift(ctx, state, epoch, z_height_cm, approach, mode)

    if state.phase == Phase.TRANSPORT:
        return _transport_move(ctx, state, epoch, point, z_height_cm, approach, mode, args)

    return _approach_move(ctx, state, epoch, point, z_height_cm, approach, mode, args)


# A learned correction beyond this (mm, per axis) means the measurement chain is wrong, not the arm.
MAX_CORRECTION_MM = 40.0


def _clamp_correction(c_mm: tuple[float, float]) -> tuple[float, float]:
    """Clamp a per-lock bias correction to +-MAX_CORRECTION_MM per axis."""
    return (
        max(-MAX_CORRECTION_MM, min(MAX_CORRECTION_MM, c_mm[0])),
        max(-MAX_CORRECTION_MM, min(MAX_CORRECTION_MM, c_mm[1])),
    )


def _approach_move(
    ctx: ToolContext,
    state: GuardrailState,
    epoch: int,
    point: list[float],
    z_height_cm: float,
    approach: str,
    mode: str,
    args: dict[str, Any],
) -> MoveToSpatialPointResponse | dict[str, Any]:
    """Hover or descend at a locked target (phase APPROACH)."""
    robot = ctx.robot
    assert robot is not None
    is_descent = z_height_cm < 3.0

    lock: TargetLock | None = None
    matched_obj: dict[str, Any] | None = None
    if ctx.get_latest_jpeg_fn is not None:
        # One frame, captured before any motion, is the only evidence the lock is built from.
        frame0 = _decode_frame(ctx.get_latest_jpeg_fn(), "target refinement", robot_moved=False)
        calib = _ready_calibrator(mode)
        resolved = _resolve_lock(ctx, state, epoch, point, args.get("label"), calib, frame0)
        if isinstance(resolved, dict):
            return resolved
        lock = resolved
    else:
        calib = None

    guardrail_warning: str | None = None
    if is_descent:
        refused, guardrail_warning = _check_descent(ctx, state, epoch, lock, point, mode)
        if refused is not None:
            state.commit(epoch, lambda s: s.record_descent_refusal())
            return refused
    elif lock is not None and lock.hovers_since_descent >= MAX_HOVERS_PER_LOCK:
        state.commit(epoch, lambda s: s.escalate("hovers"))
        return _retry_exhausted_response(state)

    if lock is not None:
        x_mm, y_mm = lock.commanded_xy_mm()
        target_xyz = np.array([x_mm / 1000.0, y_mm / 1000.0, z_height_cm / 100.0])
        matched_obj = {"color": lock.color} if lock.color else None
        overlay_point = lock.centre_1000
    else:
        target_xyz = image_point_to_xyz(point, hover_z=z_height_cm / 100.0, mode=mode)
        overlay_point = point

    log_with_tag(
        logger,
        "EXEC",
        f"move_to_spatial_point: image [{point[0]:.0f}, {point[1]:.0f}] "
        f"(z={z_height_cm:.1f}cm, approach={approach}) → XYZ "
        f"[{target_xyz[0]:.3f}, {target_xyz[1]:.3f}, {target_xyz[2]:.3f}]",
    )
    _update_target_overlay(overlay_point, z_height_cm, approach, args.get("label"), matched_obj)

    clamped_qpos, warnings = _move_site(robot, target_xyz, approach)
    t_motion_end = time.monotonic()

    perception: AlignmentMeasurement | None = None
    jaw: JawReading | None = None
    has_frames = ctx.get_latest_jpeg_fn is not None or ctx.get_fresh_jpeg_fn is not None
    if lock is not None and has_frames:
        perception, jaw, _ = _measure_alignment(lock, ctx, calib, mode, t_motion_end)
        if perception is None:
            warnings = [
                *warnings,
                "Alignment not measured: no camera frame newer than the end of the motion "
                "arrived in time.",
            ]

    # The correction this move was commanded with (before it is updated from this measurement).
    applied_correction = [lock.c_mm[0], lock.c_mm[1]] if lock is not None else None

    offset = perception["measured_offset"] if perception is not None else None
    residual = (offset["dx_mm"], offset["dy_mm"]) if offset is not None else None
    implausible = (
        not is_descent
        and jaw is not None
        and jaw.source == "aruco"
        and offset is not None
        and offset["distance_mm"] > IMPLAUSIBLE_RESIDUAL_MM
    )
    learn = (
        not is_descent
        and not implausible
        and jaw is not None
        and jaw.source == "aruco"
        and residual is not None
        and get_settings().bias_correction
    )

    def commit_motion(s: GuardrailState) -> None:
        if is_descent:
            s.last_motion = "descent"
            s.last_hover = None
            s.descent_refusals = 0
            if lock is not None:
                lock.hovers_since_descent = 0
        elif lock is not None and perception is not None and jaw is not None and not implausible:
            if learn and residual is not None:
                lock.c_mm = _clamp_correction(
                    (lock.c_mm[0] + residual[0], lock.c_mm[1] + residual[1])
                )
            s.last_motion = "hover"
            s.last_hover = HoverRecord(
                lock_id=lock.lock_id,
                commanded_xy_mm=(float(target_xyz[0] * 1000.0), float(target_xyz[1] * 1000.0)),
                is_aligned=perception["is_aligned"],
                residual_mm=residual,
                source=jaw.source,
            )
            lock.hovers_since_descent += 1
        else:
            s.last_motion = "other"
            s.last_hover = None
            if lock is not None and not is_descent:
                lock.hovers_since_descent += 1
        if not implausible:
            s.record_motion_success()

    state.commit(epoch, commit_motion)

    if implausible and offset is not None:
        text = (
            f"Implausible alignment measurement ({offset['distance_mm']:.0f} mm between the jaw "
            f"tag and the target, more than {IMPLAUSIBLE_RESIDUAL_MM:.0f} mm): not trusted, the "
            "correction was not updated. The arm moved to the hover position. Re-issue the "
            f"hover {hover_call(lock, point)} to measure again."
        )
        refusal = _refusal(text, "perception", text)
        refusal["robot_moved"] = True
        return refusal

    visual_hint = _approach_hint(z_height_cm, lock, perception, point)
    return build_move_spatial_response(
        target_xyz_m=target_xyz.tolist(),
        executed_qpos_deg=clamped_qpos,
        approach=approach,
        visual_hint=visual_hint,
        perception=perception,
        warnings=warnings if warnings else None,
        applied_correction_mm=applied_correction,
        gripper_source=jaw.source if jaw is not None else None,
        guardrail_warning=guardrail_warning,
        target_xy_mm=list(lock.centre_xy_mm) if lock is not None else None,
    )


def _approach_hint(
    z_height_cm: float,
    lock: TargetLock | None,
    perception: AlignmentMeasurement | None,
    point: list[float],
) -> str:
    """Honest visual hint for a hover/descent: report the measurement, name the next call."""
    if z_height_cm < 3.0:
        return (
            f"Arm reached contact height (z={z_height_cm:.1f}cm). Close the gripper with "
            "close_gripper(); the gripper sensor, not the image, tells whether it closed on the "
            "object."
        )
    prefix = f"Arm reached hover position (z={z_height_cm:.1f}cm). Visual Verification Loop:"
    if perception is None or lock is None:
        return (
            f"{prefix} alignment not measured (no camera frame, or no object detected at the "
            "point)."
        )
    offset = perception["measured_offset"]
    hover_text = hover_call(lock, point)
    if perception["is_aligned"] is True and offset is not None:
        c = lock.centre_1000
        return (
            f"{prefix} measured alignment confirmed "
            f"(offset [dx={offset['dx_mm']:.1f}, dy={offset['dy_mm']:.1f}] mm, within "
            f"{perception['alignment_tolerance_mm']:.1f} mm). Ready for descent: "
            f"move_to_spatial_point(point=[{round(c[0])}, {round(c[1])}], z_height_cm=0.0)."
        )
    if perception["is_aligned"] is False and offset is not None:
        return (
            f"{prefix} measured misalignment "
            f"[dx={offset['dx_mm']:.1f}, dy={offset['dy_mm']:.1f}] mm "
            f"(tolerance {perception['alignment_tolerance_mm']:.1f} mm). Re-issue the hover "
            f"{hover_text}; the controller applies the measured correction automatically. Do not "
            "offset the point yourself."
        )
    return (
        f"{prefix} alignment unobservable (gripper source: {perception['gripper_source']}, "
        "no jaw tag detected)."
    )


def _lift(
    ctx: ToolContext,
    state: GuardrailState,
    epoch: int,
    z_height_cm: float,
    approach: str,
    mode: str,
) -> MoveToSpatialPointResponse | dict[str, Any]:
    """Lift straight up from the grasp and verify it (phase HELD_UNVERIFIED / CLOSED_EMPTY).

    The XY the model sent is ignored: the arm rises at the current forward-kinematics XY, so a
    drifting point can never turn the lift into a sideways transport. The post-lift gripper
    sensor decides; vision can only veto.
    """
    robot = ctx.robot
    assert robot is not None
    requested_z_cm = z_height_cm
    z_height_cm = max(z_height_cm, MIN_LIFT_HEIGHT_CM)
    site = _site_xyz(robot)
    target_xyz = np.array([site[0], site[1], z_height_cm / 100.0])
    log_with_tag(
        logger,
        "EXEC",
        f"lift: straight up at [{site[0] * 1000:.0f}, {site[1] * 1000:.0f}] mm to "
        f"z={z_height_cm:.1f}cm",
    )
    clamped_qpos, warnings = _move_site(robot, target_xyz, approach)
    t_motion_end = time.monotonic()

    grasp = classify_grasp(robot.read_gripper_state(), get_settings())
    has_frames = ctx.get_latest_jpeg_fn is not None or ctx.get_fresh_jpeg_fn is not None
    calib = _ready_calibrator(mode) if has_frames else None
    vote = "unavailable"
    if has_frames:
        try:
            raw = _fresh_jpeg(ctx, t_motion_end)
            frame_rgb = _decode_frame(raw, "visual verification", True)
            lift_site = _site_xyz(robot)
            grasp_z = state.grasp_z_mm if state.grasp_z_mm is not None else 0.0
            dz_mm = (lift_site[2] * 1000.0) - grasp_z
            lift_height_mm = max(dz_mm, lift_site[2] * 1000.0, MIN_LIFT_HEIGHT_CM * 10.0)
            vote = vision_vote(frame_rgb, state.target, calib, lift_height_mm=lift_height_mm)
        except PerceptionUnavailable:
            vote = "unavailable"
        except PERCEPTION_ERRORS as e:
            log_with_tag(logger, "WARN", f"Post-lift vision vote failed: {e}")
            vote = "unavailable"

    verdict = decide_grasp(grasp["grasped"], vote, state.consecutive_unavailable)

    def commit_verdict(s: GuardrailState) -> None:
        s.last_verdict = verdict.to_dict()
        s.last_motion = "lift"
        s.last_hover = None
        if vote == "unavailable":
            s.consecutive_unavailable += 1
        else:
            s.consecutive_unavailable = 0
        if verdict.verdict == "confirmed":
            s.phase = Phase.TRANSPORT
        elif verdict.verdict == "missed":
            s.phase = Phase.CLOSED_EMPTY
        if verdict.verdict != "confirmed":
            s.record_miss_or_unverified(verdict.reason, consumes_attempt=verdict.consumes_attempt)
        s.record_motion_success()

    state.commit(epoch, commit_verdict)

    lock = state.target
    prefix = f"Lift complete (z={z_height_cm:.1f}cm, straight up from the grasp)."
    if verdict.verdict == "confirmed":
        visual_hint = (
            f"{prefix} Grasp confirmed by the gripper sensor (aperture "
            f"{grasp['aperture_deg']:.1f}°) after the lift; the object is no longer on its table "
            "footprint. Transport allowed."
        )
    elif verdict.verdict == "missed":
        visual_hint = (
            f"{prefix} Grasp missed ({verdict.reason}). Transport blocked. Call open_gripper(), "
            f"then {hover_call(lock)} to try again."
        )
    else:
        visual_hint = (
            f"{prefix} Grasp unverified ({verdict.reason}); the camera could not confirm the "
            "object left the table. Transport blocked. Re-verify with "
            f"{lift_call(lock, z_height_cm)} or call open_gripper() and retry."
        )
    return build_move_spatial_response(
        target_xyz_m=target_xyz.tolist(),
        executed_qpos_deg=clamped_qpos,
        approach=approach,
        visual_hint=visual_hint,
        warnings=warnings if warnings else None,
        grasp_verification=verdict.to_dict(),
        lift_note=(
            "The controller lifted straight up at the current gripper XY; "
            "the point you sent was ignored."
            + (
                f" The lift was raised from {requested_z_cm:.1f} cm to {z_height_cm:.1f} cm, the "
                "minimum height at which the camera can verify the grasp."
                if z_height_cm > requested_z_cm
                else ""
            )
        ),
    )


def _transport_move(
    ctx: ToolContext,
    state: GuardrailState,
    epoch: int,
    point: list[float],
    z_height_cm: float,
    approach: str,
    mode: str,
    args: dict[str, Any],
) -> MoveToSpatialPointResponse | dict[str, Any]:
    """Move a confirmed-held object to a raw destination (phase TRANSPORT).

    No refinement, no lock, no descent gate, no corrector: the destination is where the model
    pointed. The aperture is re-read after the move; a drop means the object was lost.
    """
    robot = ctx.robot
    assert robot is not None
    target_xyz = image_point_to_xyz(point, hover_z=z_height_cm / 100.0, mode=mode)
    log_with_tag(
        logger,
        "EXEC",
        f"transport: image [{point[0]:.0f}, {point[1]:.0f}] (z={z_height_cm:.1f}cm) → XYZ "
        f"[{target_xyz[0]:.3f}, {target_xyz[1]:.3f}, {target_xyz[2]:.3f}]",
    )
    _update_target_overlay(point, z_height_cm, approach, args.get("label") or "DESTINATION", None)
    clamped_qpos, warnings = _move_site(robot, target_xyz, approach)

    dropped = _recheck_hold(robot, state, epoch)

    def mark_transport(s: GuardrailState) -> None:
        s.last_motion = "transport"
        s.record_motion_success()

    state.commit(epoch, mark_transport)

    if dropped is not None:
        visual_hint = (
            "Transport move complete, but the gripper aperture shows the jaws closed on nothing: "
            "the object was lost on the way. Call open_gripper(), then hover again at the object."
        )
    elif z_height_cm < 3.0:
        visual_hint = (
            f"Arm reached the destination at z={z_height_cm:.1f}cm with the object still held. "
            "Call open_gripper() to release it."
        )
    else:
        visual_hint = f"Arm reached the destination (z={z_height_cm:.1f}cm) with the object held."
    return build_move_spatial_response(
        target_xyz_m=target_xyz.tolist(),
        executed_qpos_deg=clamped_qpos,
        approach=approach,
        visual_hint=visual_hint,
        warnings=warnings if warnings else None,
        grasp_verification=dropped,
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
            "Use z_height_cm=3.0 to hover over an object before grasping, "
            "5.0 for lifts and transit, and 0.0 for table contact."
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
    guardrail_state: GuardrailState | None = None,
    query_er2_locator_fn: Callable[[bytes], dict[str, list[float]] | None] | None = None,
    get_fresh_jpeg_fn: Callable[[float, float], bytes | None] | None = None,
) -> dict[str, Any] | GripperToolResponse | MoveToSpatialPointResponse:
    """Dispatch tool call to the RobotInterface asynchronously.

    Args:
        call: The tool call object with name and args.
        robot: The initialized RobotInterface instance.
        get_latest_jpeg_fn: Optional callable returning latest camera frame JPEG bytes.
        status_callback: Optional callable for streaming status messages to UI.
        mode: Operational mode ('real' or 'sim') selecting the workspace calibration. When
            None it is taken from ``robot.mode``.
        guardrail_state: Optional GuardrailState for closed-loop safety gating.
        query_er2_locator_fn: Optional callable for ER2 visual pointing queries.
        get_fresh_jpeg_fn: Optional callable ``(t_after, timeout_s) -> JPEG | None`` returning a
            frame captured after ``t_after`` (``time.monotonic()``); measurements use it so they
            never read a frame taken before the arm arrived.

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
        return {"error": err_msg, "error_type": "hardware", "robot_moved": False}

    if get_active_robot() is None:
        set_active_robot(robot)

    spec = TOOL_REGISTRY.get(name)
    if spec is None:
        err_msg = f"Unknown tool called: {name}"
        log_with_tag(logger, "ERROR", err_msg)
        return {"error": err_msg, "error_type": "argument", "robot_moved": False}

    ctx = ToolContext(
        robot=robot,
        get_latest_jpeg_fn=get_latest_jpeg_fn,
        status_callback=status_callback,
        mode=mode,
        guardrail_state=guardrail_state,
        query_er2_locator_fn=query_er2_locator_fn,
        get_fresh_jpeg_fn=get_fresh_jpeg_fn,
    )

    epoch0 = guardrail_state.epoch if guardrail_state is not None else 0
    result = await _run_handler(spec, ctx, args, robot, name)
    if guardrail_state is not None and name in _MOTION_TOOLS and isinstance(result, dict):
        error_type = result.get("error_type")
        if error_type in _REFUSAL_ERROR_TYPES:
            guardrail_state.commit(epoch0, lambda s: s.record_refusal())
    return result


# Motion tools whose refusals count toward the "three refusals in a row" escalation.
_MOTION_TOOLS = frozenset(
    {"move_to_spatial_point", "move_to_pose", "move_to_preset", "set_joint_positions"}
)
_REFUSAL_ERROR_TYPES = frozenset({"descent_gate", "transport_gate", "perception"})


async def _run_handler(
    spec: ToolSpec, ctx: ToolContext, args: dict[str, Any], robot: RobotInterface, name: str
) -> dict[str, Any] | GripperToolResponse | MoveToSpatialPointResponse:
    """Run a tool handler off the event loop and map every failure to a typed error payload."""
    try:
        result = await asyncio.to_thread(spec.handler, ctx, args)
        if isinstance(result, dict) and "error" in result and "error_type" not in result:
            result["error_type"] = "internal"
            result.setdefault("robot_moved", False)
        return result
    except ToolArgumentError as e:
        log_with_tag(logger, "ERROR", f"Tool argument error during {name}: {e}")
        return {
            "error": str(e),
            "error_type": "argument",
            "robot_moved": getattr(e, "robot_moved", False),
        }
    except CalibrationUnavailable as e:
        log_with_tag(logger, "ERROR", f"Calibration error during {name}: {e}")
        return {
            "error": str(e),
            "error_type": "calibration",
            "robot_moved": getattr(e, "robot_moved", False),
        }
    except PerceptionUnavailable as e:
        log_with_tag(logger, "ERROR", f"Perception error during {name}: {e}")
        return {
            "error": str(e),
            "error_type": "perception",
            "robot_moved": getattr(e, "robot_moved", False),
        }
    except HardwareError as e:
        log_with_tag(logger, "ERROR", f"Hardware error during {name}: {e}")
        try:
            robot.interrupt()
        except Exception as ie:
            log_with_tag(logger, "WARN", f"Interrupt failed following HardwareError: {ie}")
        return {
            "error": str(e),
            "error_type": "hardware",
            "robot_moved": getattr(e, "robot_moved", False),
        }
    except Exception as e:
        logger.exception(f"Internal error during {name}: {e}")
        log_with_tag(logger, "ERROR", f"Exception during execution of {name}: {e}")
        return {
            "error": f"Internal error during {name}: {e}",
            "error_type": "internal",
            "robot_moved": False,
        }


def execute_tool_call_sync(
    call: Any,
    robot: RobotInterface | None,
    get_latest_jpeg_fn: Callable[[], bytes | None] | None = None,
    status_callback: Callable[[str], None] | None = None,
    mode: str | None = None,
    guardrail_state: GuardrailState | None = None,
    query_er2_locator_fn: Callable[[bytes], dict[str, list[float]] | None] | None = None,
    get_fresh_jpeg_fn: Callable[[float, float], bytes | None] | None = None,
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
                    guardrail_state=guardrail_state,
                    query_er2_locator_fn=query_er2_locator_fn,
                    get_fresh_jpeg_fn=get_fresh_jpeg_fn,
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
                guardrail_state=guardrail_state,
                query_er2_locator_fn=query_er2_locator_fn,
                get_fresh_jpeg_fn=get_fresh_jpeg_fn,
            )
        )
