"""Pure helpers of the closed pick loop: the lift vision vote and the refusal hints.

Kept free of tool plumbing so they can be tested against rendered frames directly.
"""

import math
from typing import Any

import numpy as np

from agent.guardrails import GuardrailState, Phase, TargetLock
from robot.camera_model import PinholeCamera
from vision.detector import detect_workspace_objects

# At least this many of the 4 mat markers must be visible for a frame to be trusted.
MIN_VISIBLE_MARKERS = 3
# A blob of the lock colour whose bbox overlaps the lock-time bbox at least this much (IoU) is
# the object still sitting on its table footprint.
ON_TABLE_IOU = 0.5
# If the camera angle predicts an overlap at or above this threshold between table and lift height,
# vision cannot reliably discriminate the lift, so the sensor decides (inconclusive).
PREDICTED_IOU_INCONCLUSIVE_THRESHOLD = 0.35
# The jaw tag cannot be told apart from a wrong calibration below this many mm of residual: never
# act on a measured offset larger than this (a mis-detection, not a bias).
IMPLAUSIBLE_RESIDUAL_MM = 30.0


def bbox_iou(
    a: list[int] | tuple[int, ...] | list[float] | tuple[float, ...],
    b: list[int] | tuple[int, ...] | list[float] | tuple[float, ...],
) -> float:
    """Intersection over union of two (x, y, w, h) boxes.

    Args:
        a: First box.
        b: Second box.

    Returns:
        IoU in [0, 1].
    """
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ix = max(0.0, min(ax1 + aw, bx1 + bw) - max(ax1, bx1))
    iy = max(0.0, min(ay1 + ah, by1 + bh) - max(ay1, by1))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0 else 0.0


def visible_marker_count(frame_rgb: np.ndarray, calibrator: Any) -> int:
    """Count how many of the four mat corner markers are visible in a frame.

    Args:
        frame_rgb: RGB frame.
        calibrator: Workspace calibrator (provides the marker ids and the detector).

    Returns:
        Number of expected corner markers detected (0-4).
    """
    detected = calibrator.detect_markers(frame_rgb)
    marker_ids = calibrator.config.get("marker_ids", {})
    expected = [
        int(marker_ids.get("top_left", 1)),
        int(marker_ids.get("top_right", 2)),
        int(marker_ids.get("bottom_right", 3)),
        int(marker_ids.get("bottom_left", 0)),
    ]
    return sum(1 for mid in expected if mid in detected)


def predict_cube_lift_iou(
    camera: PinholeCamera,
    centre_xy_mm: tuple[float, float],
    side_mm: float,
    yaw_deg: float,
    lift_height_mm: float,
) -> float:
    """Predict the image bounding box IoU of a cube between table height and lift height.

    Args:
        camera: Pinhole camera model in the robot frame.
        centre_xy_mm: Footprint centre on the table in robot mm.
        side_mm: Side length of the cube in mm.
        yaw_deg: Yaw orientation in degrees.
        lift_height_mm: Height the cube was lifted in mm.

    Returns:
        IoU in [0, 1] between the projected box on the table and at the lift height.
    """
    cx, cy = centre_xy_mm
    s = side_mm
    half = s / 2.0
    theta = math.radians(yaw_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    corner_offsets = [
        (-half, -half),
        (half, -half),
        (half, half),
        (-half, half),
    ]
    corners_xy = [
        (cx + dx * cos_t - dy * sin_t, cy + dx * sin_t + dy * cos_t) for dx, dy in corner_offsets
    ]
    corners_table = [(x, y, 0.0) for x, y in corners_xy] + [(x, y, s) for x, y in corners_xy]
    corners_lifted = [(x, y, lift_height_mm) for x, y in corners_xy] + [
        (x, y, lift_height_mm + s) for x, y in corners_xy
    ]
    try:
        pts_table = [camera.project(p) for p in corners_table]
        pts_lifted = [camera.project(p) for p in corners_lifted]
    except (ValueError, ZeroDivisionError):
        return 1.0

    u_t = [p[0] for p in pts_table]
    v_t = [p[1] for p in pts_table]
    box_t = [min(u_t), min(v_t), max(u_t) - min(u_t), max(v_t) - min(v_t)]

    u_l = [p[0] for p in pts_lifted]
    v_l = [p[1] for p in pts_lifted]
    box_l = [min(u_l), min(v_l), max(u_l) - min(u_l), max(v_l) - min(v_l)]

    return bbox_iou(box_t, box_l)


def vision_vote(
    frame_rgb: np.ndarray | None,
    lock: TargetLock | None,
    calibrator: Any,
    lift_height_mm: float = 50.0,
    camera: PinholeCamera | None = None,
) -> str:
    """Vote on whether the locked object is still on its table footprint after the lift.

    Vision can only veto a grasp, and only on positive evidence: a blob of the lock's colour
    that overlaps the lock-time bounding box. Anything that makes the frame unusable (no frame,
    fewer than 3 of the 4 mat markers) is ``unavailable`` and keeps transport
    blocked until it is re-verified. When the frame is usable but there is no lock reference,
    ``no_reference`` is returned so the gripper sensor decides. When the camera angle cannot
    distinguish a lifted cube from its footprint (predicted overlap >= 0.35, or no camera
    model), ``inconclusive`` is returned so the gripper sensor decides.

    Args:
        frame_rgb: Post-lift frame, or None.
        lock: The target lock (provides colour, footprint centre, yaw and bbox), or None.
        calibrator: Workspace calibrator.
        lift_height_mm: Measured lift height in mm (defaults to 50.0).
        camera: Optional explicit PinholeCamera model (retrieved from calibrator if None).

    Returns:
        'on_table', 'not_seen_on_table', 'no_reference', 'inconclusive', or 'unavailable'.
    """
    if frame_rgb is None or calibrator is None:
        return "unavailable"
    if visible_marker_count(frame_rgb, calibrator) < MIN_VISIBLE_MARKERS:
        return "unavailable"
    if lock is None or lock.bbox_px is None or lock.color is None:
        return "no_reference"

    cam = camera
    if cam is None:
        if hasattr(calibrator, "geometry_snapshot"):
            snap = calibrator.geometry_snapshot()
            cam = getattr(snap, "camera", None)
        elif hasattr(calibrator, "camera"):
            cam = getattr(calibrator, "camera", None)

    if cam is None:
        return "inconclusive"

    side = lock.width_mm or 30.0
    yaw = lock.yaw_deg or 0.0
    pred_iou = predict_cube_lift_iou(
        camera=cam,
        centre_xy_mm=lock.centre_xy_mm,
        side_mm=side,
        yaw_deg=yaw,
        lift_height_mm=lift_height_mm,
    )
    if pred_iou >= PREDICTED_IOU_INCONCLUSIVE_THRESHOLD:
        return "inconclusive"

    objects = detect_workspace_objects(frame_rgb, calibrator=calibrator)
    for obj in objects:
        if obj["color"] == lock.color and bbox_iou(obj["bbox_px"], lock.bbox_px) >= ON_TABLE_IOU:
            return "on_table"
    return "not_seen_on_table"


def hover_call(lock: TargetLock | None, fallback_point: list[float] | None = None) -> str:
    """Return the exact hover call that re-measures alignment at the target.

    Args:
        lock: Active lock.
        fallback_point: Point to use when there is no lock.

    Returns:
        Call text such as ``move_to_spatial_point(point=[y, x], z_height_cm=3.0)``.
    """
    point = (lock.centre_1000 if lock is not None and lock.centre_1000 else fallback_point) or [
        500.0,
        500.0,
    ]
    return f"move_to_spatial_point(point=[{round(point[0])}, {round(point[1])}], z_height_cm=3.0)"


def lift_call(lock: TargetLock | None, z_height_cm: float = 5.0) -> str:
    """Return the lift call that verifies a grasp (the controller lifts straight up).

    Args:
        lock: Active lock.
        z_height_cm: Lift height.

    Returns:
        Call text such as ``move_to_spatial_point(point=[y, x], z_height_cm=5.0)``.
    """
    point = lock.centre_1000 if lock is not None and lock.centre_1000 else [500.0, 500.0]
    return (
        f"move_to_spatial_point(point=[{round(point[0])}, {round(point[1])}], "
        f"z_height_cm={z_height_cm:.1f})"
    )


def transport_gate_message(state: GuardrailState) -> str:
    """Build the transport-gate refusal text naming the calls that will succeed.

    Args:
        state: Current guardrail state.

    Returns:
        Error text beginning with ``Transport blocked``.
    """
    reason = state.last_grasp_reason or "unverified"
    attempts = max(1, state.retry_count)
    lock = state.target
    if state.phase == Phase.CLOSED_EMPTY:
        return (
            f"Transport blocked: grasp not verified ({reason}); the jaws are empty or the object "
            f"was left behind. Call open_gripper(), then {hover_call(lock)} to try again "
            f"(attempt {attempts} of 3)."
        )
    return (
        f"Transport blocked: grasp not verified ({reason}). Verify it by lifting straight up with "
        f"{lift_call(lock)}, or call open_gripper() and retry (attempt {attempts} of 3)."
    )
