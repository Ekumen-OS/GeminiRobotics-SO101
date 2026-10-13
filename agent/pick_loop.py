"""Pure helpers of the closed pick loop: the lift vision vote and the refusal hints.

Kept free of tool plumbing so they can be tested against rendered frames directly.
"""

from typing import Any

import numpy as np

from agent.guardrails import GuardrailState, Phase, TargetLock
from robot.vision_detector import detect_workspace_objects

# At least this many of the 4 mat markers must be visible for a frame to be trusted.
MIN_VISIBLE_MARKERS = 3
# A blob of the lock colour whose bbox overlaps the lock-time bbox at least this much (IoU) is
# the object still sitting on its table footprint.
ON_TABLE_IOU = 0.5
# The jaw tag cannot be told apart from a wrong calibration below this many mm of residual: never
# act on a measured offset larger than this (a mis-detection, not a bias).
IMPLAUSIBLE_RESIDUAL_MM = 30.0


def bbox_iou(a: list[int] | tuple[int, ...], b: list[int] | tuple[int, ...]) -> float:
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


def vision_vote(frame_rgb: np.ndarray | None, lock: TargetLock | None, calibrator: Any) -> str:
    """Vote on whether the locked object is still on its table footprint after the lift.

    Vision can only veto a grasp, and only on positive evidence: a blob of the lock's colour
    that overlaps the lock-time bounding box. Anything that makes the frame unusable (no frame,
    fewer than 3 of the 4 mat markers, no lock reference) is ``unavailable`` and keeps transport
    blocked until it is re-verified.

    Args:
        frame_rgb: Post-lift frame, or None.
        lock: The target lock (provides the colour and the lock-time bbox), or None.
        calibrator: Workspace calibrator.

    Returns:
        'on_table', 'not_seen_on_table' or 'unavailable'.
    """
    if frame_rgb is None or lock is None or lock.bbox_px is None or lock.color is None:
        return "unavailable"
    if calibrator is None or visible_marker_count(frame_rgb, calibrator) < MIN_VISIBLE_MARKERS:
        return "unavailable"
    objects = detect_workspace_objects(frame_rgb, calibrator=calibrator)
    for obj in objects:
        if obj["color"] == lock.color and bbox_iou(obj["bbox_px"], lock.bbox_px) >= ON_TABLE_IOU:
            return "on_table"
    return "not_seen_on_table"


def hover_call(lock: TargetLock | None, fallback_point: list[float] | None = None) -> str:
    """Return the exact hover call that re-measures alignment at the target.

    Args:
        lock: Active lock (its raw point is the one the model originally sent).
        fallback_point: Point to use when there is no lock.

    Returns:
        Call text such as ``move_to_spatial_point(point=[y, x], z_height_cm=3.0)``.
    """
    point = (lock.raw_point if lock is not None and lock.raw_point else fallback_point) or [
        500.0,
        500.0,
    ]
    return f"move_to_spatial_point(point=[{point[0]:.0f}, {point[1]:.0f}], z_height_cm=3.0)"


def lift_call(lock: TargetLock | None, z_height_cm: float = 5.0) -> str:
    """Return the lift call that verifies a grasp (the controller lifts straight up).

    Args:
        lock: Active lock.
        z_height_cm: Lift height.

    Returns:
        Call text such as ``move_to_spatial_point(point=[y, x], z_height_cm=5.0)``.
    """
    point = lock.raw_point if lock is not None and lock.raw_point else [500.0, 500.0]
    return (
        f"move_to_spatial_point(point=[{point[0]:.0f}, {point[1]:.0f}], "
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
