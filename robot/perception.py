"""Perception metrics engine for SO-101 visual verification and anti-hallucination guardrails."""

import math
from typing import Any

import numpy as np

from robot.contracts import AlignmentMeasurement, VisualAlignmentResponse
from robot.settings import get_settings


def compute_visual_alignment(
    frame_rgb: np.ndarray | None,
    target_point_1000: list[float] | tuple[float, float] | np.ndarray,
    gripper_point_1000: list[float] | tuple[float, float] | np.ndarray | None = None,
    robot_state: dict[str, Any] | None = None,
    tolerance: float | None = None,
    calibrator: Any | None = None,
    gripper_source: str | None = None,
) -> VisualAlignmentResponse:
    """Calculate measured visual alignment offset between gripper and target object.

    Args:
        frame_rgb: Current RGB camera frame (unused; kept for call-site compatibility).
        target_point_1000: Normalized [y, x] coordinate of the target in [0, 1000]. Used as
            given: the target is never re-detected here, callers pass the locked footprint centre.
        gripper_point_1000: Optional explicitly specified normalized [y, x] gripper position.
        robot_state: Optional dictionary containing robot telemetry or joint angles.
        tolerance: Maximum pixel error threshold in normalized [0, 1000] space.
            Defaults to settings alignment_tolerance_px.
        calibrator: Unused (kept for call-site compatibility).
        gripper_source: Optional provenance string describing how gripper position
            was estimated (e.g. 'aruco', 'kinematic', 'er2', 'provided').

    Returns:
        Dictionary adhering to ARCHITECTURE.md perception contract:
            - gripper_point_1000: [y_g, x_g] or None
            - target_point_1000: [y_t, x_t]
            - measured_offset: {"dy": float, "dx": float, "distance": float} or None
            - is_aligned: bool or None
            - alignment_tolerance: float
            - gripper_source: str ('aruco', 'kinematic', 'er2', 'provided', 'none')
    """
    if tolerance is None:
        tolerance = float(get_settings().alignment_tolerance_px)
    else:
        tolerance = float(tolerance)
    target_y = float(target_point_1000[0])
    target_x = float(target_point_1000[1])

    # Determine gripper coordinate
    if gripper_point_1000 is not None:
        grip_y = float(gripper_point_1000[0])
        grip_x = float(gripper_point_1000[1])
        source = gripper_source or "provided"
    elif robot_state and "gripper_point_1000" in robot_state:
        pt = robot_state["gripper_point_1000"]
        grip_y = float(pt[0])
        grip_x = float(pt[1])
        source = gripper_source or robot_state.get("gripper_source", "provided")
    else:
        return {
            "gripper_point_1000": None,
            "target_point_1000": [target_y, target_x],
            "measured_offset": None,
            "is_aligned": None,
            "alignment_tolerance": float(tolerance),
            "gripper_source": "none",
        }

    dy = float(target_y - grip_y)
    dx = float(target_x - grip_x)
    distance = float(math.hypot(dy, dx))
    is_aligned = bool(abs(dy) <= tolerance and abs(dx) <= tolerance)

    return {
        "gripper_point_1000": [grip_y, grip_x],
        "target_point_1000": [target_y, target_x],
        "measured_offset": {
            "dy": dy,
            "dx": dx,
            "distance": distance,
        },
        "is_aligned": is_aligned,
        "alignment_tolerance": float(tolerance),
        "gripper_source": source,
    }


def compute_alignment_mm(
    target_xy_mm: list[float] | tuple[float, float],
    jaw_xy_mm: list[float] | tuple[float, float] | None,
    source: str,
    tolerance_mm: float | None = None,
) -> AlignmentMeasurement:
    """Compare the jaw with the locked target in the table plane, in robot millimetres.

    Both points are table-plane positions (the jaw's parallax already removed by the locator),
    so subtracting them is meaningful. Alignment is only judged when the jaw was actually
    observed: the kinematic fallback is self-consistent with the IK and cannot see calibration
    or encoder bias, so it reports ``is_aligned=None`` rather than a flattering True.

    Args:
        target_xy_mm: Locked target centre [x, y] in robot mm.
        jaw_xy_mm: Jaw (grasp site) [x, y] in robot mm, or None if it was not located.
        source: Jaw locator source ('aruco', 'kinematic', 'none').
        tolerance_mm: Alignment tolerance; defaults to ``settings.alignment_tolerance_mm``.

    Returns:
        AlignmentMeasurement payload.
    """
    tol = float(get_settings().alignment_tolerance_mm if tolerance_mm is None else tolerance_mm)
    target = [float(target_xy_mm[0]), float(target_xy_mm[1])]
    if jaw_xy_mm is None:
        return {
            "gripper_xy_mm": None,
            "target_xy_mm": target,
            "measured_offset": None,
            "is_aligned": None,
            "alignment_tolerance_mm": tol,
            "gripper_source": "none",
        }
    jaw = [float(jaw_xy_mm[0]), float(jaw_xy_mm[1])]
    dx = target[0] - jaw[0]
    dy = target[1] - jaw[1]
    distance = float(math.hypot(dx, dy))
    is_aligned: bool | None = None
    if source == "aruco":
        is_aligned = bool(distance <= tol)
    return {
        "gripper_xy_mm": jaw,
        "target_xy_mm": target,
        "measured_offset": {"dx_mm": dx, "dy_mm": dy, "distance_mm": distance},
        "is_aligned": is_aligned,
        "alignment_tolerance_mm": tol,
        "gripper_source": source,
    }
