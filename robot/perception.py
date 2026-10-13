"""Perception metrics engine for SO-101 visual verification and anti-hallucination guardrails."""

import math

from robot.contracts import AlignmentMeasurement
from robot.settings import get_settings


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
