"""Shared helpers for the closed pick-loop tests (scripted measurements, hint parsing)."""

import re
from collections.abc import Callable, Iterator
from typing import Any
from unittest.mock import patch

import numpy as np

from agent.guardrails import TargetLock
from agent.tools import JawReading, ToolContext
from robot.perception import compute_alignment_mm

_CALL = re.compile(
    r"move_to_spatial_point\(point=\[\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\],\s*z_height_cm=([-\d.]+)\)"
)


def parse_move_call(text: str | None) -> tuple[list[float], float] | None:
    """Parse the first ``move_to_spatial_point(point=[y, x], z_height_cm=z)`` in a hint.

    Args:
        text: Hint or error text.

    Returns:
        ([y, x], z_height_cm) or None.
    """
    match = _CALL.search(text or "")
    if not match:
        return None
    return [float(match.group(1)), float(match.group(2))], float(match.group(3))


def next_call_from(resp: dict[str, Any]) -> tuple[list[float], float] | None:
    """Return the call a tool response tells the model to make next, if any."""
    return parse_move_call(resp.get("visual_hint")) or parse_move_call(resp.get("error"))


def scripted_measurements(
    residuals: list[tuple[float, float] | None],
    source: str = "aruco",
) -> Callable[..., Any]:
    """Build a replacement for ``agent.tools._measure_alignment`` returning scripted residuals.

    Args:
        residuals: Per-call residual (target centre minus jaw, [dx, dy] mm) or None for an
            unlocated jaw; the last entry repeats once the list is exhausted.
        source: Jaw source reported with each measurement.

    Returns:
        Callable with the ``_measure_alignment`` signature.
    """
    it: Iterator[tuple[float, float] | None] = iter(residuals)
    last: list[tuple[float, float] | None] = [residuals[-1] if residuals else (0.0, 0.0)]

    def fake(
        lock: TargetLock,
        ctx: ToolContext,
        calib: Any,
        mode: str,
        t_motion_end: float | None = None,
    ) -> tuple[Any, JawReading, None]:
        nonlocal it
        residual = next(it, last[0])
        if residual is None:
            payload = compute_alignment_mm(lock.centre_xy_mm, None, "none")
            return payload, JawReading(None, "none"), None
        jaw_xy = [lock.centre_xy_mm[0] - residual[0], lock.centre_xy_mm[1] - residual[1]]
        payload = compute_alignment_mm(lock.centre_xy_mm, jaw_xy, source)
        return payload, JawReading(jaw_xy, source), None

    return fake


def patch_measurements(residuals: list[tuple[float, float] | None], source: str = "aruco") -> Any:
    """Context manager patching ``agent.tools._measure_alignment`` with scripted residuals."""
    return patch(
        "agent.tools._measure_alignment", side_effect=scripted_measurements(residuals, source)
    )


def table_xy_of_point(point: list[float]) -> np.ndarray:
    """Return the robot-frame table XY (mm) of a normalised image point (sim calibration)."""
    from vision.calibrator import get_calibrator

    calib = get_calibrator(mode="sim")
    px, py = calib.gemini_to_pixel(point)
    return calib.pixel_to_robot_mm(px, py)
