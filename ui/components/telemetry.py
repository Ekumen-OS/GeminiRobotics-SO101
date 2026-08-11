"""Telemetry component for rendering joint states in Gradio."""

import numpy as np

from robot.constants import JOINT_NAMES
from robot.interface import RobotInterface


def read_measured_joint_degrees(
    robot: RobotInterface | None, default_deg: list[float]
) -> list[float]:
    """Read current joint state from robot, falling back to latest target values."""
    if robot is not None and hasattr(robot, "get_joint_positions"):
        try:
            curr = robot.get_joint_positions()
            if len(curr) == len(JOINT_NAMES):
                return [round(float(np.degrees(v)), 1) for v in curr]
        except Exception:
            pass
    return [round(float(v), 1) for v in default_deg]


def render_joint_state_panel(target_deg: list[float], measured_deg: list[float]) -> str:
    """Build a compact HTML joint telemetry panel for the movement section."""
    rows: list[str] = []
    for idx, (name, target, measured) in enumerate(
        zip(JOINT_NAMES, target_deg, measured_deg, strict=False),
        start=1,
    ):
        delta = measured - target
        rows.append(
            "<tr>"
            f"<td style='padding:6px 8px;border-bottom:1px solid #2f3440;'>J{idx} {name}</td>"
            f"<td style='padding:6px 8px;border-bottom:1px solid #2f3440;"
            f"text-align:right;'>{target:.1f}</td>"
            f"<td style='padding:6px 8px;border-bottom:1px solid #2f3440;"
            f"text-align:right;'>{measured:.1f}</td>"
            f"<td style='padding:6px 8px;border-bottom:1px solid #2f3440;"
            f"text-align:right;'>{delta:+.1f}</td>"
            "</tr>"
        )

    return (
        "<div style='border:1px solid #2f3440;border-radius:10px;padding:10px 12px;"
        "background:#161b22;'>"
        "<div style='font-weight:600;font-size:13px;margin-bottom:8px;color:#c9d1d9;'>"
        "Joint Telemetry (degrees)"
        "</div>"
        "<table style='width:100%;font-size:12px;border-collapse:collapse;color:#d0d7de;'>"
        "<thead><tr>"
        "<th style='text-align:left;padding:5px 8px;color:#8b949e;'>Joint</th>"
        "<th style='text-align:right;padding:5px 8px;color:#8b949e;'>Target</th>"
        "<th style='text-align:right;padding:5px 8px;color:#8b949e;'>Measured</th>"
        "<th style='text-align:right;padding:5px 8px;color:#8b949e;'>Delta</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
        "</div>"
    )
