import os
import sys

# MuJoCo picks its OpenGL backend when `mujoco` is first imported, so this must run before any
# `import mujoco`. On headless Linux (e.g. an SSH session) the default GLFW backend has no display
# and every offscreen render fails with "gladLoadGL error"; EGL renders offscreen instead.
if (
    sys.platform.startswith("linux")
    and "DISPLAY" not in os.environ
    and "WAYLAND_DISPLAY" not in os.environ
):
    os.environ.setdefault("MUJOCO_GL", "egl")

from typing import Any

from robot.calibration import (
    ArucoWorkspaceCalibrator,
    generate_aruco_markers,
    generate_calibration_sheet,
    setup_workspace_interactive,
)
from robot.cartesian_ik import CartesianIK
from robot.constants import ACTUATOR_NAMES, JOINT_NAMES, PRESET_POSES
from robot.kinematics import dh_transform, forward_kinematics, plot_robot
from robot.real_robot import RealSO101Robot
from robot.settings import Settings, get_settings, set_settings


def __getattr__(name: str) -> Any:
    """Dynamically resolve configuration defaults from central Settings."""
    if name == "DEFAULT_SERIAL_PORT":
        return get_settings().serial_port
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "ACTUATOR_NAMES",
    "DEFAULT_SERIAL_PORT",
    "JOINT_NAMES",
    "PRESET_POSES",
    "Settings",
    "get_settings",
    "set_settings",
    "ArucoWorkspaceCalibrator",
    "CartesianIK",
    "RealSO101Robot",
    "dh_transform",
    "forward_kinematics",
    "generate_aruco_markers",
    "generate_calibration_sheet",
    "plot_robot",
    "setup_workspace_interactive",
]
