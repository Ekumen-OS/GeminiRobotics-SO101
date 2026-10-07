"""Operator tooling and offline utilities."""

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

from tools.calibration_sheet import generate_aruco_markers, generate_calibration_sheet
from tools.setup_wizard import setup_workspace_interactive

__all__ = [
    "generate_aruco_markers",
    "generate_calibration_sheet",
    "setup_workspace_interactive",
]
