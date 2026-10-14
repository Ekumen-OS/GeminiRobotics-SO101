"""Web user interface and 3D simulation viewer package."""

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
