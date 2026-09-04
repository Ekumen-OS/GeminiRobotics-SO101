"""Global pytest configuration and environment setup for GeminiRobotics-SO101."""

import os
import sys

# Ensure headless Linux environments default to EGL offscreen rendering for MuJoCo
if (
    sys.platform.startswith("linux")
    and "DISPLAY" not in os.environ
    and "WAYLAND_DISPLAY" not in os.environ
):
    os.environ.setdefault("MUJOCO_GL", "egl")
