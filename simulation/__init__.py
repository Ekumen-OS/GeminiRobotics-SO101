"""SO-101 (GeminiRobotics-SO101) MuJoCo Simulation and RL Environments Package."""

import os
import sys
from typing import TYPE_CHECKING, Any

# MuJoCo picks its OpenGL backend when `mujoco` is first imported, so this must run before any
# `import mujoco`. On headless Linux (e.g. an SSH session) the default GLFW backend has no display
# and every offscreen render fails with "gladLoadGL error"; EGL renders offscreen instead.
if (
    sys.platform.startswith("linux")
    and "DISPLAY" not in os.environ
    and "WAYLAND_DISPLAY" not in os.environ
):
    os.environ.setdefault("MUJOCO_GL", "egl")

from simulation.line_trajectory import LineTrajectoryGenerator
from simulation.sim_robot import SO101Robot

if TYPE_CHECKING:
    from simulation.env import SO101ReachEnv
    from simulation.warp_env import SO101WarpEnv

__all__ = [
    "LineTrajectoryGenerator",
    "SO101ReachEnv",
    "SO101Robot",
    "SO101WarpEnv",
]
__version__ = "0.1.0"


def __getattr__(name: str) -> Any:
    if name == "SO101ReachEnv":
        from simulation.env import SO101ReachEnv

        return SO101ReachEnv
    if name == "SO101WarpEnv":
        from simulation.warp_env import SO101WarpEnv

        return SO101WarpEnv
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
