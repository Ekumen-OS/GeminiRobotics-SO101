"""MuJoCo must see MUJOCO_GL=egl at its first import on headless Linux.

MuJoCo fixes its OpenGL backend when ``mujoco`` is first imported. If anything imports it before
the headless default is set, every offscreen render fails with ``gladLoadGL error`` (seen over
SSH, where there is no display). Each case runs in a fresh interpreter because the backend is
chosen once per process.
"""

import os
import subprocess
import sys
import textwrap

import pytest

pytestmark = [
    pytest.mark.skipif(
        not sys.platform.startswith("linux"), reason="headless EGL default is Linux-only"
    ),
    pytest.mark.slow,
]

PROBE = textwrap.dedent(
    """
    import builtins, os, sys
    real_import = builtins.__import__
    seen = {{}}
    def spy(name, *args, **kwargs):
        if name.split(".")[0] == "mujoco" and "gl" not in seen:
            seen["gl"] = os.environ.get("MUJOCO_GL")
        return real_import(name, *args, **kwargs)
    builtins.__import__ = spy
    import {module}
    print("MUJOCO_GL_AT_FIRST_IMPORT=" + str(seen.get("gl")))
    """
)


def _first_import_gl(module: str, **env_overrides: str) -> str:
    """Return MUJOCO_GL as seen by the first ``import mujoco`` while importing ``module``."""
    env = {k: v for k, v in os.environ.items() if k not in ("MUJOCO_GL", "DISPLAY")}
    env.pop("WAYLAND_DISPLAY", None)
    env.update(env_overrides)
    result = subprocess.run(
        [sys.executable, "-c", PROBE.format(module=module)],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
        check=True,
    )
    line = next(ln for ln in result.stdout.splitlines() if ln.startswith("MUJOCO_GL_AT_FIRST"))
    return line.split("=", 1)[1]


@pytest.mark.parametrize(
    "module",
    [
        "robot.cartesian_ik",
        "simulation.sim_robot",
        "vision.sim_runtime",
        "ui.app",
        "main",
    ],
)
def test_headless_linux_selects_egl_before_mujoco_is_imported(module: str) -> None:
    """Without a display, importing the package must pick EGL before MuJoCo loads."""
    assert _first_import_gl(module) == "egl"


def test_desktop_session_keeps_mujoco_default_backend() -> None:
    """With a display available, the default (GLFW) backend must be left alone."""
    assert _first_import_gl("robot.cartesian_ik", DISPLAY=":0") == "None"
