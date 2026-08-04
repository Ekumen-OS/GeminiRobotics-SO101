"""Interactive 3D simulation viewer launcher for SO-101 in MuJoCo."""

import os
import sys

import mujoco
import mujoco.viewer

from simulation.sim_robot import SO101Robot


def ensure_mjpython() -> None:
    """Relaunch under mjpython on macOS if running under default CPython."""
    if sys.platform != "darwin":
        return

    if os.environ.get("MJPYTHON_RUNNING") == "1":
        return

    import shutil

    mjpython_path = shutil.which("mjpython")
    if not mjpython_path:
        python_dir = os.path.dirname(sys.executable)
        candidate = os.path.join(python_dir, "mjpython")
        if os.path.isfile(candidate):
            mjpython_path = candidate

    if mjpython_path:
        os.environ["MJPYTHON_RUNNING"] = "1"
        os.execv(mjpython_path, [mjpython_path] + sys.argv)


def main(robot: SO101Robot | None = None) -> None:
    """Launch interactive 3D simulation viewer where users can control joints via MuJoCo UI.

    Args:
        robot: Optional robot instance (SO101Robot or TwinSO101Robot).
    """
    ensure_mjpython()

    if robot is None:
        robot = SO101Robot()
        robot.reset("HOME")

    print("=" * 60)
    print("Launching SO-101 Interactive MuJoCo Simulation Viewer")
    print("------------------------------------------------------------")
    print("Use the 'Control' / 'Actuators' sliders on the right panel in MuJoCo")
    print("to interactively move the SO-101 robot joints in real time!")
    print("Close the viewer window or press Ctrl+C in terminal to finish.")
    print("=" * 60)

    with mujoco.viewer.launch_passive(robot.model, robot.data) as viewer:
        robot.viewer = viewer
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
        while viewer.is_running():
            if hasattr(robot, "step"):
                robot.step(1)
            viewer.sync()


if __name__ == "__main__":
    main()
