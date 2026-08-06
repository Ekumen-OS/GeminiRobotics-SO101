"""Unified main entry point for GeminiRobotics-SO101 simulation, hardware, and twin execution."""

import argparse
import time

from robot import PRESET_POSES, RealSO101Robot
from robot.twin_robot import TwinSO101Robot
from simulation.sim_robot import SO101Robot


def main() -> None:
    """Execute main CLI for GeminiRobotics-SO101 (Simulation & Physical Hardware)."""
    parser = argparse.ArgumentParser(
        description="GeminiRobotics-SO101 Main CLI — Run VLA Web UI or verification routines."
    )
    parser.add_argument(
        "--mode",
        choices=["sim", "real", "twin"],
        default="real",
        help=(
            "Execution target mode: 'real' (Physical SO-101 arm & UVC camera stream), "
            "'sim' (MuJoCo simulation), or 'twin' (Sim + Real Digital Twin Mirror)."
        ),
    )
    parser.add_argument(
        "--task",
        choices=["check_environment", "test_simulation"],
        default=None,
        help=(
            "Optional verification task: 'check_environment' (System & hardware "
            "diagnostics) or 'test_simulation' (Kinematics validation via line sweeps). "
            "If omitted, defaults to launching the Gradio VLA Web UI."
        ),
    )
    parser.add_argument(
        "--port",
        default="/dev/tty.usbmodem5B415332861",
        help="Serial port for physical arm (e.g., /dev/tty.usbmodem5B415332861).",
    )

    args = parser.parse_args()

    # On macOS, switch to mjpython before headers only if opening GLFW window in test_simulation
    if args.task == "test_simulation":
        from simulation.simulate import ensure_mjpython

        ensure_mjpython()

    task_display = args.task.upper() if args.task is not None else "WEB_UI"
    print("=" * 60)
    print(f"GeminiRobotics-SO101 Execution — Mode: [{args.mode.upper()}] | Task: [{task_display}]")
    print("=" * 60)

    # 1. Check for standalone diagnostic / verification tasks first
    if args.task == "check_environment":
        from simulation.check_env import check_environment

        check_environment()
        return

    elif args.task == "test_simulation":
        if args.mode != "sim":
            print("Notice: 'test_simulation' runs via the simulated MuJoCo physics scene.")
        from simulation.test_movement import test_robot_movement

        test_robot_movement()
        return

    # 2. Initialize Robot based on mode for Web UI operation without native window popups
    robot = None
    if args.mode == "sim":
        print("Initializing MuJoCo simulation robot arm in headless UI mode...")
        robot = SO101Robot(render_viewer=False)
        robot.reset("HOME")
    elif args.mode == "twin":
        print(f"Initializing Twin (Sim + Real on port {args.port}) without viewer...")
        robot = TwinSO101Robot(port=args.port, render_viewer=False)
        robot.reset("HOME")
    else:
        print(f"Connecting real SO-101 arm on port {args.port} for Real Mode VLA UI...")
        robot = RealSO101Robot(port=args.port)

    # 3. Execute Unified Gradio VLA Web UI server
    try:
        from ui.app import launch_web_ui

        launch_web_ui(robot=robot, mode=args.mode, server_name="0.0.0.0", server_port=7860)
    except KeyboardInterrupt:
        print("\nExecution interrupted by user.")
    finally:
        # Always park robot safely to HOME pose on completion/exit
        print("\nSafely parking robot to HOME pose...")
        if hasattr(robot, "move_to_pose"):
            robot.move_to_pose(PRESET_POSES["HOME"], duration_sec=1.0)
        elif robot is not None and hasattr(robot, "set_joint_positions"):
            robot.set_joint_positions(PRESET_POSES["HOME"])
            time.sleep(1.0)

        if robot is not None and hasattr(robot, "disconnect"):
            robot.disconnect()

        print("Task execution finished successfully.")


if __name__ == "__main__":
    main()
