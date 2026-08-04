"""Unified main entry point for GeminiRobotics-SO101 simulation, hardware, and twin execution."""

import argparse
import time

from robot import PRESET_POSES, RealSO101Robot
from robot.twin_robot import TwinSO101Robot
from simulation.sim_robot import SO101Robot


def main() -> None:
    """Execute main CLI for GeminiRobotics-SO101 (Simulation & Physical Hardware)."""
    parser = argparse.ArgumentParser(
        description=(
            "GeminiRobotics-SO101 Main CLI — Run MuJoCo simulation or physical SO-101 commands."
        )
    )
    parser.add_argument(
        "--mode",
        choices=["sim", "real", "twin"],
        default="sim",
        help=(
            "Execution target mode: 'sim' (MuJoCo simulation), "
            "'real' (Physical SO-101 arm), or "
            "'twin' (Sim + Real Digital Twin Mirror)."
        ),
    )
    parser.add_argument(
        "--task",
        choices=["interactive", "check_environment", "test_simulation"],
        default="interactive",
        help=(
            "Task routine to execute: 'interactive' (3D MuJoCo viewer with "
            "UI actuator slider controls), 'check_environment' (System & "
            "hardware diagnostics), or 'test_simulation' (Kinematics validation "
            "via Cartesian line sweeps)."
        ),
    )
    parser.add_argument(
        "--port",
        default="/dev/tty.usbmodem1201",
        help="Serial port for real physical arm (e.g., /dev/tty.usbmodem1201 or /dev/ttyUSB0).",
    )

    args = parser.parse_args()

    print("=" * 60)
    print(
        f"GeminiRobotics-SO101 Execution — "
        f"Mode: [{args.mode.upper()}] | Task: [{args.task.upper()}]"
    )
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

    # 2. Initialize Robot based on execution mode for interactive operation
    robot = None
    if args.mode == "sim":
        print("Initializing MuJoCo simulation robot arm...")
        robot = SO101Robot()
        robot.reset("HOME")
    elif args.mode == "twin":
        print(f"Initializing Digital Twin (Sim + Real on port {args.port})...")
        robot = TwinSO101Robot(port=args.port)
        robot.reset("HOME")
    else:
        print(f"Connecting to real SO-101 robot arm on port {args.port}...")
        robot = RealSO101Robot(port=args.port)

    # 3. Execute interactive control loop
    try:
        if args.mode in ["sim", "twin"]:
            from simulation.simulate import main as launch_simulate

            launch_simulate(robot=robot)

        else:
            print("=" * 60)
            print("Physical Arm Interactive Control Session Connected.")
            print("Press Ctrl+C in terminal to disconnect and park robot.")
            print("=" * 60)
            while True:
                time.sleep(1.0)

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
