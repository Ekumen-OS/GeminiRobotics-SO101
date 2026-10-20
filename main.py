"""Unified main entry point for GeminiRobotics-SO101 simulation and hardware execution."""

import argparse
import logging
import sys

from dotenv import load_dotenv

from robot.constants import PRESET_POSES
from robot.errors import HardwareError
from robot.interface import RobotInterface
from robot.logger import setup_logging
from robot.real_robot import RealSO101Robot
from robot.settings import Settings, get_settings, set_settings
from simulation.sim_robot import SO101Robot

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        description="GeminiRobotics-SO101 Main CLI — Run VLA Web UI or verification routines."
    )
    parser.add_argument(
        "--mode",
        choices=["sim", "real"],
        default="real",
        help=(
            "Execution target mode: 'real' (Physical SO-101 arm & UVC camera stream) "
            "or 'sim' (MuJoCo simulation)."
        ),
    )
    parser.add_argument(
        "--task",
        choices=[
            "check_environment",
            "generate_markers",
        ],
        default=None,
        help=(
            "Optional verification or setup task: 'check_environment' (System & hardware "
            "diagnostics), or 'generate_markers' (Interactive workspace calibration & sheet "
            "generator). If omitted, defaults to launching the Gradio VLA Web UI."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="assets/aruco_markers",
        help="Directory to save generated ArUco markers and calibration sheet.",
    )
    parser.add_argument(
        "--reuse",
        action="store_true",
        help="Reuse existing markers in output directory without regenerating image files.",
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Force regeneration and overwrite of all marker image files and sheet.",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Skip interactive prompts during setup tasks and use default/existing config.",
    )
    parser.add_argument(
        "--port",
        default=None,
        help="Serial port for physical arm (defaults to ROBOT_PORT or the platform default).",
    )
    return parser


def _run_once(args: argparse.Namespace) -> None:
    """Execute a single run of the selected mode/task.

    Args:
        args: Parsed command-line arguments namespace.
    """
    task_display = args.task.upper() if args.task is not None else "WEB_UI"
    print("=" * 60)
    print(f"GeminiRobotics-SO101 Execution — Mode: [{args.mode.upper()}] | Task: [{task_display}]")
    print("=" * 60)

    # 1. Check for standalone diagnostic / verification tasks first
    if args.task == "check_environment":
        from tools.check_env import check_environment

        sys.exit(check_environment(mode=args.mode))

    elif args.task == "generate_markers":
        from tools.setup_wizard import setup_workspace_interactive

        reuse_flag = True if args.reuse else (False if args.regenerate else None)

        setup_workspace_interactive(
            output_dir=args.output_dir,
            interactive=not args.non_interactive,
            reuse=reuse_flag,
        )
        return

    # 2. Initialize Robot based on mode for Web UI operation without native window popups
    settings = get_settings()
    port = settings.serial_port

    robot: RobotInterface
    if args.mode == "sim":
        print("Initializing MuJoCo simulation robot arm in headless UI mode...")
        robot = SO101Robot()
        robot.reset("HOME")
    else:
        print(f"Connecting real SO-101 arm on port {port} for Real Mode VLA UI...")
        try:
            robot = RealSO101Robot(port=port)
        except ImportError as e:
            logger.error(f"Missing dependency for real mode: {e}")
            print(
                f"\n❌ {e}\nTo use real mode, install the hw extra: `uv sync --extra hw --extra ui`"
            )
            sys.exit(1)
        except FileNotFoundError as e:
            logger.error(f"Missing motor calibration for real mode: {e}")
            print(f"\n❌ {e}")
            sys.exit(1)
        except HardwareError as e:
            logger.error(f"Hardware connection failed on port {port}: {e}")
            print(f"\n⚠️ Hardware connection failed on port {port}: {e}")
            print("Starting UI in degraded/disconnected real mode.")
            robot = RealSO101Robot(port=port, connect_on_init=False)

    # 3. Execute Unified Gradio VLA Web UI server
    try:
        from ui.app import launch_web_ui

        launch_web_ui(
            robot=robot,
            mode=args.mode,
            server_name=settings.server_host,
            server_port=settings.server_port,
        )
    except KeyboardInterrupt:
        print("\nExecution interrupted by user.")
    finally:
        # Always park robot safely to HOME pose on completion/exit
        print("\nSafely parking robot to HOME pose...")
        robot.move_to_pose(PRESET_POSES["HOME"], duration_sec=1.0)
        robot.disconnect()
        print("Task execution finished successfully.")


def main() -> None:
    """Execute main CLI for GeminiRobotics-SO101 (Simulation & Physical Hardware)."""
    load_dotenv()
    setup_logging()
    argv = sys.argv[1:]
    parser = _build_parser()
    args = parser.parse_args(argv)

    settings = Settings.from_env(serial_port=args.port)
    set_settings(settings)

    _run_once(args)


if __name__ == "__main__":
    main()
