"""Unified main entry point for GeminiRobotics-SO101 simulation, hardware, and twin execution."""

import argparse
import signal
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from robot import DEFAULT_SERIAL_PORT, PRESET_POSES, RealSO101Robot
from robot.twin_robot import TwinSO101Robot
from simulation.sim_robot import SO101Robot

load_dotenv()


def _build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
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
        choices=[
            "check_environment",
            "test_simulation",
            "generate_markers",
            "setup_workspace",
            "calibrate_workspace",
        ],
        default=None,
        help=(
            "Optional verification or setup task: 'check_environment' (System & hardware "
            "diagnostics), 'test_simulation' (Kinematics validation), or 'generate_markers' / "
            "'setup_workspace' (Interactive workspace calibration & sheet generator). "
            "If omitted, defaults to launching the Gradio VLA Web UI."
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
        default=DEFAULT_SERIAL_PORT,
        help=(
            f"Serial port for physical arm (default: {DEFAULT_SERIAL_PORT}). "
            "Can also be configured via ROBOT_PORT or SO101_PORT environment variable."
        ),
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help=(
            "Development mode: watch Python source files and automatically restart "
            "the Web UI process on change."
        ),
    )
    return parser


def _scan_python_sources(project_root: Path) -> dict[Path, int]:
    """Collect current mtime snapshot for project Python files.

    Args:
        project_root: Repository root folder.

    Returns:
        Mapping from absolute Python file paths to modification timestamps (ns).
    """
    exclude_dirs = {".git", ".venv", "__pycache__", ".ruff_cache", ".mypy_cache"}
    watch_roots = [
        project_root / "main.py",
        project_root / "robot",
        project_root / "simulation",
        project_root / "ui",
    ]

    snapshot: dict[Path, int] = {}

    for root in watch_roots:
        if root.is_file() and root.suffix == ".py":
            snapshot[root] = root.stat().st_mtime_ns
            continue
        if not root.exists():
            continue

        for file_path in root.rglob("*.py"):
            if any(part in exclude_dirs for part in file_path.parts):
                continue
            try:
                snapshot[file_path] = file_path.stat().st_mtime_ns
            except FileNotFoundError:
                # File may disappear between glob and stat during save operations.
                continue

    return snapshot


def _stop_child_process(proc: subprocess.Popen[bytes]) -> None:
    """Stop child process gracefully, escalating only if needed."""
    if proc.poll() is not None:
        return

    try:
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=5)
        return
    except (subprocess.TimeoutExpired, ProcessLookupError):
        pass

    try:
        proc.terminate()
        proc.wait(timeout=3)
        return
    except (subprocess.TimeoutExpired, ProcessLookupError):
        pass

    try:
        proc.kill()
    except ProcessLookupError:
        return


def _spawn_child(project_root: Path, forward_args: list[str]) -> subprocess.Popen[bytes]:
    """Spawn the actual app process used by reload supervisor."""
    cmd = [sys.executable, str(project_root / "main.py"), *forward_args]
    cmd_display = " ".join(cmd)
    print(f"[RELOAD] Starting child process: {cmd_display}")
    return subprocess.Popen(cmd, cwd=str(project_root))


def _run_with_reload(forward_args: list[str]) -> None:
    """Run a file-watching supervisor that restarts app process on Python changes."""
    project_root = Path(__file__).resolve().parent
    snapshot = _scan_python_sources(project_root)
    proc = _spawn_child(project_root, forward_args)

    try:
        while True:
            time.sleep(0.8)

            if proc.poll() is not None:
                code = proc.returncode
                print(f"[RELOAD] Child exited with code {code}. Restarting...")
                proc = _spawn_child(project_root, forward_args)
                snapshot = _scan_python_sources(project_root)
                continue

            new_snapshot = _scan_python_sources(project_root)
            if new_snapshot == snapshot:
                continue

            changed_count = len(set(snapshot.items()) ^ set(new_snapshot.items()))
            print(f"[RELOAD] Detected {changed_count} Python file change(s). Restarting...")
            snapshot = new_snapshot
            _stop_child_process(proc)
            proc = _spawn_child(project_root, forward_args)
    except KeyboardInterrupt:
        print("\n[RELOAD] Stopping development reloader...")
    finally:
        _stop_child_process(proc)


def _run_once(args: argparse.Namespace) -> None:
    """Execute a single run of the selected mode/task."""
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

    elif args.task in ("generate_markers", "setup_workspace", "calibrate_workspace"):
        from robot.calibration import setup_workspace_interactive

        reuse_flag = True if args.reuse else (False if args.regenerate else None)

        setup_workspace_interactive(
            output_dir=args.output_dir,
            interactive=not args.non_interactive,
            reuse=reuse_flag,
        )
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


def main() -> None:
    """Execute main CLI for GeminiRobotics-SO101 (Simulation & Physical Hardware)."""
    argv = sys.argv[1:]
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.reload:
        if args.task is not None:
            parser.error("--reload can only be used when launching the Web UI (without --task).")
        forward_args = [arg for arg in argv if arg != "--reload"]
        _run_with_reload(forward_args)
        return

    _run_once(args)


if __name__ == "__main__":
    main()
