"""Environment diagnostics and preflight verification script."""

import os
import sys

# On headless Linux systems without an active display server, default to EGL offscreen rendering
if (
    sys.platform.startswith("linux")
    and "DISPLAY" not in os.environ
    and "WAYLAND_DISPLAY" not in os.environ
):
    os.environ.setdefault("MUJOCO_GL", "egl")

import json
import socket
from pathlib import Path
from typing import Callable


def check_python_and_imports() -> tuple[str, str]:
    """Verify Python version and critical third-party dependencies.

    Returns:
        Tuple of (status, message).
    """
    if sys.version_info < (3, 10):
        return (
            "FAIL",
            f"Python {sys.version.split()[0]} is unsupported. Python >= 3.10 is required.",
        )
    try:
        import cv2  # noqa: F401
        import google.genai  # noqa: F401
        import gradio  # noqa: F401
        import mujoco  # noqa: F401
    except ImportError as e:
        return "FAIL", f"Missing required dependency: {e}. Fix: Run `uv sync --extra ui`."
    ver = sys.version.split()[0]
    return (
        "OK",
        f"Python {ver} and core dependencies (mujoco, cv2, gradio, google.genai) imported.",
    )


def check_gemini_credentials() -> tuple[str, str]:
    """Verify that Gemini AI Studio or Vertex AI credentials are configured.

    Returns:
        Tuple of (status, message).
    """
    from robot.settings import get_settings

    settings = get_settings()
    if settings.credentials_mode == "none":
        return (
            "FAIL",
            "Gemini credentials not configured. Fix: Set GOOGLE_API_KEY or Vertex AI in .env.",
        )
    return "OK", f"Gemini credentials configured (mode: {settings.credentials_mode})."


def check_calibration_config() -> tuple[str, str]:
    """Verify that the workspace calibration JSON configuration is readable and valid.

    Returns:
        Tuple of (status, message).
    """
    from robot.settings import get_settings

    cfg_path = Path(get_settings().calibration_config_path)
    if not cfg_path.is_file():
        return (
            "FAIL",
            f"Calibration config '{cfg_path}' not found. Fix: Run `uv run python main.py --setup`.",
        )
    try:
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        return (
            "FAIL",
            f"Calibration config '{cfg_path}' is not valid JSON ({e}). "
            "Fix: Re-run `uv run python main.py --setup`.",
        )
    if not isinstance(data, dict):
        return (
            "FAIL",
            f"Calibration config '{cfg_path}' must be a JSON object.",
        )
    if "marker_ids" not in data or "workspace_corners_mm" not in data:
        return (
            "FAIL",
            f"Calibration config '{cfg_path}' missing 'marker_ids' or 'workspace_corners_mm'. "
            "Fix: Re-run `uv run python main.py --setup`.",
        )
    return "OK", f"Workspace calibration config valid ('{cfg_path}')."


def check_server_port() -> tuple[str, str]:
    """Verify that the configured web server port is available.

    Returns:
        Tuple of (status, message).
    """
    from robot.settings import get_settings

    port = get_settings().server_port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("0.0.0.0", port))
        except OSError:
            return (
                "FAIL",
                f"Server port {port} is already in use. "
                f"Fix: Free port (`lsof -ti :{port} | xargs kill -9`) or change SERVER_PORT.",
            )
    return "OK", f"Server port {port} is free."


def check_sim_scene() -> tuple[str, str]:
    """Verify that the MuJoCo simulation scene loads successfully.

    Returns:
        Tuple of (status, message).
    """
    from simulation.sim_robot import SO101Robot

    robot = SO101Robot()
    nq = robot.model.nq
    acts = len(robot.actuator_ids)
    robot.close()
    return "OK", f"MuJoCo scene loaded ({nq} DOFs, {acts} actuators)."


def check_sim_ik() -> tuple[str, str]:
    """Verify that the Cartesian IK solver resolves a reachable target.

    Returns:
        Tuple of (status, message).
    """
    import numpy as np

    from robot.cartesian_ik import CartesianIK

    ik = CartesianIK()
    target = np.array([0.18, 0.05, 0.15])
    qpos = ik.solve_ik(target_pos=target)
    if qpos is None or not np.all(np.isfinite(qpos)):
        return "FAIL", f"IK solver failed to resolve target {target}."
    return "OK", f"IK solved target {target} -> qpos: {np.round(qpos, 3)}."


def check_sim_render() -> tuple[str, str]:
    """Verify that offscreen rendering returns an RGB frame.

    Returns:
        Tuple of (status, message).
    """
    from simulation.sim_robot import SO101Robot

    robot = SO101Robot()
    try:
        frame = robot.render_rgb_frame(width=640, height=480)
        if frame is None or getattr(frame, "shape", None) != (480, 640, 3):
            shape_info = getattr(frame, "shape", None)
            return (
                "FAIL",
                f"Offscreen render returned invalid frame shape: {shape_info}.",
            )
        return "OK", f"Offscreen render returned {frame.shape[1]}x{frame.shape[0]} frame."
    finally:
        robot.close()


def check_real_lerobot() -> tuple[str, str]:
    """Verify that the lerobot package is installed for hardware control.

    Returns:
        Tuple of (status, message).
    """
    from robot.real_robot import LEROBOT_AVAILABLE

    if not LEROBOT_AVAILABLE:
        return (
            "FAIL",
            "LeRobot package not installed. Fix: Run `uv sync --extra hw --extra ui`.",
        )
    return "OK", "LeRobot package and FeetechMotorsBus available."


def check_real_serial_port() -> tuple[str, str]:
    """Verify that the configured serial port device exists and has read/write permissions.

    Returns:
        Tuple of (status, message).
    """
    from robot.settings import get_settings

    serial_port = get_settings().serial_port
    port_path = Path(serial_port)
    if not port_path.exists():
        return (
            "FAIL",
            f"Serial port '{serial_port}' not found on system. "
            "Fix: Connect SO-101 USB cable or configure ROBOT_PORT in .env.",
        )
    has_rw = os.access(port_path, os.R_OK) and os.access(port_path, os.W_OK)
    if not has_rw:
        return (
            "FAIL",
            f"Serial port '{serial_port}' exists but lacks read/write permissions.\n"
            f"       Immediate fix: sudo chmod 666 {serial_port}\n"
            f"       Permanent fix: sudo usermod -a -G dialout $USER",
        )
    return "OK", f"Serial port '{serial_port}' exists and is accessible (rw)."


def check_real_motor_calibration() -> tuple[str, str]:
    """Verify that a motor calibration JSON file exists.

    Returns:
        Tuple of (status, message).
    """
    from robot.settings import get_settings

    settings = get_settings()
    for candidate in settings.motor_calibration_candidates:
        if candidate.is_file():
            return "OK", f"Motor calibration found at '{candidate}'."
    searched = "\n  ".join(str(p) for p in settings.motor_calibration_candidates)
    fid = settings.follower_robot_id
    return (
        "FAIL",
        f"No motor calibration file found. Searched:\n  {searched}\n"
        f"Fix: Run `lerobot-calibrate --robot.type=so101_follower --robot.id={fid}`",
    )


def check_real_camera() -> tuple[str, str]:
    """Verify that the configured camera opens and returns a frame.

    Returns:
        Tuple of (status, message).
    """
    import cv2

    from vision.camera import _open_capture, get_default_camera_index

    cam_idx = get_default_camera_index()
    cap = _open_capture(cam_idx, attempts=1)
    if not cap.isOpened():
        cap.release()
        return (
            "FAIL",
            f"Could not open camera at index {cam_idx}. "
            "Fix: Ensure USB camera is connected and accessible.",
        )
    try:
        from robot.settings import get_settings

        settings = get_settings()
        cap.set(
            cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc("M", "J", "P", "G"),  # type: ignore[attr-defined]
        )
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.stream_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.stream_height)
        ret, frame = cap.read()
        if not ret or frame is None:
            return (
                "FAIL",
                f"Camera at index {cam_idx} opened but failed to capture a frame. "
                "Fix: Check device or camera permissions.",
            )
        h, w = frame.shape[:2]
        from robot.calibration import ArucoWorkspaceCalibrator
        from vision.gripper_locator import JAW_TAG_ID, detect_jaw_tag

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        corners = detect_jaw_tag(rgb, tag_id=JAW_TAG_ID)
        offset = settings.gripper_tag_local_offset_mm
        if corners is not None:
            cx, cy = corners.mean(axis=0)
            tag_info = f"jaw tag ID {JAW_TAG_ID} detected at ({cx:.1f}, {cy:.1f}) px"
        else:
            tag_info = f"jaw tag ID {JAW_TAG_ID} not in view"

        af_val = cap.get(cv2.CAP_PROP_AUTOFOCUS)
        focus_val = cap.get(cv2.CAP_PROP_FOCUS)
        focus_info = f"autofocus={af_val:.0f}, focus={focus_val:.0f}"

        calib = ArucoWorkspaceCalibrator(mode="real")
        markers = calib.detect_markers(rgb)
        if len(markers) >= 4:
            calib.update_homography(markers)
        snap = calib.geometry_snapshot()
        if snap is not None and snap.camera is not None:
            c = snap.camera.centre_mm
            cam_pose_info = (
                f"centre_mm=[{c[0]:.0f}, {c[1]:.0f}, {c[2]:.0f}] "
                f"height_mm={c[2]:.0f} hfov={snap.hfov_deg:.1f}"
            )
        else:
            cam_pose_info = f"camera centre unavailable (hfov={settings.camera_hfov_deg:.1f})"

        return (
            "OK",
            f"Camera index {cam_idx} opened and returned {w}x{h} frame "
            f"({focus_info}; {cam_pose_info}; {tag_info}; offset={offset} mm).",
        )
    finally:
        cap.release()


def check_real_homography() -> tuple[str, str]:
    """Verify that the workspace homography file exists.

    Returns:
        Tuple of (status, message).
    """
    from robot.settings import get_settings

    save_path = Path(get_settings().calibration_save_path)
    if not save_path.is_file():
        return (
            "WARN",
            f"Workspace homography matrix not found at '{save_path}'. "
            "Fix: Calibrate from the UI panel.",
        )
    return "OK", f"Workspace homography found at '{save_path}'."


def _run_single_check(name: str, check_fn: Callable[[], tuple[str, str]]) -> tuple[str, str]:
    """Execute a single check function with diagnostic safety boundary.

    Args:
        name: Name of the check being run.
        check_fn: Diagnostic check callable.

    Returns:
        Tuple of (status, message).
    """
    try:
        return check_fn()
    except Exception as exc:
        return "FAIL", f"{name} check raised unexpected error: {exc}"


def check_environment(mode: str = "sim") -> int:
    """Run preflight checks for the specified mode without moving the robot.

    Args:
        mode: Execution mode ('sim' or 'real').

    Returns:
        0 if all checks passed (OK or WARN), 1 if any check failed (FAIL).
    """
    common_checks: list[tuple[str, Callable[[], tuple[str, str]]]] = [
        ("Dependencies", check_python_and_imports),
        ("Gemini Credentials", check_gemini_credentials),
        ("Calibration Config", check_calibration_config),
        ("Server Port", check_server_port),
    ]

    sim_checks: list[tuple[str, Callable[[], tuple[str, str]]]] = [
        ("MuJoCo Scene", check_sim_scene),
        ("Inverse Kinematics", check_sim_ik),
        ("Offscreen Render", check_sim_render),
    ]

    real_checks: list[tuple[str, Callable[[], tuple[str, str]]]] = [
        ("LeRobot", check_real_lerobot),
        ("Serial Port", check_real_serial_port),
        ("Motor Calibration", check_real_motor_calibration),
        ("Camera Capture", check_real_camera),
        ("Workspace Homography", check_real_homography),
    ]

    checks = list(common_checks)
    if mode == "sim":
        checks.extend(sim_checks)
    elif mode == "real":
        checks.extend(real_checks)
    else:
        print(f"[FAIL] Invalid mode '{mode}'. Choose 'sim' or 'real'.")
        return 1

    any_fail = False
    for name, fn in checks:
        status, message = _run_single_check(name, fn)
        print(f"[{status}] {message}")
        if status == "FAIL":
            any_fail = True

    return 1 if any_fail else 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preflight environment checks.")
    parser.add_argument("--mode", choices=["sim", "real"], default="sim", help="Target mode")
    cli_args = parser.parse_args()
    sys.exit(check_environment(mode=cli_args.mode))
