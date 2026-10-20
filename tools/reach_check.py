"""Operator reach-check tool for verifying physical and simulated workspace geometry."""

import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

# On headless Linux systems without an active display server, default to EGL offscreen rendering
if (
    sys.platform.startswith("linux")
    and "DISPLAY" not in os.environ
    and "WAYLAND_DISPLAY" not in os.environ
):
    os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import numpy as np

from agent.motion import MotionSegment, plan_approach_segments, run_segments
from agent.tools import validate_and_clamp_joints
from robot.calibration import ArucoWorkspaceCalibrator, GeometrySnapshot
from robot.cartesian_ik import CartesianIK
from robot.constants import PRESET_POSES
from robot.errors import HardwareError
from robot.interface import RobotInterface
from robot.logger import log_with_tag
from robot.real_robot import RealSO101Robot
from robot.registry import get_active_robot, set_active_robot
from robot.settings import get_settings, table_height_to_robot_z_m
from simulation.sim_calibration import init_sim_workspace_calibration
from simulation.sim_robot import SO101Robot
from vision.camera import _open_capture, get_default_camera_index
from vision.gripper_locator import detect_jaw_tag, locate_jaw_xy_mm

logger = logging.getLogger(__name__)

REACH_CHECK_HOVER_M = 0.05  # 50 mm table height


@dataclass(frozen=True)
class ReachPointResult:
    """Measured telemetry at a single reach-check waypoint."""

    name: str
    commanded_xy_mm: tuple[float, float]
    fk_xy_mm: tuple[float, float]
    locator_xy_mm: tuple[float, float] | None
    loc_minus_fk_mm: tuple[float, float] | None
    truth_xy_mm: tuple[float, float] | None
    fk_vs_truth_mm: float | None
    loc_vs_truth_mm: float | None
    tag_side_px: float | None


def build_reach_check_points(
    config_path: str | Path | None = None,
) -> list[tuple[str, float, float]]:
    """Build the ordered 5-point reach-check list (4 marker centres + mat centre).

    Args:
        config_path: Optional path to workspace calibration JSON.

    Returns:
        List of (name, x_mm, y_mm) in robot base frame.
    """
    cfg_file = Path(config_path) if config_path else get_settings().calibration_config_path
    cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
    corners = cfg.get("workspace_corners_mm", {})
    marker_ids = cfg.get("marker_ids", {})

    order = ["top_left", "top_right", "bottom_right", "bottom_left"]
    points: list[tuple[str, float, float]] = []
    xs: list[float] = []
    ys: list[float] = []

    for key in order:
        pt = corners[key]
        x_mm = float(pt["x"])
        y_mm = float(pt["y"])
        mid = marker_ids.get(key)
        label = f"{key} (ID {mid})" if mid is not None else key
        points.append((label, x_mm, y_mm))
        xs.append(x_mm)
        ys.append(y_mm)

    centre_x = sum(xs) / len(xs)
    centre_y = sum(ys) / len(ys)
    points.append(("mat_centre", centre_x, centre_y))
    return points


def _move_to_hover(
    robot: RobotInterface,
    ik: CartesianIK,
    x_mm: float,
    y_mm: float,
    mode: str,
) -> None:
    """Plan and execute safe approach segments to hover at 50 mm table height over (x_mm, y_mm)."""
    target_z_m = table_height_to_robot_z_m(REACH_CHECK_HOVER_M, mode=mode)
    target_xyz = np.array([x_mm / 1000.0, y_mm / 1000.0, target_z_m], dtype=np.float64)
    segments = plan_approach_segments(
        ik,
        robot.get_joint_positions(),
        target_xyz,
        mode=mode,
    )
    clamped_segments: list[MotionSegment] = []
    for seg in segments:
        clamped_deg, _ = validate_and_clamp_joints([float(v) for v in np.degrees(seg.qpos_rad)])
        clamped_segments.append(
            MotionSegment(np.radians(clamped_deg), seg.duration_sec, seg.site_xyz_m)
        )
    run_segments(robot, clamped_segments)


def _format_point_line(res: ReachPointResult) -> str:
    """Format single-line telemetry for one reach-check waypoint."""
    parts = [
        f"[{res.name}]",
        f"cmd=({res.commanded_xy_mm[0]:+.1f}, {res.commanded_xy_mm[1]:+.1f}) mm",
        f"fk=({res.fk_xy_mm[0]:+.1f}, {res.fk_xy_mm[1]:+.1f}) mm",
    ]
    if res.locator_xy_mm is not None and res.loc_minus_fk_mm is not None:
        dist_loc_fk = float(np.hypot(*res.loc_minus_fk_mm))
        parts.append(f"loc=({res.locator_xy_mm[0]:+.1f}, {res.locator_xy_mm[1]:+.1f}) mm")
        parts.append(
            f"loc-fk=({res.loc_minus_fk_mm[0]:+.1f}, {res.loc_minus_fk_mm[1]:+.1f}) mm "
            f"({dist_loc_fk:.1f} mm)"
        )
    else:
        parts.append("loc=none")
        parts.append("loc-fk=none")

    if res.truth_xy_mm is not None and res.fk_vs_truth_mm is not None:
        parts.append(f"truth=({res.truth_xy_mm[0]:+.1f}, {res.truth_xy_mm[1]:+.1f}) mm")
        parts.append(f"fk_err={res.fk_vs_truth_mm:.2f} mm")
        if res.loc_vs_truth_mm is not None:
            parts.append(f"loc_err={res.loc_vs_truth_mm:.2f} mm")
        else:
            parts.append("loc_err=none")

    if res.tag_side_px is not None:
        parts.append(f"tag_px={res.tag_side_px:.1f}")

    return " | ".join(parts)


def run_reach_check(
    mode: str = "sim",
    interactive: bool | None = None,
    robot: RobotInterface | None = None,
    calibrator: ArucoWorkspaceCalibrator | None = None,
    frame_fn: Callable[[], np.ndarray | None] | None = None,
    config_path: str | Path | None = None,
) -> list[ReachPointResult]:
    """Run the 5-point operator reach check and return per-point results.

    Moves the arm at normal speed to HOME, then hovers at 50 mm table height over each of the
    4 marker centres and the mat centre. Waits for Enter before each move when running in an
    interactive terminal, and returns HOME at the end (or on Ctrl-C).

    Args:
        mode: Operational mode ('real' or 'sim').
        interactive: Whether to prompt for Enter before each move. Defaults to
            ``sys.stdin.isatty()``.
        robot: Optional pre-initialized robot controller.
        calibrator: Optional pre-initialized workspace calibrator.
        frame_fn: Optional frame provider callable returning an RGB numpy array.
        config_path: Optional path to workspace calibration JSON.

    Returns:
        List of ReachPointResult for all visited points.

    Raises:
        HardwareError: If robot motion fails or is interrupted (fail-closed).
    """
    if interactive is None:
        interactive = sys.stdin.isatty()

    settings = get_settings()
    points = build_reach_check_points(config_path=config_path)
    ik = CartesianIK()

    own_robot = robot is None
    cap: cv2.VideoCapture | None = None
    prev_active = get_active_robot()

    if robot is None:
        if mode == "sim":
            robot = SO101Robot()
        else:
            robot = RealSO101Robot(port=settings.serial_port)

    set_active_robot(robot)

    try:
        if calibrator is None:
            if mode == "sim":
                calibrator = init_sim_workspace_calibration(config_path=config_path)
            else:
                calibrator = ArucoWorkspaceCalibrator(config_path=config_path, mode="real")

        if frame_fn is None:
            if mode == "sim":
                assert isinstance(robot, SO101Robot)
                sim_bot = robot

                def _sim_frame() -> np.ndarray | None:
                    return sim_bot.render_rgb_frame(640, 480)

                frame_fn = _sim_frame
            else:
                cam_idx = get_default_camera_index()
                cap = _open_capture(cam_idx)
                if cap.isOpened():
                    cap.set(
                        cv2.CAP_PROP_FOURCC,
                        cv2.VideoWriter_fourcc("M", "J", "P", "G"),  # type: ignore[attr-defined]
                    )
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.stream_width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.stream_height)

                def _real_frame() -> np.ndarray | None:
                    if cap is None or not cap.isOpened():
                        return None
                    if settings.vision_settle_s > 0:
                        time.sleep(settings.vision_settle_s)
                    ret, bgr = False, None
                    for _ in range(4):
                        ret, bgr = cap.read()
                    if not ret or bgr is None:
                        return None
                    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

                frame_fn = _real_frame

        print(
            "Operator instruction: at each hover point, measure the miss from the fingertip "
            "midpoint to the marker centre."
        )

        results: list[ReachPointResult] = []
        try:
            robot.move_to_pose(PRESET_POSES["HOME"], duration_sec=1.0)
            for name, x_mm, y_mm in points:
                if interactive:
                    input(
                        f"Press Enter to move to {name} ({x_mm:+.1f}, {y_mm:+.1f}) mm "
                        "at 50 mm height..."
                    )
                _move_to_hover(robot, ik, x_mm, y_mm, mode=mode)

                qpos = robot.get_joint_positions()
                fk_m = ik.forward_kinematics(qpos)
                fk_xy = (float(fk_m[0] * 1000.0), float(fk_m[1] * 1000.0))

                frame = frame_fn() if frame_fn is not None else None
                snapshot: GeometrySnapshot | None = None
                if calibrator is not None:
                    if frame is not None and mode == "real":
                        detected = calibrator.detect_markers(frame)
                        if len(detected) >= 4:
                            calibrator.update_homography(detected)
                    snapshot = calibrator.geometry_snapshot()

                corners = detect_jaw_tag(frame) if frame is not None else None
                tag_side_px: float | None = None
                if corners is not None:
                    edge_lengths = [
                        np.linalg.norm(corners[(i + 1) % 4] - corners[i]) for i in range(4)
                    ]
                    tag_side_px = float(np.mean(edge_lengths))

                loc = locate_jaw_xy_mm(
                    frame,
                    snapshot,
                    qpos,
                    tag_offset_mm=settings.gripper_tag_local_offset_mm,
                    ik=ik,
                    mode=mode,
                )
                loc_xy = loc.xy_mm if loc is not None else None
                loc_minus_fk = (
                    (loc_xy[0] - fk_xy[0], loc_xy[1] - fk_xy[1]) if loc_xy is not None else None
                )

                truth_xy: tuple[float, float] | None = None
                fk_vs_truth: float | None = None
                loc_vs_truth: float | None = None
                if mode == "sim" and hasattr(robot, "get_end_effector_pose"):
                    ee_pos_m = robot.get_end_effector_pose()[0]
                    truth_xy = (float(ee_pos_m[0] * 1000.0), float(ee_pos_m[1] * 1000.0))
                    fk_vs_truth = float(np.hypot(fk_xy[0] - truth_xy[0], fk_xy[1] - truth_xy[1]))
                    if loc_xy is not None:
                        loc_vs_truth = float(
                            np.hypot(loc_xy[0] - truth_xy[0], loc_xy[1] - truth_xy[1])
                        )

                res = ReachPointResult(
                    name=name,
                    commanded_xy_mm=(x_mm, y_mm),
                    fk_xy_mm=fk_xy,
                    locator_xy_mm=loc_xy,
                    loc_minus_fk_mm=loc_minus_fk,
                    truth_xy_mm=truth_xy,
                    fk_vs_truth_mm=fk_vs_truth,
                    loc_vs_truth_mm=loc_vs_truth,
                    tag_side_px=tag_side_px,
                )
                results.append(res)
                print(_format_point_line(res))

            robot.move_to_pose(PRESET_POSES["HOME"], duration_sec=1.0)
        except KeyboardInterrupt:
            print("\nInterrupted by operator (Ctrl-C); returning to HOME...")
            robot.move_to_pose(PRESET_POSES["HOME"], duration_sec=1.0)
            return results
        except HardwareError as e:
            log_with_tag(logger, "ERROR", f"HardwareError during reach_check: {e}")
            try:
                robot.interrupt()
            except HardwareError:
                pass
            raise

        return results
    finally:
        set_active_robot(prev_active)
        if cap is not None and cap.isOpened():
            cap.release()
        if own_robot:
            if hasattr(robot, "close"):
                robot.close()
            if hasattr(robot, "disconnect"):
                robot.disconnect()
