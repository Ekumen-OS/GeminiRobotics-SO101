"""Operator reach-check tool for verifying physical and simulated workspace geometry."""

import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

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
from robot.errors import CalibrationUnavailable, HardwareError
from robot.interface import RobotInterface
from robot.logger import log_with_tag
from robot.real_robot import RealSO101Robot
from robot.registry import get_active_robot, set_active_robot
from robot.settings import (
    get_settings,
    robot_z_to_table_height_mm,
    table_height_to_robot_z_m,
)
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
    unreachable: bool = False
    ik_err_mm: float = 0.0


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
    status_str = f" [UNREACHABLE (IK err={res.ik_err_mm:.1f} mm)]" if res.unreachable else ""
    parts = [
        f"[{res.name}]{status_str}",
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
    telemetry_dir: str | Path | None = None,
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
        telemetry_dir: Optional directory to persist raw frames and telemetry JSON.
            Defaults to ``~/.cache/geminirobotics/reach_check/<timestamp>/``.

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

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = (
        Path(telemetry_dir)
        if telemetry_dir is not None
        else Path.home() / ".cache" / "geminirobotics" / "reach_check" / timestamp
    )
    out_dir.mkdir(parents=True, exist_ok=True)

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
            for idx, (name, x_mm, y_mm) in enumerate(points):
                if interactive:
                    input(
                        f"Press Enter to move to {name} ({x_mm:+.1f}, {y_mm:+.1f}) mm "
                        "at 50 mm height..."
                    )
                _move_to_hover(robot, ik, x_mm, y_mm, mode=mode)

                qpos = robot.get_joint_positions()
                site_m, rot = ik.gripper_frame_pose(qpos)
                fk_xy = (float(site_m[0] * 1000.0), float(site_m[1] * 1000.0))

                ik_err_mm = float(np.hypot(fk_xy[0] - x_mm, fk_xy[1] - y_mm))
                unreachable = ik_err_mm > 5.0

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

                tag_offset_world = rot @ np.asarray(
                    settings.gripper_tag_local_offset_mm, dtype=np.float64
                )
                tag_z_robot_mm = float(site_m[2] * 1000.0 + tag_offset_world[2])
                tag_height_mm = robot_z_to_table_height_mm(tag_z_robot_mm, mode=mode)

                plane_intersection_mm: tuple[float, float] | None = None
                if snapshot is not None and corners is not None:
                    centre_px = corners.mean(axis=0)
                    try:
                        plane_intersection_mm = snapshot.pixel_to_plane_mm(
                            float(centre_px[0]), float(centre_px[1]), tag_height_mm
                        )
                    except CalibrationUnavailable:
                        plane_intersection_mm = None

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
                    unreachable=unreachable,
                    ik_err_mm=ik_err_mm,
                )
                results.append(res)
                print(_format_point_line(res))

                # Save raw telemetry and frame outside repo
                slug = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
                slug = slug.replace("/", "_")
                if frame is not None:
                    bgr_save = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(out_dir / f"point_{idx:02d}_{slug}.png"), bgr_save)

                cam_data: dict[str, Any] = {}
                if snapshot is not None and snapshot.camera is not None:
                    cam_data = {
                        "centre_mm": [float(v) for v in snapshot.camera.centre_mm],
                        "hfov_deg": float(snapshot.hfov_deg),
                        "image_size": [int(v) for v in snapshot.image_size],
                    }

                point_record: dict[str, Any] = {
                    "name": name,
                    "commanded_xy_mm": [float(x_mm), float(y_mm)],
                    "fk_xy_mm": [float(fk_xy[0]), float(fk_xy[1])],
                    "ik_err_mm": ik_err_mm,
                    "unreachable": unreachable,
                    "qpos_rad": [float(v) for v in qpos],
                    "site_fk_m": [float(v) for v in site_m],
                    "tag_offset_world_mm": [float(v) for v in tag_offset_world],
                    "tag_height_mm": float(tag_height_mm),
                    "tag_corner_pixels": (
                        [[float(c[0]), float(c[1])] for c in corners]
                        if corners is not None
                        else None
                    ),
                    "plane_intersection_mm": (
                        [float(plane_intersection_mm[0]), float(plane_intersection_mm[1])]
                        if plane_intersection_mm is not None
                        else None
                    ),
                    "locator_xy_mm": (
                        [float(loc_xy[0]), float(loc_xy[1])] if loc_xy is not None else None
                    ),
                    "loc_minus_fk_mm": (
                        [float(loc_minus_fk[0]), float(loc_minus_fk[1])]
                        if loc_minus_fk is not None
                        else None
                    ),
                    "camera": cam_data,
                }
                (out_dir / f"point_{idx:02d}_{slug}.json").write_text(
                    json.dumps(point_record, indent=2), encoding="utf-8"
                )

            robot.move_to_pose(PRESET_POSES["HOME"], duration_sec=1.0)
            print(f"\nSaved reach-check telemetry and frames to: {out_dir}")
        except KeyboardInterrupt:
            print("\nInterrupted by operator (Ctrl-C); returning to HOME...")
            robot.move_to_pose(PRESET_POSES["HOME"], duration_sec=1.0)
            return results
        except HardwareError as e:
            log_with_tag(logger, "ERROR", f"HardwareError during reach_check: {e}")
            try:
                robot.interrupt()
            except HardwareError as interrupt_err:
                sys.stderr.write(f"Failed to interrupt robot: {interrupt_err}\n")
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


def replay_reach_point(
    json_path: str | Path,
    frame_path: str | Path | None = None,
    ik: CartesianIK | None = None,
) -> dict[str, Any]:
    """Recompute jaw locator from a saved reach_check telemetry JSON and frame.

    Args:
        json_path: Path to saved waypoint JSON.
        frame_path: Optional path to saved RGB/BGR frame PNG. If None, looks for matching PNG.
        ik: Optional CartesianIK instance.

    Returns:
        Dict with recomputed locator values and validation comparison against saved result.
    """
    json_file = Path(json_path)
    data = json.loads(json_file.read_text(encoding="utf-8"))

    qpos = np.asarray(data["qpos_rad"], dtype=np.float64)
    solver = ik if ik is not None else CartesianIK()
    site_m, rot = solver.gripper_frame_pose(qpos)
    fk_xy = (float(site_m[0] * 1000.0), float(site_m[1] * 1000.0))

    tag_offset_world = np.asarray(data["tag_offset_world_mm"], dtype=np.float64)
    plane_intersection = (
        tuple(data["plane_intersection_mm"])
        if data.get("plane_intersection_mm") is not None
        else None
    )

    replayed_loc_xy: tuple[float, float] | None = None
    if plane_intersection is not None:
        replayed_loc_xy = (
            float(plane_intersection[0] - tag_offset_world[0]),
            float(plane_intersection[1] - tag_offset_world[1]),
        )

    saved_loc_xy = tuple(data["locator_xy_mm"]) if data.get("locator_xy_mm") is not None else None
    matches = (
        bool(np.allclose(replayed_loc_xy, saved_loc_xy, atol=1e-3))
        if replayed_loc_xy and saved_loc_xy
        else False
    )

    return {
        "name": data.get("name"),
        "commanded_xy_mm": tuple(data["commanded_xy_mm"]),
        "fk_xy_mm": fk_xy,
        "replayed_loc_xy_mm": replayed_loc_xy,
        "saved_loc_xy_mm": saved_loc_xy,
        "matches": matches,
    }
