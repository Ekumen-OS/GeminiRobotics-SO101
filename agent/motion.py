"""Motion planning helpers for the spatial tools: safe approach segments and cancellable runs."""

import math
from dataclasses import dataclass

import numpy as np

from robot.cartesian_ik import CartesianIK
from robot.errors import HardwareError
from robot.interface import RobotInterface

# A site below this height (m) is "low": a long sweep from there would drag across the mat.
LOW_POSE_Z_M = 0.03
# Moves shorter than this (m) in XY are local adjustments and never need a lift-first approach.
LOCAL_MOVE_XY_M = 0.03
# Height (m) the arm is kept above while travelling away from a low pose.
SAFE_TRAVEL_Z_M = 0.06


@dataclass(frozen=True)
class MotionSegment:
    """One joint-space move of a multi-segment motion.

    Attributes:
        qpos_rad: Target joint angles (radians), gripper included.
        duration_sec: Move duration in seconds.
        site_xyz_m: Cartesian grasp-site target of this segment (metres).
    """

    qpos_rad: np.ndarray
    duration_sec: float
    site_xyz_m: tuple[float, float, float]


def plan_approach_segments(
    ik: CartesianIK,
    current_qpos_rad: np.ndarray,
    target_xyz_m: np.ndarray,
    approach: str = "top_down",
    travel_duration_sec: float = 1.5,
) -> list[MotionSegment]:
    """Plan the joint-space segments that bring the grasp site to ``target_xyz_m``.

    A single segment is used unless the arm starts low (site below 3 cm, e.g. at HOME or after a
    grasp descent) and the target is more than 30 mm away: then the arm first goes straight up to
    6 cm, travels at or above 6 cm, and only descends vertically over the target. This keeps the
    first move from sweeping the jaws across the workspace below 6 cm, where they would push
    objects.

    Args:
        ik: Cartesian IK solver (also used for the forward kinematics of the start pose).
        current_qpos_rad: Current joint angles (radians); the gripper joint is held.
        target_xyz_m: Target grasp-site position (metres).
        approach: IK approach mode ('top_down' or 'horizontal').
        travel_duration_sec: Duration of the (single or travel) segment.

    Returns:
        Ordered list of MotionSegment, never empty.
    """
    current = np.asarray(current_qpos_rad, dtype=np.float64)
    target = np.asarray(target_xyz_m, dtype=np.float64)
    site = ik.forward_kinematics(current)
    gripper = float(current[5]) if len(current) >= 6 else 0.0

    waypoints: list[tuple[np.ndarray, float]] = []
    far = math.hypot(target[0] - site[0], target[1] - site[1]) > LOCAL_MOVE_XY_M
    if site[2] < LOW_POSE_Z_M and far:
        waypoints.append((np.array([site[0], site[1], SAFE_TRAVEL_Z_M]), 1.0))
        travel = np.array([target[0], target[1], max(float(target[2]), SAFE_TRAVEL_Z_M)])
        waypoints.append((travel, travel_duration_sec))
        if target[2] < SAFE_TRAVEL_Z_M:
            waypoints.append((target, 1.0))
    else:
        waypoints.append((target, travel_duration_sec))

    segments: list[MotionSegment] = []
    seed = current
    for xyz, duration in waypoints:
        solved = ik.solve_ik(
            target_pos=xyz,
            initial_qpos=seed,
            approach=approach,
            null_weight=1e-3,
        )
        solved = np.asarray(solved, dtype=np.float64).copy()
        if len(solved) >= 6:
            solved[5] = gripper
        segments.append(
            MotionSegment(
                qpos_rad=solved,
                duration_sec=float(duration),
                site_xyz_m=(float(xyz[0]), float(xyz[1]), float(xyz[2])),
            )
        )
        seed = solved
    return segments


def run_segments(robot: RobotInterface, segments: list[MotionSegment]) -> None:
    """Execute segments in order, aborting if the robot is interrupted between them.

    The robot's ``interrupt_epoch`` is captured once and handed to every ``move_to_pose`` call,
    so an interrupt that arrives between two segments cancels the rest instead of being cleared
    by the next ``move_to_pose``.

    Args:
        robot: Robot controller.
        segments: Segments to run.

    Raises:
        HardwareError: If the robot is interrupted; ``robot_moved`` tells whether any earlier
            segment already ran.
    """
    epoch = getattr(robot, "interrupt_epoch", None)
    for i, segment in enumerate(segments):
        try:
            if epoch is None:
                robot.move_to_pose(segment.qpos_rad, duration_sec=segment.duration_sec)
            else:
                robot.move_to_pose(
                    segment.qpos_rad, duration_sec=segment.duration_sec, cancel_epoch=epoch
                )
        except HardwareError as e:
            if i > 0:
                e.robot_moved = True
            raise
