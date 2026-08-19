"""Trajectory planner and quintic minimum-jerk motion interpolation for SO-101 robot arm.

Implements C2-continuous quintic polynomial interpolation in joint space and Cartesian
task space to eliminate servo acceleration spikes on Feetech STS3215 actuators, and
orchestrates phased pick-and-place trajectories.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from robot.cartesian_ik import CartesianIK
from robot.constants import JOINT_LIMITS, PRESET_POSES
from robot.interface import RobotInterface

# Gripper default positions in radians
GRIPPER_OPEN_RAD = np.radians(JOINT_LIMITS["gripper"][1])  # +75 deg
GRIPPER_CLOSE_RAD = np.radians(JOINT_LIMITS["gripper"][0])  # -75 deg


def quintic_minimum_jerk(tau: float) -> float:
    """Evaluate normalized quintic minimum-jerk polynomial at tau in [0, 1].

    Formula: s(tau) = 10*tau^3 - 15*tau^4 + 6*tau^5
    Boundary conditions: s(0)=0, s(1)=1, s'(0)=s'(1)=0, s''(0)=s''(1)=0 (C2 continuous).

    Args:
        tau: Normalized time parameter in range [0.0, 1.0].

    Returns:
        Interpolation scaling factor in range [0.0, 1.0].
    """
    tau = max(0.0, min(1.0, float(tau)))
    return 10.0 * (tau**3) - 15.0 * (tau**4) + 6.0 * (tau**5)


def quintic_minimum_jerk_derivative(tau: float, duration: float) -> float:
    """Evaluate velocity derivative of quintic polynomial ds/dt at normalized time tau.

    Args:
        tau: Normalized time parameter in range [0.0, 1.0].
        duration: Total duration T of the motion in seconds.

    Returns:
        First derivative value ds/dt.
    """
    if duration <= 0.0:
        return 0.0
    tau = max(0.0, min(1.0, float(tau)))
    return (30.0 * (tau**2) - 60.0 * (tau**3) + 30.0 * (tau**4)) / duration


def quintic_minimum_jerk_second_derivative(tau: float, duration: float) -> float:
    """Evaluate acceleration derivative of quintic polynomial d2s/dt2 at normalized time tau.

    Args:
        tau: Normalized time parameter in range [0.0, 1.0].
        duration: Total duration T of the motion in seconds.

    Returns:
        Second derivative value d2s/dt2.
    """
    if duration <= 0.0:
        return 0.0
    tau = max(0.0, min(1.0, float(tau)))
    return (60.0 * tau - 180.0 * (tau**2) + 120.0 * (tau**3)) / (duration**2)


@dataclass
class TrajectoryPhase:
    """Single discrete phase in a multi-step trajectory.

    Attributes:
        name: Identifier name for this phase (e.g. 'A_transit_to_hover').
        waypoints: Array of shape (N, 6) containing 6-DOF joint angles in radians.
        phase_type: Type of motion ('joint' or 'cartesian').
        duration_sec: Expected total duration of the phase in seconds.
    """

    name: str
    waypoints: np.ndarray
    phase_type: str
    duration_sec: float


class TrajectoryPlanner:
    """Generates and executes minimum-jerk joint and Cartesian trajectories for the SO-101 arm."""

    def __init__(
        self,
        ik_solver: CartesianIK | None = None,
        robot: RobotInterface | None = None,
    ) -> None:
        """Initialize trajectory planner.

        Args:
            ik_solver: Optional CartesianIK instance. Lazily creates one if None.
            robot: Optional RobotInterface instance.
        """
        self._ik_solver = ik_solver
        self.robot = robot

    @property
    def ik_solver(self) -> CartesianIK:
        """Return the CartesianIK solver instance, lazily initializing if needed."""
        if self._ik_solver is None:
            self._ik_solver = CartesianIK()
        return self._ik_solver

    def plan_joint_trajectory(
        self,
        q_start: np.ndarray,
        q_goal: np.ndarray,
        duration_sec: float = 1.0,
        dt: float = 0.02,
    ) -> np.ndarray:
        """Generate C2-continuous quintic minimum-jerk joint trajectory.

        Args:
            q_start: Starting joint configuration array of shape (6,).
            q_goal: Target goal joint configuration array of shape (6,).
            duration_sec: Duration of the motion trajectory in seconds.
            dt: Timestep spacing between waypoints in seconds.

        Returns:
            NumPy array of shape (N, 6) with interpolated joint angles in radians.
        """
        q_start = np.asarray(q_start, dtype=np.float32)
        q_goal = np.asarray(q_goal, dtype=np.float32)
        num_steps = max(2, int(round(duration_sec / dt)))

        waypoints = np.zeros((num_steps, len(q_start)), dtype=np.float32)
        for i in range(num_steps):
            tau = i / (num_steps - 1)
            s = quintic_minimum_jerk(tau)
            waypoints[i] = (1.0 - s) * q_start + s * q_goal

        return waypoints

    def plan_cartesian_linear(
        self,
        p_start: np.ndarray,
        p_goal: np.ndarray,
        duration_sec: float = 1.0,
        dt: float = 0.02,
        initial_qpos: np.ndarray | None = None,
        nominal_qpos: np.ndarray | None = None,
        gripper_val: float | None = None,
    ) -> np.ndarray:
        """Generate straight-line Cartesian path with minimum-jerk profile and IK tracking.

        Args:
            p_start: Starting 3D Cartesian position [X, Y, Z] in meters.
            p_goal: Target 3D Cartesian position [X, Y, Z] in meters.
            duration_sec: Duration of Cartesian sweep in seconds.
            dt: Timestep spacing in seconds.
            initial_qpos: Initial joint angle seed for Inverse Kinematics.
            nominal_qpos: Rest posture for null-space centering.
            gripper_val: Fixed gripper angle in radians to maintain throughout the path.

        Returns:
            NumPy array of shape (N, 6) containing joint angle waypoints.
        """
        p_start = np.asarray(p_start, dtype=np.float64)
        p_goal = np.asarray(p_goal, dtype=np.float64)
        num_steps = max(2, int(round(duration_sec / dt)))

        if initial_qpos is None:
            if self.robot is not None:
                initial_qpos = self.robot.get_joint_positions()
            else:
                initial_qpos = PRESET_POSES["HOME"].copy()

        curr_q = np.asarray(initial_qpos, dtype=np.float32).copy()
        if nominal_qpos is None:
            nominal_qpos = curr_q.copy()

        fixed_gripper = gripper_val if gripper_val is not None else curr_q[5]

        waypoints = np.zeros((num_steps, 6), dtype=np.float32)
        for i in range(num_steps):
            tau = i / (num_steps - 1)
            s = quintic_minimum_jerk(tau)
            target_pt = (1.0 - s) * p_start + s * p_goal

            solved_q = self.ik_solver.solve_ik(
                target_pos=target_pt,
                initial_qpos=curr_q,
                nominal_qpos=nominal_qpos,
                preserve_orientation=False,
                max_iters=150,
                tol=1e-4,
            )
            solved_q[5] = fixed_gripper
            waypoints[i] = solved_q
            curr_q = solved_q.copy()

        return waypoints

    def generate_pick_and_place(
        self,
        pick_xyz: np.ndarray,
        place_xyz: np.ndarray | None = None,
        hover_height_m: float = 0.05,
        grasp_z_m: float = 0.005,
        current_qpos: np.ndarray | None = None,
        gripper_open_rad: float = GRIPPER_OPEN_RAD,
        gripper_close_rad: float = GRIPPER_CLOSE_RAD,
        transit_duration: float = 1.5,
        descend_duration: float = 1.0,
        retract_duration: float = 1.0,
        dt: float = 0.02,
    ) -> list[TrajectoryPhase]:
        """Orchestrate discrete phased pick-and-place trajectories.

        Phases:
            A: Joint-space quintic transit to pre-grasp hover pose (Z + hover_height_m).
            B: Linear Cartesian descent to grasp height (grasp_z_m).
            C: Gripper close actuation.
            D: Linear Cartesian vertical retract back to hover height.
            E (optional): Joint-space transit to destination hover pose.
            F (optional): Linear Cartesian descent to place height, gripper open, retract.

        Args:
            pick_xyz: Target pick 3D coordinates [X, Y, Z] in meters.
            place_xyz: Optional target place 3D coordinates [X, Y, Z] in meters.
            hover_height_m: Vertical clearance offset above grasp point in meters.
            grasp_z_m: Table surface grasp height in meters.
            current_qpos: Current 6-DOF joint angles array.
            gripper_open_rad: Gripper open angle in radians.
            gripper_close_rad: Gripper close angle in radians.
            transit_duration: Transit duration in seconds.
            descend_duration: Descent duration in seconds.
            retract_duration: Retract duration in seconds.
            dt: Timestep duration in seconds.

        Returns:
            List of TrajectoryPhase instances.
        """
        if current_qpos is None:
            if self.robot is not None:
                current_qpos = self.robot.get_joint_positions()
            else:
                current_qpos = PRESET_POSES["HOME"].copy()

        current_qpos = np.asarray(current_qpos, dtype=np.float32).copy()
        phases: list[TrajectoryPhase] = []

        pick_hover_xyz = pick_xyz.copy()
        pick_hover_xyz[2] = pick_xyz[2] + hover_height_m

        pick_grasp_xyz = pick_xyz.copy()
        pick_grasp_xyz[2] = grasp_z_m

        # Phase A: Transit to pre-grasp hover pose with gripper open
        q_hover_seed = current_qpos.copy()
        q_hover_seed[5] = gripper_open_rad
        q_hover = self.ik_solver.solve_ik(
            target_pos=pick_hover_xyz,
            initial_qpos=q_hover_seed,
            null_weight=1e-3,
        )
        q_hover[5] = gripper_open_rad

        w_a = self.plan_joint_trajectory(
            current_qpos, q_hover, duration_sec=transit_duration, dt=dt
        )
        phases.append(
            TrajectoryPhase(
                name="A_transit_to_hover",
                waypoints=w_a,
                phase_type="joint",
                duration_sec=transit_duration,
            )
        )

        # Phase B: Linear Cartesian descent to target grasp height
        w_b = self.plan_cartesian_linear(
            p_start=pick_hover_xyz,
            p_goal=pick_grasp_xyz,
            duration_sec=descend_duration,
            dt=dt,
            initial_qpos=q_hover,
            gripper_val=gripper_open_rad,
        )
        phases.append(
            TrajectoryPhase(
                name="B_descend",
                waypoints=w_b,
                phase_type="cartesian",
                duration_sec=descend_duration,
            )
        )

        # Phase C: Gripper close actuation (hold arm joints stationary, interpolate gripper)
        q_grasp_open = w_b[-1].copy()
        q_grasp_closed = q_grasp_open.copy()
        q_grasp_closed[5] = gripper_close_rad
        w_c = self.plan_joint_trajectory(q_grasp_open, q_grasp_closed, duration_sec=0.6, dt=dt)
        phases.append(
            TrajectoryPhase(
                name="C_grasp",
                waypoints=w_c,
                phase_type="joint",
                duration_sec=0.6,
            )
        )

        # Phase D: Linear Cartesian vertical retract back to hover height with gripper closed
        w_d = self.plan_cartesian_linear(
            p_start=pick_grasp_xyz,
            p_goal=pick_hover_xyz,
            duration_sec=retract_duration,
            dt=dt,
            initial_qpos=q_grasp_closed,
            gripper_val=gripper_close_rad,
        )
        phases.append(
            TrajectoryPhase(
                name="D_retract",
                waypoints=w_d,
                phase_type="cartesian",
                duration_sec=retract_duration,
            )
        )

        # Optional Place phases (E & F)
        if place_xyz is not None:
            place_hover_xyz = place_xyz.copy()
            place_hover_xyz[2] = place_xyz[2] + hover_height_m

            place_grasp_xyz = place_xyz.copy()
            place_grasp_xyz[2] = grasp_z_m

            q_retract_end = w_d[-1].copy()
            q_place_hover = self.ik_solver.solve_ik(
                target_pos=place_hover_xyz,
                initial_qpos=q_retract_end,
                null_weight=1e-3,
            )
            q_place_hover[5] = gripper_close_rad

            # Phase E: Transit to destination hover pose
            w_e = self.plan_joint_trajectory(
                q_retract_end, q_place_hover, duration_sec=transit_duration, dt=dt
            )
            phases.append(
                TrajectoryPhase(
                    name="E_transit_to_place",
                    waypoints=w_e,
                    phase_type="joint",
                    duration_sec=transit_duration,
                )
            )

            # Phase F1: Linear descent to place height
            w_f1 = self.plan_cartesian_linear(
                p_start=place_hover_xyz,
                p_goal=place_grasp_xyz,
                duration_sec=descend_duration,
                dt=dt,
                initial_qpos=q_place_hover,
                gripper_val=gripper_close_rad,
            )

            # Phase F2: Open gripper to release object
            q_place_closed = w_f1[-1].copy()
            q_place_opened = q_place_closed.copy()
            q_place_opened[5] = gripper_open_rad
            w_f2 = self.plan_joint_trajectory(
                q_place_closed, q_place_opened, duration_sec=0.5, dt=dt
            )

            # Phase F3: Retract back to place hover height
            w_f3 = self.plan_cartesian_linear(
                p_start=place_grasp_xyz,
                p_goal=place_hover_xyz,
                duration_sec=retract_duration,
                dt=dt,
                initial_qpos=q_place_opened,
                gripper_val=gripper_open_rad,
            )

            w_f = np.vstack([w_f1, w_f2, w_f3])
            total_f_sec = descend_duration + 0.5 + retract_duration
            phases.append(
                TrajectoryPhase(
                    name="F_place_and_release",
                    waypoints=w_f,
                    phase_type="cartesian",
                    duration_sec=total_f_sec,
                )
            )

        return phases

    def execute_trajectory(
        self,
        robot: RobotInterface,
        phases: list[TrajectoryPhase],
        stop_flag: Callable[[], bool] | None = None,
        dt: float = 0.02,
        phase_callback: Callable[[str, int, int], None] | None = None,
    ) -> bool:
        """Execute a sequence of trajectory phases sequentially on robot hardware/simulation.

        Args:
            robot: Target RobotInterface instance.
            phases: List of TrajectoryPhase objects to execute.
            stop_flag: Optional callable returning True if execution was interrupted.
            dt: Timestep duration to sleep between waypoints in seconds.
            phase_callback: Optional callable(phase_name, phase_idx, total_phases).

        Returns:
            True if all phases completed successfully, False if interrupted.
        """
        for p_idx, phase in enumerate(phases):
            if stop_flag is not None and stop_flag():
                return False

            if phase_callback is not None:
                phase_callback(phase.name, p_idx + 1, len(phases))

            for wp in phase.waypoints:
                if stop_flag is not None and stop_flag():
                    return False

                robot.set_joint_positions(wp)
                if hasattr(robot, "step"):
                    robot.step(1)

                time.sleep(dt)

        return True
