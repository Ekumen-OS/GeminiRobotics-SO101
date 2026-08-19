"""Unit tests for quintic minimum-jerk trajectory planner and phased motion generation."""

import numpy as np

from robot.cartesian_ik import CartesianIK
from robot.constants import PRESET_POSES
from robot.interface import RobotInterface
from robot.trajectory import (
    TrajectoryPlanner,
    quintic_minimum_jerk,
    quintic_minimum_jerk_derivative,
    quintic_minimum_jerk_second_derivative,
)


def test_quintic_boundary_conditions() -> None:
    """Test quintic polynomial satisfies C2 boundary conditions at t=0 and t=T."""
    # Values at endpoints
    assert np.isclose(quintic_minimum_jerk(0.0), 0.0)
    assert np.isclose(quintic_minimum_jerk(1.0), 1.0)

    # First derivative (velocity) at endpoints must be zero
    duration = 2.0
    assert np.isclose(quintic_minimum_jerk_derivative(0.0, duration), 0.0)
    assert np.isclose(quintic_minimum_jerk_derivative(1.0, duration), 0.0)

    # Second derivative (acceleration) at endpoints must be zero
    assert np.isclose(quintic_minimum_jerk_second_derivative(0.0, duration), 0.0)
    assert np.isclose(quintic_minimum_jerk_second_derivative(1.0, duration), 0.0)

    # Finite-difference verification for velocity
    eps = 1e-6
    v_start_fd = (quintic_minimum_jerk(eps) - quintic_minimum_jerk(0.0)) / eps
    v_end_fd = (quintic_minimum_jerk(1.0) - quintic_minimum_jerk(1.0 - eps)) / eps
    assert np.isclose(v_start_fd, 0.0, atol=1e-4)
    assert np.isclose(v_end_fd, 0.0, atol=1e-4)


def test_quintic_monotonicity() -> None:
    """Test quintic polynomial is strictly non-decreasing over tau in [0, 1]."""
    taus = np.linspace(0.0, 1.0, 500)
    s_vals = [quintic_minimum_jerk(t) for t in taus]
    diffs = np.diff(s_vals)
    assert np.all(diffs >= -1e-8)


def test_joint_trajectory_interpolation() -> None:
    """Test joint trajectory planning preserves boundary values and generates correct shape."""
    planner = TrajectoryPlanner()
    q_start = np.array([0.0, -1.0, 0.5, 0.5, 0.0, -0.5], dtype=np.float32)
    q_goal = np.array([0.5, 0.0, 0.0, -0.5, 0.2, 0.5], dtype=np.float32)

    duration = 1.0
    dt = 0.02
    waypoints = planner.plan_joint_trajectory(q_start, q_goal, duration_sec=duration, dt=dt)

    expected_steps = int(round(duration / dt))
    assert waypoints.shape == (expected_steps, 6)
    assert np.allclose(waypoints[0], q_start)
    assert np.allclose(waypoints[-1], q_goal)

    # Check that joint velocities start and end at zero
    vel = np.diff(waypoints, axis=0) / dt
    assert np.allclose(vel[0], np.zeros(6), atol=0.05)
    assert np.allclose(vel[-1], np.zeros(6), atol=0.05)


def test_cartesian_linear_trajectory() -> None:
    """Test Cartesian linear path generation with IK convergence."""
    ik = CartesianIK()
    planner = TrajectoryPlanner(ik_solver=ik)

    p_start = np.array([0.20, 0.05, 0.10])
    p_goal = np.array([0.25, -0.05, 0.05])

    waypoints = planner.plan_cartesian_linear(
        p_start=p_start,
        p_goal=p_goal,
        duration_sec=0.5,
        dt=0.05,
        gripper_val=0.5,
    )

    assert waypoints.shape[1] == 6
    assert waypoints.shape[0] >= 10
    # Gripper angle is fixed throughout Cartesian translation
    assert np.allclose(waypoints[:, 5], 0.5)


def test_pick_and_place_phased_structure() -> None:
    """Test phased pick-and-place generator creates complete 6-phase sequence."""
    planner = TrajectoryPlanner()
    pick_xyz = np.array([0.22, 0.08, 0.02])
    place_xyz = np.array([0.18, -0.12, 0.02])

    phases = planner.generate_pick_and_place(
        pick_xyz=pick_xyz,
        place_xyz=place_xyz,
        hover_height_m=0.06,
        grasp_z_m=0.005,
        current_qpos=PRESET_POSES["HOME"],
        transit_duration=0.5,
        descend_duration=0.4,
        retract_duration=0.4,
    )

    assert len(phases) == 6
    names = [p.name for p in phases]
    assert "A_transit_to_hover" in names
    assert "B_descend" in names
    assert "C_grasp" in names
    assert "D_retract" in names
    assert "E_transit_to_place" in names
    assert "F_place_and_release" in names

    # Check waypoints exist in every phase
    for p in phases:
        assert p.waypoints.shape[0] >= 2
        assert p.waypoints.shape[1] == 6


class MockRobot(RobotInterface):
    """Mock robot for testing trajectory execution."""

    def __init__(self) -> None:
        """Initialize mock robot."""
        self.positions: list[np.ndarray] = []
        self._curr = np.zeros(6, dtype=np.float32)

    def set_joint_positions(self, qpos: np.ndarray) -> None:
        """Set mock joint positions."""
        self._curr = np.asarray(qpos, dtype=np.float32)
        self.positions.append(self._curr.copy())

    def get_joint_positions(self) -> np.ndarray:
        """Get mock joint positions."""
        return self._curr.copy()


def test_trajectory_execution_and_interruption() -> None:
    """Test execution engine runs all waypoints and respects stop flag."""
    planner = TrajectoryPlanner()
    robot = MockRobot()

    q_start = np.zeros(6, dtype=np.float32)
    q_goal = np.ones(6, dtype=np.float32)
    w_a = planner.plan_joint_trajectory(q_start, q_goal, duration_sec=0.1, dt=0.02)
    from robot.trajectory import TrajectoryPhase

    phases = [TrajectoryPhase(name="P1", waypoints=w_a, phase_type="joint", duration_sec=0.1)]

    # Successful execution
    ok = planner.execute_trajectory(robot, phases, dt=0.001)
    assert ok is True
    assert len(robot.positions) == len(w_a)
    assert np.allclose(robot.get_joint_positions(), q_goal)

    # Interrupted execution
    robot_stopped = MockRobot()
    stop_count = 0

    def should_stop() -> bool:
        nonlocal stop_count
        stop_count += 1
        return stop_count >= 3

    ok_stopped = planner.execute_trajectory(robot_stopped, phases, stop_flag=should_stop, dt=0.001)
    assert ok_stopped is False
    assert len(robot_stopped.positions) < len(w_a)
