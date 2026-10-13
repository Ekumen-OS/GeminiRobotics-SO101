"""Unit and boundary tests for the Cartesian Inverse Kinematics (IK) solver."""

import mujoco
import numpy as np
import pytest

from agent.tools import validate_and_clamp_joints
from robot.cartesian_ik import CartesianIK
from robot.constants import JOINT_LIMITS, PRESET_POSES


@pytest.fixture
def ik_solver() -> CartesianIK:
    """Provide a fresh CartesianIK solver instance.

    Returns:
        CartesianIK solver instance initialized with the SO-101 scene model.
    """
    return CartesianIK()


def test_cartesian_ik_initialization(ik_solver: CartesianIK) -> None:
    """Verify CartesianIK initializes with valid MuJoCo model, data, and joint IDs."""
    assert ik_solver.model is not None
    assert ik_solver.data is not None
    assert len(ik_solver.joint_ids) == 5  # 5 arm joints
    assert len(ik_solver.all_joint_ids) == 6  # 5 arm joints + gripper
    assert ik_solver.site_id >= 0


def test_forward_kinematics_site_position(ik_solver: CartesianIK) -> None:
    """Verify forward kinematics computation for standard preset poses."""
    home_rad = np.radians(PRESET_POSES["HOME"])
    for i, j_id in enumerate(ik_solver.all_joint_ids):
        ik_solver.data.qpos[ik_solver.model.jnt_qposadr[j_id]] = home_rad[i]
    mujoco.mj_forward(ik_solver.model, ik_solver.data)

    ee_pos = ik_solver.data.site_xpos[ik_solver.site_id].copy()
    # SO-101 HOME pose end effector should be in front of robot and above table
    assert 0.10 < ee_pos[0] < 0.45, f"Unexpected X for HOME pose: {ee_pos[0]}"
    assert abs(ee_pos[1]) < 0.05, f"Unexpected lateral Y for HOME pose: {ee_pos[1]}"
    assert 0.05 < ee_pos[2] < 0.30, f"Unexpected Z for HOME pose: {ee_pos[2]}"


def test_solve_ik_workspace_nominal_convergence(ik_solver: CartesianIK) -> None:
    """Verify that a reachable workspace target converges with < 10 mm error."""
    target_xyz = np.array([0.22, 0.0, 0.05], dtype=np.float64)
    solution_qpos = ik_solver.solve_ik(
        target_pos=target_xyz,
        initial_qpos=np.radians(PRESET_POSES["HOME"]),
        approach="top_down",
    )

    assert len(solution_qpos) == 6

    # Forward kinematics verification of achieved position
    for i, j_id in enumerate(ik_solver.all_joint_ids):
        ik_solver.data.qpos[ik_solver.model.jnt_qposadr[j_id]] = solution_qpos[i]
    mujoco.mj_forward(ik_solver.model, ik_solver.data)
    achieved_pos = ik_solver.data.site_xpos[ik_solver.site_id].copy()

    pos_error_m = np.linalg.norm(target_xyz - achieved_pos)
    assert pos_error_m < 0.010, f"Positional error too high: {pos_error_m * 1000:.2f} mm"


@pytest.mark.parametrize("approach", ["top_down", "horizontal"])
def test_solve_ik_approach_modes(ik_solver: CartesianIK, approach: str) -> None:
    """Verify that both top_down and horizontal approach modes solve successfully."""
    target_xyz = np.array([0.20, 0.05, 0.06], dtype=np.float64)
    solution = ik_solver.solve_ik(
        target_pos=target_xyz,
        initial_qpos=np.radians(PRESET_POSES["HOME"]),
        approach=approach,
    )
    assert len(solution) == 6
    assert np.all(np.isfinite(solution))


def test_singularity_and_hostile_seed_recovery(ik_solver: CartesianIK) -> None:
    """Verify solver recovers when seeded from extreme or hostile configurations."""
    target_xyz = np.array([0.20, -0.05, 0.04], dtype=np.float64)

    # Hostile seed: all zeros (singularity pose where arm is stretched flat)
    zero_seed = np.zeros(6, dtype=np.float32)
    solution_zeros = ik_solver.solve_ik(
        target_pos=target_xyz,
        initial_qpos=zero_seed,
        approach="top_down",
    )
    assert np.all(np.isfinite(solution_zeros))

    # Check achieved position
    for i, j_id in enumerate(ik_solver.all_joint_ids):
        ik_solver.data.qpos[ik_solver.model.jnt_qposadr[j_id]] = solution_zeros[i]
    mujoco.mj_forward(ik_solver.model, ik_solver.data)
    achieved = ik_solver.data.site_xpos[ik_solver.site_id].copy()
    err_m = np.linalg.norm(target_xyz - achieved)
    assert err_m < 0.015, f"Recovery from zero seed failed with error {err_m * 1000:.2f} mm"


def test_validate_and_clamp_joints_safety() -> None:
    """Verify joint limits validation and defensive clamping."""
    # Joint within valid range remains untouched
    valid_qpos = [0.0, 10.0, 45.0, 60.0, 0.0, 0.0]
    clamped, warnings = validate_and_clamp_joints(valid_qpos)
    assert clamped == valid_qpos
    assert len(warnings) == 0

    # Wildly out-of-range joints are clamped strictly to limits
    out_of_range_qpos = [500.0, -500.0, 1000.0, -1000.0, 360.0, -200.0]
    clamped, warnings = validate_and_clamp_joints(out_of_range_qpos)
    assert len(clamped) == 6
    assert len(warnings) > 0

    for val, (min_l, max_l) in zip(clamped, JOINT_LIMITS.values(), strict=True):
        assert min_l <= val <= max_l, f"Value {val} outside [{min_l}, {max_l}]"


def test_workspace_envelope_reachability_sample(ik_solver: CartesianIK) -> None:
    """Sample reachable workspace points and verify >= 95% convergence."""
    x_coords = np.linspace(0.18, 0.30, 5)
    y_coords = np.linspace(-0.12, 0.12, 5)
    z_m = 0.04

    home_seed = np.radians(PRESET_POSES["HOME"])
    success_count = 0
    total_count = len(x_coords) * len(y_coords)

    for x in x_coords:
        for y in y_coords:
            target = np.array([x, y, z_m], dtype=np.float64)
            sol = ik_solver.solve_ik(
                target_pos=target,
                initial_qpos=home_seed,
                approach="top_down",
            )
            for i, j_id in enumerate(ik_solver.all_joint_ids):
                ik_solver.data.qpos[ik_solver.model.jnt_qposadr[j_id]] = sol[i]
            mujoco.mj_forward(ik_solver.model, ik_solver.data)
            achieved = ik_solver.data.site_xpos[ik_solver.site_id].copy()
            if np.linalg.norm(target - achieved) < 0.015:
                success_count += 1

    success_rate = success_count / total_count
    assert success_rate >= 0.95, f"Expected >= 95% reachability, got {success_rate:.1%}"


def test_out_of_reach_target_graceful_handling(ik_solver: CartesianIK) -> None:
    """Verify that targeting an unreachable point does not crash or raise exceptions."""
    unreachable_xyz = np.array([1.50, 0.0, 0.50], dtype=np.float64)
    solution = ik_solver.solve_ik(
        target_pos=unreachable_xyz,
        initial_qpos=np.radians(PRESET_POSES["HOME"]),
        approach="top_down",
    )
    assert len(solution) == 6
    assert np.all(np.isfinite(solution))
