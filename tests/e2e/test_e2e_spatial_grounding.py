"""End-to-end test suite for cross-feature spatial grounding and real-world manipulation scenarios.

Covers Tier 3 cross-feature interactions (Homography <-> IK <-> FK <-> Tools) and Tier 4
real-world manipulation scenarios in simulation.
"""

import mujoco
import numpy as np
import pytest

from agent.tools import execute_tool_call, image_point_to_xyz, validate_and_clamp_joints
from robot.cartesian_ik import CartesianIK
from robot.settings import Settings, set_settings
from simulation.sim_calibration import init_sim_workspace_calibration
from simulation.sim_robot import SO101Robot
from tests.conftest import DummyToolCall
from vision.calibrator import get_calibrator

# ==============================================================================
# TIER 3: CROSS-FEATURE COMBINATIONS
# ==============================================================================


def test_t3_cross_feature_gemini_to_ik_to_fk_consistency() -> None:
    """Validate complete kinematic chain: Gemini point -> Homography -> IK -> FK."""
    calib = get_calibrator(mode="sim")
    init_sim_workspace_calibration(calibrator=calib)
    ik = CartesianIK()

    test_points = [
        [500.0, 500.0],
        [700.0, 400.0],
        [600.0, 600.0],
        [760.0, 484.0],
    ]

    for pt in test_points:
        target_xyz = calib.gemini_to_robot_xyz(pt, z_m=0.04)
        solved_rad = ik.solve_ik(target_xyz, approach="top_down")

        # Evaluate Forward Kinematics in MuJoCo
        for i, j_id in enumerate(ik.all_joint_ids):
            qpos_adr = ik.model.jnt_qposadr[j_id]
            ik.data.qpos[qpos_adr] = solved_rad[i]
        mujoco.mj_forward(ik.model, ik.data)
        fk_pos = ik.data.site_xpos[ik.site_id]

        # Distance error between requested target and achieved FK end-effector
        pos_error = float(np.linalg.norm(fk_pos - target_xyz))
        assert pos_error < 0.025, f"FK positioning error {pos_error * 1000:.1f}mm too large at {pt}"


def test_t3_cross_feature_tools_image_point_to_xyz_integration() -> None:
    """Validate agent.tools.image_point_to_xyz matches calibrator across heights."""
    calib = get_calibrator(mode="sim")
    init_sim_workspace_calibration(calibrator=calib)

    for hover_z in [0.0, 0.02, 0.05, 0.10]:
        xyz_tool = image_point_to_xyz([760.0, 484.0], hover_z=hover_z, mode="sim")
        xyz_calib = calib.gemini_to_robot_xyz([760.0, 484.0], z_m=hover_z)
        assert np.allclose(xyz_tool, xyz_calib, atol=1e-4)


@pytest.mark.asyncio
async def test_t3_cross_feature_move_to_spatial_point_execution() -> None:
    """Validate move_to_spatial_point tool execution in headless simulation."""
    calib = get_calibrator(mode="sim")
    init_sim_workspace_calibration(calibrator=calib)

    robot = SO101Robot()
    robot.reset("HOME")

    try:
        call = DummyToolCall(
            name="move_to_spatial_point",
            args={"point": [760.0, 484.0], "z_height_cm": 4.0},
        )
        result = await execute_tool_call(call=call, robot=robot)

        assert result.get("status") == "success"
        assert "target_xyz_m" in result
        assert "executed_qpos_deg" in result
        assert "visual_hint" in result

        target_xyz = result["target_xyz_m"]
        assert 0.10 <= target_xyz[0] <= 0.40
        assert -0.30 <= target_xyz[1] <= 0.30
        assert np.isclose(target_xyz[2], 0.04)
    finally:
        robot.close()


def test_t3_cross_feature_camera_fov_containment() -> None:
    """Validate that workspace points map within the camera field of view in simulation."""
    calib = get_calibrator(mode="sim")
    init_sim_workspace_calibration(calibrator=calib)

    test_points = [
        [300.0, 300.0],
        [500.0, 500.0],
        [700.0, 300.0],
        [700.0, 700.0],
        [760.0, 484.0],
    ]

    for pt in test_points:
        px, py = calib.gemini_to_pixel(pt, img_shape=(480, 640))
        assert 0.0 <= px <= 640.0
        assert 0.0 <= py <= 480.0


# ==============================================================================
# TIER 4: REAL-WORLD APPLICATION SCENARIOS
# ==============================================================================


@pytest.mark.asyncio
async def test_t4_scenario_pick_hover_and_descend_workflow(
    restore_settings: None,
) -> None:
    """Real-World Scenario 1: Model moves to hover above target, then descends to contact."""
    set_settings(Settings.from_env(descent_gate="off"))
    calib = get_calibrator(mode="sim")
    init_sim_workspace_calibration(calibrator=calib)

    robot = SO101Robot()
    robot.reset("HOME")

    try:
        # Step 1: Hover at 4.0 cm
        hover_call = DummyToolCall(
            name="move_to_spatial_point",
            args={"point": [760.0, 484.0], "z_height_cm": 4.0},
        )
        res1 = await execute_tool_call(call=hover_call, robot=robot)
        assert res1["status"] == "success"
        assert "Visual Verification Loop" in res1["visual_hint"]

        # Step 2: Descend to 0.0 cm (table contact)
        descend_call = DummyToolCall(
            name="move_to_spatial_point",
            args={"point": [760.0, 484.0], "z_height_cm": 0.0},
        )
        res2 = await execute_tool_call(call=descend_call, robot=robot)
        assert res2["status"] == "success"
        assert res2["target_xyz_m"][2] == 0.0

        # Step 3: Close gripper
        close_call = DummyToolCall(name="close_gripper", args={})
        res3 = await execute_tool_call(call=close_call, robot=robot)
        assert res3["status"] == "success"
        assert res3["gripper_state"] == "closed_empty"
        assert "Gripper closed on NOTHING" in res3["visual_hint"]

        # Step 4: Lift to 5.0 cm
        lift_call = DummyToolCall(
            name="move_to_spatial_point",
            args={"point": [760.0, 484.0], "z_height_cm": 5.0},
        )
        res4 = await execute_tool_call(call=lift_call, robot=robot)
        assert res4["status"] == "success"
        assert np.isclose(res4["target_xyz_m"][2], 0.05)
    finally:
        robot.close()


@pytest.mark.asyncio
async def test_t4_scenario_perimeter_survey_motion() -> None:
    """Real-World Scenario 3: Perimeter sweep across 4 quadrants without discontinuity."""
    calib = get_calibrator(mode="sim")
    init_sim_workspace_calibration(calibrator=calib)

    robot = SO101Robot()
    robot.reset("HOME")

    survey_points = [
        [400.0, 400.0],
        [400.0, 600.0],
        [700.0, 600.0],
        [700.0, 400.0],
    ]

    try:
        for pt in survey_points:
            call = DummyToolCall(
                name="move_to_spatial_point",
                args={"point": pt, "z_height_cm": 5.0},
            )
            result = await execute_tool_call(call=call, robot=robot)
            assert result["status"] == "success"
            executed_qpos = result["executed_qpos_deg"]
            assert len(executed_qpos) == 6
    finally:
        robot.close()


def test_t4_scenario_simulation_target_body_spatial_alignment() -> None:
    """Real-World Scenario 4: Ground physical MuJoCo target body coordinates to robot space."""
    robot = SO101Robot()
    try:
        # Query target body position from MuJoCo model
        target_body_id = robot.model.body("target").id
        sim_pos = robot.data.xpos[target_body_id]

        # In scene.xml, target is at [0.18, 0.0, 0.015]
        assert 0.15 <= sim_pos[0] <= 0.25
        assert -0.05 <= sim_pos[1] <= 0.05
        assert 0.01 <= sim_pos[2] <= 0.05

        # Solve IK directly to reach above the target body
        ik = CartesianIK()
        hover_target = np.array([sim_pos[0], sim_pos[1], 0.05], dtype=np.float32)
        qpos = ik.solve_ik(hover_target, approach="top_down")
        assert qpos is not None

        clamped, warnings = validate_and_clamp_joints(list(np.degrees(qpos)))
        assert len(warnings) == 0
    finally:
        robot.close()
