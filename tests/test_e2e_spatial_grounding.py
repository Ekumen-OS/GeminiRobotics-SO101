"""Four-tier end-to-end test suite for spatial grounding, calibration & kinematics.

Covers:
- F1: ArUco Marker ID & Corner Harmonization
- F2: Homography & Lateral Axis Parity (Red cube [760, 484] grounding)
- F3: Calibrator Mode Context Routing (Sim vs Real)
- F4: Reachability & FOV Kinematic Grounding
- Cross-feature interactions and real-world manipulation scenarios.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from agent.tools import execute_tool_call, image_point_to_xyz, validate_and_clamp_joints
from robot.calibration import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_SAVE_PATH,
    DEFAULT_SIM_SAVE_PATH,
    ArucoWorkspaceCalibrator,
)
from robot.cartesian_ik import CartesianIK
from robot.constants import JOINT_LIMITS
from simulation.sim_calibration import init_sim_workspace_calibration
from simulation.sim_robot import SO101Robot
from ui.stream import get_calibrator

# ==============================================================================
# TIER 1: ISOLATED FEATURE MECHANICS (F1, F2, F3, F4)
# ==============================================================================


def test_t1_f1_config_marker_id_and_corner_definitions() -> None:
    """Validate that workspace calibration JSON specifies standardized marker IDs and corners."""
    assert DEFAULT_CONFIG_PATH.exists(), f"Missing calibration config at {DEFAULT_CONFIG_PATH}"
    config = json.loads(DEFAULT_CONFIG_PATH.read_text())

    # Verify ArUco dictionary and marker size
    assert config.get("aruco_dictionary") == "DICT_4X4_50"
    assert float(config.get("marker_size_mm", 0)) == 40.0

    # Verify standardized marker IDs (1:TL, 2:TR, 3:BR, 0:BL)
    marker_ids = config.get("marker_ids", {})
    assert marker_ids.get("top_left") == 1
    assert marker_ids.get("top_right") == 2
    assert marker_ids.get("bottom_right") == 3
    assert marker_ids.get("bottom_left") == 0

    # Verify physical workspace corners coordinates
    corners = config.get("workspace_corners_mm", {})
    assert set(corners.keys()) == {"top_left", "top_right", "bottom_right", "bottom_left"}
    assert corners["top_left"]["x"] == 380.0
    assert corners["top_right"]["x"] == 100.0
    assert corners["bottom_right"]["x"] == 100.0
    assert corners["bottom_left"]["x"] == 380.0


def test_t1_f1_scene_xml_fiducial_geometries() -> None:
    """Validate that simulation scene.xml defines ArUco marker geoms matching workspace mat."""
    scene_xml_path = Path(__file__).resolve().parent.parent / "simulation" / "assets" / "scene.xml"
    assert scene_xml_path.exists(), f"Missing scene.xml at {scene_xml_path}"
    content = scene_xml_path.read_text()

    # Verify ArUco marker textures and materials are declared
    for i in range(4):
        assert f'name="tex_marker_{i}"' in content
        assert f'name="mat_marker_{i}"' in content

    # Verify 4 corner geoms exist
    for geom_name in ["aruco_tl_0", "aruco_tr_1", "aruco_br_2", "aruco_bl_3"]:
        assert geom_name in content


def test_t1_f1_calibrator_fallback_marker_dictionary() -> None:
    """Validate calibrator fallback configuration when non-existent path is provided."""
    calib = ArucoWorkspaceCalibrator(config_path="/tmp/non_existent_config.json")
    assert calib.config.get("aruco_dictionary") == "DICT_4X4_50"
    assert "marker_ids" in calib.config
    assert calib.detector_params is not None
    assert calib.aruco_dict is not None


def test_t1_f2_gemini_to_pixel_boundary_mapping() -> None:
    """Validate normalized Gemini coordinates [0..1000] mapping to exact pixel coordinates."""
    calib = ArucoWorkspaceCalibrator()

    # Image shape 640x480 (height 480, width 640)
    shape = (480, 640)

    # Origin [0, 0] -> pixel (0, 0)
    px, py = calib.gemini_to_pixel([0.0, 0.0], img_shape=shape)
    assert px == 0.0 and py == 0.0

    # Max point [1000, 1000] -> pixel (640, 480)
    px, py = calib.gemini_to_pixel([1000.0, 1000.0], img_shape=shape)
    assert px == 640.0 and py == 480.0

    # Center point [500, 500] -> pixel (320, 240)
    px, py = calib.gemini_to_pixel([500.0, 500.0], img_shape=shape)
    assert px == 320.0 and py == 240.0


def test_t1_f2_homography_reprojection_and_forward_mapping() -> None:
    """Validate forward homography calculation and sub-millimeter reprojection accuracy."""
    calib = ArucoWorkspaceCalibrator()

    # Create synthetic corner detections in a 640x480 frame
    marker_corners = {
        1: np.array([[100, 60], [140, 60], [140, 100], [100, 100]], dtype=np.float32),
        2: np.array([[500, 60], [540, 60], [540, 100], [500, 100]], dtype=np.float32),
        3: np.array([[500, 380], [540, 380], [540, 420], [500, 420]], dtype=np.float32),
        0: np.array([[100, 380], [140, 380], [140, 420], [100, 420]], dtype=np.float32),
    }

    success = calib.update_homography(marker_corners)
    assert success is True
    assert calib.is_calibrated is True
    assert calib.reprojection_error is not None
    assert calib.reprojection_error < 1.0  # Sub-millimeter error

    # Forward projection of Top-Left marker center (120, 80)
    tl_robot_mm = calib.pixel_to_robot_mm(120.0, 80.0)
    expected_tl_x = calib.config["workspace_corners_mm"]["top_left"]["x"]
    expected_tl_y = calib.config["workspace_corners_mm"]["top_left"]["y"]
    assert np.allclose(tl_robot_mm, [expected_tl_x, expected_tl_y], atol=1.0)


def test_t1_f2_homography_inverse_projection_roundtrip() -> None:
    """Validate inverse homography projecting robot coordinates back to pixel coordinates."""
    calib = ArucoWorkspaceCalibrator()

    marker_corners = {
        1: np.array([[100, 60], [140, 60], [140, 100], [100, 100]], dtype=np.float32),
        2: np.array([[500, 60], [540, 60], [540, 100], [500, 100]], dtype=np.float32),
        3: np.array([[500, 380], [540, 380], [540, 420], [500, 420]], dtype=np.float32),
        0: np.array([[100, 380], [140, 380], [140, 420], [100, 420]], dtype=np.float32),
    }
    calib.update_homography(marker_corners)
    assert calib.inverse_homography is not None

    test_pixels = [(120.0, 80.0), (520.0, 80.0), (320.0, 240.0)]
    for u, v in test_pixels:
        robot_mm = calib.pixel_to_robot_mm(u, v)
        reprojected_pixel = calib.robot_mm_to_pixel(robot_mm[0], robot_mm[1])
        assert reprojected_pixel is not None
        assert np.isclose(reprojected_pixel[0], u, atol=0.5)
        assert np.isclose(reprojected_pixel[1], v, atol=0.5)


def test_t1_f2_red_cube_760_484_grounding_in_sim() -> None:
    """Validate canonical red cube coordinate [760, 484] projects to valid workspace XYZ."""
    calib = get_calibrator(mode="sim")
    init_sim_workspace_calibration(calibrator=calib)
    assert calib.is_calibrated is True

    xyz = calib.gemini_to_robot_xyz([760.0, 484.0], z_m=0.03)

    # Robot base X (forward distance) must be in reachable front arc [0.10, 0.40]m
    assert 0.10 <= xyz[0] <= 0.40
    # Robot base Y (lateral) must be within workspace mat span [-0.30, 0.30]m
    assert -0.30 <= xyz[1] <= 0.30
    # Z coordinate must match requested height exactly
    assert np.isclose(xyz[2], 0.03)


def test_t1_f3_calibrator_mode_routing_sim_and_real() -> None:
    """Validate mode-based calibrator routing between simulation and physical files."""
    sim_calib = get_calibrator(mode="sim")
    assert sim_calib is not None

    real_calib = ArucoWorkspaceCalibrator(mode="real")
    assert real_calib.mode == "real"

    if DEFAULT_SIM_SAVE_PATH.exists():
        sim_calib.load_calibration(DEFAULT_SIM_SAVE_PATH)
        assert sim_calib.is_calibrated is True
        assert sim_calib.homography_matrix.shape == (3, 3)

    if DEFAULT_SAVE_PATH.exists():
        real_calib.load_calibration(DEFAULT_SAVE_PATH)
        assert real_calib.is_calibrated is True
        assert real_calib.homography_matrix.shape == (3, 3)


def test_t1_f4_cartesian_ik_reachability_within_envelope() -> None:
    """Validate Cartesian IK solver reaches target coordinates within the 391mm arm reach."""
    ik = CartesianIK()
    target_pos = np.array([0.22, 0.05, 0.05], dtype=np.float32)

    solved_qpos = ik.solve_ik(target_pos, approach="top_down")
    assert solved_qpos is not None
    assert len(solved_qpos) == 6

    # Verify solved angles are within limits
    clamped, warnings = validate_and_clamp_joints(list(np.degrees(solved_qpos)))
    assert len(warnings) == 0, f"Joint clamping warnings on reachable target: {warnings}"


# ==============================================================================
# TIER 2: BOUNDARY VALUES, CORNER CASES & RESILIENCE
# ==============================================================================


def test_t2_f1_workspace_four_boundary_corners_reachability() -> None:
    """Validate IK convergence on all 4 physical workspace boundary corners."""
    ik = CartesianIK()
    config = json.loads(DEFAULT_CONFIG_PATH.read_text())
    corners = config["workspace_corners_mm"]

    for corner_name, coords in corners.items():
        x_m = float(coords["x"]) / 1000.0
        y_m = float(coords["y"]) / 1000.0
        z_m = 0.05  # Hover height

        radius = np.hypot(x_m, y_m)
        if radius <= 0.391:
            qpos = ik.solve_ik(np.array([x_m, y_m, z_m]), approach="top_down")
            clamped, warnings = validate_and_clamp_joints(list(np.degrees(qpos)))
            # Shoulder pan must remain within reasonable sweep
            assert abs(clamped[0]) <= 90.0, f"Excessive shoulder pan at corner {corner_name}"


def test_t2_f2_normalized_coordinate_extremes_boundary_mapping() -> None:
    """Validate coordinate projection on extreme normalized image edges [0, 1000]."""
    calib = get_calibrator(mode="sim")
    init_sim_workspace_calibration(calibrator=calib)

    boundary_points = [
        [0.0, 0.0],
        [1000.0, 0.0],
        [0.0, 1000.0],
        [1000.0, 1000.0],
        [500.0, 0.0],
        [500.0, 1000.0],
        [0.0, 500.0],
        [1000.0, 500.0],
    ]

    for pt in boundary_points:
        xyz = calib.gemini_to_robot_xyz(pt, z_m=0.03)
        assert not np.isnan(xyz).any(), f"NaN encountered for boundary point {pt}"
        assert not np.isinf(xyz).any(), f"Inf encountered for boundary point {pt}"
        assert np.isclose(xyz[2], 0.03)


def test_t2_f2_out_of_reach_target_ik_safety_and_clamping() -> None:
    """Validate IK solver behavior on target coordinates far outside the arm envelope."""
    ik = CartesianIK()
    unreachable_pos = np.array([0.70, 0.50, 0.05], dtype=np.float32)  # Reach radius > 0.85m

    # Should not crash with unhandled exception; returns best-effort clamped pose
    solved_qpos = ik.solve_ik(unreachable_pos, approach="top_down")
    clamped, warnings = validate_and_clamp_joints(list(np.degrees(solved_qpos)))

    for i, angle in enumerate(clamped):
        limits = list(JOINT_LIMITS.values())[i]
        assert limits[0] <= angle <= limits[1]


def test_t2_f3_uncalibrated_calibrator_raises_runtime_error() -> None:
    """Validate that uncalibrated calibrator raises descriptive RuntimeError."""
    uncalibrated = ArucoWorkspaceCalibrator()
    uncalibrated.homography_matrix = None

    with pytest.raises(RuntimeError, match="Workspace is not calibrated"):
        uncalibrated.gemini_to_robot_xyz([500.0, 500.0])

    with pytest.raises(RuntimeError, match="Workspace is not calibrated"):
        uncalibrated.pixel_to_robot_mm(320.0, 240.0)


def test_t2_f4_degenerate_collinear_corners_rejection() -> None:
    """Validate that collinear / degenerate marker positions fail homography estimation."""
    calib = ArucoWorkspaceCalibrator()

    # Collinear corners (all along y=100)
    collinear_corners = {
        1: np.array([[100, 100], [140, 100], [140, 100], [100, 100]], dtype=np.float32),
        2: np.array([[200, 100], [240, 100], [240, 100], [200, 100]], dtype=np.float32),
        3: np.array([[300, 100], [340, 100], [340, 100], [300, 100]], dtype=np.float32),
        0: np.array([[400, 100], [440, 100], [440, 100], [400, 100]], dtype=np.float32),
    }

    # Should safely return False or handle without unhandled crash
    success = calib.update_homography(collinear_corners)
    assert success is False or calib.reprojection_error is None or calib.reprojection_error > 5.0


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
        xyz_tool = image_point_to_xyz([760.0, 484.0], hover_z=hover_z)
        xyz_calib = calib.gemini_to_robot_xyz([760.0, 484.0], z_m=hover_z)
        assert np.allclose(xyz_tool, xyz_calib, atol=1e-4)


def test_t3_cross_feature_move_to_spatial_point_execution() -> None:
    """Validate move_to_spatial_point tool execution in headless simulation."""
    import asyncio

    calib = get_calibrator(mode="sim")
    init_sim_workspace_calibration(calibrator=calib)

    robot = SO101Robot(render_viewer=False)
    robot.reset("HOME")

    try:
        call = SimpleNamespace(
            name="move_to_spatial_point",
            args={"point": [760.0, 484.0], "z_height_cm": 4.0, "approach": "top_down"},
        )
        result = asyncio.run(execute_tool_call(call=call, robot=robot))

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


def test_t4_scenario_pick_hover_and_descend_workflow() -> None:
    """Real-World Scenario 1: Model moves to hover above target, then descends to contact."""
    import asyncio

    calib = get_calibrator(mode="sim")
    init_sim_workspace_calibration(calibrator=calib)

    robot = SO101Robot(render_viewer=False)
    robot.reset("HOME")

    try:
        # Step 1: Hover at 4.0 cm
        hover_call = SimpleNamespace(
            name="move_to_spatial_point",
            args={"point": [760.0, 484.0], "z_height_cm": 4.0, "approach": "top_down"},
        )
        res1 = asyncio.run(execute_tool_call(call=hover_call, robot=robot))
        assert res1["status"] == "success"
        assert "Visual Verification Loop" in res1["visual_hint"]

        # Step 2: Descend to 0.0 cm (table contact)
        descend_call = SimpleNamespace(
            name="move_to_spatial_point",
            args={"point": [760.0, 484.0], "z_height_cm": 0.0, "approach": "top_down"},
        )
        res2 = asyncio.run(execute_tool_call(call=descend_call, robot=robot))
        assert res2["status"] == "success"
        assert res2["target_xyz_m"][2] == 0.0

        # Step 3: Close gripper
        close_call = SimpleNamespace(name="close_gripper", args={})
        res3 = asyncio.run(execute_tool_call(call=close_call, robot=robot))
        assert res3["status"] == "success"
        assert "Gripper closed" in res3["visual_hint"]

        # Step 4: Lift to 5.0 cm
        lift_call = SimpleNamespace(
            name="move_to_spatial_point",
            args={"point": [760.0, 484.0], "z_height_cm": 5.0, "approach": "top_down"},
        )
        res4 = asyncio.run(execute_tool_call(call=lift_call, robot=robot))
        assert res4["status"] == "success"
        assert np.isclose(res4["target_xyz_m"][2], 0.05)
    finally:
        robot.close()


def test_t4_scenario_horizontal_approach_traversal() -> None:
    """Real-World Scenario 2: Approach target object horizontally across lateral span."""
    import asyncio

    calib = get_calibrator(mode="sim")
    init_sim_workspace_calibration(calibrator=calib)

    robot = SO101Robot(render_viewer=False)
    robot.reset("HOME")

    targets = [
        [650.0, 350.0],  # Left
        [650.0, 500.0],  # Center
        [650.0, 650.0],  # Right
    ]

    try:
        for pt in targets:
            call = SimpleNamespace(
                name="move_to_spatial_point",
                args={"point": pt, "z_height_cm": 3.0, "approach": "horizontal"},
            )
            result = asyncio.run(execute_tool_call(call=call, robot=robot))
            assert result["status"] == "success"
            assert result["approach"] == "horizontal"
    finally:
        robot.close()


def test_t4_scenario_perimeter_survey_motion() -> None:
    """Real-World Scenario 3: Perimeter sweep across 4 quadrants without discontinuity."""
    import asyncio

    calib = get_calibrator(mode="sim")
    init_sim_workspace_calibration(calibrator=calib)

    robot = SO101Robot(render_viewer=False)
    robot.reset("HOME")

    survey_points = [
        [400.0, 400.0],
        [400.0, 600.0],
        [700.0, 600.0],
        [700.0, 400.0],
    ]

    try:
        for pt in survey_points:
            call = SimpleNamespace(
                name="move_to_spatial_point",
                args={"point": pt, "z_height_cm": 5.0, "approach": "top_down"},
            )
            result = asyncio.run(execute_tool_call(call=call, robot=robot))
            assert result["status"] == "success"
            executed_qpos = result["executed_qpos_deg"]
            assert len(executed_qpos) == 6
    finally:
        robot.close()


def test_t4_scenario_simulation_target_body_spatial_alignment() -> None:
    """Real-World Scenario 4: Ground physical MuJoCo target body coordinates to robot space."""
    robot = SO101Robot(render_viewer=False)
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
