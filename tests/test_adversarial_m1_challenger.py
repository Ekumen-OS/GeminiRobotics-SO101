"""Adversarial stress testing and empirical validation for Milestone 1.

Probes:
- Dense 2D grid projection and reachability across [0, 1000].
- Numerical IK convergence, joint limits satisfaction, and forward kinematics verification.
- Benchmark target [760, 484] lateral parity and shoulder pan angle (~ -42 deg).
- Extreme boundary coordinates, degenerate homographies, and mode routing resilience.
- Homography lateral Y axis positive sign verification and prompt text harmonization.
"""

import json
from pathlib import Path

import cv2
import mujoco
import numpy as np
import pytest

from agent.tools import image_point_to_xyz, validate_and_clamp_joints
from robot.calibration import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_SAVE_PATH,
    ArucoWorkspaceCalibrator,
)
from robot.cartesian_ik import CartesianIK
from robot.constants import PRESET_POSES
from simulation.sim_calibration import init_sim_workspace_calibration
from ui.stream import get_calibrator


def test_adversarial_dense_grid_projection_and_ik_convergence() -> None:
    """Stress-test 2D-to-3D projection and IK across a dense 21x21 grid (441 points)."""
    calib = get_calibrator(mode="sim")
    init_sim_workspace_calibration(calibrator=calib)
    assert calib.is_calibrated is True

    # Calibrated workspace polygon from physical corner definitions
    corners_robot_mm = [
        [380.0, -190.0],  # TL
        [100.0, -190.0],  # TR
        [100.0, 190.0],  # BR
        [380.0, 190.0],  # BL
    ]
    corners_img_px = [calib.robot_mm_to_pixel(x_mm, y_mm) for x_mm, y_mm in corners_robot_mm]
    poly_px = np.array(corners_img_px, dtype=np.float32)

    grid_y = np.linspace(0.0, 1000.0, 21)
    grid_x = np.linspace(0.0, 1000.0, 21)

    total_points = 0
    inside_polygon_points = 0
    reachable_points = 0
    converged_within_10mm = 0
    converged_within_15mm = 0
    joint_limit_violations = 0
    fk_errors = []

    home_seed = PRESET_POSES["HOME"]

    for y in grid_y:
        for x in grid_x:
            total_points += 1
            pt = [float(y), float(x)]
            xyz = calib.gemini_to_robot_xyz(pt, z_m=0.03)

            assert not np.isnan(xyz).any(), f"NaN in projection for {pt}"
            assert not np.isinf(xyz).any(), f"Inf in projection for {pt}"
            assert np.isclose(xyz[2], 0.03)

            px = (x / 1000.0) * 640.0
            py = (y / 1000.0) * 480.0
            dist = cv2.pointPolygonTest(poly_px, (px, py), False)

            if dist >= 0:
                inside_polygon_points += 1
                r = float(np.hypot(xyz[0], xyz[1]))
                # Filter points within physical arm reach envelope
                if r <= 0.380:
                    reachable_points += 1
                    ik = CartesianIK()
                    solved_qpos = ik.solve_ik(xyz, initial_qpos=home_seed, approach="top_down")

                    # Verify all joints within physical limits
                    clamped, warnings = validate_and_clamp_joints(list(np.degrees(solved_qpos)))
                    if len(warnings) > 0:
                        joint_limit_violations += 1

                    # Evaluate Forward Kinematics in MuJoCo
                    for i, j_id in enumerate(ik.all_joint_ids):
                        ik.data.qpos[ik.model.jnt_qposadr[j_id]] = solved_qpos[i]
                    mujoco.mj_forward(ik.model, ik.data)
                    achieved_pos = ik.data.site_xpos[ik.site_id].copy()

                    err_dist = float(np.linalg.norm(xyz - achieved_pos))
                    fk_errors.append(err_dist)

                    if err_dist <= 0.010:
                        converged_within_10mm += 1
                    if err_dist <= 0.015:
                        converged_within_15mm += 1

    assert total_points == 441
    assert inside_polygon_points > 50, f"Expected >50 polygon points, got {inside_polygon_points}"
    assert reachable_points > 40, f"Expected >40 reachable points, got {reachable_points}"
    assert joint_limit_violations == 0, f"Encountered {joint_limit_violations} limit violations"

    # Convergence rate for reachable workspace points must be >= 95% within 15mm
    conv_rate_15mm = converged_within_15mm / reachable_points
    assert conv_rate_15mm >= 0.95, f"Low convergence rate: {conv_rate_15mm:.2%}"


def test_adversarial_benchmark_red_cube_760_484_grounding_audit() -> None:
    """Deep forensic verification of the benchmark red cube target [760, 484]."""
    sim_calib = get_calibrator(mode="sim")
    init_sim_workspace_calibration(calibrator=sim_calib)

    real_calib = ArucoWorkspaceCalibrator(mode="real")
    if DEFAULT_SAVE_PATH.exists():
        real_calib.load_calibration(DEFAULT_SAVE_PATH)

    pt = [760.0, 484.0]
    xyz_sim = sim_calib.gemini_to_robot_xyz(pt, z_m=0.03)

    # 1. Physical base frame assertions
    # Forward distance X should be ~0.197 m (between 0.18 and 0.22 m)
    assert 0.18 <= xyz_sim[0] <= 0.22, f"Unexpected X coordinate: {xyz_sim[0]}"
    # Lateral distance Y should be positive (~+0.145 m), reflecting camera view towards robot
    assert 0.12 <= xyz_sim[1] <= 0.17, f"Unexpected Y coordinate (lateral inversion?): {xyz_sim[1]}"
    assert np.isclose(xyz_sim[2], 0.03)

    # 2. Sim vs Real lateral parity check
    if real_calib.is_calibrated:
        xyz_real = real_calib.gemini_to_robot_xyz(pt, z_m=0.03)
        assert np.allclose(xyz_sim, xyz_real, atol=2e-3), (
            f"Sim ({xyz_sim}) and Real ({xyz_real}) lateral parity mismatch!"
        )

    # 3. IK Solution and Shoulder Pan Angle Check
    ik = CartesianIK()
    qpos = ik.solve_ik(xyz_sim, initial_qpos=PRESET_POSES["HOME"], approach="top_down")
    shoulder_pan_deg = float(np.degrees(qpos[0]))

    # Target is on positive Y (right side of camera view, left of robot base facing camera)
    # Robot shoulder pan angle should rotate negative towards the target (~ -42 deg)
    assert -45.0 <= shoulder_pan_deg <= -39.0, (
        f"Shoulder pan angle {shoulder_pan_deg:.2f} deg out of expected [-45, -39] range!"
    )

    # 4. Forward Kinematics verification in MuJoCo
    for i, j_id in enumerate(ik.all_joint_ids):
        ik.data.qpos[ik.model.jnt_qposadr[j_id]] = qpos[i]
    mujoco.mj_forward(ik.model, ik.data)
    fk_pos = ik.data.site_xpos[ik.site_id].copy()
    fk_error = float(np.linalg.norm(xyz_sim - fk_pos))
    assert fk_error < 0.005, f"FK tracking error {fk_error * 1000:.2f}mm exceeded 5mm threshold"


def test_adversarial_extreme_boundary_and_out_of_bounds_points() -> None:
    """Stress-test extreme boundaries, corners, and out-of-range normalized inputs."""
    calib = get_calibrator(mode="sim")
    init_sim_workspace_calibration(calibrator=calib)

    extreme_points = [
        [0.0, 0.0],
        [0.0, 1000.0],
        [1000.0, 0.0],
        [1000.0, 1000.0],
        [-200.0, 500.0],
        [1200.0, 500.0],
        [500.0, -300.0],
        [500.0, 1400.0],
    ]

    for pt in extreme_points:
        xyz = calib.gemini_to_robot_xyz(pt, z_m=0.03)
        assert not np.isnan(xyz).any(), f"NaN generated for extreme point {pt}"
        assert not np.isinf(xyz).any(), f"Inf generated for extreme point {pt}"
        assert np.isclose(xyz[2], 0.03)


def test_adversarial_calibrator_degenerate_and_faulty_inputs() -> None:
    """Stress-test calibrator against degenerate geometries and missing markers."""
    # Force uncalibrated calibrator instance
    calib = ArucoWorkspaceCalibrator(save_path="/tmp/non_existent_calibration.npz")
    calib.homography_matrix = None
    calib.inverse_homography = None

    # Case 1: Incomplete markers (only 3 corners detected)
    partial_markers = {
        1: np.array([[100, 60], [140, 60], [140, 100], [100, 100]], dtype=np.float32),
        2: np.array([[500, 60], [540, 60], [540, 100], [500, 100]], dtype=np.float32),
        3: np.array([[500, 380], [540, 380], [540, 420], [500, 420]], dtype=np.float32),
    }
    success = calib.update_homography(partial_markers)
    assert success is False
    assert calib.is_calibrated is False

    # Case 2: Degenerate collinear points
    collinear_markers = {
        1: np.array([[100, 100], [140, 100], [140, 100], [100, 100]], dtype=np.float32),
        2: np.array([[200, 100], [240, 100], [240, 100], [200, 100]], dtype=np.float32),
        3: np.array([[300, 100], [340, 100], [340, 100], [300, 100]], dtype=np.float32),
        0: np.array([[400, 100], [440, 100], [440, 100], [400, 100]], dtype=np.float32),
    }
    success_collinear = calib.update_homography(collinear_markers)
    assert (
        success_collinear is False
        or calib.reprojection_error is None
        or calib.reprojection_error > 5.0
    )


def test_adversarial_mode_routing_and_uncalibrated_guardrails() -> None:
    """Verify mode routing in agent.tools.image_point_to_xyz and error guardrails."""
    # Test explicit mode routing
    xyz_sim = image_point_to_xyz([760.0, 484.0], hover_z=0.03, mode="sim")
    assert 0.12 <= xyz_sim[1] <= 0.17

    if DEFAULT_SAVE_PATH.exists():
        xyz_real = image_point_to_xyz([760.0, 484.0], hover_z=0.03, mode="real")
        assert np.allclose(xyz_sim, xyz_real, atol=2e-3)

    # Test error when forced to uncalibrated instance
    calib_dummy = ArucoWorkspaceCalibrator(config_path="/tmp/empty.json")
    calib_dummy.homography_matrix = None
    with pytest.raises(RuntimeError, match="Workspace is not calibrated"):
        calib_dummy.gemini_to_robot_xyz([500.0, 500.0])


def test_adversarial_ik_singularity_and_hostile_seed_recovery() -> None:
    """Stress-test IK solver convergence starting from hostile initial seeds."""
    ik = CartesianIK()
    target = np.array([0.20, 0.10, 0.03], dtype=np.float32)

    # Test from zero pose
    q_zero = np.zeros(6, dtype=np.float32)
    solved_from_zero = ik.solve_ik(target, initial_qpos=q_zero, approach="top_down")
    assert solved_from_zero is not None

    # Verify joint limits
    clamped, warnings = validate_and_clamp_joints(list(np.degrees(solved_from_zero)))
    assert len(warnings) == 0

    # Test from extreme joint limit pose
    q_extreme = np.array([1.5, 0.8, -1.5, 1.2, 0.0, 0.0], dtype=np.float32)
    solved_from_extreme = ik.solve_ik(target, initial_qpos=q_extreme, approach="top_down")
    assert solved_from_extreme is not None
    clamped_ext, warnings_ext = validate_and_clamp_joints(list(np.degrees(solved_from_extreme)))
    assert len(warnings_ext) == 0


def test_adversarial_homography_files_lateral_parity_matrix_elements(
    sim_homography_path: Path,
    real_homography_path: Path,
) -> None:
    """Directly verify homography matrices for lateral axis inversion bug elimination."""
    data_sim = np.load(sim_homography_path)
    data_real = np.load(real_homography_path)

    h_sim = data_sim["homography"]
    h_real = data_real["homography"]

    # In positive lateral coordinate system, H[1, 1] must be positive
    assert h_sim[1, 1] > 0, f"Sim homography H[1,1] is negative: {h_sim[1, 1]}"
    assert h_real[1, 1] > 0, f"Real homography H[1,1] is negative: {h_real[1, 1]}"

    # Element-wise close match between sim and physical homography
    np.testing.assert_allclose(h_sim, h_real, rtol=0.05, atol=1e-3)


def test_adversarial_system_prompt_marker_id_harmonization() -> None:
    """Ensure system prompt in agent/vla_controller.py uses harmonized ArUco IDs."""
    from agent.vla_controller import VLAController

    vla = VLAController()
    system_prompt = vla.history[0].get("content", "")

    # Must contain the harmonized IDs
    assert (
        "ID 1 (Top-Left), ID 2 (Top-Right), ID 3 (Bottom-Right), ID 0 (Bottom-Left)"
        in system_prompt
    ), "System prompt does not contain harmonized marker IDs"

    # Must NOT contain legacy mismatched ID definition
    assert "ID 0 (Top-Left), ID 1 (Top-Right)" not in system_prompt


def test_adversarial_scene_xml_geom_and_config_parity() -> None:
    """Verify scene.xml geom coordinates and materials perfectly match workspace config."""
    config = json.loads(DEFAULT_CONFIG_PATH.read_text())
    corners = config["workspace_corners_mm"]

    scene_xml = (
        Path(__file__).resolve().parent.parent / "simulation" / "assets" / "scene.xml"
    ).read_text()

    # Check top_left (X=0.380, Y=-0.190) -> aruco_tl_0 with mat_marker_1
    assert 'name="aruco_tl_0"' in scene_xml
    assert 'material="mat_marker_1"' in scene_xml
    assert 'pos="0.380 -0.190 0.0008"' in scene_xml
    assert corners["top_left"]["x"] == 380.0
    assert corners["top_left"]["y"] == -190.0

    # Check top_right (X=0.100, Y=-0.190) -> aruco_tr_1 with mat_marker_2
    assert 'name="aruco_tr_1"' in scene_xml
    assert 'material="mat_marker_2"' in scene_xml
    assert 'pos="0.100 -0.190 0.0008"' in scene_xml
    assert corners["top_right"]["x"] == 100.0
    assert corners["top_right"]["y"] == -190.0

    # Check bottom_right (X=0.100, Y=+0.190) -> aruco_br_2 with mat_marker_3
    assert 'name="aruco_br_2"' in scene_xml
    assert 'material="mat_marker_3"' in scene_xml
    assert 'pos="0.100 0.190 0.0008"' in scene_xml
    assert corners["bottom_right"]["x"] == 100.0
    assert corners["bottom_right"]["y"] == 190.0

    # Check bottom_left (X=0.380, Y=+0.190) -> aruco_bl_3 with mat_marker_0
    assert 'name="aruco_bl_3"' in scene_xml
    assert 'material="mat_marker_0"' in scene_xml
    assert 'pos="0.380 0.190 0.0008"' in scene_xml
    assert corners["bottom_left"]["x"] == 380.0
    assert corners["bottom_left"]["y"] == 190.0
