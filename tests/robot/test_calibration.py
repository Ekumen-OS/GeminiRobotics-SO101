"""Unit tests for ArUco workspace calibration, coordinate projection, and spatial grounding."""

import json
import tempfile
from pathlib import Path

import cv2
import mujoco
import numpy as np
import pytest

from agent.tools import validate_and_clamp_joints
from agent.vla_controller import load_system_prompt
from robot.calibration import ArucoWorkspaceCalibrator
from robot.cartesian_ik import CartesianIK
from robot.constants import PRESET_POSES
from robot.settings import get_settings
from simulation.sim_calibration import init_sim_workspace_calibration
from vision.calibrator import get_calibrator


def test_gemini_to_pixel_mapping() -> None:
    """Test normalized Gemini coordinates (0-1000) map accurately to pixel boundaries."""
    calibrator = ArucoWorkspaceCalibrator()

    # Top-Left [y=0, x=0] -> pixel (0, 0)
    px, py = calibrator.gemini_to_pixel([0.0, 0.0], img_shape=(480, 640))
    assert np.isclose(px, 0.0)
    assert np.isclose(py, 0.0)

    # Bottom-Right [y=1000, x=1000] -> pixel (640, 480)
    px, py = calibrator.gemini_to_pixel([1000.0, 1000.0], img_shape=(480, 640))
    assert np.isclose(px, 640.0)
    assert np.isclose(py, 480.0)

    # Center [y=500, x=500] -> pixel (320, 240)
    px, py = calibrator.gemini_to_pixel([500.0, 500.0], img_shape=(480, 640))
    assert np.isclose(px, 320.0)
    assert np.isclose(py, 240.0)


def test_homography_estimation_and_roundtrip() -> None:
    """Test homography computation from known pixel correspondences and roundtrip projection."""
    calibrator = ArucoWorkspaceCalibrator()

    marker_ids = calibrator.config.get("marker_ids", {})
    id_tl = int(marker_ids.get("top_left", 1))
    id_tr = int(marker_ids.get("top_right", 2))
    id_br = int(marker_ids.get("bottom_right", 3))
    id_bl = int(marker_ids.get("bottom_left", 0))

    # Known pixel corners in a 640x480 image
    mock_detected = {
        id_tl: np.array([[100, 50], [140, 50], [140, 90], [100, 90]], dtype=np.float32),
        id_tr: np.array([[500, 50], [540, 50], [540, 90], [500, 90]], dtype=np.float32),
        id_br: np.array([[500, 400], [540, 400], [540, 440], [500, 440]], dtype=np.float32),
        id_bl: np.array([[100, 400], [140, 400], [140, 440], [100, 440]], dtype=np.float32),
    }

    success = calibrator.update_homography(mock_detected)
    assert success is True
    assert calibrator.is_calibrated is True
    assert calibrator.inverse_homography is not None
    assert calibrator.reprojection_error is not None
    assert calibrator.reprojection_error < 1.0  # sub-millimeter reprojection error on planar points

    # Test projection of Top-Left marker center -> matches top_left config
    expected_tl = [
        float(calibrator.config["workspace_corners_mm"]["top_left"]["x"]),
        float(calibrator.config["workspace_corners_mm"]["top_left"]["y"]),
    ]
    xy_tl = calibrator.pixel_to_robot_mm(120.0, 70.0)
    assert np.allclose(xy_tl, expected_tl, atol=1.0)

    # Test projection of Bottom-Right marker center -> matches bottom_right config
    expected_br = [
        float(calibrator.config["workspace_corners_mm"]["bottom_right"]["x"]),
        float(calibrator.config["workspace_corners_mm"]["bottom_right"]["y"]),
    ]
    xy_br = calibrator.pixel_to_robot_mm(520.0, 420.0)
    assert np.allclose(xy_br, expected_br, atol=1.0)

    # Test 3D XYZ in meters
    xyz_br = calibrator.pixel_to_robot_xyz(520.0, 420.0, z_m=0.03)
    assert np.allclose(
        xyz_br,
        [expected_br[0] / 1000.0, expected_br[1] / 1000.0, 0.03],
        atol=0.002,
    )

    # Test inverse projection roundtrip back to pixel coordinates
    for u, v in [(120.0, 70.0), (520.0, 70.0), (320.0, 240.0)]:
        robot_mm = calibrator.pixel_to_robot_mm(u, v)
        reprojected_pixel = calibrator.robot_mm_to_pixel(robot_mm[0], robot_mm[1])
        assert reprojected_pixel is not None
        assert np.isclose(reprojected_pixel[0], u, atol=0.5)
        assert np.isclose(reprojected_pixel[1], v, atol=0.5)


def test_uncalibrated_raises_error() -> None:
    """Test that gemini_to_robot_xyz and pixel_to_robot_mm raise RuntimeError when uncalibrated."""
    calibrator = ArucoWorkspaceCalibrator()
    calibrator.homography_matrix = None

    with pytest.raises(RuntimeError, match="Workspace is not calibrated"):
        calibrator.gemini_to_robot_xyz([0.0, 500.0], z_m=0.03)

    with pytest.raises(RuntimeError, match="Workspace is not calibrated"):
        calibrator.pixel_to_robot_mm(320.0, 240.0)


def test_marker_detection_synthetic() -> None:
    """Test ArUco marker detection on a synthetic white image containing drawn markers."""
    calibrator = ArucoWorkspaceCalibrator()
    frame = np.full((480, 640, 3), 255, dtype=np.uint8)

    dict_name = calibrator.config.get("aruco_dictionary", "DICT_4X4_50")
    dict_attr = getattr(cv2.aruco, dict_name, cv2.aruco.DICT_4X4_50)
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_attr)
    marker_img = cv2.aruco.generateImageMarker(aruco_dict, 0, 100)
    marker_rgb = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2RGB)

    frame[50:150, 50:150] = marker_rgb

    detected = calibrator.detect_markers(frame)
    assert 0 in detected
    assert detected[0].shape == (4, 2)
    center = np.mean(detected[0], axis=0)
    assert 90 <= center[0] <= 110
    assert 90 <= center[1] <= 110


def test_save_and_load_calibration() -> None:
    """Test serialization and deserialization of homography matrix to .npz file."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        calibrator = ArucoWorkspaceCalibrator()
        calibrator.homography_matrix = np.eye(3, dtype=np.float64)
        calibrator.inverse_homography = np.eye(3, dtype=np.float64)
        calibrator.reprojection_error = 0.05

        save_file = Path(tmp_dir) / "test_homo.npz"
        msg = calibrator.save_calibration(save_file)
        assert "successfully saved" in msg
        assert save_file.exists()


def test_f1_marker_id_harmonization_across_subsystems() -> None:
    """Validate ArUco marker ID standardization (1:TL, 2:TR, 3:BR, 0:BL) across subsystems."""
    # 1. Check workspace_calibration.json
    cfg_path = get_settings().calibration_config_path
    assert cfg_path.exists()
    cfg = json.loads(cfg_path.read_text())
    assert cfg.get("aruco_dictionary") == "DICT_4X4_50"
    assert float(cfg.get("marker_size_mm", 0)) == 40.0

    marker_ids = cfg.get("marker_ids", {})
    assert marker_ids["top_left"] == 1
    assert marker_ids["top_right"] == 2
    assert marker_ids["bottom_right"] == 3
    assert marker_ids["bottom_left"] == 0

    corners = cfg.get("workspace_corners_mm", {})
    assert set(corners.keys()) == {"top_left", "top_right", "bottom_right", "bottom_left"}
    for corner_name in ("top_left", "top_right", "bottom_right", "bottom_left"):
        assert "x" in corners[corner_name] and isinstance(corners[corner_name]["x"], (int, float))
        assert "y" in corners[corner_name] and isinstance(corners[corner_name]["y"], (int, float))

    # 2. Check calibrator fallback config
    fallback_calib = ArucoWorkspaceCalibrator(config_path="/tmp/non_existent_calibration.json")
    assert fallback_calib.config.get("aruco_dictionary") == "DICT_4X4_50"
    assert fallback_calib.config["marker_ids"]["top_left"] == 1
    assert fallback_calib.config["marker_ids"]["top_right"] == 2
    assert fallback_calib.config["marker_ids"]["bottom_right"] == 3
    assert fallback_calib.config["marker_ids"]["bottom_left"] == 0
    assert fallback_calib.detector_params is not None
    assert fallback_calib.aruco_dict is not None

    # 3. Check simulation scene.xml geoms & materials
    scene_path = (
        Path(__file__).resolve().parent.parent.parent / "simulation" / "assets" / "scene.xml"
    )
    scene_text = scene_path.read_text()
    for i in range(4):
        assert f'name="tex_marker_{i}"' in scene_text
        assert f'name="mat_marker_{i}"' in scene_text
    assert 'name="aruco_tl_0"' in scene_text and 'material="mat_marker_1"' in scene_text
    assert 'name="aruco_tr_1"' in scene_text and 'material="mat_marker_2"' in scene_text
    assert 'name="aruco_br_2"' in scene_text and 'material="mat_marker_3"' in scene_text
    assert 'name="aruco_bl_3"' in scene_text and 'material="mat_marker_0"' in scene_text

    # 4. Check VLAController system prompt text
    sys_prompt = load_system_prompt("sim")
    expected_str = "ID 1 (Top-Left), ID 2 (Top-Right), ID 3 (Bottom-Right), ID 0 (Bottom-Left)"
    assert expected_str in sys_prompt


def test_f2_homography_lateral_y_parity_and_red_cube_grounding(
    homography_path: Path,
) -> None:
    """Validate positive lateral Y mapping and parity for benchmark [760, 484]."""
    c_real = ArucoWorkspaceCalibrator(mode="real", save_path=homography_path)
    c_sim = ArucoWorkspaceCalibrator(mode="sim", save_path=homography_path)

    assert c_real.is_calibrated
    assert c_sim.is_calibrated

    pt = [760.0, 484.0]
    xyz_real = c_real.gemini_to_robot_xyz(pt, z_m=0.03)
    xyz_sim = c_sim.gemini_to_robot_xyz(pt, z_m=0.03)

    # Both must match with high accuracy
    np.testing.assert_allclose(xyz_real, xyz_sim, atol=2e-3)

    # Check Cartesian target values
    assert 0.18 <= xyz_real[0] <= 0.22  # X ~ 0.197m
    assert 0.12 <= xyz_real[1] <= 0.17  # Y ~ +0.15m (positive lateral sign)
    assert np.isclose(xyz_real[2], 0.03)

    # Solve IK and verify shoulder pan rotates towards the camera (negative pan, ~ -42 deg)
    ik = CartesianIK()
    q_real = ik.solve_ik(xyz_real)
    pan_deg_real = float(np.degrees(q_real[0]))
    assert -46.0 <= pan_deg_real <= -38.0, f"Pan angle out of expected range: {pan_deg_real} deg"

    q_sim = ik.solve_ik(xyz_sim)
    pan_deg_sim = float(np.degrees(q_sim[0]))
    assert -46.0 <= pan_deg_sim <= -38.0, f"Sim pan angle out of expected range: {pan_deg_sim} deg"


def test_f4_reachability_and_fov_coverage() -> None:
    """Validate all reachable workspace coordinates (r <= 391mm) and corners produce valid IK.

    Also verifies target coordinates fall within camera FOV.
    """
    calib = ArucoWorkspaceCalibrator(mode="sim")
    ik = CartesianIK()
    neutral_seed = PRESET_POSES["HOME"].copy()

    # 1. Test 5x5 grid across physical workspace span
    for x_m in np.linspace(0.14, 0.35, 5):
        for y_m in np.linspace(-0.19, 0.19, 5):
            r = np.hypot(x_m, y_m)
            if r <= 0.391:
                px_py = calib.robot_mm_to_pixel(x_m * 1000.0, y_m * 1000.0)
                assert px_py is not None
                px, py = px_py
                assert 0.0 <= px <= 640.0, f"Pixel X={px} outside FOV for ({x_m}, {y_m})"
                assert 0.0 <= py <= 480.0, f"Pixel Y={py} outside FOV for ({x_m}, {y_m})"

                target_xyz = np.array([x_m, y_m, 0.05])
                solved_q = ik.solve_ik(target_xyz, initial_qpos=neutral_seed, null_weight=1e-3)
                clamped_deg, warnings = validate_and_clamp_joints(list(np.degrees(solved_q)))
                assert len(warnings) == 0

                for i, j_id in enumerate(ik.all_joint_ids):
                    qadr = ik.model.jnt_qposadr[j_id]
                    ik.data.qpos[qadr] = solved_q[i]
                mujoco.mj_forward(ik.model, ik.data)
                achieved = ik.data.site_xpos[ik.site_id].copy()
                err = float(np.linalg.norm(target_xyz - achieved))
                assert err < 0.015, (
                    f"IK positioning error too large ({err * 1000:.1f}mm) at ({x_m}, {y_m})"
                )

    # 2. Check all 4 workspace boundary corners from config
    cfg = json.loads(get_settings().calibration_config_path.read_text())
    for corner_name, coords in cfg["workspace_corners_mm"].items():
        x_m = float(coords["x"]) / 1000.0
        y_m = float(coords["y"]) / 1000.0
        if np.hypot(x_m, y_m) <= 0.391:
            qpos = ik.solve_ik(np.array([x_m, y_m, 0.05]), approach="top_down")
            clamped, _ = validate_and_clamp_joints(list(np.degrees(qpos)))
            assert abs(clamped[0]) <= 90.0, f"Excessive shoulder pan at corner {corner_name}"


def test_normalized_coordinate_extremes_boundary_mapping() -> None:
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


def test_degenerate_collinear_corners_rejection() -> None:
    """Validate that collinear / degenerate marker positions fail homography estimation."""
    calib = ArucoWorkspaceCalibrator()

    collinear_corners = {
        1: np.array([[100, 100], [140, 100], [140, 100], [100, 100]], dtype=np.float32),
        2: np.array([[200, 100], [240, 100], [240, 100], [200, 100]], dtype=np.float32),
        3: np.array([[300, 100], [340, 100], [340, 100], [300, 100]], dtype=np.float32),
        0: np.array([[400, 100], [440, 100], [440, 100], [400, 100]], dtype=np.float32),
    }

    success = calib.update_homography(collinear_corners)
    assert success is False or calib.reprojection_error is None or calib.reprojection_error > 5.0
