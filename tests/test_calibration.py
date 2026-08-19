"""Unit tests for ArUco workspace calibration and coordinate projection."""

import tempfile
from pathlib import Path

import cv2
import numpy as np

from robot.calibration import ArucoWorkspaceCalibrator, generate_aruco_markers


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

    # Known pixel corners in a 640x480 image
    # Top-Left (ID 0), Top-Right (ID 1), Bottom-Right (ID 2), Bottom-Left (ID 3)
    mock_detected = {
        0: np.array([[100, 50], [140, 50], [140, 90], [100, 90]], dtype=np.float32),
        1: np.array([[500, 50], [540, 50], [540, 90], [500, 90]], dtype=np.float32),
        2: np.array([[500, 400], [540, 400], [540, 440], [500, 440]], dtype=np.float32),
        3: np.array([[100, 400], [140, 400], [140, 440], [100, 440]], dtype=np.float32),
    }

    success = calibrator.update_homography(mock_detected)
    assert success is True
    assert calibrator.is_calibrated is True
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


def test_uncalibrated_raises_error() -> None:
    """Test that gemini_to_robot_xyz and pixel_to_robot_mm raise RuntimeError when uncalibrated."""
    import pytest

    calibrator = ArucoWorkspaceCalibrator()
    calibrator.homography_matrix = None

    with pytest.raises(RuntimeError, match="Workspace is not calibrated"):
        calibrator.gemini_to_robot_xyz([0.0, 500.0], z_m=0.03)

    with pytest.raises(RuntimeError, match="Workspace is not calibrated"):
        calibrator.pixel_to_robot_mm(320.0, 240.0)


def test_marker_detection_synthetic() -> None:
    """Test ArUco marker detection on a synthetic white image containing drawn markers."""
    calibrator = ArucoWorkspaceCalibrator()
    # Create white canvas
    frame = np.full((480, 640, 3), 255, dtype=np.uint8)

    # Draw marker ID 0 at pixel location
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


def test_overlay_drawing() -> None:
    """Test overlay drawing functions return an annotated image with matching dimensions."""
    calibrator = ArucoWorkspaceCalibrator()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    mock_detected = {
        0: np.array([[100, 50], [140, 50], [140, 90], [100, 90]], dtype=np.float32),
        1: np.array([[500, 50], [540, 50], [540, 90], [500, 90]], dtype=np.float32),
        2: np.array([[500, 400], [540, 400], [540, 440], [500, 440]], dtype=np.float32),
        3: np.array([[100, 400], [140, 400], [140, 440], [100, 440]], dtype=np.float32),
    }

    annotated = calibrator.draw_workspace_overlay(
        frame_rgb=frame,
        detected_markers=mock_detected,
        show_axes=True,
        show_markers=True,
        active_target_px=(320.0, 240.0),
    )

    assert annotated.shape == frame.shape
    assert annotated.dtype == frame.dtype
    # Check that drawing actually modified some pixels
    assert not np.array_equal(annotated, frame)


def test_generate_aruco_markers() -> None:
    """Test marker generation utility creates expected files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        files = generate_aruco_markers(output_dir=tmp_dir, marker_ids=[0, 1, 2, 3])
        assert len(files) == 4
        for f in files:
            assert f.exists()
            assert f.suffix == ".png"
            img = cv2.imread(str(f))
            assert img is not None
            assert img.shape[0] > 100


def test_generate_calibration_sheet() -> None:
    """Test generating the all-in-one workspace calibration mat sheet."""
    from robot.calibration import generate_calibration_sheet

    with tempfile.TemporaryDirectory() as tmp_dir:
        sheet_path = generate_calibration_sheet(output_dir=tmp_dir)
        assert sheet_path.exists()
        assert sheet_path.name == "workspace_calibration_sheet.png"
        img = cv2.imread(str(sheet_path))
        assert img is not None
        assert img.shape == (1900, 2400, 3)


def test_setup_workspace_interactive() -> None:
    """Test non-interactive setup workspace config updating and artifact generation."""
    from robot.calibration import setup_workspace_interactive

    with tempfile.TemporaryDirectory() as tmp_dir:
        cfg_path = Path(tmp_dir) / "test_ws.json"
        out_dir = Path(tmp_dir) / "markers"

        updated = setup_workspace_interactive(
            config_path=cfg_path,
            output_dir=out_dir,
            interactive=False,
            x_max=400,
            x_min=100,
            y_max=300,
            y_min=-250,
            marker_size=50,
        )

        assert cfg_path.exists()
        assert updated["workspace_corners_mm"]["top_left"]["x"] == 400
        assert updated["workspace_corners_mm"]["top_left"]["y"] == 300
        assert updated["marker_size_mm"] == 50
        assert (out_dir / "workspace_calibration_sheet.png").exists()
        assert (out_dir / "marker_0.png").exists()


def test_setup_workspace_reuse_existing_markers() -> None:
    """Test setup workspace detects existing markers and updates JSON without regenerating files."""
    from robot.calibration import setup_workspace_interactive

    with tempfile.TemporaryDirectory() as tmp_dir:
        cfg_path = Path(tmp_dir) / "test_ws.json"
        out_dir = Path(tmp_dir) / "markers"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Create dummy existing marker files with known mtimes
        dummy_marker = out_dir / "marker_0.png"
        dummy_marker.write_text("fake image content")
        initial_mtime = dummy_marker.stat().st_mtime

        # Run setup with reuse=True
        updated = setup_workspace_interactive(
            config_path=cfg_path,
            output_dir=out_dir,
            interactive=False,
            reuse=True,
            x_max=450,
            x_min=50,
        )

        assert updated["workspace_corners_mm"]["top_left"]["x"] == 450
        assert updated["workspace_corners_mm"]["bottom_left"]["x"] == 50
        assert dummy_marker.read_text() == "fake image content"
        assert dummy_marker.stat().st_mtime == initial_mtime


def test_compute_sim_calibration() -> None:
    """Test synthetic homography computation from MuJoCo camera projection."""
    from simulation.sim_calibration import compute_sim_calibration, init_sim_workspace_calibration

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = Path(tmp_dir) / "sim_homography.npz"
        h_mat, inv_h, err = compute_sim_calibration(save_path=save_path)

        assert h_mat.shape == (3, 3)
        assert inv_h.shape == (3, 3)
        assert err < 0.01  # Sub-millimeter error for exact geometric projection
        assert save_path.exists()

        calibrator = init_sim_workspace_calibration(save_path=save_path)
        assert calibrator.is_calibrated is True

        # Test center point projection roundtrip
        px, py = calibrator.gemini_to_pixel([500.0, 500.0])
        xyz = calibrator.pixel_to_robot_xyz(px, py, z_m=0.03)
        assert 0.10 <= xyz[0] <= 0.45  # In front of robot base in workspace
        assert -0.30 <= xyz[1] <= 0.30  # Within workspace lateral span
        assert np.isclose(xyz[2], 0.03)


def test_sim_robot_camera_rendering() -> None:
    """Test SO101Robot rendering from workspace_cam in MuJoCo simulation."""
    from simulation.sim_robot import SO101Robot

    robot = SO101Robot(render_viewer=False)
    try:
        frame = robot.render_rgb_frame(width=640, height=480, camera="workspace_cam")
        assert frame.shape == (480, 640, 3)
        assert frame.dtype == np.uint8
        # Ensure frame is not completely blank / black
        assert np.mean(frame) > 10.0
    finally:
        robot.close()


def test_sim_spatial_point_execution() -> None:
    """Test spatial point IK execution end-to-end in simulation mode."""
    import asyncio
    from types import SimpleNamespace

    from agent.tools import execute_tool_call
    from simulation.sim_calibration import init_sim_workspace_calibration
    from simulation.sim_robot import SO101Robot
    from ui.stream import get_calibrator

    calib = get_calibrator(mode="sim")
    init_sim_workspace_calibration(calibrator=calib)

    robot = SO101Robot(render_viewer=False)
    robot.reset("HOME")

    try:
        call = SimpleNamespace(
            name="move_to_spatial_point",
            args={"point": [500.0, 500.0], "z_height_cm": 3.0},
        )
        result = asyncio.run(execute_tool_call(call=call, robot=robot))
        assert result.get("status") == "success"
        assert "target_xyz_m" in result
        target_xyz = result["target_xyz_m"]
        assert 0.10 <= target_xyz[0] <= 0.45
        assert -0.30 <= target_xyz[1] <= 0.30
        assert np.isclose(target_xyz[2], 0.03)
    finally:
        robot.close()
