"""Unit tests for ArUco marker and calibration sheet generation utilities."""

import tempfile

import cv2

from tools.calibration_sheet import generate_aruco_markers, generate_calibration_sheet


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
    with tempfile.TemporaryDirectory() as tmp_dir:
        sheet_path = generate_calibration_sheet(output_dir=tmp_dir)
        assert sheet_path.exists()
        assert sheet_path.name == "workspace_calibration_sheet.png"
        img = cv2.imread(str(sheet_path))
        assert img is not None
        assert img.shape == (1900, 2400, 3)
