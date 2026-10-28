"""Unit tests for ArUco marker and placement guide generation utilities."""

import tempfile
from pathlib import Path

import cv2
import pytest

from tools.calibration_sheet import (
    _FONT,
    _font,
    generate_aruco_markers,
    generate_jaw_tag_sheet,
    generate_placement_guide,
)

_SAMPLE_CORNERS_PAN: dict[str, tuple[float, float]] = {
    "top_left": (414.0, -240.0),
    "top_right": (97.0, -240.0),
    "bottom_right": (97.0, 102.0),
    "bottom_left": (414.0, 102.0),
}


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


def test_generate_placement_guide() -> None:
    """Test generating the workspace placement guide diagram."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        guide_path = generate_placement_guide(
            corners_pan_mm=_SAMPLE_CORNERS_PAN,
            marker_size_mm=40.0,
            dictionary_name="DICT_4X4_50",
            pan_offset_x_mm=38.8,
            output_dir=tmp_dir,
            date_str="2026-10-27",
        )
        assert guide_path.exists()
        assert guide_path.name == "workspace_placement_guide.png"
        img = cv2.imread(str(guide_path))
        assert img is not None
        assert img.shape == (2480, 3508, 3)


def test_generate_placement_guide_deterministic_for_fixed_inputs() -> None:
    """Verify that placement guide image bytes are identical for identical inputs."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        out1 = Path(tmp_dir) / "dir1"
        out2 = Path(tmp_dir) / "dir2"
        p1 = generate_placement_guide(
            corners_pan_mm=_SAMPLE_CORNERS_PAN,
            marker_size_mm=40.0,
            dictionary_name="DICT_4X4_50",
            pan_offset_x_mm=38.8,
            output_dir=out1,
            date_str="2026-10-27",
        )
        p2 = generate_placement_guide(
            corners_pan_mm=_SAMPLE_CORNERS_PAN,
            marker_size_mm=40.0,
            dictionary_name="DICT_4X4_50",
            pan_offset_x_mm=38.8,
            output_dir=out2,
            date_str="2026-10-27",
        )

        assert p1.read_bytes() == p2.read_bytes()


def test_generate_jaw_tag_sheet() -> None:
    """Test jaw tag printable sheet generation."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        sheet_path = generate_jaw_tag_sheet(output_dir=tmp_dir)
        assert sheet_path.exists()
        assert sheet_path.name == "jaw_tag_sheet.png"
        img = cv2.imread(str(sheet_path))
        assert img is not None
        assert img.shape[0] > 500


@pytest.mark.parametrize("size_pt", [10.0, 13.0, 20.0])
def test_font_prints_at_requested_point_size(size_pt: float) -> None:
    """Text cap height on the 300 dpi page is 0.7 em of the requested point size, within 10 %."""
    scale, thickness = _font(size_pt)
    cap_px = cv2.getTextSize("H", _FONT, scale, thickness)[0][1]
    expected_px = 0.7 * size_pt * 300 / 72
    assert abs(cap_px - expected_px) / expected_px < 0.10
