"""Unit tests for workspace setup wizard CLI utility."""

import json
import tempfile
from pathlib import Path

import pytest

from tools.setup_wizard import (
    base_to_pan_xy_mm,
    get_pan_origin_offset_x_mm,
    pan_to_base_xy_mm,
    setup_workspace_interactive,
)


def test_setup_workspace_interactive() -> None:
    """Test non-interactive setup workspace config updating and artifact generation."""
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


def test_setup_wizard_non_interactive_reuse_preserves_existing_corners() -> None:
    """A non-interactive --reuse run without overrides must not rewrite calibrated corners.

    The checked-in config has corners that are not an axis-aligned rectangle; rebuilding them
    from x/y extremes used to flip the lateral sign of two corners on every `--reuse` run.
    """
    corners = {
        "top_left": {"x": 380.0, "y": -190.0},
        "top_right": {"x": 100.0, "y": -190.0},
        "bottom_right": {"x": 100.0, "y": 190.0},
        "bottom_left": {"x": 380.0, "y": 190.0},
    }
    with tempfile.TemporaryDirectory() as tmp_dir:
        cfg_path = Path(tmp_dir) / "ws.json"
        cfg_path.write_text(json.dumps({"workspace_corners_mm": corners, "marker_size_mm": 50}))
        out_dir = Path(tmp_dir) / "markers"
        out_dir.mkdir()
        (out_dir / "marker_0.png").write_text("existing")

        updated = setup_workspace_interactive(
            config_path=cfg_path, output_dir=out_dir, interactive=False, reuse=True
        )

    assert updated["workspace_corners_mm"] == corners

    # Explicit overrides still rebuild the rectangle (existing behaviour).
    with tempfile.TemporaryDirectory() as tmp_dir:
        cfg_path = Path(tmp_dir) / "ws.json"
        cfg_path.write_text(json.dumps({"workspace_corners_mm": corners, "marker_size_mm": 50}))
        updated = setup_workspace_interactive(
            config_path=cfg_path,
            output_dir=Path(tmp_dir) / "m",
            interactive=False,
            x_max=400,
            x_min=100,
            y_max=300,
            y_min=-250,
        )
    assert updated["workspace_corners_mm"]["top_left"] == {"x": 400, "y": 300}


def test_pan_origin_offset_and_coordinate_conversions() -> None:
    """Verify offset is extracted dynamically from the model and coordinate conversions work."""
    offset = get_pan_origin_offset_x_mm()
    # MuJoCo model shoulder body pos x = 0.0388353 m (~38.8 mm)
    assert offset == pytest.approx(38.8353, abs=0.01)

    # Forward: Pan-axis tape measurements -> Robot base frame (+38.8 mm in X)
    assert pan_to_base_xy_mm(414.0, -240.0) == (452.8, -240.0)
    assert pan_to_base_xy_mm(97.0, -240.0) == (135.8, -240.0)
    assert pan_to_base_xy_mm(97.0, 102.0) == (135.8, 102.0)
    assert pan_to_base_xy_mm(414.0, 102.0) == (452.8, 102.0)

    # Reverse: Robot base frame -> Pan-axis tape measurements (-38.8 mm in X)
    assert base_to_pan_xy_mm(452.8, -240.0) == (414.0, -240.0)
    assert base_to_pan_xy_mm(135.8, -240.0) == (97.0, -240.0)
    assert base_to_pan_xy_mm(135.8, 102.0) == (97.0, 102.0)
    assert base_to_pan_xy_mm(452.8, 102.0) == (414.0, 102.0)


def test_setup_wizard_interactive_prompts_convert_pan_to_base(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify interactive wizard converts pan tape measurements to base-frame corners."""
    import sys

    # Simulate interactive TTY
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

    inputs = iter(
        [
            "414, -240",  # top_left (from pan axis)
            "97, -240",  # top_right (from pan axis)
            "97, 102",  # bottom_right (from pan axis)
            "414, 102",  # bottom_left (from pan axis)
            "40",  # marker_size
        ]
    )
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    cfg_path = tmp_path / "ws_interactive.json"
    out_dir = tmp_path / "markers"

    updated = setup_workspace_interactive(
        config_path=cfg_path,
        output_dir=out_dir,
        interactive=True,
    )

    corners = updated["workspace_corners_mm"]
    assert corners["top_left"] == {"x": 452.8, "y": -240.0}
    assert corners["top_right"] == {"x": 135.8, "y": -240.0}
    assert corners["bottom_right"] == {"x": 135.8, "y": 102.0}
    assert corners["bottom_left"] == {"x": 452.8, "y": 102.0}
