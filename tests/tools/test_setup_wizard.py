"""Unit tests for workspace setup wizard CLI utility."""

import json
import tempfile
from pathlib import Path

from tools.setup_wizard import setup_workspace_interactive


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
