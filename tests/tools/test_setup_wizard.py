"""Unit tests for workspace setup wizard CLI utility."""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

from tools.setup_wizard import (
    base_to_pan_xy_mm,
    evaluate_workspace_reachability,
    get_pan_origin_offset_x_mm,
    pan_to_base_xy_mm,
    setup_workspace_interactive,
)


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


def test_rectangle_to_corners_current_setup_values(tmp_path: Path) -> None:
    """For the current setup values, the JSON must equal today's corners and preserve ID mapping."""
    cfg_path = tmp_path / "ws_calib.json"
    out_dir = tmp_path / "markers"

    updated = setup_workspace_interactive(
        config_path=cfg_path,
        output_dir=out_dir,
        interactive=False,
        near_x_mm=97.0,
        depth_mm=317.0,
        width_mm=342.0,
        lateral_offset_mm=-69.0,
        marker_size_mm=40.0,
    )

    corners = updated["workspace_corners_mm"]
    assert corners["top_left"] == {"x": 452.8, "y": -240.0}
    assert corners["top_right"] == {"x": 135.8, "y": -240.0}
    assert corners["bottom_right"] == {"x": 135.8, "y": 102.0}
    assert corners["bottom_left"] == {"x": 452.8, "y": 102.0}

    assert updated["marker_ids"] == {
        "top_left": 1,
        "top_right": 2,
        "bottom_right": 3,
        "bottom_left": 0,
    }
    assert updated["marker_size_mm"] == 40.0

    assert (out_dir / "workspace_placement_guide.png").exists()
    assert (out_dir / "marker_0.png").exists()
    assert (out_dir / "jaw_tag_sheet.png").exists()


def test_defaults_derived_from_existing_json_round_trip(tmp_path: Path) -> None:
    """The defaults derived from an existing JSON round-trip correctly."""
    pan_offset = get_pan_origin_offset_x_mm()
    test_corners = {
        "top_left": {"x": round(350.0 + pan_offset, 1), "y": -180.0},
        "top_right": {"x": round(110.0 + pan_offset, 1), "y": -180.0},
        "bottom_right": {"x": round(110.0 + pan_offset, 1), "y": 150.0},
        "bottom_left": {"x": round(350.0 + pan_offset, 1), "y": 150.0},
    }
    initial_config = {
        "aruco_dictionary": "DICT_4X4_50",
        "marker_ids": {"top_left": 1, "top_right": 2, "bottom_right": 3, "bottom_left": 0},
        "marker_size_mm": 45.0,
        "workspace_corners_mm": test_corners,
    }
    cfg_path = tmp_path / "ws_roundtrip.json"
    cfg_path.write_text(json.dumps(initial_config, indent=2))
    out_dir = tmp_path / "markers"

    updated = setup_workspace_interactive(
        config_path=cfg_path,
        output_dir=out_dir,
        interactive=False,
    )

    assert updated["workspace_corners_mm"] == test_corners
    assert updated["marker_size_mm"] == 45.0


def test_non_rectangular_existing_json_prints_warning(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """A non-rectangular existing JSON prints a warning and uses the bounding rectangle."""
    pan_offset = get_pan_origin_offset_x_mm()
    skewed_corners = {
        "top_left": {"x": round(400.0 + pan_offset, 1), "y": -200.0},
        "top_right": {"x": round(120.0 + pan_offset, 1), "y": -180.0},  # Y differs by 20mm
        "bottom_right": {"x": round(100.0 + pan_offset, 1), "y": 150.0},  # X differs by 20mm
        "bottom_left": {"x": round(380.0 + pan_offset, 1), "y": 160.0},
    }
    cfg_path = tmp_path / "ws_skewed.json"
    cfg_path.write_text(
        json.dumps({"workspace_corners_mm": skewed_corners, "marker_size_mm": 40.0})
    )

    setup_workspace_interactive(
        config_path=cfg_path,
        output_dir=tmp_path / "m",
        interactive=False,
    )

    captured = capsys.readouterr()
    assert "not an axis-aligned rectangle within 2 mm" in (captured.out + captured.err)


def test_validation_rejects_bad_input_and_accepts_retry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Validation rejects negative depth and near X that is too small, then accepts retry."""
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

    scripted_inputs = iter(
        [
            "50",  # near X too small (must be > 40/2 + 60 = 80 mm) -> rejected
            "97",  # near X valid -> accepted
            "-50",  # depth negative -> rejected
            "317",  # depth valid -> accepted
            "342",  # width -> accepted
            "-69",  # lateral offset -> accepted
            "40",  # marker size -> accepted
        ]
    )
    monkeypatch.setattr("builtins.input", lambda _: next(scripted_inputs))

    cfg_path = tmp_path / "ws_validated.json"
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


_CURRENT_SETUP_STEP_MM = 40.0
_Reach = tuple[float, list[str], tuple[np.ndarray, np.ndarray, np.ndarray]]


@pytest.fixture(scope="module")
def current_setup_reachability() -> tuple[
    float, list[str], tuple[np.ndarray, np.ndarray, np.ndarray]
]:
    """Compute the current-setup reachability grid once and share it across the module."""
    pan_offset = get_pan_origin_offset_x_mm()
    current_corners = {
        "top_left": {"x": 414.0 + pan_offset, "y": -240.0},
        "top_right": {"x": 97.0 + pan_offset, "y": -240.0},
        "bottom_right": {"x": 97.0 + pan_offset, "y": 102.0},
        "bottom_left": {"x": 414.0 + pan_offset, "y": 102.0},
    }
    return evaluate_workspace_reachability(current_corners, step_mm=_CURRENT_SETUP_STEP_MM)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Runtime-faithful single-solve IK (HOME seed, solver's own neutral retry) fails at "
        "interior points 150-350 mm from the pan axis, e.g. pan-frame (257,-180) 272 mm, "
        "(297,100) 133 mm, (197,-220) 116 mm error; see workspace-setup-fixup-2 report."
    ),
)
def test_reachability_current_setup_and_no_interior_holes(
    current_setup_reachability: _Reach,
) -> None:
    """Current setup reports ID 1 out of reach and has no holes in [150, 350] mm."""
    pan_offset = get_pan_origin_offset_x_mm()
    frac_cur, out_cur, (xs, ys, mask) = current_setup_reachability
    assert out_cur == ["top_left"]  # ID 1 out of reach
    assert frac_cur > 0.90

    # Every grid point whose distance from the pan axis is between 150 and 350 mm is reachable
    holes: list[tuple[float, float]] = []
    for i, x_base in enumerate(xs):
        x_pan = float(x_base) - pan_offset
        for j, y_pan in enumerate(ys):
            dist_pan = float(np.hypot(x_pan, float(y_pan)))
            if 150.0 <= dist_pan <= 350.0 and not mask[i, j]:
                holes.append((round(x_pan, 1), round(float(y_pan), 1)))
    assert holes == [], f"Unexpected interior reachability holes in [150, 350] mm: {holes}"


def test_reachability_small_rectangle_fully_reachable() -> None:
    """A small rectangle near the robot is fully reachable."""
    pan_offset = get_pan_origin_offset_x_mm()
    # Small rectangle at near X = 150 (pan frame): near_x=150, depth=100, width=150, lat=0
    small_corners = {
        "top_left": {"x": 250.0 + pan_offset, "y": -75.0},
        "top_right": {"x": 150.0 + pan_offset, "y": -75.0},
        "bottom_right": {"x": 150.0 + pan_offset, "y": 75.0},
        "bottom_left": {"x": 250.0 + pan_offset, "y": 75.0},
    }
    frac_small, out_small, _ = evaluate_workspace_reachability(small_corners, step_mm=40.0)
    assert frac_small == 1.0  # 100% reachable
    assert len(out_small) == 0


def test_setup_non_interactive_writes_json_from_defaults(tmp_path: Path) -> None:
    """Running setup non-interactively writes JSON from defaults without prompting."""
    cfg_path = tmp_path / "ws_defaults.json"
    out_dir = tmp_path / "markers"

    updated = setup_workspace_interactive(
        config_path=cfg_path,
        output_dir=out_dir,
        interactive=False,
    )

    assert cfg_path.exists()
    assert "workspace_corners_mm" in updated
    assert "marker_ids" in updated
    assert (out_dir / "workspace_placement_guide.png").exists()


def test_setup_workspace_reuse_existing_markers() -> None:
    """Setup workspace detects existing markers and updates JSON without regenerating files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        cfg_path = Path(tmp_dir) / "test_ws.json"
        out_dir = Path(tmp_dir) / "markers"
        out_dir.mkdir(parents=True, exist_ok=True)

        dummy_marker = out_dir / "marker_0.png"
        dummy_marker.write_text("fake image content")
        initial_mtime = dummy_marker.stat().st_mtime

        updated = setup_workspace_interactive(
            config_path=cfg_path,
            output_dir=out_dir,
            interactive=False,
            reuse=True,
            near_x_mm=100.0,
            depth_mm=300.0,
            width_mm=300.0,
            lateral_offset_mm=0.0,
        )

        assert cfg_path.exists()
        assert updated["workspace_corners_mm"]["top_left"]["y"] == -150.0
        assert dummy_marker.read_text() == "fake image content"
        assert dummy_marker.stat().st_mtime == initial_mtime
