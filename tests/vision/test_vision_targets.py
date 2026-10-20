"""Tests for stream overlay toggles, workspace overlay, and model target blackboard."""

import hashlib
import tempfile
from pathlib import Path

import numpy as np
import pytest

from robot.calibration import ArucoWorkspaceCalibrator
from simulation.sim_calibration import init_sim_workspace_calibration
from vision.overlays import (
    draw_workspace_overlay,
    is_calibration_overlay_enabled,
    is_target_overlay_enabled,
    set_calibration_overlay,
    set_target_overlay_enabled,
)
from vision.targets import (
    clear_model_targets,
    draw_model_target_overlay,
    get_model_targets,
    set_model_targets,
)


@pytest.fixture(autouse=True)
def reset_stream_overlay_state():
    """Reset stream overlay toggles and model targets before and after each test."""
    set_target_overlay_enabled(True)
    set_calibration_overlay(False)
    clear_model_targets()
    yield
    set_target_overlay_enabled(True)
    set_calibration_overlay(False)
    clear_model_targets()


def test_overlay_toggles_calibration() -> None:
    """Verify calibration overlay toggle state transitions."""
    assert not is_calibration_overlay_enabled()
    set_calibration_overlay(True)
    assert is_calibration_overlay_enabled()
    set_calibration_overlay(False)
    assert not is_calibration_overlay_enabled()


def test_overlay_toggles_target() -> None:
    """Verify model target overlay toggle state transitions."""
    assert is_target_overlay_enabled()
    set_target_overlay_enabled(False)
    assert not is_target_overlay_enabled()
    set_target_overlay_enabled(True)
    assert is_target_overlay_enabled()


def test_model_targets_set_and_get() -> None:
    """Verify setting structured model targets dicts and retrieving them."""
    targets = [
        {
            "point": (450.0, 550.0),
            "label": "red block",
            "box_2d": [400.0, 500.0, 500.0, 600.0],
            "color": (255, 0, 0),
            "active": True,
        },
        {
            "box_2d": [100.0, 200.0, 250.0, 350.0],
            "label": "box without explicit point",
        },
        [200.0, 300.0],
    ]
    set_model_targets(targets)
    retrieved = get_model_targets()
    assert len(retrieved) == 3
    assert retrieved[0]["point"] == (450.0, 550.0)
    assert retrieved[0]["label"] == "red block"
    assert retrieved[0]["box_2d"] == [400.0, 500.0, 500.0, 600.0]
    assert retrieved[0]["active"] is True

    # Target 1: derived point from center of box_2d
    assert retrieved[1]["point"] == (175.0, 275.0)
    assert retrieved[1]["label"] == "box without explicit point"
    assert retrieved[1]["box_2d"] == [100.0, 200.0, 250.0, 350.0]
    assert retrieved[1]["active"] is False

    # Target 2: point from coordinates list
    assert retrieved[2]["point"] == (200.0, 300.0)
    assert retrieved[2]["label"] is None
    assert retrieved[2]["active"] is False


def test_model_targets_clear() -> None:
    """Verify clearing model targets resets blackboard."""
    set_model_targets([{"point": (100.0, 100.0), "label": "test"}])
    assert len(get_model_targets()) == 1

    clear_model_targets()
    assert get_model_targets() == []


def test_draw_model_target_overlay() -> None:
    """Verify drawing single and multiple model target overlays on an image canvas."""
    frame = np.full((480, 640, 3), 128, dtype=np.uint8)

    # When target overlay is disabled, frame is untouched
    set_target_overlay_enabled(False)
    set_model_targets([{"point": (250.0, 250.0), "label": "test"}])
    res = draw_model_target_overlay(frame)
    assert np.array_equal(res, frame)

    # When target overlay is enabled, single and multi-target overlays are rendered onto frame
    set_target_overlay_enabled(True)
    set_model_targets(
        [
            {"point": (300.0, 400.0), "label": "RED CUBE [300, 400]"},
            {"point": (600.0, 700.0), "label": "BLUE CYLINDER [600, 700]"},
            {"box_2d": [100.0, 200.0, 250.0, 350.0], "label": "GREEN BOWL"},
        ]
    )
    res2 = draw_model_target_overlay(frame.copy())
    assert res2.shape == (480, 640, 3)
    assert not np.array_equal(res2, frame)


def test_draw_model_target_overlay_no_box() -> None:
    """Verify that draw_model_target_overlay does not draw bounding boxes around objects."""
    frame = np.full((480, 640, 3), 128, dtype=np.uint8)
    set_target_overlay_enabled(True)
    box = [100.0, 100.0, 400.0, 400.0]
    center = (250.0, 250.0)
    set_model_targets([{"point": center, "box_2d": box, "active": True}])

    rendered = draw_model_target_overlay(frame)

    h, w = frame.shape[:2]
    cx = int(np.clip((center[1] / 1000.0) * w, 0, w - 1))
    cy = int(np.clip((center[0] / 1000.0) * h, 0, h - 1))
    ring_radius = max(4, int(round(7.0 * (w / 640.0))))
    max_marker_dist = 3.0 * ring_radius

    diff = np.any(rendered != frame, axis=-1)
    changed_ys, changed_xs = np.where(diff)
    assert len(changed_xs) > 0, "Overlay should render marker pixels"

    # All changed pixels without label must stay within 3x ring_radius of marker center
    dists = np.hypot(changed_xs - cx, changed_ys - cy)
    assert np.all(dists <= max_marker_dist)

    # Box boundary lines must have zero modifications
    ymin_px = int(box[0] / 1000.0 * h)
    xmin_px = int(box[1] / 1000.0 * w)
    ymax_px = int(box[2] / 1000.0 * h)
    xmax_px = int(box[3] / 1000.0 * w)
    assert not diff[ymin_px, xmin_px:xmax_px].any()
    assert not diff[ymax_px, xmin_px:xmax_px].any()
    assert not diff[ymin_px:ymax_px, xmin_px].any()
    assert not diff[ymin_px:ymax_px, xmax_px].any()


def test_draw_model_target_overlay_active_vs_stale_opacity() -> None:
    """Verify active target is drawn at full opacity while stale targets are alpha-blended."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    set_target_overlay_enabled(True)
    set_model_targets(
        [
            {"point": (200.0, 200.0), "color": (200, 200, 200), "active": True},
            {"point": (400.0, 400.0), "color": (200, 200, 200), "active": False},
        ]
    )
    rendered = draw_model_target_overlay(frame)

    h, w = frame.shape[:2]
    cx_act = int((200.0 / 1000.0) * w)
    cy_act = int((200.0 / 1000.0) * h)
    cx_stale = int((400.0 / 1000.0) * w)
    cy_stale = int((400.0 / 1000.0) * h)

    # Center dot of active target is at full colour (200)
    act_val = int(rendered[cy_act, cx_act, 0])
    assert act_val == 200

    # Center dot of stale target is blended at ~50% opacity
    stale_val = int(rendered[cy_stale, cx_stale, 0])
    assert 90 <= stale_val <= 110
    assert stale_val < act_val


def test_draw_model_target_overlay_destination_shape() -> None:
    """Verify destination target uses open square marker distinct from circular ring."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    set_target_overlay_enabled(True)

    set_model_targets([{"point": (500.0, 500.0), "label": "red cube", "active": True}])
    rendered_std = draw_model_target_overlay(frame.copy())

    set_model_targets([{"point": (500.0, 500.0), "label": "destination", "active": True}])
    rendered_dest = draw_model_target_overlay(frame.copy())

    h, w = frame.shape[:2]
    cx = int((500.0 / 1000.0) * w)
    cy = int((500.0 / 1000.0) * h)
    scale = w / 640.0
    ring_radius = max(4, int(round(7.0 * scale)))

    # Diagonal corner of square (cx + ring_radius, cy + ring_radius) is painted by square, not ring
    corner_y = cy + ring_radius
    corner_x = cx + ring_radius
    assert not np.array_equal(rendered_std[corner_y, corner_x], rendered_dest[corner_y, corner_x])
    assert rendered_dest[corner_y, corner_x].max() > 0
    assert rendered_std[corner_y, corner_x].max() == 0


def test_draw_model_target_overlay_preserves_pristine_frame() -> None:
    """Verify draw_model_target_overlay never mutates the pristine input frame array."""
    frame = np.full((480, 640, 3), 128, dtype=np.uint8)
    frame_backup = frame.copy()
    set_target_overlay_enabled(True)
    set_model_targets([{"point": (300.0, 300.0), "label": "cube", "active": True}])

    rendered = draw_model_target_overlay(frame)
    assert not np.shares_memory(rendered, frame)
    assert np.array_equal(frame, frame_backup)


def test_overlay_drawing() -> None:
    """Test draw_workspace_overlay returns an annotated image with matching dimensions."""
    calibrator = ArucoWorkspaceCalibrator()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    mock_detected = {
        0: np.array([[100, 50], [140, 50], [140, 90], [100, 90]], dtype=np.float32),
        1: np.array([[500, 50], [540, 50], [540, 90], [500, 90]], dtype=np.float32),
        2: np.array([[500, 400], [540, 400], [540, 440], [500, 440]], dtype=np.float32),
        3: np.array([[100, 400], [140, 400], [140, 440], [100, 440]], dtype=np.float32),
    }

    annotated = draw_workspace_overlay(
        calibrator=calibrator,
        frame_rgb=frame,
        detected_markers=mock_detected,
        show_markers=True,
    )

    assert annotated.shape == frame.shape
    assert annotated.dtype == frame.dtype
    assert not np.array_equal(annotated, frame)


def test_draw_workspace_overlay_byte_identical_regression() -> None:
    """Verify draw_workspace_overlay produces byte-identical output to original method."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        calibrator = init_sim_workspace_calibration(save_path=Path(tmp_dir) / "sim.npz")
        rng = np.random.default_rng(42)
        frame = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
        mock_detected = {
            0: np.array([[100, 400], [140, 400], [140, 440], [100, 440]], dtype=np.float32),
            1: np.array([[100, 50], [140, 50], [140, 90], [100, 90]], dtype=np.float32),
            2: np.array([[500, 50], [540, 50], [540, 90], [500, 90]], dtype=np.float32),
            3: np.array([[500, 400], [540, 400], [540, 440], [500, 440]], dtype=np.float32),
        }

        out_markers = draw_workspace_overlay(
            calibrator, frame.copy(), detected_markers=mock_detected, show_markers=True
        )
        out_no_markers = draw_workspace_overlay(
            calibrator, frame.copy(), detected_markers={}, show_markers=False
        )

        assert (
            hashlib.sha256(out_markers.tobytes()).hexdigest()
            == "50afa0cd2382098bb1e39b8806913f291fd85a916e8e092d0a7314870eda3595"
        )
        assert (
            hashlib.sha256(out_no_markers.tobytes()).hexdigest()
            == "696be2cf207bad65ed02d8b65f382b52a661aad0b6ec1a00f3caf969ab430c06"
        )
