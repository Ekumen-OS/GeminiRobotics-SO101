"""Characterization tests for stream overlay toggles and model-target blackboard."""

import numpy as np
import pytest

import vision as stream_mod


@pytest.fixture(autouse=True)
def reset_stream_overlay_state():
    """Reset stream overlay toggles and model targets before and after each test."""
    stream_mod.set_target_overlay_enabled(True)
    stream_mod.set_calibration_overlay(False)
    stream_mod.set_bounding_boxes_overlay(False)
    stream_mod.set_grid_overlay(False)
    stream_mod.set_crosshair_overlay(False)
    stream_mod.clear_model_targets()
    yield
    stream_mod.set_target_overlay_enabled(True)
    stream_mod.set_calibration_overlay(False)
    stream_mod.set_bounding_boxes_overlay(False)
    stream_mod.set_grid_overlay(False)
    stream_mod.set_crosshair_overlay(False)
    stream_mod.clear_model_targets()


def test_overlay_toggles_calibration() -> None:
    """Verify calibration overlay toggle state transitions."""
    assert not stream_mod.is_calibration_overlay_enabled()
    stream_mod.set_calibration_overlay(True)
    assert stream_mod.is_calibration_overlay_enabled()
    stream_mod.set_calibration_overlay(False)
    assert not stream_mod.is_calibration_overlay_enabled()


def test_overlay_toggles_bounding_boxes() -> None:
    """Verify bounding boxes overlay toggle state transitions."""
    assert not stream_mod.is_bounding_boxes_overlay_enabled()
    stream_mod.set_bounding_boxes_overlay(True)
    assert stream_mod.is_bounding_boxes_overlay_enabled()
    stream_mod.set_bounding_boxes_overlay(False)
    assert not stream_mod.is_bounding_boxes_overlay_enabled()


def test_overlay_toggles_grid() -> None:
    """Verify grid overlay toggle state transitions."""
    assert not stream_mod.is_grid_overlay_enabled()
    stream_mod.set_grid_overlay(True)
    assert stream_mod.is_grid_overlay_enabled()
    stream_mod.set_grid_overlay(False)
    assert not stream_mod.is_grid_overlay_enabled()


def test_overlay_toggles_crosshair() -> None:
    """Verify crosshair overlay toggle state transitions."""
    assert not stream_mod.is_crosshair_overlay_enabled()
    stream_mod.set_crosshair_overlay(True)
    assert stream_mod.is_crosshair_overlay_enabled()
    stream_mod.set_crosshair_overlay(False)
    assert not stream_mod.is_crosshair_overlay_enabled()


def test_overlay_toggles_target() -> None:
    """Verify model target overlay toggle state transitions."""
    assert stream_mod.is_target_overlay_enabled()
    stream_mod.set_target_overlay_enabled(False)
    assert not stream_mod.is_target_overlay_enabled()
    stream_mod.set_target_overlay_enabled(True)
    assert stream_mod.is_target_overlay_enabled()


def test_model_targets_set_and_get() -> None:
    """Verify setting structured model targets dicts and retrieving them."""
    targets = [
        {
            "point": (450.0, 550.0),
            "label": "red block",
            "box_2d": [400.0, 500.0, 500.0, 600.0],
            "color": (255, 0, 0),
        },
        {
            "box_2d": [100.0, 100.0, 300.0, 300.0],
            "label": "box without explicit point",
        },
        [200.0, 300.0],
    ]
    stream_mod.set_model_targets(targets)
    retrieved = stream_mod.get_model_targets()
    assert len(retrieved) == 3
    assert retrieved[0]["point"] == (450.0, 550.0)
    assert retrieved[0]["label"] == "red block"
    assert retrieved[0]["box_2d"] == [400.0, 500.0, 500.0, 600.0]

    # Target 1: derived point from center of box_2d
    assert retrieved[1]["point"] == (200.0, 200.0)
    assert retrieved[1]["label"] == "box without explicit point"

    # Target 2: point from coordinates list
    assert retrieved[2]["point"] == (200.0, 300.0)
    assert retrieved[2]["label"] is None

    # Primary target point
    pt, label = stream_mod.get_model_target_point()
    assert pt == (450.0, 550.0)
    assert label == "red block"


def test_model_targets_clear() -> None:
    """Verify clearing model targets resets blackboard and primary point."""
    stream_mod.set_model_targets([{"point": (100.0, 100.0), "label": "test"}])
    assert len(stream_mod.get_model_targets()) == 1

    stream_mod.clear_model_targets()
    assert stream_mod.get_model_targets() == []
    assert stream_mod.get_model_target_point() == (None, None)


def test_model_targets_add_incremental() -> None:
    """Verify adding model targets incrementally preserves primary target."""
    assert stream_mod.get_model_targets() == []
    stream_mod.add_model_target(
        point=(300.0, 400.0),
        label="first",
        box_2d=[250.0, 350.0, 350.0, 450.0],
        color=(0, 255, 0),
    )
    assert len(stream_mod.get_model_targets()) == 1
    pt, lbl = stream_mod.get_model_target_point()
    assert pt == (300.0, 400.0)
    assert lbl == "first"

    stream_mod.add_model_target(point=(600.0, 700.0), label="second")
    assert len(stream_mod.get_model_targets()) == 2
    # Primary remains first target
    pt2, lbl2 = stream_mod.get_model_target_point()
    assert pt2 == (300.0, 400.0)
    assert lbl2 == "first"

    # Invalid additions should be no-ops
    stream_mod.add_model_target(point=None)  # type: ignore[arg-type]
    stream_mod.add_model_target(point=[1.0])  # type: ignore[arg-type]
    assert len(stream_mod.get_model_targets()) == 2


def test_set_and_clear_model_target_point() -> None:
    """Verify setting and clearing single target point."""
    stream_mod.set_model_target_point([350.0, 450.0], label="single point")
    pt, lbl = stream_mod.get_model_target_point()
    assert pt == (350.0, 450.0)
    assert lbl == "single point"

    stream_mod.clear_model_target_point()
    assert stream_mod.get_model_target_point() == (None, None)

    # Calling set_model_target_point with None clears
    stream_mod.set_model_target_point([100.0, 100.0])
    assert stream_mod.get_model_target_point()[0] == (100.0, 100.0)
    stream_mod.set_model_target_point(None)
    assert stream_mod.get_model_target_point() == (None, None)


def test_draw_model_target_overlay() -> None:
    """Verify drawing model target overlay on an image canvas."""
    frame = np.full((480, 640, 3), 128, dtype=np.uint8)

    # When target overlay is disabled, frame is untouched
    stream_mod.set_target_overlay_enabled(False)
    stream_mod.set_model_targets([{"point": (250.0, 250.0), "label": "test"}])
    res = stream_mod.draw_model_target_overlay(frame)
    assert np.array_equal(res, frame)

    # When target overlay is enabled, overlay is rendered onto frame
    stream_mod.set_target_overlay_enabled(True)
    res2 = stream_mod.draw_model_target_overlay(frame.copy())
    # Frame should be modified by drawing markers/labels
    assert not np.array_equal(res2, frame)
