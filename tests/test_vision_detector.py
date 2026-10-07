"""Characterization tests for tabletop object detector and workspace polygon extraction."""

from unittest.mock import MagicMock

import numpy as np

from robot.vision_detector import (
    detect_workspace_objects,
    get_workspace_polygon_from_calibrator,
    refine_point_to_nearest_object,
)


def test_detect_objects_synthetic_red_block() -> None:
    """Verify detection of a synthetic red block on neutral tabletop."""
    frame = np.full((480, 640, 3), 180, dtype=np.uint8)
    frame[200:240, 300:340] = [220, 20, 20]

    detected = detect_workspace_objects(frame, min_area=50)
    assert len(detected) == 1
    obj = detected[0]
    assert obj["color"] == "red"
    assert obj["area"] >= 1500

    norm_y, norm_x = obj["bottom_contact_1000"]
    assert abs(norm_y - 500.0) < 5.0
    assert abs(norm_x - 500.0) < 5.0


def test_detect_objects_multi_color() -> None:
    """Verify detection of multiple colored blocks (red, blue, green, yellow)."""
    frame = np.full((480, 640, 3), 180, dtype=np.uint8)
    frame[100:130, 100:130] = [220, 20, 20]  # Red
    frame[100:130, 400:430] = [20, 50, 220]  # Blue
    frame[300:330, 100:130] = [20, 200, 30]  # Green
    frame[300:330, 400:430] = [240, 220, 20]  # Yellow

    detected = detect_workspace_objects(frame, min_area=50)
    detected_colors = {d["color"] for d in detected}
    assert "red" in detected_colors
    assert "blue" in detected_colors
    assert "green" in detected_colors
    assert "yellow" in detected_colors


def test_workspace_polygon_filter() -> None:
    """Verify objects outside the workspace polygon are ignored."""
    frame = np.full((480, 640, 3), 180, dtype=np.uint8)
    # Inside polygon: red block at (250, 250)
    frame[240:270, 240:270] = [220, 20, 20]
    # Outside polygon: blue block at (50, 50)
    frame[40:70, 40:70] = [20, 50, 220]

    poly = np.array([[150, 150], [450, 150], [450, 450], [150, 450]], dtype=np.float32)
    detected = detect_workspace_objects(frame, workspace_polygon_px=poly, min_area=50)
    assert len(detected) == 1
    assert detected[0]["color"] == "red"


def test_refine_point_to_nearest_object() -> None:
    """Verify point snapping to nearest detected object contact center."""
    frame = np.full((480, 640, 3), 180, dtype=np.uint8)
    frame[260:300, 300:340] = [220, 20, 20]

    candidate = [580.0, 470.0]
    refined, match = refine_point_to_nearest_object(candidate, frame, max_dist_1000=200.0)

    assert match is not None
    assert match["color"] == "red"
    assert abs(refined[0] - 625.0) < 5.0
    assert abs(refined[1] - 500.0) < 5.0


def test_refine_point_target_color_priority() -> None:
    """Verify snapping prioritizes the specified target color."""
    frame = np.full((480, 640, 3), 180, dtype=np.uint8)
    # Red cube closer to candidate
    frame[200:240, 200:240] = [220, 20, 20]
    # Blue cube slightly further away
    frame[250:290, 250:290] = [20, 50, 220]

    candidate = [500.0, 400.0]
    refined, match = refine_point_to_nearest_object(
        candidate, frame, target_color="blue", max_dist_1000=300.0
    )
    assert match is not None
    assert match["color"] == "blue"


def test_refine_point_no_match_far_away() -> None:
    """Verify original point is retained if object is beyond max distance."""
    frame = np.full((480, 640, 3), 180, dtype=np.uint8)
    frame[400:440, 500:540] = [220, 20, 20]
    frame[50:80, 50:80] = [20, 50, 220]

    candidate = [100.0, 500.0]
    refined, match = refine_point_to_nearest_object(candidate, frame, max_dist_1000=100.0)
    assert match is None
    assert refined == candidate


def test_detect_objects_empty_frame() -> None:
    """Verify empty and black frames return empty lists without error."""
    assert detect_workspace_objects(None) == []  # type: ignore[arg-type]
    empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    assert detect_workspace_objects(empty_frame) == []


def test_get_workspace_polygon_from_calibrator() -> None:
    """Verify extraction of 4-corner workspace polygon from calibrator markers."""
    frame = np.full((480, 640, 3), 180, dtype=np.uint8)

    mock_calib = MagicMock()
    mock_calib.config = {
        "marker_ids": {"top_left": 1, "top_right": 2, "bottom_right": 3, "bottom_left": 0}
    }
    mock_calib.detect_markers.return_value = {
        1: np.array([[100, 100], [120, 100], [120, 120], [100, 120]]),
        2: np.array([[500, 100], [520, 100], [520, 120], [500, 120]]),
        3: np.array([[500, 400], [520, 400], [520, 420], [500, 420]]),
        0: np.array([[100, 400], [120, 400], [120, 420], [100, 420]]),
    }

    poly = get_workspace_polygon_from_calibrator(frame, calibrator=mock_calib)
    assert poly is not None
    assert poly.shape == (4, 2)

    # When calibrator is None
    assert get_workspace_polygon_from_calibrator(frame, calibrator=None) is None
    assert get_workspace_polygon_from_calibrator(frame) is None

    # When a marker is missing
    mock_calib.detect_markers.return_value = {1: np.zeros((4, 2))}
    assert get_workspace_polygon_from_calibrator(frame, calibrator=mock_calib) is None
