"""Unit tests for the OpenCV multi-color salient object detector."""

import numpy as np

from robot.vision_detector import (
    detect_workspace_objects,
    refine_point_to_nearest_object,
)


def test_detect_objects_synthetic_red_block() -> None:
    """Verify detection of a synthetic red cube on neutral tabletop."""
    # Create 480x640 neutral table image
    frame = np.full((480, 640, 3), 180, dtype=np.uint8)

    # Draw red square at y in [200, 240], x in [300, 340]
    # In RGB: Red has high R, low G, B
    frame[200:240, 300:340] = [220, 20, 20]

    detected = detect_workspace_objects(frame, min_area=50)
    assert len(detected) == 1
    obj = detected[0]
    assert obj["color"] == "red"
    assert obj["area"] >= 1500

    # Bottom contact should be at y=240, x=320
    # Normalized: y = (240 / 480) * 1000 = 500, x = (320 / 640) * 1000 = 500
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
    # Red cube at bottom contact (y=300, x=320) -> normalized (y=625, x=500)
    frame[260:300, 300:340] = [220, 20, 20]

    # Candidate point slightly offset: [580, 470]
    candidate = [580.0, 470.0]
    refined, match = refine_point_to_nearest_object(candidate, frame, max_dist_1000=200.0)

    assert match is not None
    assert match["color"] == "red"
    assert abs(refined[0] - 625.0) < 5.0
    assert abs(refined[1] - 500.0) < 5.0


def test_refine_point_no_match_far_away() -> None:
    """Verify original point is retained if object is beyond max distance."""
    frame = np.full((480, 640, 3), 180, dtype=np.uint8)
    frame[400:440, 500:540] = [220, 20, 20]

    # Two objects so fallback does not trigger
    frame[50:80, 50:80] = [20, 50, 220]

    candidate = [100.0, 500.0]
    refined, match = refine_point_to_nearest_object(candidate, frame, max_dist_1000=100.0)
    assert match is None
    assert refined == candidate


def test_detect_objects_empty_frame() -> None:
    """Verify empty and black frames return empty lists without error."""
    assert detect_workspace_objects(None) == []
    empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    assert detect_workspace_objects(empty_frame) == []
