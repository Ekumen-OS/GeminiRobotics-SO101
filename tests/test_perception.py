"""Unit and boundary tests for the perception metrics engine and vision detector."""

import random
import string
from typing import Any

import numpy as np
import pytest

from agent.vla_controller import parse_targets_from_text
from robot.perception import compute_visual_alignment
from robot.vision_detector import (
    detect_workspace_objects,
    refine_point_to_nearest_object,
)

# ==============================================================================
# 1. Computer Vision Object Detection & Snapping
# ==============================================================================


def test_detect_objects_synthetic_red_block(red_cube_frame: np.ndarray) -> None:
    """Verify detection of a synthetic red block on tabletop."""
    detected = detect_workspace_objects(red_cube_frame, min_area=50)
    assert len(detected) == 1
    obj = detected[0]
    assert obj["color"] == "red"
    assert obj["area"] >= 1200

    # Bottom contact: y in [200, 240], x in [300, 340]
    # Expected normalized: y = 240/480*1000 = 500, x = 320/640*1000 = 500
    norm_y, norm_x = obj["bottom_contact_1000"]
    assert abs(norm_y - 500.0) < 5.0
    assert abs(norm_x - 500.0) < 5.0


def test_detect_objects_multi_color(neutral_frame: np.ndarray) -> None:
    """Verify detection of multiple colored blocks (red, blue, green, yellow)."""
    frame = neutral_frame.copy()
    frame[100:130, 100:130] = [220, 20, 20]  # Red
    frame[100:130, 400:430] = [20, 50, 220]  # Blue
    frame[300:330, 100:130] = [20, 200, 30]  # Green
    frame[300:330, 400:430] = [240, 220, 20]  # Yellow

    detected = detect_workspace_objects(frame, min_area=50)
    detected_colors = {d["color"] for d in detected}
    assert {"red", "blue", "green", "yellow"}.issubset(detected_colors)


def test_workspace_polygon_filter(neutral_frame: np.ndarray) -> None:
    """Verify objects outside the workspace polygon are ignored."""
    frame = neutral_frame.copy()
    # Inside polygon: red block
    frame[240:270, 240:270] = [220, 20, 20]
    # Outside polygon: blue block
    frame[40:70, 40:70] = [20, 50, 220]

    poly = np.array([[150, 150], [450, 150], [450, 450], [150, 450]], dtype=np.float32)
    detected = detect_workspace_objects(frame, workspace_polygon_px=poly, min_area=50)
    assert len(detected) == 1
    assert detected[0]["color"] == "red"


def test_detect_objects_empty_or_none_frame() -> None:
    """Verify empty and None frames return empty lists gracefully."""
    assert detect_workspace_objects(None) == []
    empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    assert detect_workspace_objects(empty_frame) == []


def test_refine_point_to_nearest_object(neutral_frame: np.ndarray) -> None:
    """Verify point snapping to nearest detected object contact center."""
    frame = neutral_frame.copy()
    frame[260:300, 300:340] = [220, 20, 20]  # contact at y=300 (norm 625), x=320 (norm 500)

    candidate = [580.0, 470.0]
    refined, match = refine_point_to_nearest_object(candidate, frame, max_dist_1000=200.0)

    assert match is not None
    assert match["color"] == "red"
    assert abs(refined[0] - 625.0) < 5.0
    assert abs(refined[1] - 500.0) < 5.0


def test_refine_point_no_match_fallback(neutral_frame: np.ndarray) -> None:
    """Verify original point is retained if object is beyond max distance."""
    frame = neutral_frame.copy()
    frame[400:440, 500:540] = [220, 20, 20]
    frame[50:80, 50:80] = [20, 50, 220]

    candidate = [100.0, 500.0]
    refined, match = refine_point_to_nearest_object(candidate, frame, max_dist_1000=100.0)
    assert match is None
    assert refined == candidate


# ==============================================================================
# 2. Visual Alignment Metrics Computation
# ==============================================================================


def test_compute_visual_alignment_math() -> None:
    """Verify exact formula: dy = target_y - grip_y, dx = target_x - grip_x."""
    metrics = compute_visual_alignment(
        frame_rgb=None,
        target_point_1000=[750.0, 520.0],
        gripper_point_1000=[700.0, 500.0],
        tolerance=25.0,
    )

    assert metrics["measured_offset"]["dy"] == 50.0
    assert metrics["measured_offset"]["dx"] == 20.0
    assert np.isclose(metrics["measured_offset"]["distance"], np.hypot(50.0, 20.0))
    assert metrics["is_aligned"] is False


@pytest.mark.parametrize(
    "dy,dx,expected_aligned",
    [
        (0.0, 0.0, True),
        (25.0, 25.0, True),
        (-25.0, -25.0, True),
        (25.0, 0.0, True),
        (0.0, 25.0, True),
        (25.1, 0.0, False),
        (0.0, 25.1, False),
        (-25.1, 0.0, False),
        (0.0, -25.1, False),
        (25.0, 25.1, False),
        (100.0, 100.0, False),
    ],
)
def test_compute_visual_alignment_boundary_matrix(
    dy: float, dx: float, expected_aligned: bool
) -> None:
    """Verify strict tolerance boundaries at |dy| <= 25.0 and |dx| <= 25.0."""
    grip = [500.0, 500.0]
    target = [500.0 + dy, 500.0 + dx]
    metrics = compute_visual_alignment(
        frame_rgb=None,
        target_point_1000=target,
        gripper_point_1000=grip,
        tolerance=25.0,
    )
    assert metrics["is_aligned"] is expected_aligned


def test_compute_visual_alignment_robot_state_fallback() -> None:
    """Verify gripper point resolution from robot_state dictionary."""
    state: dict[str, Any] = {"gripper_point_1000": [600.0, 450.0]}
    metrics = compute_visual_alignment(
        frame_rgb=None,
        target_point_1000=[610.0, 455.0],
        gripper_point_1000=None,
        robot_state=state,
        tolerance=25.0,
    )
    assert metrics["gripper_point_1000"] == [600.0, 450.0]
    assert metrics["is_aligned"] is True


# ==============================================================================
# 4. Target Coordinate Parsing from LLM Responses
# ==============================================================================


def test_parse_targets_valid_points_and_boxes() -> None:
    """Verify extraction of 2D points and 4D bounding boxes from text."""
    # 2D point with label
    text_2d = "I will reach for the RED_BLOCK at [760, 484] now."
    targets_2d = parse_targets_from_text(text_2d)
    assert len(targets_2d) == 1
    assert targets_2d[0]["point"] == (760.0, 484.0)
    assert "RED_BLOCK" in targets_2d[0]["label"]

    # 4D bounding box
    text_4d = "Detected TARGET box: [200, 300, 240, 340]."
    targets_4d = parse_targets_from_text(text_4d)
    assert len(targets_4d) == 1
    assert targets_4d[0]["point"] == (220.0, 320.0)  # Box center
    assert targets_4d[0]["box_2d"] == [200.0, 300.0, 240.0, 340.0]


def test_parse_targets_out_of_bounds_filtering() -> None:
    """Verify that coordinates outside [0, 1000] are rejected."""
    text_oob = "Coordinates out of bounds: [1200, 500], [-50, 200], [500, 1005]"
    targets = parse_targets_from_text(text_oob)
    assert len(targets) == 0


def test_parse_targets_fuzzer_resilience() -> None:
    """Stress test coordinate parser against random noise and malformed text."""
    random.seed(42)
    corpus = string.ascii_letters + string.digits + string.punctuation + " \t\n\r"

    for _ in range(200):
        length = random.randint(10, 200)
        noisy_text = "".join(random.choice(corpus) for _ in range(length))
        # Must execute without throwing any exception
        results = parse_targets_from_text(noisy_text)
        assert isinstance(results, list)
        for t in results:
            assert "point" in t
            assert len(t["point"]) == 2
            assert 0.0 <= t["point"][0] <= 1000.0
            assert 0.0 <= t["point"][1] <= 1000.0
