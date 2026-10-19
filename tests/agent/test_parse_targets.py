"""Unit and fuzz tests for target coordinate parsing from LLM responses."""

import random
import string

from agent.vla_controller import parse_targets_from_text


def test_parse_targets_valid_points_and_boxes() -> None:
    """Verify extraction of single and multiple 2D points and 4D bounding boxes from text."""
    # Single 2D point with label
    text_2d = "I will reach for the RED_BLOCK at [760, 484] now."
    targets_2d = parse_targets_from_text(text_2d)
    assert len(targets_2d) == 1
    assert targets_2d[0]["point"] == (760.0, 484.0)
    assert "RED_BLOCK" in targets_2d[0]["label"]

    # Multiple 2D points with labels
    multi_text = (
        "I observe 2 objects on the table: red cube at [320, 450] and blue cylinder at [600, 750]."
    )
    targets_multi = parse_targets_from_text(multi_text)
    assert len(targets_multi) == 2
    assert targets_multi[0]["point"] == (320.0, 450.0)
    assert "RED CUBE" in targets_multi[0]["label"]
    assert targets_multi[1]["point"] == (600.0, 750.0)
    assert "BLUE CYLINDER" in targets_multi[1]["label"]

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


def test_parse_targets_signed_delta_tokens() -> None:
    """Verify coordinate parser does not misinterpret signed delta tokens as absolute targets."""
    text_with_deltas = (
        "Gripper is misaligned by dy=+35, dx=-20 relative to target. "
        "Correcting to [760, 484] before descent."
    )
    targets = parse_targets_from_text(text_with_deltas)

    # Must extract the valid absolute target [760, 484], NOT [35, -20]
    points = [t["point"] for t in targets]
    assert (760.0, 484.0) in points
    assert (35.0, -20.0) not in points


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
