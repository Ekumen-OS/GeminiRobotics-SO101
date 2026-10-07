"""Tests that production callers supply the workspace calibrator to the detector.

Phase 2 made the calibrator an explicit parameter of ``robot/vision_detector.py``. Without it the
detector scans the whole frame instead of the ArUco-bounded workspace, so every production path
must pass one through.
"""

from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from agent.tools import execute_tool_call
from robot.perception import compute_visual_alignment, verify_grasp_elevation
from tests.conftest import DummyToolCall, MockRobot
from tests.test_e2e_perception_guardrails import make_synthetic_red_cube_frame


def _red_cube_jpeg() -> bytes:
    frame = make_synthetic_red_cube_frame(cube_y_1000=760.0, cube_x_1000=484.0)
    _, enc = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    return enc.tobytes()


@pytest.mark.asyncio
async def test_move_to_spatial_point_passes_calibrator_to_refinement_and_perception() -> None:
    """Both detector entry points used by move_to_spatial_point receive the shared calibrator."""
    calib = MagicMock()
    calib.is_calibrated = True
    calib.mode = "real"
    calib.gemini_to_robot_xyz.return_value = np.array([0.2, 0.05, 0.04])
    call = DummyToolCall(
        "move_to_spatial_point",
        {"point": [755.0, 480.0], "z_height_cm": 4.0, "approach": "top_down"},
    )
    with (
        patch("agent.tools.get_calibrator", return_value=calib),
        patch(
            "agent.tools.refine_point_to_nearest_object",
            return_value=([760.0, 484.0], {"color": "red"}),
        ) as refine,
        patch("agent.tools.compute_visual_alignment", return_value={"is_aligned": True}) as cva,
    ):
        res = await execute_tool_call(call, robot=MockRobot(), get_latest_jpeg_fn=_red_cube_jpeg)

    assert res.get("status") == "success"
    assert refine.call_args.kwargs.get("calibrator") is calib
    assert cva.call_args.kwargs.get("calibrator") is calib


def test_compute_visual_alignment_forwards_calibrator() -> None:
    """compute_visual_alignment hands the calibrator to the point refiner."""
    calib = MagicMock()
    frame = make_synthetic_red_cube_frame(cube_y_1000=760.0, cube_x_1000=484.0)
    with patch(
        "robot.perception.refine_point_to_nearest_object", return_value=([760.0, 484.0], None)
    ) as refine:
        compute_visual_alignment(frame, [755.0, 480.0], calibrator=calib)
    assert refine.call_args.kwargs.get("calibrator") is calib


def test_verify_grasp_elevation_forwards_calibrator() -> None:
    """verify_grasp_elevation hands the calibrator to both detector calls."""
    calib = MagicMock()
    frame = make_synthetic_red_cube_frame(cube_y_1000=760.0, cube_x_1000=484.0)
    with patch("robot.perception.detect_workspace_objects", return_value=[]) as detect:
        verify_grasp_elevation(frame, frame, target_point_1000=[760.0, 484.0], calibrator=calib)
    assert detect.call_count == 2
    assert all(c.kwargs.get("calibrator") is calib for c in detect.call_args_list)
