"""Tests that production callers supply the workspace calibrator to the detector.

Phase 2 made the calibrator an explicit parameter of ``vision/detector.py``. Without it the
detector scans the whole frame instead of the ArUco-bounded workspace, so every production path
must pass one through.
"""

from unittest.mock import MagicMock, patch

import cv2
import pytest

from agent.tools import execute_tool_call
from tests.conftest import DummyToolCall, MockRobot
from tests.e2e.test_e2e_perception_guardrails import make_synthetic_red_cube_frame


def _red_cube_jpeg() -> bytes:
    frame = make_synthetic_red_cube_frame(cube_y_1000=760.0, cube_x_1000=484.0)
    _, enc = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    return enc.tobytes()


@pytest.mark.asyncio
async def test_move_to_spatial_point_passes_calibrator_to_refinement_and_snapshot_to_locator() -> (
    None
):
    """Lock detection receives the shared calibrator and the jaw locator its geometry snapshot."""
    from simulation.sim_calibration import init_sim_workspace_calibration
    from vision.calibrator import get_calibrator

    calib = get_calibrator(mode="sim")
    init_sim_workspace_calibration(calibrator=calib)
    call = DummyToolCall(
        "move_to_spatial_point",
        {"point": [755.0, 480.0], "z_height_cm": 4.0, "approach": "top_down"},
    )
    from vision.detector import refine_point_to_nearest_object as real_refine
    from vision.gripper_locator import locate_jaw_xy_mm as real_locate

    with (
        patch("vision.calibrator.get_calibrator", return_value=calib),
        patch("agent.tools.refine_point_to_nearest_object", side_effect=real_refine) as refine,
        patch("agent.tools.locate_jaw_xy_mm", side_effect=real_locate) as locate,
    ):
        res = await execute_tool_call(
            call, robot=MockRobot(), mode="sim", get_latest_jpeg_fn=_red_cube_jpeg
        )

    assert res.get("status") == "success"
    assert refine.call_args.kwargs.get("calibrator") is calib
    # The jaw locator receives the calibrator's geometry snapshot (one per tool call).
    assert locate.call_args.args[1] is calib.geometry_snapshot()


def test_vision_vote_forwards_calibrator_to_the_detector() -> None:
    """The post-lift vision vote hands the calibrator to the detector (workspace mask)."""
    from agent.guardrails import TargetLock
    from agent.pick_loop import vision_vote
    from simulation.sim_calibration import init_sim_workspace_calibration
    from vision.calibrator import get_calibrator

    sim_calib = get_calibrator(mode="sim")
    init_sim_workspace_calibration(calibrator=sim_calib)

    calib = MagicMock()
    calib.detect_markers.return_value = {0: 0, 1: 0, 2: 0, 3: 0}
    calib.config = {"marker_ids": {}}
    calib.geometry_snapshot.return_value.camera = sim_calib.geometry_snapshot().camera
    frame = make_synthetic_red_cube_frame(cube_y_1000=760.0, cube_x_1000=484.0)
    lock = TargetLock(1, "red", (240.0, 0.0), [760.0, 484.0], bbox_px=[300, 300, 25, 25])
    with patch("agent.pick_loop.detect_workspace_objects", return_value=[]) as detect:
        assert vision_vote(frame, lock, calib) == "not_seen_on_table"
    assert detect.call_args.kwargs.get("calibrator") is calib
