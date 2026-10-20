"""Tests for table-relative height calculations and TABLE_Z_MM support."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from agent.guardrails import GuardrailState, Phase
from agent.motion import plan_approach_segments
from agent.tools import execute_tool_call
from robot.cartesian_ik import CartesianIK
from robot.constants import PRESET_POSES
from robot.settings import (
    Settings,
    robot_z_to_table_height_mm,
    set_settings,
    table_height_to_robot_z_m,
)
from tests.conftest import DummyToolCall, MockRobot
from vision.gripper_locator import locate_jaw_xy_mm


def test_get_table_z_mm() -> None:
    """Verify get_table_z_mm returns 0 in sim regardless of env, and table_z_mm in real."""
    custom_settings = Settings(table_z_mm=-24.0)
    assert custom_settings.get_table_z_mm("sim") == 0.0
    assert custom_settings.get_table_z_mm("real") == -24.0


def test_table_height_helpers_round_trip(restore_settings: None) -> None:
    """Verify table_height_to_robot_z_m and robot_z_to_table_height_mm round-trip."""
    set_settings(Settings.from_env(table_z_mm=-24.0))

    for mode in ("real", "sim"):
        height_m = 0.035
        robot_z_m = table_height_to_robot_z_m(height_m, mode=mode)
        recovered_height_mm = robot_z_to_table_height_mm(robot_z_m * 1000.0, mode=mode)
        assert recovered_height_mm == pytest.approx(height_m * 1000.0, abs=1e-5)


@pytest.mark.asyncio
async def test_spatial_tools_with_table_z(restore_settings: None) -> None:
    """Verify tool commands map to correct robot z coordinates with TABLE_Z_MM=-24 in real mode."""
    set_settings(Settings.from_env(table_z_mm=-24.0, descent_gate="off"))
    robot = MockRobot(qpos=PRESET_POSES["HOME"].copy())

    mock_calib = MagicMock()
    mock_calib.is_calibrated = True

    def dummy_gemini(point, img_shape=(480, 640), z_m=0.03):
        return np.array([0.22, 0.0, z_m])

    mock_calib.gemini_to_robot_xyz.side_effect = dummy_gemini

    with patch("vision.calibrator.get_calibrator", return_value=mock_calib):
        # 1. Hover at 3 cm commands robot z = 0.006 m (0.03 + (-0.024))
        call_hover = DummyToolCall(
            "move_to_spatial_point",
            {"point": [500.0, 500.0], "z_height_cm": 3.0},
        )
        res_hover = await execute_tool_call(call_hover, robot=robot)
        assert res_hover.get("status") == "success"
        assert res_hover["target_xyz_m"][2] == pytest.approx(0.006, abs=1e-4)

        # 2. Contact at 0 cm commands robot z = -0.024 m (0.0 + (-0.024))
        call_contact = DummyToolCall(
            "move_to_spatial_point",
            {"point": [500.0, 500.0], "z_height_cm": 0.0},
        )
        res_contact = await execute_tool_call(call_contact, robot=robot)
        assert res_contact.get("status") == "success"
        assert res_contact["target_xyz_m"][2] == pytest.approx(-0.024, abs=1e-4)

        # 3. Lift to 5 cm commands robot z = 0.026 m (0.05 + (-0.024))
        state = GuardrailState()
        state.phase = Phase.HELD_UNVERIFIED
        call_lift = DummyToolCall(
            "move_to_spatial_point",
            {"point": [500.0, 500.0], "z_height_cm": 5.0},
        )
        res_lift = await execute_tool_call(call_lift, robot=robot, guardrail_state=state)
        assert res_lift.get("status") == "success"
        assert res_lift["target_xyz_m"][2] == pytest.approx(0.026, abs=1e-4)


def test_plan_approach_segments_table_height(restore_settings: None) -> None:
    """Verify that with table at -24 mm, a site at robot z=15 mm is not a low pose in real mode."""
    set_settings(Settings.from_env(table_z_mm=-24.0))
    ik = CartesianIK()
    target_start = np.array([0.25, 0.0, 0.015])
    qpos_start = ik.solve_ik(target_start, approach="top_down")
    target_far = np.array([0.25, 0.10, 0.05])

    # In real mode with table at -24 mm:
    # low_pose_z = 0.03 + (-0.024) = 0.006 m.
    # Current site z ≈ 0.015 m > 0.006 m, so it is NOT a low pose -> 1 segment.
    segs_real = plan_approach_segments(ik, qpos_start, target_far, mode="real")
    assert len(segs_real) == 1

    # In sim mode (table at 0 mm):
    # low_pose_z = 0.03 m.
    # Current site z ≈ 0.015 m < 0.03 m, so it IS a low pose -> 3 segments.
    segs_sim = plan_approach_segments(ik, qpos_start, target_far, mode="sim")
    assert len(segs_sim) == 3


def test_locate_jaw_xy_mm_parallax_height(restore_settings: None) -> None:
    """Verify locate_jaw_xy_mm passes parallax height = robot z + 24 mm when TABLE_Z_MM=-24."""
    set_settings(Settings.from_env(table_z_mm=-24.0))
    ik = CartesianIK()
    q = np.radians([0.0, 20.0, 20.0, 20.0, 0.0, 0.0])

    mock_snapshot = MagicMock()
    mock_snapshot.pixel_to_plane_mm.return_value = (100.0, 50.0)
    mock_snapshot.table_mm_to_pixel.return_value = (320.0, 240.0)

    site_m, rot = ik.gripper_frame_pose(q)
    tag_z_robot = site_m[2] * 1000.0 + (rot @ np.array(Settings().gripper_tag_local_offset_mm))[2]

    fake_corners = np.array([[10, 10], [20, 10], [20, 20], [10, 20]])
    with patch("vision.gripper_locator.detect_jaw_tag", return_value=fake_corners):
        blank = np.zeros((100, 100, 3), dtype=np.uint8)

        # Real mode: height_passed = robot z - (-24) = robot z + 24 mm
        loc_real = locate_jaw_xy_mm(blank, mock_snapshot, q, ik=ik, mode="real")
        assert loc_real is not None
        call_args = mock_snapshot.pixel_to_plane_mm.call_args[0]
        height_passed_real = call_args[2]
        assert height_passed_real == pytest.approx(tag_z_robot + 24.0, abs=1e-4)

        # Sim mode: height_passed = robot z - 0 = robot z
        loc_sim = locate_jaw_xy_mm(blank, mock_snapshot, q, ik=ik, mode="sim")
        assert loc_sim is not None
        call_args_sim = mock_snapshot.pixel_to_plane_mm.call_args[0]
        height_passed_sim = call_args_sim[2]
        assert height_passed_sim == pytest.approx(tag_z_robot, abs=1e-4)


@pytest.mark.asyncio
async def test_actual_position_logged_on_move(restore_settings: None) -> None:
    """Verify that move_to_spatial_point logs the reached FK position and projected camera point."""
    set_settings(Settings.from_env(table_z_mm=-24.0, descent_gate="off"))
    robot = MockRobot(qpos=PRESET_POSES["HOME"].copy())

    mock_calib = MagicMock()
    mock_calib.is_calibrated = True
    mock_calib.gemini_to_robot_xyz.return_value = np.array([0.22, 0.0, 0.006])
    mock_calib.robot_mm_to_pixel.return_value = (320.0, 240.0)

    logged_exec_messages = []

    def mock_log(logger, tag, msg):
        if tag == "EXEC" and "reached:" in msg:
            logged_exec_messages.append(msg)

    with (
        patch("vision.calibrator.get_calibrator", return_value=mock_calib),
        patch("agent.tools.log_with_tag", side_effect=mock_log),
    ):
        call = DummyToolCall(
            "move_to_spatial_point",
            {"point": [500.0, 500.0], "z_height_cm": 3.0},
        )
        res = await execute_tool_call(call, robot=robot, mode="real")
        assert res.get("status") == "success"
        assert len(logged_exec_messages) == 1
        msg = logged_exec_messages[0]
        assert "FK XYZ" in msg
        assert "table Z=" in msg
        assert "table XY" in msg
        assert "image [500, 500]" in msg
        assert "error: dX=" in msg
