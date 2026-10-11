"""Characterization and closed-loop grasping tests for Phase 3d.

Pins today's baseline behavior before flipping:
- compute_visual_alignment with no gripper returns is_aligned=True, dy=dx=0
  (Task 0 -> flipped in Task 2)
- spatial move far away after close_gripper is executed without gate rejection
  (Task 0 -> flipped in Task 4)
"""

import os
from typing import Any
from unittest.mock import AsyncMock, patch

import cv2
import numpy as np
import pytest
from google.genai import types

from agent.guardrails import GuardrailState, OffsetCorrector
from agent.tools import (
    ToolContext,
    _handle_close_gripper,
    _handle_move_to_pose,
    _handle_move_to_preset,
    _handle_move_to_spatial_point,
    _handle_open_gripper,
    _handle_set_joint_positions,
    image_point_to_xyz,
)
from agent.vla_controller import VLAController
from robot.cartesian_ik import CartesianIK
from robot.perception import compute_visual_alignment
from simulation.sim_calibration import init_sim_workspace_calibration
from simulation.sim_robot import SO101Robot
from tests.conftest import MockRobot, fake_live_session
from vision.calibrator import get_calibrator
from vision.gripper_locator import locate_gripper_aruco, locate_gripper_kinematic


def test_compute_visual_alignment_no_gripper_returns_none_flipped() -> None:
    """Task 2 FLIPPED: compute_visual_alignment without gripper returns None metrics.

    Returns gripper_source='none'. Flipped from Task 0 baseline where it fell back
    to gripper = target.
    """
    target = [760.0, 484.0]
    result = compute_visual_alignment(
        frame_rgb=None,
        target_point_1000=target,
        gripper_point_1000=None,
        tolerance=25.0,
    )

    assert result["is_aligned"] is None
    assert result["measured_offset"] is None
    assert result["gripper_point_1000"] is None
    assert result["gripper_source"] == "none"
    assert result["target_point_1000"] == target


@pytest.mark.asyncio
async def test_spatial_move_after_close_blocked_flipped() -> None:
    """Task 4 FLIPPED: scripted close_gripper followed by transport move is blocked.

    Flipped from Task 0 baseline where transport move executed without verification.
    Now transport_gate rejects unverified transport and the arm does not move.
    """
    robot = MockRobot()
    controller = VLAController(mode="sim", robot=robot)
    controller._api_available = True

    turns = [
        types.LiveServerMessage(
            tool_call=types.LiveServerToolCall(
                function_calls=[types.FunctionCall(name="close_gripper", args={}, id="call_close")]
            )
        ),
        types.LiveServerMessage(
            tool_call=types.LiveServerToolCall(
                function_calls=[
                    types.FunctionCall(
                        name="move_to_spatial_point",
                        args={"point": [200.0, 800.0], "z_height_cm": 5.0, "approach": "top_down"},
                        id="call_transport",
                    )
                ]
            )
        ),
    ]

    mock_session = fake_live_session(turns)
    controller._session = mock_session

    chunks = list(controller.process_instruction_stream("Close and move far away"))
    assert len(chunks) >= 1

    # Both tools yielded responses (send_tool_response called once per turn)
    assert mock_session.send_tool_response.call_count == 2

    # Check transport response payload
    second_call_responses = mock_session.send_tool_response.call_args_list[1][0][0]
    transport_resp = second_call_responses[0].response
    assert transport_resp.get("error_type") == "transport_gate"
    assert transport_resp.get("robot_moved") is False
    assert "Transport blocked" in transport_resp.get("error", "")

    # Only 1 move occurred (close_gripper), transport motion was blocked
    assert len(robot.move_history) == 1


def test_sim_gripper_locator_validation_10_targets() -> None:
    """Task 2.6: Sim validation across 10 reachable targets for ArUco and FK locators.

    Hover arm over 10 reachable targets in simulation, locate gripper using both
    ArUco locator and forward kinematics, and assert each estimate lies within 25 units.
    """
    calib = get_calibrator("sim")
    init_sim_workspace_calibration(calibrator=calib)
    robot = SO101Robot(render_viewer=False)
    robot.reset("HOME")
    solver = CartesianIK()

    targets = [
        [700.0, 450.0],
        [760.0, 484.0],
        [720.0, 520.0],
        [680.0, 400.0],
        [740.0, 430.0],
        [710.0, 490.0],
        [750.0, 550.0],
        [690.0, 460.0],
        [730.0, 510.0],
        [770.0, 460.0],
    ]

    aruco_errors: list[float] = []
    fk_errors: list[float] = []

    try:
        for tgt in targets:
            xyz = image_point_to_xyz(tgt, hover_z=0.04, mode="sim")
            qpos = solver.solve_ik(
                xyz, initial_qpos=robot.get_joint_positions(), approach="top_down"
            )
            robot.move_to_pose(qpos)
            frame = robot.render_rgb_frame(640, 480)

            loc_aruco = locate_gripper_aruco(frame, offset_px=(-7.0, 8.0))
            loc_fk = locate_gripper_kinematic(calib, robot)

            assert loc_aruco is not None, f"ArUco tag not detected at target {tgt}"
            assert loc_fk is not None, f"FK locator failed at target {tgt}"

            err_aruco = float(
                np.hypot(loc_aruco.point_1000[0] - tgt[0], loc_aruco.point_1000[1] - tgt[1])
            )
            err_fk = float(np.hypot(loc_fk.point_1000[0] - tgt[0], loc_fk.point_1000[1] - tgt[1]))

            aruco_errors.append(err_aruco)
            fk_errors.append(err_fk)

            assert err_aruco <= 25.0, f"ArUco error {err_aruco:.2f} exceeded 25 units at {tgt}"
            assert err_fk <= 25.0, f"FK error {err_fk:.2f} exceeded 25 units at {tgt}"
    finally:
        robot.close()

    assert max(aruco_errors) <= 25.0
    assert max(fk_errors) <= 25.0


@pytest.mark.asyncio
async def test_close_on_air_blocks_transport_gate() -> None:
    """Close on air: reading reports empty jaws, far move refused, open_gripper allowed."""
    robot = MockRobot()
    controller = VLAController(mode="sim", robot=robot)
    controller._api_available = True

    turns = [
        types.LiveServerMessage(
            tool_call=types.LiveServerToolCall(
                function_calls=[types.FunctionCall(name="close_gripper", args={}, id="c1")]
            )
        ),
        types.LiveServerMessage(
            tool_call=types.LiveServerToolCall(
                function_calls=[
                    types.FunctionCall(
                        name="move_to_spatial_point",
                        args={"point": [300.0, 800.0], "z_height_cm": 5.0, "approach": "top_down"},
                        id="c2",
                    )
                ]
            )
        ),
        types.LiveServerMessage(
            tool_call=types.LiveServerToolCall(
                function_calls=[types.FunctionCall(name="open_gripper", args={}, id="c3")]
            )
        ),
    ]

    mock_session = fake_live_session(turns)
    controller._session = mock_session

    list(controller.process_instruction_stream("Pick and transport"))
    assert mock_session.send_tool_response.call_count == 3

    # Check close response
    r1 = mock_session.send_tool_response.call_args_list[0][0][0][0].response
    assert r1.get("gripper_state") == "closed_empty"

    # Check transport response: rejected
    r2 = mock_session.send_tool_response.call_args_list[1][0][0][0].response
    assert r2.get("status") == "error"
    assert r2.get("error_type") == "transport_gate"
    assert r2.get("robot_moved") is False
    assert "Transport blocked" in r2.get("error", "")

    # Check open response: allowed
    r3 = mock_session.send_tool_response.call_args_list[2][0][0][0].response
    assert r3.get("status") == "success"
    assert r3.get("gripper_state") == "open"

    # Robot moved only for close and open; transport move was not executed
    assert len(robot.move_history) == 2


@pytest.mark.asyncio
async def test_aperture_object_elevation_table_blocks_transport() -> None:
    """Aperture reports object, elevation says still on table: verdict missed, transport blocked."""
    from tests.test_e2e_perception_guardrails import make_synthetic_red_cube_frame

    robot = MockRobot()
    robot.gripper_reading = {"aperture_deg": 25.0, "load": 150.0, "source": "measured"}
    controller = VLAController(mode="sim", robot=robot)
    controller._api_available = True

    # Pre-grasp frame has red cube, post-lift frame STILL has red cube on table
    frame_with_cube = cv2.imencode(".jpg", make_synthetic_red_cube_frame(760.0, 484.0))[1].tobytes()

    turns = [
        types.LiveServerMessage(
            tool_call=types.LiveServerToolCall(
                function_calls=[
                    types.FunctionCall(
                        name="move_to_spatial_point",
                        args={"point": [760.0, 484.0], "z_height_cm": 0.0, "approach": "top_down"},
                        id="c1_contact",
                    )
                ]
            )
        ),
        types.LiveServerMessage(
            tool_call=types.LiveServerToolCall(
                function_calls=[types.FunctionCall(name="close_gripper", args={}, id="c2_close")]
            )
        ),
        types.LiveServerMessage(
            tool_call=types.LiveServerToolCall(
                function_calls=[
                    types.FunctionCall(
                        name="move_to_spatial_point",
                        args={"point": [760.0, 484.0], "z_height_cm": 5.0, "approach": "top_down"},
                        id="c3_lift",
                    )
                ]
            )
        ),
        types.LiveServerMessage(
            tool_call=types.LiveServerToolCall(
                function_calls=[
                    types.FunctionCall(
                        name="move_to_spatial_point",
                        args={"point": [200.0, 800.0], "z_height_cm": 5.0, "approach": "top_down"},
                        id="c4_transport",
                    )
                ]
            )
        ),
    ]

    mock_session = fake_live_session(turns)
    controller._session = mock_session

    with patch.dict(os.environ, {"DESCENT_GATE": "off"}):
        list(
            controller.process_instruction_stream(
                "Pick and transport", get_latest_jpeg_fn=lambda: frame_with_cube
            )
        )

    # Lift tool call payload
    r_lift = mock_session.send_tool_response.call_args_list[2][0][0][0].response
    assert "grasp_verification" in r_lift
    assert r_lift["grasp_verification"]["verdict"] == "missed"
    assert controller.guardrail_state.retry_count == 1

    # Transport tool call payload: rejected
    r_trans = mock_session.send_tool_response.call_args_list[3][0][0][0].response
    assert r_trans.get("status") == "error"
    assert r_trans.get("error_type") == "transport_gate"
    assert r_trans.get("robot_moved") is False


@pytest.mark.asyncio
async def test_aperture_object_elevation_elevated_allows_transport() -> None:
    """Aperture reports object, elevation says elevated: verdict confirmed, transport allowed."""
    from tests.test_e2e_perception_guardrails import make_synthetic_red_cube_frame

    robot = MockRobot()
    robot.gripper_reading = {"aperture_deg": 25.0, "load": 150.0, "source": "measured"}
    controller = VLAController(mode="sim", robot=robot)
    controller._api_available = True

    pre_frame = cv2.imencode(".jpg", make_synthetic_red_cube_frame(760.0, 484.0))[1].tobytes()
    post_frame = cv2.imencode(".jpg", np.full((480, 640, 3), 180, dtype=np.uint8))[1].tobytes()

    current_frame = [pre_frame]

    def get_frame() -> bytes:
        return current_frame[0]

    turns = [
        types.LiveServerMessage(
            tool_call=types.LiveServerToolCall(
                function_calls=[
                    types.FunctionCall(
                        name="move_to_spatial_point",
                        args={"point": [760.0, 484.0], "z_height_cm": 0.0, "approach": "top_down"},
                        id="c1_contact",
                    )
                ]
            )
        ),
        types.LiveServerMessage(
            tool_call=types.LiveServerToolCall(
                function_calls=[types.FunctionCall(name="close_gripper", args={}, id="c2_close")]
            )
        ),
        types.LiveServerMessage(
            tool_call=types.LiveServerToolCall(
                function_calls=[
                    types.FunctionCall(
                        name="move_to_spatial_point",
                        args={"point": [760.0, 484.0], "z_height_cm": 5.0, "approach": "top_down"},
                        id="c3_lift",
                    )
                ]
            )
        ),
        types.LiveServerMessage(
            tool_call=types.LiveServerToolCall(
                function_calls=[
                    types.FunctionCall(
                        name="move_to_spatial_point",
                        args={"point": [200.0, 800.0], "z_height_cm": 5.0, "approach": "top_down"},
                        id="c4_transport",
                    )
                ]
            )
        ),
    ]

    mock_session = fake_live_session(turns)
    controller._session = mock_session

    with patch.dict(os.environ, {"DESCENT_GATE": "off"}):
        orig_exec = controller._execute_tool

        async def intercept_exec(call, ctx=None):
            res = await orig_exec(call, ctx)
            if call.name == "close_gripper":
                current_frame[0] = post_frame
            return res

        controller._execute_tool = intercept_exec

        list(
            controller.process_instruction_stream(
                "Pick and transport", get_latest_jpeg_fn=get_frame
            )
        )

    # Lift tool call payload
    r_lift = mock_session.send_tool_response.call_args_list[2][0][0][0].response
    assert "grasp_verification" in r_lift
    assert r_lift["grasp_verification"]["verdict"] == "confirmed"

    # Transport tool call payload: success!
    r_trans = mock_session.send_tool_response.call_args_list[3][0][0][0].response
    assert r_trans.get("status") == "success"
    assert robot.move_to_pose_called is True


def test_descent_gate_block_and_warn_on_misalignment() -> None:
    """Hover with gripper 45 units off: descent blocked in block mode, allowed with warn."""
    robot = MockRobot()
    state = GuardrailState()
    # Record hover at [760, 484] with 45 units offset: gripper at [715, 484], offset [45, 0]
    state.last_hover_target = [760.0, 484.0]
    state.last_hover_offset = [45.0, 0.0]
    state.last_hover_is_aligned = False

    ctx = ToolContext(robot=robot, guardrail_state=state, mode="sim")

    # In block mode: descent is blocked
    with patch.dict(os.environ, {"DESCENT_GATE": "block"}):
        res_block = _handle_move_to_spatial_point(
            ctx, {"point": [760.0, 484.0], "z_height_cm": 0.0, "approach": "top_down"}
        )
        assert res_block.get("status") == "error"
        assert res_block.get("error_type") == "descent_gate"
        assert res_block.get("robot_moved") is False
        assert robot.move_to_pose_called is False
        assert (
            "Re-issue the same target [760, 484]; the controller applies the measured "
            "correction [45.0, 0.0] automatically." in res_block.get("error", "")
        )

    # In warn mode: descent proceeds with warning
    with patch.dict(os.environ, {"DESCENT_GATE": "warn"}):
        res_warn = _handle_move_to_spatial_point(
            ctx, {"point": [760.0, 484.0], "z_height_cm": 0.0, "approach": "top_down"}
        )
        assert res_warn.get("status") == "success"
        assert robot.move_to_pose_called is True
        assert "DESCENT_GATE=warn" in res_warn.get("guardrail_warning", "")


def test_descent_gate_perception_unavailable_when_no_hover_measurement() -> None:
    """Descent attempted without prior hover measurement: refused with error_type='perception'."""
    robot = MockRobot()
    state = GuardrailState()
    ctx = ToolContext(robot=robot, guardrail_state=state, mode="sim")

    with patch.dict(os.environ, {"DESCENT_GATE": "block"}):
        res = _handle_move_to_spatial_point(
            ctx, {"point": [760.0, 484.0], "z_height_cm": 0.0, "approach": "top_down"}
        )
        assert res.get("status") == "error"
        assert res.get("error_type") == "perception"
        assert res.get("robot_moved") is False
        assert robot.move_to_pose_called is False


@pytest.mark.asyncio
async def test_three_misses_escalation_status_and_user_message() -> None:
    """3 consecutive misses trigger retry_exhausted, escalation status, and send_message works."""
    robot = MockRobot()
    controller = VLAController(mode="sim", robot=robot)
    controller._api_available = True

    turns = [
        types.LiveServerMessage(
            tool_call=types.LiveServerToolCall(
                function_calls=[
                    types.FunctionCall(
                        name="move_to_spatial_point",
                        args={"point": [760.0, 484.0], "z_height_cm": 5.0, "approach": "top_down"},
                        id="lift_3",
                    )
                ]
            )
        ),
        types.LiveServerMessage(
            tool_call=types.LiveServerToolCall(
                function_calls=[
                    types.FunctionCall(
                        name="move_to_spatial_point",
                        args={"point": [760.0, 484.0], "z_height_cm": 0.0, "approach": "top_down"},
                        id="move_blocked",
                    )
                ]
            )
        ),
        types.LiveServerMessage(
            tool_call=types.LiveServerToolCall(
                function_calls=[
                    types.FunctionCall(
                        name="send_message",
                        args={"message": "I was unable to pick the object after 3 attempts."},
                        id="msg_after_esc",
                    )
                ]
            )
        ),
    ]

    mock_session = fake_live_session(turns)
    controller._session = mock_session

    orig_exec = controller._execute_tool

    async def intercept_exec(call, ctx=None):
        # Configure guardrail state to simulate 2 prior misses when lift_3 runs
        if call.id == "lift_3":
            controller.guardrail_state.retry_count = 2
            controller.guardrail_state.gripper_closed = True
            controller.guardrail_state.last_grasp_target = [760.0, 484.0]
            controller.guardrail_state.transport_allowed = False
            robot.gripper_reading = {"aperture_deg": -75.0, "load": 0.0, "source": "measured"}
        return await orig_exec(call, ctx)

    controller._execute_tool = intercept_exec

    chunks = list(controller.process_instruction_stream("Pick cube"))
    assert controller.guardrail_state.escalated is True

    # Lift 3 completed recording third miss
    r_lift = mock_session.send_tool_response.call_args_list[0][0][0][0].response
    assert r_lift.get("status") == "success"
    assert r_lift.get("grasp_verification", {}).get("verdict") == "missed"

    # Subsequent motion rejected with retry_exhausted
    r_blocked = mock_session.send_tool_response.call_args_list[1][0][0][0].response
    assert r_blocked.get("status") == "error"
    assert r_blocked.get("error_type") == "retry_exhausted"
    assert r_blocked.get("robot_moved") is False

    # send_message still succeeds even in escalated state
    r_msg = mock_session.send_tool_response.call_args_list[2][0][0][0].response
    assert r_msg.get("status") == "success"

    output = "".join(chunks)
    assert "🛑 Grasp failed 3 times — waiting for the user." in output


def test_guardrail_state_resets_on_interrupt_and_new_instruction() -> None:
    """GuardrailState is reset on interrupt() and on a new instruction."""
    robot = MockRobot()
    controller = VLAController(mode="sim", robot=robot)
    controller.guardrail_state.retry_count = 2
    controller.guardrail_state.gripper_closed = True
    controller.guardrail_state.escalated = True

    # Test 1: interrupt() resets
    controller.interrupt()
    assert controller.guardrail_state.retry_count == 0
    assert controller.guardrail_state.gripper_closed is False
    assert controller.guardrail_state.escalated is False

    # Dirty again
    controller.guardrail_state.retry_count = 3
    controller.guardrail_state.escalated = True

    # Test 2: new instruction in process_instruction_stream resets
    mock_session = fake_live_session([])
    controller._session = mock_session
    controller._api_available = True
    list(controller.process_instruction_stream("New task"))
    assert controller.guardrail_state.retry_count == 0
    assert controller.guardrail_state.escalated is False


def test_guardrail_state_one_retry_per_grasp_attempt() -> None:
    """Close on air then lift -> retry_count == 1; open, close -> 2; third -> 3 and escalated."""
    state = GuardrailState()
    assert state.retry_count == 0
    assert state.attempt_counted is False

    # Attempt 1: close on air, then lift (both record miss)
    state.record_miss_or_unverified("closed_on_air")
    assert state.retry_count == 1
    assert state.attempt_counted is True
    # Lift fails verification in same attempt -> retry_count does not increment again
    state.record_miss_or_unverified("object_still_on_table")
    assert state.retry_count == 1
    assert state.escalated is False

    # Open resets attempt_counted
    state.open_gripper()
    assert state.retry_count == 1
    assert state.attempt_counted is False

    # Attempt 2: close on air again -> increments to 2
    state.record_miss_or_unverified("closed_on_air")
    assert state.retry_count == 2
    assert state.attempt_counted is True
    state.record_miss_or_unverified("closed_on_air")
    assert state.retry_count == 2
    assert state.escalated is False

    # Open resets attempt_counted
    state.open_gripper()
    assert state.attempt_counted is False

    # Attempt 3: third failure -> increments to 3 and escalates
    state.record_miss_or_unverified("closed_on_air")
    assert state.retry_count == 3
    assert state.escalated is True

    # Reset clears all
    state.reset()
    assert state.retry_count == 0
    assert state.attempt_counted is False
    assert state.escalated is False


def test_one_shot_bias_corrector_pin_sign_and_no_stacking() -> None:
    """One-shot corrector: point_corrected = point + (target - gripper), applied once, no stack."""
    corrector = OffsetCorrector()
    # Hover at target [500, 500]. Gripper located at [455, 500].
    # Offset = target - gripper = [500 - 455, 500 - 500] = [+45, 0].
    corrector.record_hover(target_1000=[500.0, 500.0], offset=[45.0, 0.0])

    # (a) Re-issued target -> corrected once
    corrected_pt, applied, note = corrector.consume_correction([500.0, 500.0])
    assert applied == [45.0, 0.0]
    assert np.isclose(corrected_pt[0], 545.0)
    assert np.isclose(corrected_pt[1], 500.0)
    assert note is None

    # (c) A second move after (a) -> no correction (consumed)
    second_pt, second_applied, second_note = corrector.consume_correction([500.0, 500.0])
    assert second_applied is None
    assert second_pt == [500.0, 500.0]
    assert second_note is None

    # (b) Model-corrected target -> not corrected again
    corrector.record_hover(target_1000=[500.0, 500.0], offset=[45.0, 0.0])
    mc_pt, mc_applied, mc_note = corrector.consume_correction([545.0, 500.0])
    assert mc_applied is None
    assert mc_pt == [545.0, 500.0]
    assert mc_note == "model-corrected"

    # Consumed after (b) as well
    after_mc_pt, after_mc_applied, after_mc_note = corrector.consume_correction([545.0, 500.0])
    assert after_mc_applied is None
    assert after_mc_pt == [545.0, 500.0]
    assert after_mc_note is None


def test_refinement_then_bias_correction_reaches_arm() -> None:
    """Fix 1: Target refinement happens first, then bias correction applies to refined point."""
    robot = MockRobot()
    state = GuardrailState()
    state.last_hover_target = [700.0, 500.0]
    state.last_hover_offset = [10.0, -15.0]
    state.last_hover_is_aligned = True
    state.corrector.record_hover(target_1000=[700.0, 500.0], offset=[10.0, -15.0])

    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    _, dummy_enc = cv2.imencode(".jpg", dummy_img)
    jpeg_bytes = dummy_enc.tobytes()

    ctx = ToolContext(
        robot=robot,
        guardrail_state=state,
        mode="sim",
        get_latest_jpeg_fn=lambda: jpeg_bytes,
    )

    with (
        patch(
            "agent.tools.refine_point_to_nearest_object",
            return_value=([700.0, 500.0], {"color": "red"}),
        ),
        patch("agent.tools.image_point_to_xyz", return_value=np.array([0.2, 0.0, 0.0])) as mock_xyz,
    ):
        res = _handle_move_to_spatial_point(
            ctx, {"point": [700.0, 500.0], "z_height_cm": 0.0, "approach": "top_down"}
        )

    assert mock_xyz.call_count == 1
    passed_pt = mock_xyz.call_args[0][0]
    assert np.allclose(passed_pt, [710.0, 485.0])
    assert res.get("applied_correction_1000") == [10.0, -15.0]


def test_elevation_vote_unknown_when_frame_missing_blocks_transport() -> None:
    """Fix 4: Elevation vote is None when pre_grasp_frame is missing, blocking transport."""
    robot = MockRobot()
    robot.gripper_reading = {"aperture_deg": 25.0, "load": 150.0, "source": "measured"}
    state = GuardrailState()
    state.gripper_closed = True
    state.last_grasp_target = [700.0, 500.0]
    state.pre_grasp_frame = None

    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    _, dummy_enc = cv2.imencode(".jpg", dummy_img)
    jpeg_bytes = dummy_enc.tobytes()

    ctx = ToolContext(
        robot=robot,
        guardrail_state=state,
        mode="sim",
        get_latest_jpeg_fn=lambda: jpeg_bytes,
    )

    with patch("agent.tools.image_point_to_xyz", return_value=np.array([0.2, 0.0, 0.05])):
        res_lift = _handle_move_to_spatial_point(
            ctx, {"point": [700.0, 500.0], "z_height_cm": 5.0, "approach": "top_down"}
        )

    gv = res_lift.get("grasp_verification")
    assert gv is not None
    assert gv.get("verdict") == "unverified"
    assert gv.get("reason") == "perception_unavailable"
    assert state.transport_allowed is False

    res_far = _handle_move_to_spatial_point(
        ctx, {"point": [200.0, 800.0], "z_height_cm": 5.0, "approach": "top_down"}
    )
    assert res_far.get("status") == "error"
    assert res_far.get("error_type") == "transport_gate"
    assert res_far.get("robot_moved") is False


def test_transport_gate_blocks_preset_pose_until_gripper_opened() -> None:
    """Fix 5: After closed-on-air, preset/pose/joint moves refused; after open they execute."""
    robot = MockRobot()
    robot.gripper_reading = {"aperture_deg": -75.0, "load": 0.0, "source": "measured"}
    state = GuardrailState()
    ctx = ToolContext(robot=robot, guardrail_state=state, mode="sim")

    close_res = _handle_close_gripper(ctx, {})
    assert close_res.get("status") == "success"
    assert state.gripper_closed is True
    assert state.transport_allowed is False

    robot.move_to_pose_called = False
    preset_blocked = _handle_move_to_preset(ctx, {"preset": "HOME"})
    assert preset_blocked.get("status") == "error"
    assert preset_blocked.get("error_type") == "transport_gate"
    assert preset_blocked.get("robot_moved") is False
    assert robot.move_to_pose_called is False

    pose_blocked = _handle_move_to_pose(ctx, {"qpos": [0.0] * 6, "duration_sec": 1.0})
    assert pose_blocked.get("status") == "error"
    assert pose_blocked.get("error_type") == "transport_gate"
    assert pose_blocked.get("robot_moved") is False
    assert robot.move_to_pose_called is False

    joints_blocked = _handle_set_joint_positions(ctx, {"qpos": [0.0] * 6})
    assert joints_blocked.get("status") == "error"
    assert joints_blocked.get("error_type") == "transport_gate"
    assert joints_blocked.get("robot_moved") is False
    assert robot.move_to_pose_called is False

    open_res = _handle_open_gripper(ctx, {})
    assert open_res.get("status") == "success"
    assert state.gripper_closed is False

    preset_allowed = _handle_move_to_preset(ctx, {"preset": "HOME"})
    assert preset_allowed.get("status") == "success"
    assert robot.move_to_pose_called is True


@pytest.mark.asyncio
async def test_er2_locator_uses_generate_content_and_feeds_alignment() -> None:
    """Fix 6: ER2 locator uses generate_content and feeds visual alignment in real mode."""
    with patch.dict(os.environ, {"GRIPPER_LOCATOR_ER2": "1"}):
        robot = MockRobot()
        controller = VLAController(mode="real", robot=robot)
        controller._api_available = True

        turns = [
            types.LiveServerMessage(
                tool_call=types.LiveServerToolCall(
                    function_calls=[
                        types.FunctionCall(
                            name="move_to_spatial_point",
                            args={
                                "point": [700.0, 500.0],
                                "z_height_cm": 5.0,
                                "approach": "top_down",
                            },
                            id="call_hover_er2",
                        )
                    ]
                )
            ),
            types.LiveServerMessage(server_content=types.LiveServerContent(turn_complete=True)),
        ]
        mock_session = fake_live_session(turns)
        controller._session = mock_session

        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        _, enc = cv2.imencode(".jpg", blank_frame)
        frame_jpeg = enc.tobytes()

        query_in_progress = False
        send_instruction_during_query = False
        send_text_during_query = False

        orig_send_instruction = mock_session.send_instruction
        orig_send_text = mock_session.send_text

        async def spy_send_instruction(*args: Any, **kwargs: Any) -> Any:
            nonlocal send_instruction_during_query
            if query_in_progress:
                send_instruction_during_query = True
            return await orig_send_instruction(*args, **kwargs)

        async def spy_send_text(*args: Any, **kwargs: Any) -> Any:
            nonlocal send_text_during_query
            if query_in_progress:
                send_text_during_query = True
            return await orig_send_text(*args, **kwargs)

        mock_session.send_instruction = AsyncMock(side_effect=spy_send_instruction)
        mock_session.send_text = AsyncMock(side_effect=spy_send_text)

        async def mock_generate_content(*args: Any, **kwargs: Any) -> Any:
            nonlocal query_in_progress
            query_in_progress = True
            try:
                return '{"gripper_fingertips": [690, 515], "target": [700, 500]}'
            finally:
                query_in_progress = False

        mock_session.client.aio.models.generate_content = AsyncMock(
            side_effect=mock_generate_content
        )

        with patch(
            "google.genai.models.AsyncModels.generate_content",
            new_callable=AsyncMock,
            side_effect=mock_generate_content,
        ):
            chunks = list(
                controller.process_instruction_stream(
                    "hover at target", get_latest_jpeg_fn=lambda: frame_jpeg
                )
            )

        assert len(chunks) >= 1
        assert mock_session.send_tool_response.call_count == 1
        tool_resp = mock_session.send_tool_response.call_args[0][0][0].response

        gripper_source = tool_resp.get("gripper_source")
        mo = tool_resp.get("perception", {}).get("measured_offset", {})
        measured_offset = {"dy": int(mo.get("dy")), "dx": int(mo.get("dx"))}

        assert gripper_source == "er2"
        assert measured_offset == {"dy": 10, "dx": -15}
        assert not send_instruction_during_query
        assert not send_text_during_query
        assert mock_session.client.aio.models.generate_content.call_count == 1
        gen_kwargs = mock_session.client.aio.models.generate_content.call_args.kwargs
        from robot.settings import get_settings

        assert gen_kwargs["model"] == get_settings().gemini_model_id
        assert len(gen_kwargs["contents"]) == 2


def test_bias_corrector_model_already_corrected_returns_note() -> None:
    """Fix 2: If requested point matches target + offset, consume without applying and set note."""
    robot = MockRobot()
    state = GuardrailState()
    state.last_hover_target = [700.0, 500.0]
    state.last_hover_offset = [10.0, -15.0]
    state.last_hover_is_aligned = True
    state.corrector.record_hover(target_1000=[700.0, 500.0], offset=[10.0, -15.0])

    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    _, dummy_enc = cv2.imencode(".jpg", dummy_img)
    jpeg_bytes = dummy_enc.tobytes()

    ctx = ToolContext(
        robot=robot,
        guardrail_state=state,
        mode="sim",
        get_latest_jpeg_fn=lambda: jpeg_bytes,
    )

    with patch(
        "agent.tools.image_point_to_xyz", return_value=np.array([0.2, 0.0, 0.05])
    ) as mock_xyz:
        res = _handle_move_to_spatial_point(
            ctx, {"point": [710.0, 485.0], "z_height_cm": 5.0, "approach": "top_down"}
        )

    assert res.get("status") == "success"
    assert res.get("applied_correction_1000") is None
    assert res.get("correction_note") == "model-corrected"
    assert np.allclose(mock_xyz.call_args[0][0], [710.0, 485.0])


def test_hover_bias_correction_preserves_target_for_subsequent_descent() -> None:
    """Hover with bias correction shifts arm without corrupting last_hover_target for descent."""
    robot = MockRobot()
    state = GuardrailState()
    # Initial hover target at [700, 500] with offset [30, -30] (distance 42.4 > 35)
    state.last_hover_target = [700.0, 500.0]
    state.last_hover_offset = [30.0, -30.0]
    state.last_hover_is_aligned = True
    state.corrector.record_hover(target_1000=[700.0, 500.0], offset=[30.0, -30.0])

    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    _, dummy_enc = cv2.imencode(".jpg", dummy_img)
    jpeg_bytes = dummy_enc.tobytes()

    ctx = ToolContext(
        robot=robot,
        guardrail_state=state,
        mode="sim",
        get_latest_jpeg_fn=lambda: jpeg_bytes,
    )

    # 1. Re-hover with the same target -> correction [30, -30] applied, arm receives [730, 470]
    aligned_perception = {
        "is_aligned": True,
        "measured_offset": {"dy": 0.0, "dx": 0.0, "distance": 0.0},
    }
    with (
        patch(
            "agent.tools.image_point_to_xyz", return_value=np.array([0.2, 0.0, 0.05])
        ) as mock_xyz,
        patch(
            "agent.tools._measure_alignment",
            return_value=(aligned_perception, "aruco", dummy_img),
        ),
    ):
        res_hover = _handle_move_to_spatial_point(
            ctx, {"point": [700.0, 500.0], "z_height_cm": 5.0, "approach": "top_down"}
        )
    assert res_hover.get("status") == "success"
    assert res_hover.get("applied_correction_1000") == [30.0, -30.0]
    assert np.allclose(mock_xyz.call_args[0][0], [730.0, 470.0])
    # The recorded hover target remains [700, 500] (the object), NOT [730, 470]
    assert state.last_hover_target == [700.0, 500.0]

    # 2. Descend to grasp at [700, 500] -> descent gate matches hover target within 35 units!
    with patch.dict(os.environ, {"DESCENT_GATE": "block"}):
        with patch(
            "agent.tools.image_point_to_xyz", return_value=np.array([0.2, 0.0, 0.0])
        ) as mock_xyz_desc:
            res_desc = _handle_move_to_spatial_point(
                ctx, {"point": [700.0, 500.0], "z_height_cm": 0.0, "approach": "top_down"}
            )
        assert res_desc.get("status") == "success"
        assert res_desc.get("error_type") is None
        assert np.allclose(mock_xyz_desc.call_args[0][0], [700.0, 500.0])
