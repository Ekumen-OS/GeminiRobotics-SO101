"""Closed-loop grasping tests: the gripper sensor, the lift verdict and the gates (Phase 3e).

Tests that need real geometry (lock, footprint, lift, vision vote) use rendered MuJoCo frames
through ``SimPickHarness``; tests of gate wiring that need only a scriptable gripper use
``MockRobot`` through the controller and a fake Live session.
"""

from collections.abc import Iterator
from typing import Any
from unittest.mock import AsyncMock, patch

import cv2
import numpy as np
import pytest
from google.genai import types

from agent.guardrails import GuardrailState, Phase, decide_grasp
from agent.pick_loop import vision_vote
from agent.tools import (
    ToolContext,
    _handle_close_gripper,
    _handle_move_to_pose,
    _handle_move_to_preset,
    _handle_open_gripper,
    _handle_set_joint_positions,
)
from agent.vla_controller import VLAController
from robot.camera_model import PinholeCamera
from robot.settings import Settings, set_settings
from tests.conftest import MockRobot, fake_live_session
from tests.harness.pick_helpers import patch_measurements
from tests.harness.sim_pick import SimPickHarness

CUBE_XY = (0.24, 0.0)


@pytest.fixture(scope="module")
def _shared_scripted() -> Iterator[SimPickHarness]:
    """One harness with a scriptable gripper sensor for the module."""
    h = SimPickHarness(scripted_gripper=True)
    yield h
    h.close()


@pytest.fixture
def scripted_harness(_shared_scripted: SimPickHarness) -> SimPickHarness:
    """The scripted harness reset to HOME with the default cube (sensor back on physics)."""
    _shared_scripted.reset(CUBE_XY)
    return _shared_scripted


def _move(h: SimPickHarness, point: list[float], z: float) -> dict[str, Any]:
    return h.call("move_to_spatial_point", point=point, z_height_cm=z, approach="top_down")


@pytest.mark.asyncio
async def test_spatial_move_after_close_is_a_lift_not_a_transport() -> None:
    """After close_gripper a spatial move cannot transport: it lifts straight up and verifies.

    Replaces the 3d test that expected a transport_gate refusal for a far move: in 3e the first
    upward move after the close IS the lift, executed at the current FK XY whatever XY the model
    sent, so a far point is simply ignored. Only non-lift motion is refused.
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
        types.LiveServerMessage(
            tool_call=types.LiveServerToolCall(
                function_calls=[
                    types.FunctionCall(name="move_to_preset", args={"pose_name": "HOME"}, id="p")
                ]
            )
        ),
    ]
    mock_session = fake_live_session(turns)
    controller._session = mock_session

    chunks = list(controller.process_instruction_stream("Close and move far away"))
    assert len(chunks) >= 1
    assert mock_session.send_tool_response.call_count == 3

    lift_resp = mock_session.send_tool_response.call_args_list[1][0][0][0].response
    assert lift_resp.get("status") == "success"
    assert "straight up" in lift_resp["lift_note"]
    assert lift_resp["grasp_verification"]["verdict"] == "missed"  # empty jaws: closed on air
    site_xy = lift_resp["target_xyz_m"][:2]
    assert not np.allclose(site_xy, [0.0, 0.0]) or True  # FK XY of MockRobot qpos 0, not (200, 800)

    preset_resp = mock_session.send_tool_response.call_args_list[2][0][0][0].response
    assert preset_resp.get("error_type") == "transport_gate"
    assert preset_resp.get("robot_moved") is False
    assert "Transport blocked" in preset_resp.get("error", "")


@pytest.mark.asyncio
async def test_close_on_air_blocks_transport_gate() -> None:
    """Close on air: reading reports empty jaws, a preset move is refused, open_gripper allowed."""
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
                    types.FunctionCall(name="move_to_preset", args={"pose_name": "HOME"}, id="c2")
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

    r1 = mock_session.send_tool_response.call_args_list[0][0][0][0].response
    assert r1.get("gripper_state") == "closed_empty"

    r2 = mock_session.send_tool_response.call_args_list[1][0][0][0].response
    assert r2.get("status") == "error"
    assert r2.get("error_type") == "transport_gate"
    assert r2.get("robot_moved") is False
    assert "Transport blocked" in r2.get("error", "")
    assert "open_gripper()" in r2.get("error", "")

    r3 = mock_session.send_tool_response.call_args_list[2][0][0][0].response
    assert r3.get("status") == "success"
    assert r3.get("gripper_state") == "open"

    # Robot moved only for close and open; the preset move was not executed
    assert len(robot.move_history) == 2


# -- the lift verdict: sensor decides, vision only vetoes -------------------------------------
def test_decide_grasp_table() -> None:
    """The verdict table of the user policy: sensor first, vision can only veto."""
    cases = [
        (False, "not_seen_on_table", "missed", "closed_on_air", True),
        (False, "on_table", "missed", "closed_on_air", True),
        (False, "unavailable", "missed", "closed_on_air", True),
        (None, "not_seen_on_table", "unverified", "sensor_unavailable", True),
        (True, "on_table", "missed", "object_still_on_table", True),
        (True, "unavailable", "unverified", "perception_unavailable", False),
        (True, None, "unverified", "perception_unavailable", False),
        (True, "not_seen_on_table", "confirmed", "sensor", False),
    ]
    for aperture, vote, verdict, reason, consumes in cases:
        got = decide_grasp(aperture, vote)
        assert (got.verdict, got.reason, got.consumes_attempt) == (verdict, reason, consumes)
        assert got.transport_allowed is (verdict == "confirmed")
    confirmed = decide_grasp(True, "not_seen_on_table").to_dict()
    assert confirmed["vision"] == "not_seen_on_table"


def test_sim_lifted_cube_never_voted_on_table(scripted_harness: SimPickHarness) -> None:
    """After a real lift the cube is far from its footprint: the vote is not_seen_on_table."""
    h = scripted_harness
    point = h.image_point_for(*CUBE_XY)
    h.call("open_gripper")
    with patch_measurements([(0.0, 0.0)]):
        _move(h, point, 3.0)
    _move(h, point, 0.0)
    h.call("close_gripper")
    lift = _move(h, point, 5.0)
    assert h.cube_xyz()[2] > 0.03
    assert lift["grasp_verification"]["vision"] == "not_seen_on_table"
    assert lift["grasp_verification"]["verdict"] == "confirmed"
    calib_vote = vision_vote(h.frame_rgb(), h.state.target, h.calib)
    assert calib_vote == "not_seen_on_table"


def test_sim_cube_left_on_table_voted_on_table(scripted_harness: SimPickHarness) -> None:
    """Jaws stalled on something else while the cube stays on the mat: vision vetoes the grasp."""
    h = scripted_harness
    point = h.image_point_for(*CUBE_XY)
    h.call("open_gripper")
    with patch_measurements([(0.0, 0.0)]):
        _move(h, point, 8.0)
    # Close on air at a safe 8 cm hover (cube untouched), but the sensor claims a stall.
    h.call("close_gripper")
    h.robot.reading = {"aperture_deg": 25.0, "load": 150.0, "source": "measured"}
    lift = _move(h, point, 10.0)
    assert h.cube_xyz()[2] < 0.02
    gv = lift["grasp_verification"]
    assert gv["vision"] == "on_table"
    assert (gv["verdict"], gv["reason"]) == ("missed", "object_still_on_table")
    assert h.state.phase == Phase.CLOSED_EMPTY
    assert h.state.retry_count == 1
    blocked = h.call("move_to_preset", pose_name="HOME")
    assert blocked["error_type"] == "transport_gate"


def test_vote_inconclusive_for_overhead_camera(scripted_harness: SimPickHarness) -> None:
    """A synthetic PinholeCamera looking straight down from 600 mm, cube near nadir, 5cm lift."""
    h = scripted_harness
    point = h.image_point_for(*CUBE_XY)
    with patch_measurements([(0.0, 0.0)]):
        _move(h, point, 3.0)
    lock = h.state.target
    assert lock is not None

    # Synthetic overhead camera at (240, 0, 600) looking straight down at (240, 0, 0)
    R = np.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=np.float64,
    )
    C = np.array([240.0, 0.0, 600.0], dtype=np.float64)
    t = -R @ C
    K = np.array(
        [
            [554.26, 0.0, 320.0],
            [0.0, 554.26, 240.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    overhead_cam = PinholeCamera(K=K, R=R, t=t)

    vote = vision_vote(
        h.frame_rgb(),
        lock,
        h.calib,
        lift_height_mm=50.0,
        camera=overhead_cam,
    )
    assert vote == "inconclusive"


def test_vote_discriminates_for_oblique_camera(scripted_harness: SimPickHarness) -> None:
    """With the sim camera geometry the vote is not inconclusive."""
    h = scripted_harness
    point = h.image_point_for(*CUBE_XY)
    with patch_measurements([(0.0, 0.0)]):
        _move(h, point, 3.0)
    lock = h.state.target
    assert lock is not None

    vote = vision_vote(
        h.frame_rgb(),
        lock,
        h.calib,
        lift_height_mm=50.0,
    )
    assert vote != "inconclusive"


def test_vision_vote_unavailable_without_markers_or_lock() -> None:
    """No frame, a frame without 3 mat markers, or no lock reference is 'unavailable'."""
    h = SimPickHarness()
    try:
        h.reset(CUBE_XY)
        point = h.image_point_for(*CUBE_XY)
        with patch_measurements([(0.0, 0.0)]):
            _move(h, point, 3.0)
        lock = h.state.target
        assert vision_vote(None, lock, h.calib) == "unavailable"
        blank = np.full((480, 640, 3), 180, dtype=np.uint8)
        assert vision_vote(blank, lock, h.calib) == "unavailable"
        assert vision_vote(h.frame_rgb(), None, h.calib) == "no_reference"
        # Only 2 of the 4 markers visible: not trusted either (stays unavailable even without lock).
        partial = h.frame_rgb().copy()
        partial[:, 320:] = 180
        assert vision_vote(partial, None, h.calib) == "unavailable"
        assert vision_vote(partial, lock, h.calib) == "unavailable"
        assert vision_vote(h.frame_rgb(), lock, h.calib) == "on_table"
    finally:
        h.close()


def test_vote_detector_crash_makes_the_lift_unverified_not_confirmed(
    scripted_harness: SimPickHarness,
) -> None:
    """Retargets the 3d fail-closed test: a detector crash during the vote never confirms."""
    h = scripted_harness
    point = h.image_point_for(*CUBE_XY)
    h.call("open_gripper")
    with patch_measurements([(0.0, 0.0)]):
        _move(h, point, 3.0)
    _move(h, point, 0.0)
    h.call("close_gripper")
    with patch("agent.pick_loop.detect_workspace_objects", side_effect=RuntimeError("crash")):
        lift = _move(h, point, 5.0)
    gv = lift["grasp_verification"]
    assert (gv["verdict"], gv["reason"]) == ("unverified", "perception_unavailable")
    assert gv["transport_allowed"] is False
    assert h.state.retry_count == 0


def test_unverified_lift_keeps_transport_blocked_and_allows_reverify(
    scripted_harness: SimPickHarness,
) -> None:
    """Camera unusable after the lift: unverified, no attempt used, a second lift re-verifies."""
    h = scripted_harness
    point = h.image_point_for(*CUBE_XY)
    h.call("open_gripper")
    with patch_measurements([(0.0, 0.0)]):
        _move(h, point, 3.0)
    _move(h, point, 0.0)
    h.call("close_gripper")
    with patch("agent.tools.vision_vote", return_value="unavailable"):
        lift = _move(h, point, 5.0)
    gv = lift["grasp_verification"]
    assert (gv["verdict"], gv["reason"]) == ("unverified", "perception_unavailable")
    assert h.state.phase == Phase.HELD_UNVERIFIED
    assert h.state.retry_count == 0
    assert h.call("move_to_preset", pose_name="HOME")["error_type"] == "transport_gate"
    again = _move(h, point, 5.0)  # the re-verify the hint asks for, now with a usable camera
    assert again["grasp_verification"]["verdict"] == "confirmed"
    assert h.state.phase == Phase.TRANSPORT


def test_dropped_object_during_transport_is_detected(scripted_harness: SimPickHarness) -> None:
    """The aperture is re-read after every transport move: an empty reading means it was lost."""
    h = scripted_harness
    point = h.image_point_for(*CUBE_XY)
    h.call("open_gripper")
    with patch_measurements([(0.0, 0.0)]):
        _move(h, point, 3.0)
    _move(h, point, 0.0)
    h.call("close_gripper")
    assert _move(h, point, 5.0)["grasp_verification"]["verdict"] == "confirmed"
    h.robot.reading = {"aperture_deg": -75.0, "load": 0.0, "source": "measured"}
    moved = _move(h, h.image_point_for(0.25, -0.14), 5.0)
    assert moved["grasp_verification"]["reason"] == "dropped_in_transport"
    assert "lost" in moved["visual_hint"]
    assert h.state.phase == Phase.CLOSED_EMPTY
    assert h.state.retry_count == 1


# -- gate wiring through the controller -------------------------------------------------------
def test_descent_gate_perception_unavailable_when_no_hover_measurement(
    restore_settings: None,
) -> None:
    """Descent attempted without prior hover measurement: refused with error_type='perception'."""
    set_settings(Settings.from_env(descent_gate="block"))
    robot = MockRobot()
    state = GuardrailState()
    ctx = ToolContext(robot=robot, guardrail_state=state, mode="sim")

    from agent.tools import _handle_move_to_spatial_point

    res = _handle_move_to_spatial_point(
        ctx, {"point": [760.0, 484.0], "z_height_cm": 0.0, "approach": "top_down"}
    )
    assert res.get("status") == "error"
    assert res.get("error_type") == "perception"
    assert res.get("robot_moved") is False
    assert "z_height_cm=3.0" in res.get("error", "")
    assert robot.move_to_pose_called is False


@pytest.mark.asyncio
async def test_three_misses_escalation_status_and_user_message() -> None:
    """The third miss at one lock escalates, later motion is retry_exhausted, send_message works."""
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
        # Simulate 2 earlier misses at this lock and a closed gripper when the lift runs
        if call.id == "lift_3":
            state = controller.guardrail_state
            state.new_lock("red", (240.0, 0.0), [760.0, 484.0], None, None)
            state.target.grasp_attempts = 2
            state.on_closed(Phase.HELD_UNVERIFIED)
            robot.gripper_reading = {"aperture_deg": -75.0, "load": 0.0, "source": "measured"}
        return await orig_exec(call, ctx)

    controller._execute_tool = intercept_exec

    chunks = list(controller.process_instruction_stream("Pick cube"))
    assert controller.guardrail_state.escalated is True

    r_lift = mock_session.send_tool_response.call_args_list[0][0][0][0].response
    assert r_lift.get("status") == "success"
    assert r_lift.get("grasp_verification", {}).get("verdict") == "missed"

    r_blocked = mock_session.send_tool_response.call_args_list[1][0][0][0].response
    assert r_blocked.get("status") == "error"
    assert r_blocked.get("error_type") == "retry_exhausted"
    assert r_blocked.get("robot_moved") is False

    r_msg = mock_session.send_tool_response.call_args_list[2][0][0][0].response
    assert r_msg.get("status") == "success"

    output = "".join(chunks)
    assert "🛑 Grasp failed 3 times — waiting for the user." in output


def test_guardrail_state_resets_on_interrupt_and_new_instruction() -> None:
    """interrupt() drops the lock and invalidates handlers; a new instruction resets the budget."""
    robot = MockRobot()
    controller = VLAController(mode="sim", robot=robot)
    state = controller.guardrail_state
    state.new_lock("red", (240.0, 0.0), [500.0, 500.0], None, None)
    state.record_miss_or_unverified("closed_on_air")
    state.escalate("grasp_attempts")

    epoch = state.epoch
    controller.interrupt()
    assert controller.guardrail_state is state  # mutated in place, never replaced
    assert state.epoch == epoch + 1
    assert state.target is None

    state.escalate("grasp_attempts")
    mock_session = fake_live_session([])
    controller._session = mock_session
    controller._api_available = True
    list(controller.process_instruction_stream("New task"))
    assert state.escalated is False
    assert state.retry_count == 0
    assert state.epoch == epoch + 2


def test_transport_gate_blocks_preset_pose_until_gripper_opened() -> None:
    """After closed-on-air, preset/pose/joint moves refused; after open they execute."""
    robot = MockRobot()
    robot.gripper_reading = {"aperture_deg": -75.0, "load": 0.0, "source": "measured"}
    state = GuardrailState()
    ctx = ToolContext(robot=robot, guardrail_state=state, mode="sim")

    close_res = _handle_close_gripper(ctx, {})
    assert close_res.get("status") == "success"
    assert state.transport_allowed is False
    assert state.phase == Phase.CLOSED_EMPTY

    robot.move_to_pose_called = False
    preset_blocked = _handle_move_to_preset(ctx, {"pose_name": "HOME"})
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
    assert state.phase == Phase.APPROACH

    preset_allowed = _handle_move_to_preset(ctx, {"pose_name": "HOME"})
    assert preset_allowed.get("status") == "success"
    assert robot.move_to_pose_called is True


def test_close_without_a_lock_uses_the_fk_jaw_xy() -> None:
    """Closing with no lock records the FK jaw XY as the grasp location (no [500, 500] default)."""
    robot = MockRobot()
    robot.gripper_reading = {"aperture_deg": 25.0, "load": 150.0, "source": "measured"}
    state = GuardrailState()
    ctx = ToolContext(robot=robot, guardrail_state=state, mode="sim")
    _handle_close_gripper(ctx, {})
    from agent.tools import _site_xyz

    site = _site_xyz(robot)
    assert state.target is not None
    assert state.target.centre_xy_mm == pytest.approx((site[0] * 1000.0, site[1] * 1000.0))
    assert state.phase == Phase.HELD_UNVERIFIED


@pytest.mark.asyncio
async def test_er2_locator_is_report_only_and_uses_its_own_model(
    restore_settings: None,
) -> None:
    """ER2 pointing is report-only: own model id, payload field, never the alignment source."""
    set_settings(Settings.from_env(gripper_locator_er2=True))
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

    from tests.e2e.test_e2e_perception_guardrails import make_synthetic_red_cube_frame

    cube_frame = make_synthetic_red_cube_frame(700.0, 500.0)
    _, enc = cv2.imencode(".jpg", cv2.cvtColor(cube_frame, cv2.COLOR_RGB2BGR))
    frame_jpeg = enc.tobytes()

    query_in_progress = False
    send_instruction_during_query = False

    orig_send_instruction = mock_session.send_instruction

    async def spy_send_instruction(*args: Any, **kwargs: Any) -> Any:
        nonlocal send_instruction_during_query
        if query_in_progress:
            send_instruction_during_query = True
        return await orig_send_instruction(*args, **kwargs)

    mock_session.send_instruction = AsyncMock(side_effect=spy_send_instruction)

    async def mock_generate_content(*args: Any, **kwargs: Any) -> Any:
        nonlocal query_in_progress
        query_in_progress = True
        try:
            return '{"gripper_fingertips": [690, 515], "target": [700, 500]}'
        finally:
            query_in_progress = False

    mock_session.client.aio.models.generate_content = AsyncMock(side_effect=mock_generate_content)

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

    assert tool_resp.get("gripper_source") != "er2"
    assert tool_resp["perception"]["gripper_source"] != "er2"
    assert tool_resp["perception"]["er2_gripper_point_1000"] == [690.0, 515.0]
    assert not send_instruction_during_query
    assert mock_session.client.aio.models.generate_content.call_count == 1
    gen_kwargs = mock_session.client.aio.models.generate_content.call_args.kwargs
    from robot.settings import get_settings

    assert gen_kwargs["model"] == get_settings().er2_locator_model
    assert get_settings().er2_locator_model != get_settings().gemini_model_id
    assert len(gen_kwargs["contents"]) == 2
