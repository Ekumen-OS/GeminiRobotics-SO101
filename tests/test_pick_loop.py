"""Pick-loop state machine tests (Phase 3e, Step 4): phases, target lock, refusals, escalation.

Geometry-sensitive tests drive the real tool handlers with a real ``SO101Robot`` and rendered
MuJoCo frames (``SimPickHarness``). Alignment measurements are scripted where a test needs a
specific residual; everything else (lock, footprint, IK, physics, gripper sensor, vision vote)
is real.
"""

from collections.abc import Iterator
from typing import Any
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from agent.guardrails import (
    MAX_HOVERS_PER_LOCK,
    GuardrailState,
    Phase,
)
from agent.tools import (
    ToolContext,
    _handle_move_to_pose,
    _handle_move_to_preset,
    _handle_open_gripper,
    execute_tool_call_sync,
)
from agent.vla_controller import VLAController
from robot.constants import PRESET_POSES
from robot.vision_detector import detect_workspace_objects
from tests.conftest import DummyToolCall, MockRobot
from tests.pick_helpers import (
    next_call_from,
    parse_move_call,
    patch_measurements,
)
from tests.sim_pick_harness import DESTINATION_XY_M, SimPickHarness

CUBE_XY = (0.24, 0.0)


@pytest.fixture(scope="module")
def _shared_harness() -> Iterator[SimPickHarness]:
    """One sim harness for the whole module (model load and renderer setup are slow)."""
    h = SimPickHarness()
    yield h
    h.close()


@pytest.fixture
def harness(_shared_harness: SimPickHarness) -> SimPickHarness:
    """The shared harness reset to HOME with the default cube and a fresh guardrail state."""
    _shared_harness.reset(CUBE_XY)
    return _shared_harness


def _move(h: SimPickHarness, point: list[float], z: float, **extra: Any) -> dict[str, Any]:
    return h.call("move_to_spatial_point", point=point, z_height_cm=z, approach="top_down", **extra)


def _pick_until_lift(h: SimPickHarness, point: list[float]) -> dict[str, Any]:
    """open, hover (scripted aligned), descend, close; return the close response."""
    h.call("open_gripper")
    with patch_measurements([(0.0, 0.0)]):
        assert _move(h, point, 3.0)["status"] == "success"
    assert _move(h, point, 0.0)["status"] == "success"
    return h.call("close_gripper")


# -- lock identity ---------------------------------------------------------------------------
def test_lock_is_created_once_and_never_rerefined(harness: SimPickHarness) -> None:
    """Two requests within 40 mm of the lock centre are the same lock; detection runs once."""
    point = harness.image_point_for(*CUBE_XY)
    from robot.vision_detector import refine_point_to_nearest_object as real_refine

    with (
        patch("agent.tools.refine_point_to_nearest_object", side_effect=real_refine) as refine,
        patch_measurements([(0.0, 0.0)]),
    ):
        first = _move(harness, point, 3.0)
        lock_id = harness.state.target.lock_id
        # The model drifts by ~25 units (about 20 mm) between calls.
        second = _move(harness, [point[0] + 25.0, point[1] - 10.0], 3.0)
    assert first["status"] == "success" and second["status"] == "success"
    assert harness.state.target.lock_id == lock_id
    assert refine.call_count == 1


def test_request_for_a_different_object_creates_a_new_lock(harness: SimPickHarness) -> None:
    """A point far from the lock (> 40 mm) is a different request and locks anew."""
    point = harness.image_point_for(*CUBE_XY)
    with patch_measurements([(0.0, 0.0)]):
        _move(harness, point, 3.0)
        first_id = harness.state.target.lock_id
        harness.sim.reset("HOME")
        harness.place_cube(0.27, 0.09)
        far_point = harness.image_point_for(0.27, 0.09)
        _move(harness, far_point, 3.0)
    assert harness.state.target.lock_id != first_id
    assert harness.state.target.hovers_since_descent == 1


def test_lock_centre_is_the_footprint_centre(harness: SimPickHarness) -> None:
    """The lock sits within 5 mm of the cube's true centre, not on its near edge."""
    point = harness.image_point_for(*CUBE_XY)
    with patch_measurements([(0.0, 0.0)]):
        _move(harness, point, 3.0)
    lock = harness.state.target
    truth = harness.cube_xyz()[:2] * 1000.0
    assert float(np.linalg.norm(np.array(lock.centre_xy_mm) - truth)) < 5.0


def test_merged_blobs_are_target_ambiguous(harness: SimPickHarness) -> None:
    """A blob wider than one graspable object (two cubes touching) is refused as ambiguous."""
    point = harness.image_point_for(*CUBE_XY)
    frame = harness.frame_rgb().copy()
    # Paint a second red cube touching the first: one 60 mm-wide blob.
    objs = [o for o in detect_workspace_objects(frame) if o["color"] == "red"]
    x, y, w, hgt = objs[0]["bbox_px"]
    frame[y : y + hgt, x + w : x + 2 * w] = frame[y : y + hgt, x : x + w]
    ok, enc = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    jpeg = enc.tobytes()
    res = execute_tool_call_sync(
        DummyToolCall(
            "move_to_spatial_point", {"point": point, "z_height_cm": 3.0, "approach": "top_down"}
        ),
        harness.robot,
        get_latest_jpeg_fn=lambda: jpeg,
        mode="sim",
        guardrail_state=harness.state,
    )
    assert res["error_type"] == "perception"
    assert res["reason"] == "target_ambiguous"
    assert res["robot_moved"] is False
    assert harness.state.target is None


def test_stale_lock_is_reacquired_and_keeps_its_correction(harness: SimPickHarness) -> None:
    """After a miss and open_gripper the lock is stale; the next request re-acquires it."""
    point = harness.image_point_for(*CUBE_XY)
    with patch_measurements([(0.0, 0.0)]):
        _move(harness, point, 3.0)
    lock = harness.state.target
    lock.c_mm = (3.0, -1.5)
    lock.grasp_attempts = 1
    harness.state.phase = Phase.CLOSED_EMPTY
    harness.call("open_gripper")
    assert harness.state.target is lock and lock.stale is True
    with patch_measurements([(0.0, 0.0)]):
        _move(harness, point, 3.0)
    assert harness.state.target is lock
    assert lock.stale is False
    assert lock.c_mm == (3.0, -1.5)
    assert lock.grasp_attempts == 1


# -- lift is never a transport ---------------------------------------------------------------
def test_lift_after_refinement_drift_is_not_transport(harness: SimPickHarness) -> None:
    """The lift is straight up at the FK XY, whatever point the model sends (no transport_gate)."""
    point = harness.image_point_for(*CUBE_XY)
    close = _pick_until_lift(harness, point)
    assert close["gripper_state"] == "closed"
    site_before = harness.site_xyz().copy()
    # The model re-sends a point 120 units away from the lock (far beyond the old 35-unit radius).
    drifted = [point[0] + 120.0, point[1] - 90.0]
    lift = _move(harness, drifted, 5.0)
    assert lift["status"] == "success", lift
    assert lift.get("error_type") is None
    xyz = lift["target_xyz_m"]
    assert abs(xyz[0] - site_before[0]) < 1e-6 and abs(xyz[1] - site_before[1]) < 1e-6
    assert "straight up" in lift["lift_note"]
    assert lift["grasp_verification"]["verdict"] == "confirmed"
    assert harness.state.phase == Phase.TRANSPORT
    assert harness.cube_xyz()[2] > 0.03


def test_place_path_not_refined_or_gated(harness: SimPickHarness) -> None:
    """In TRANSPORT the destination is raw: no refinement, no lock change, no descent gate."""
    point = harness.image_point_for(*CUBE_XY)
    _pick_until_lift(harness, point)
    assert _move(harness, point, 5.0)["grasp_verification"]["verdict"] == "confirmed"
    lock_id = harness.state.target.lock_id
    dest = harness.image_point_for(*DESTINATION_XY_M)

    from robot.vision_detector import refine_point_to_nearest_object as real_refine

    with patch("agent.tools.refine_point_to_nearest_object", side_effect=real_refine) as refine:
        travel = _move(harness, dest, 5.0)
        place = _move(harness, dest, 1.0)
    assert travel["status"] == "success" and place["status"] == "success"
    assert "perception" not in travel and "perception" not in place
    assert refine.call_count == 0
    assert harness.state.target.lock_id == lock_id
    assert harness.state.phase == Phase.TRANSPORT
    release = harness.call("open_gripper")
    assert release["gripper_state"] == "open"
    assert harness.state.target is None
    assert harness.state.phase == Phase.APPROACH
    placed = np.linalg.norm(harness.cube_xyz()[:2] - np.array(DESTINATION_XY_M)) * 1000.0
    assert placed < 15.0


def test_non_lift_motion_after_close_is_transport_gate_with_actionable_hint(
    harness: SimPickHarness,
) -> None:
    """After the close, presets and low moves are refused; the hint names the exact next calls."""
    point = harness.image_point_for(*CUBE_XY)
    _pick_until_lift(harness, point)
    refused = harness.call("move_to_preset", pose_name="HOME")
    assert refused["error_type"] == "transport_gate"
    assert refused["robot_moved"] is False
    assert "open_gripper()" in refused["error"]
    lift_call = parse_move_call(refused["error"])
    assert lift_call is not None and lift_call[1] == 5.0
    low = _move(harness, point, 0.0)
    assert low["error_type"] == "transport_gate"
    # ...and the named lift call works.
    assert _move(harness, lift_call[0], lift_call[1])["status"] == "success"


# -- descent gate and hints ------------------------------------------------------------------
def test_descent_gate_hint_leads_to_successful_descent(harness: SimPickHarness) -> None:
    """A refused descent names the hover call; issuing it verbatim leads to a successful descent."""
    point = harness.image_point_for(*CUBE_XY)
    harness.call("open_gripper")
    with patch_measurements([(6.0, 0.0), (0.5, 0.0)]):
        first = _move(harness, point, 3.0)
        assert first["perception"]["is_aligned"] is False
        assert first["status"] == "success"
        refused = _move(harness, point, 0.0)
        assert refused["error_type"] == "descent_gate"
        assert refused["robot_moved"] is False
        nxt = next_call_from(refused)
        assert nxt is not None and nxt[1] == 3.0
        second = _move(harness, nxt[0], nxt[1])
        assert second["perception"]["is_aligned"] is True
        descent = _move(harness, point, 0.0)
    assert descent["status"] == "success"
    assert descent.get("error_type") is None


def test_descent_requires_a_still_valid_hover(harness: SimPickHarness) -> None:
    """A preset move between the hover and the descent invalidates the hover."""
    point = harness.image_point_for(*CUBE_XY)
    with patch_measurements([(0.0, 0.0)]):
        _move(harness, point, 3.0)
    harness.call("move_to_preset", pose_name="HOME")
    refused = _move(harness, point, 0.0)
    assert refused["error_type"] == "perception"
    nxt = next_call_from(refused)
    assert nxt is not None and nxt[1] == 3.0


def test_unobservable_alignment_blocks_and_escalates_in_block_mode(harness: SimPickHarness) -> None:
    """FK-only alignment is unobservable: block refuses and escalates at once (no loop)."""
    point = harness.image_point_for(*CUBE_XY)
    harness.call("open_gripper")
    with patch_measurements([(0.0, 0.0)], source="kinematic"):
        hover = _move(harness, point, 3.0)
        assert hover["perception"]["is_aligned"] is None
        with patch.dict("os.environ", {"DESCENT_GATE": "block"}):
            refused = _move(harness, point, 0.0)
    assert refused["error_type"] == "descent_gate"
    assert "unobservable" in refused["error"]
    assert harness.state.escalated is True
    assert harness.state.escalation_reason == "alignment_unobservable"
    assert _move(harness, point, 3.0)["error_type"] == "retry_exhausted"


def test_unobservable_alignment_warns_and_descends_in_warn_mode(harness: SimPickHarness) -> None:
    """With DESCENT_GATE=warn an unobservable alignment descends with a warning."""
    point = harness.image_point_for(*CUBE_XY)
    harness.call("open_gripper")
    with patch_measurements([(0.0, 0.0)], source="kinematic"):
        _move(harness, point, 3.0)
        with patch.dict("os.environ", {"DESCENT_GATE": "warn"}):
            descent = _move(harness, point, 0.0)
    assert descent["status"] == "success"
    assert "unobservable" in descent["guardrail_warning"]


# -- escalation ------------------------------------------------------------------------------
def test_consecutive_refusals_escalate(harness: SimPickHarness) -> None:
    """Three refusals in a row without a successful motion escalate; a success resets the count."""
    point = harness.image_point_for(*CUBE_XY)
    harness.call("open_gripper")
    for expected in (1, 2):
        assert _move(harness, point, 0.0)["error_type"] == "perception"
        assert harness.state.consecutive_refusals == expected
    with patch_measurements([(0.0, 0.0)]):
        assert _move(harness, point, 3.0)["status"] == "success"
    assert harness.state.consecutive_refusals == 0

    harness.sim.reset("HOME")  # a fresh descent attempt without a valid hover
    for _ in range(3):
        _move(harness, point, 0.0)
    assert harness.state.escalated is True
    assert harness.state.escalation_reason == "refusals"
    blocked = _move(harness, point, 3.0)
    assert blocked["error_type"] == "retry_exhausted"
    assert blocked["robot_moved"] is False


def test_hover_cap_escalates(harness: SimPickHarness) -> None:
    """Four hovers at one lock without a successful descent escalate on the fifth request."""
    point = harness.image_point_for(*CUBE_XY)
    harness.call("open_gripper")
    with patch_measurements([(8.0, 0.0)]):
        for _ in range(MAX_HOVERS_PER_LOCK):
            assert _move(harness, point, 3.0)["status"] == "success"
        fifth = _move(harness, point, 3.0)
    assert fifth["error_type"] == "retry_exhausted"
    assert harness.state.escalation_reason == "hovers"


def test_open_gripper_allowed_when_escalated(harness: SimPickHarness) -> None:
    """open_gripper (and send_message) still work after escalation; motion does not."""
    harness.state.escalate("grasp_attempts")
    assert harness.call("move_to_preset", pose_name="HOME")["error_type"] == "retry_exhausted"
    assert harness.call("close_gripper")["error_type"] == "retry_exhausted"
    opened = harness.call("open_gripper")
    assert opened["status"] == "success"
    assert opened["gripper_state"] == "open"
    assert harness.call("send_message", message="stuck")["status"] == "success"


def test_three_missed_grasps_at_one_lock_escalate() -> None:
    """The third miss at the same lock escalates; counters are per lock."""
    state = GuardrailState()
    state.new_lock("red", (240.0, 0.0), [500.0, 500.0], [10, 10, 20, 20], 30.0, [500.0, 500.0])
    for expected in (1, 2, 3):
        state.on_closed(Phase.CLOSED_EMPTY, (240.0, 0.0))
        state.record_miss_or_unverified("closed_on_air")
        state.record_miss_or_unverified("object_still_on_table")  # same attempt: no double count
        assert state.retry_count == expected
        state.on_opened()
    assert state.escalated is True
    assert state.escalation_reason == "grasp_attempts"
    # A different object starts with a clean budget.
    state.new_lock("blue", (200.0, 50.0), [400.0, 400.0], [10, 10, 20, 20], 30.0, [400.0, 400.0])
    assert state.retry_count == 0


def test_perception_unavailable_verdict_does_not_consume_an_attempt() -> None:
    """A re-verify request (camera unusable) leaves the retry budget untouched."""
    state = GuardrailState()
    state.new_lock("red", (240.0, 0.0), [500.0, 500.0], [10, 10, 20, 20], 30.0, [500.0, 500.0])
    state.record_miss_or_unverified("perception_unavailable", consumes_attempt=False)
    assert state.retry_count == 0
    assert state.last_grasp_reason == "perception_unavailable"


# -- instruction boundary and epochs ---------------------------------------------------------
def test_new_instruction_keeps_closed_gripper_gated() -> None:
    """A gripper commanded closed that still reads stalled stays gated across instructions."""
    robot = MockRobot()
    robot.gripper_reading = {"aperture_deg": 25.0, "load": 150.0, "source": "measured"}
    controller = VLAController(mode="sim", robot=robot)
    controller.guardrail_state.on_closed(Phase.HELD_UNVERIFIED, (240.0, 0.0))
    controller._begin_guardrail_instruction()
    state = controller.guardrail_state
    assert state.phase == Phase.HELD_UNVERIFIED
    ctx = ToolContext(robot=robot, guardrail_state=state, mode="sim")
    refused = _handle_move_to_preset(ctx, {"pose_name": "HOME"})
    assert refused["error_type"] == "transport_gate"

    # A confirmed hold survives the boundary as TRANSPORT.
    state.phase = Phase.TRANSPORT
    state.last_verdict = {"verdict": "confirmed"}
    controller._begin_guardrail_instruction()
    assert state.phase == Phase.TRANSPORT


def test_home_pose_is_not_a_grasp() -> None:
    """HOME's partly closed gripper (-42 deg) reads 'stalled' but is not a grasp."""
    robot = MockRobot(qpos=PRESET_POSES["HOME"].copy())
    controller = VLAController(mode="sim", robot=robot)
    from robot.grasp import classify_grasp

    assert classify_grasp(robot.read_gripper_state())["grasped"] is True  # the trap
    assert controller.guardrail_state.last_commanded_gripper is None
    controller._begin_guardrail_instruction()
    assert controller.guardrail_state.phase == Phase.APPROACH
    ctx = ToolContext(robot=robot, guardrail_state=controller.guardrail_state, mode="sim")
    assert _handle_move_to_preset(ctx, {"pose_name": "HOME"})["status"] == "success"


def test_begin_instruction_mutates_in_place_and_resets_budget() -> None:
    """The state object survives; epoch advances; counters and escalation reset."""
    state = GuardrailState()
    same = state
    state.new_lock("red", (240.0, 0.0), [500.0, 500.0], None, None, [500.0, 500.0])
    state.record_miss_or_unverified("closed_on_air")
    state.escalate("grasp_attempts")
    epoch = state.epoch
    state.begin_instruction()
    assert state is same
    assert state.epoch == epoch + 1
    assert state.escalated is False
    assert state.target is None
    assert state.retry_count == 0


def test_interrupt_keeps_phase_but_drops_lock_and_hover() -> None:
    """An interrupt does not open the gripper; it invalidates the lock, hover and handlers."""
    state = GuardrailState()
    state.new_lock("red", (240.0, 0.0), [500.0, 500.0], None, None, [500.0, 500.0])
    state.on_closed(Phase.HELD_UNVERIFIED, (240.0, 0.0))
    epoch = state.epoch
    state.on_interrupt()
    assert state.epoch == epoch + 1
    assert state.target is None
    assert state.phase == Phase.HELD_UNVERIFIED


def test_stale_epoch_handler_does_not_write_state() -> None:
    """A handler that outlives an interrupt cannot touch the next instruction's state."""
    robot = MockRobot()
    state = GuardrailState()
    ctx = ToolContext(robot=robot, guardrail_state=state, mode="sim")

    def interrupt_mid_move(step_idx: int) -> None:
        if step_idx == 3:
            state.on_interrupt()

    robot.step_hook = interrupt_mid_move
    res = _handle_move_to_pose(ctx, {"qpos": [0.0] * 6, "duration_sec": 0.05})
    assert res["status"] == "success"
    assert state.last_motion is None  # the commit was dropped as stale

    robot.step_hook = None
    state.on_interrupt = lambda: None  # type: ignore[method-assign]
    epoch_before = state.epoch

    def bump(step_idx: int) -> None:
        if step_idx == 2:
            state.epoch += 1

    robot.step_hook = bump
    _handle_open_gripper(ctx, {})
    assert state.last_commanded_gripper is None
    assert state.epoch == epoch_before + 1


def test_lift_below_verification_height_is_raised_to_5cm(harness: SimPickHarness) -> None:
    """A 3 cm 'lift' is raised to 5 cm: lower, a lifted cube still overlaps its footprint."""
    point = harness.image_point_for(*CUBE_XY)
    _pick_until_lift(harness, point)
    lift = _move(harness, point, 3.0)
    assert lift["status"] == "success"
    assert lift["target_xyz_m"][2] == pytest.approx(0.05)
    assert "raised from 3.0 cm to 5.0 cm" in lift["lift_note"]
    assert lift["grasp_verification"]["verdict"] == "confirmed"
