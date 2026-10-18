"""Per-lock mm bias corrector, measurement freshness and real-arm convergence (3e, Step 7)."""

import threading
import time
from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from agent.guardrails import GuardrailState, Phase
from agent.pick_loop import IMPLAUSIBLE_RESIDUAL_MM
from robot.constants import JOINT_NAMES
from robot.real_robot import CONVERGENCE_TOL_DEG, RealSO101Robot
from tests.harness.pick_helpers import next_call_from, patch_measurements
from tests.harness.sim_pick import SimPickHarness, settings_override
from tests.harness.tools_sync import execute_tool_call_sync

CUBE_XY = (0.24, 0.0)


@pytest.fixture(scope="module")
def _shared_biased() -> Iterator[SimPickHarness]:
    """One harness with a hidden 1.5 deg shoulder-pan encoder offset (FK cannot see it)."""
    h = SimPickHarness(pan_offset_deg=1.5)
    yield h
    h.close()


@pytest.fixture(scope="module")
def _shared_unbiased() -> Iterator[SimPickHarness]:
    """One unbiased sim harness for the module."""
    h = SimPickHarness()
    yield h
    h.close()


@pytest.fixture
def biased_harness(_shared_biased: SimPickHarness) -> SimPickHarness:
    """The biased harness reset to HOME with the default cube."""
    _shared_biased.reset(CUBE_XY)
    return _shared_biased


@pytest.fixture
def harness(_shared_unbiased: SimPickHarness) -> SimPickHarness:
    """The unbiased harness reset to HOME with the default cube."""
    _shared_unbiased.reset(CUBE_XY)
    return _shared_unbiased


def _move(h: SimPickHarness, point: list[float], z: float) -> dict[str, Any]:
    return h.call("move_to_spatial_point", point=point, z_height_cm=z, approach="top_down")


def _jaw_to_lock_mm(h: SimPickHarness) -> float:
    lock = h.state.target
    return float(np.linalg.norm(h.site_xyz()[:2] * 1000.0 - np.array(lock.centre_xy_mm)))


def test_corrector_converges_in_one_update(biased_harness: SimPickHarness) -> None:
    """Constant bias b: hover 1 measures b, hover 2 lands within 2 mm, descent = centre + c."""
    h = biased_harness
    point = h.image_point_for(*CUBE_XY)
    h.call("open_gripper")

    first = _move(h, point, 3.0)
    lock = h.state.target
    miss_mm = _jaw_to_lock_mm(h)
    assert miss_mm > 5.0  # the encoder offset is a real 6-7 mm miss
    assert first["perception"]["gripper_source"] == "aruco"
    assert first["perception"]["is_aligned"] is False
    assert first["applied_correction_mm"] == [0.0, 0.0]
    c_after_first = lock.c_mm
    assert float(np.hypot(*c_after_first)) == pytest.approx(miss_mm, abs=2.0)

    second = _move(h, point, 3.0)
    assert second["applied_correction_mm"] == [c_after_first[0], c_after_first[1]]
    assert second["perception"]["is_aligned"] is True
    assert _jaw_to_lock_mm(h) < 2.0

    descent = _move(h, point, 0.0)
    assert descent["status"] == "success"
    commanded = np.array(descent["target_xyz_m"][:2]) * 1000.0
    assert commanded == pytest.approx(
        [lock.centre_xy_mm[0] + lock.c_mm[0], lock.centre_xy_mm[1] + lock.c_mm[1]], abs=1e-6
    )
    assert _jaw_to_lock_mm(h) < 2.0


def test_corrector_sign_is_target_minus_jaw(harness: SimPickHarness) -> None:
    """The residual r = target - jaw is added to c; the next command moves by +r."""
    h = harness
    point = h.image_point_for(*CUBE_XY)
    with patch_measurements([(3.0, -2.0), (0.0, 0.0)]):
        _move(h, point, 3.0)
        lock = h.state.target
        assert lock.c_mm == pytest.approx((3.0, -2.0))
        second = _move(h, point, 3.0)
    assert second["applied_correction_mm"] == pytest.approx([3.0, -2.0])
    commanded = np.array(second["target_xyz_m"][:2]) * 1000.0
    assert commanded == pytest.approx(
        [lock.centre_xy_mm[0] + 3.0, lock.centre_xy_mm[1] - 2.0], abs=1e-6
    )


def test_corrector_accumulates_residuals_per_lock(harness: SimPickHarness) -> None:
    """Successive residuals accumulate: c <- c + r."""
    point = harness.image_point_for(*CUBE_XY)
    with patch_measurements([(4.0, 0.0), (1.5, 1.0), (0.2, 0.0)]):
        for _ in range(3):
            _move(harness, point, 3.0)
    assert harness.state.target.c_mm == pytest.approx((5.7, 1.0))


def test_corrector_rejects_implausible_residual(harness: SimPickHarness) -> None:
    """A residual over 30 mm is a mis-detection: perception refusal, no update, no hover record."""
    point = harness.image_point_for(*CUBE_XY)
    with patch_measurements([(IMPLAUSIBLE_RESIDUAL_MM + 5.0, 0.0)]):
        res = _move(harness, point, 3.0)
    assert res["error_type"] == "perception"
    assert res["robot_moved"] is True  # the arm did move to the hover position
    assert "Implausible" in res["error"]
    assert harness.state.target.c_mm == (0.0, 0.0)
    assert harness.state.last_hover is None
    assert harness.state.consecutive_refusals == 1
    refused = _move(harness, point, 0.0)
    assert refused["error_type"] == "perception"
    nxt = next_call_from(refused)
    assert nxt is not None and nxt[1] == 3.0


def test_bias_correction_off_never_learns(harness: SimPickHarness) -> None:
    """BIAS_CORRECTION=0 keeps c at zero (for A/B runs)."""
    point = harness.image_point_for(*CUBE_XY)
    with settings_override(bias_correction=False), patch_measurements([(6.0, 0.0)]):
        _move(harness, point, 3.0)
        second = _move(harness, point, 3.0)
    assert harness.state.target.c_mm == (0.0, 0.0)
    assert second["applied_correction_mm"] == [0.0, 0.0]


def test_corrector_ignores_fk_measurements(harness: SimPickHarness) -> None:
    """Only the jaw tag feeds the corrector: FK agrees with the IK by construction."""
    point = harness.image_point_for(*CUBE_XY)
    with patch_measurements([(6.0, 0.0)], source="kinematic"):
        _move(harness, point, 3.0)
    assert harness.state.target.c_mm == (0.0, 0.0)


def test_corrector_not_updated_by_the_descent(harness: SimPickHarness) -> None:
    """The residual is learned at the hover only; a descent measurement leaves c alone."""
    point = harness.image_point_for(*CUBE_XY)
    harness.call("open_gripper")
    with patch_measurements([(0.0, 0.0), (5.0, 5.0)]):
        _move(harness, point, 3.0)
        _move(harness, point, 0.0)
    assert harness.state.target.c_mm == (0.0, 0.0)


def test_corrector_kept_across_retry_reset_on_new_lock() -> None:
    """C survives a retry at the same lock; a new lock, an instruction or an interrupt resets it."""
    state = GuardrailState()
    lock = state.new_lock("red", (240.0, 0.0), [500.0, 500.0], [10, 10, 20, 20], 30.0)
    lock.c_mm = (3.0, -1.0)
    state.on_closed(Phase.CLOSED_EMPTY, (240.0, 0.0))
    state.record_miss_or_unverified("closed_on_air")
    state.on_opened()
    state.reacquire(lock, (241.0, 0.5), [501.0, 500.0], [10, 10, 20, 20], 30.0)
    assert state.target is lock and lock.c_mm == (3.0, -1.0)  # kept across the retry

    other = state.new_lock("blue", (200.0, 60.0), [400.0, 400.0], [1, 1, 2, 2], 30.0)
    assert other.c_mm == (0.0, 0.0)
    other.c_mm = (2.0, 2.0)
    state.on_interrupt()
    assert state.target is None
    again = state.new_lock("blue", (200.0, 60.0), [400.0, 400.0], [1, 1, 2, 2], 30.0)
    again.c_mm = (2.0, 2.0)
    state.begin_instruction()
    assert state.target is None


# -- measurement freshness ---------------------------------------------------------------------
@pytest.mark.usefixtures("real_sleep")
def test_measurement_waits_for_fresh_frame(harness: SimPickHarness) -> None:
    """The measurement asks for a frame newer than the end of the motion, never an old one."""
    point = harness.image_point_for(*CUBE_XY)
    requests: list[tuple[float, float, float]] = []

    def spy(t_after: float, timeout_s: float) -> bytes | None:
        requests.append((t_after, time.monotonic(), timeout_s))
        return harness.fresh_jpeg(t_after, timeout_s)

    t_start = time.monotonic()
    from tests.conftest import DummyToolCall

    res = execute_tool_call_sync(
        DummyToolCall(
            "move_to_spatial_point", {"point": point, "z_height_cm": 3.0, "approach": "top_down"}
        ),
        harness.robot,
        get_latest_jpeg_fn=harness.jpeg,
        get_fresh_jpeg_fn=spy,
        mode="sim",
        guardrail_state=harness.state,
    )
    assert res["status"] == "success"
    assert len(requests) == 1
    t_after, t_called, timeout_s = requests[0]
    assert t_start < t_after <= t_called  # motion end is after the call began, before the read
    assert 0.0 < timeout_s <= 1.0
    assert res["perception"]["gripper_source"] == "aruco"


def test_no_fresh_frame_means_unmeasured_not_failed(harness: SimPickHarness) -> None:
    """If no newer frame arrives the motion still succeeded; the measurement is unavailable."""
    point = harness.image_point_for(*CUBE_XY)
    from tests.conftest import DummyToolCall

    res = execute_tool_call_sync(
        DummyToolCall(
            "move_to_spatial_point", {"point": point, "z_height_cm": 3.0, "approach": "top_down"}
        ),
        harness.robot,
        get_latest_jpeg_fn=harness.jpeg,
        get_fresh_jpeg_fn=lambda t_after, timeout_s: None,
        mode="sim",
        guardrail_state=harness.state,
    )
    assert res["status"] == "success"
    assert "perception" not in res
    assert any("newer than the end of the motion" in w for w in res["warnings"])
    assert harness.state.last_hover is None
    harness.call("open_gripper")
    refused = harness.call(
        "move_to_spatial_point", point=point, z_height_cm=0.0, approach="top_down"
    )
    assert refused["error_type"] == "perception"  # no measured hover to descend from


# -- real-arm convergence ----------------------------------------------------------------------
def _real_robot(bus: MagicMock) -> RealSO101Robot:
    robot = RealSO101Robot.__new__(RealSO101Robot)
    robot.port = "/dev/mock_tty"
    robot.connected = True
    robot._bus_lock = threading.Lock()
    robot.bus = bus
    robot.calibration = {"joint": MagicMock()}
    robot._last_qpos = np.zeros(6, dtype=np.float32)
    robot._stop_requested = False
    return robot


def test_real_move_waits_for_convergence() -> None:
    """move_to_pose does not return while Present_Position is still travelling to the goal."""
    goal_deg = 20.0
    reads = {"n": 0}

    def sync_read(register: str, normalize: bool = True) -> dict[str, float]:
        reads["n"] += 1
        # The servos lag: 3 reads still on the way, then within tolerance.
        progress = {1: 8.0, 2: 14.0, 3: 18.0}.get(reads["n"], goal_deg - 0.4)
        return {name: progress for name in JOINT_NAMES}

    bus = MagicMock()
    bus.sync_read.side_effect = sync_read
    robot = _real_robot(bus)
    robot.convergence_timeout_s = 2.0
    target = np.radians(np.full(6, goal_deg, dtype=np.float32))
    with patch("robot.real_robot.time.sleep"):
        robot.move_to_pose(target, duration_sec=0.04, dt=0.02)
    assert reads["n"] == 4  # 3 reads while travelling + the one that converged
    assert bus.sync_write.call_count == 2


def test_real_move_convergence_timeout_warns_with_per_joint_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """If the servos never arrive the move returns after a WARN naming each joint's error."""
    bus = MagicMock()
    bus.sync_read.return_value = {name: 0.0 for name in JOINT_NAMES}
    robot = _real_robot(bus)
    robot.convergence_timeout_s = 0.05
    target = np.radians(np.full(6, 10.0, dtype=np.float32))
    robot.move_to_pose(target, duration_sec=0.04, dt=0.02)
    err = capsys.readouterr().err
    assert "convergence timeout" in err
    assert "shoulder_pan=10.0" in err
    assert CONVERGENCE_TOL_DEG < 10.0


def test_real_move_convergence_can_be_interrupted() -> None:
    """An interrupt while waiting for the servos aborts the wait (robot_moved is True)."""
    from robot.errors import HardwareError

    bus = MagicMock()
    bus.sync_read.return_value = {name: 0.0 for name in JOINT_NAMES}
    robot = _real_robot(bus)
    robot.convergence_timeout_s = 5.0
    target = np.radians(np.full(6, 10.0, dtype=np.float32))

    def interrupt_soon(_: float) -> None:
        robot._stop_requested = True

    with patch("robot.real_robot.time.sleep", side_effect=interrupt_soon):
        with pytest.raises(HardwareError) as exc:
            robot.move_to_pose(target, duration_sec=0.02, dt=0.02)
    assert exc.value.robot_moved is True
