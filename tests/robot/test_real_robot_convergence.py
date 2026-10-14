"""Unit tests for RealSO101Robot closed-loop sag compensation and convergence.

Validates gravity sag correction on STS3215 servos, contact guard detection,
offset clipping, gripper invariance, intent preservation, and geometric TCP convergence.
"""

import threading
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from robot.cartesian_ik import CartesianIK
from robot.constants import JOINT_NAMES
from robot.errors import HardwareError
from robot.real_robot import (
    MAX_SAG_COMP_DEG,
    SAG_TOL_DEG,
    RealSO101Robot,
)


class FakeClock:
    """Deterministic simulated clock advancing on sleep calls."""

    def __init__(self, start: float = 1000.0) -> None:
        """Initialize simulated clock.

        Args:
            start: Initial simulated epoch timestamp.
        """
        self.time = start

    def monotonic(self) -> float:
        """Return current simulated monotonic time in seconds."""
        return self.time

    def sleep(self, seconds: float) -> None:
        """Advance simulated time without consuming wall-clock time.

        Args:
            seconds: Duration in seconds to advance the clock.
        """
        self.time += seconds


class FakeFeetechBus:
    """Hermetic mock Feetech bus supporting per-joint sag and floor obstacle contact."""

    def __init__(
        self,
        sag_deg: dict[str, float] | None = None,
        floor_hold_tcp_dz_mm: float | None = None,
    ) -> None:
        """Initialize fake bus with joint sag specifications and optional contact floor.

        Args:
            sag_deg: Mapping of joint name to sag delta in degrees (droop under gravity).
            floor_hold_tcp_dz_mm: Optional displacement above target z where floor halts TCP.
        """
        self.sag_deg: dict[str, float] = sag_deg or {}
        self.floor_hold_tcp_dz_mm = floor_hold_tcp_dz_mm
        self.goals_deg: dict[str, float] = {name: 0.0 for name in JOINT_NAMES}
        self.written_goals: list[dict[str, float]] = []

    def sync_write(self, register: str, values: dict[str, float], normalize: bool = True) -> None:
        """Record sync_write commands and update internal servo targets.

        Args:
            register: Motor register name (e.g. 'Goal_Position').
            values: Mapping of joint name to angle in degrees.
            normalize: Whether calibration normalization is enabled.
        """
        if register == "Goal_Position":
            self.written_goals.append(dict(values))
            self.goals_deg.update(values)

    def sync_read(self, register: str, normalize: bool = True) -> dict[str, float]:
        """Return simulated encoder positions with sag applied.

        Args:
            register: Motor register name (e.g. 'Present_Position').
            normalize: Whether calibration normalization is enabled.

        Returns:
            Dictionary mapping joint names to present encoder positions in degrees.
        """
        if register == "Present_Position":
            res = {}
            for name in JOINT_NAMES:
                g = self.goals_deg.get(name, 0.0)
                sag = self.sag_deg.get(name, 0.0)
                res[name] = g - sag
            if self.floor_hold_tcp_dz_mm is not None:
                # Floor halts descent: shoulder_lift partial derivative dz/d(lift) is ~ -4.23 mm/deg
                res["shoulder_lift"] = self.goals_deg["shoulder_lift"] - (
                    self.floor_hold_tcp_dz_mm / 4.23
                )
            return res
        return {}


@pytest.fixture
def fake_clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    """Fixture providing deterministic time advancement for convergence polling.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        Configured FakeClock instance.
    """
    clock = FakeClock()
    monkeypatch.setattr("robot.real_robot.time.monotonic", clock.monotonic)
    monkeypatch.setattr("robot.real_robot.time.sleep", clock.sleep)
    return clock


def _make_mock_robot(bus: Any) -> RealSO101Robot:
    """Construct an initialized hermetic RealSO101Robot wrapping the provided bus.

    Args:
        bus: Mock bus instance.

    Returns:
        Configured RealSO101Robot ready for command dispatch.
    """
    robot = RealSO101Robot.__new__(RealSO101Robot)
    robot.port = "/dev/mock_feetech"
    robot.connected = True
    robot._bus_lock = threading.Lock()
    robot.bus = bus
    robot.calibration = {"joint": MagicMock()}
    robot._last_qpos = np.zeros(6, dtype=np.float32)
    robot._stop_requested = False
    robot.convergence_timeout_s = 2.0
    return robot


def test_sag_compensation_converges_within_two_rounds(
    fake_clock: FakeClock, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify sag of 3.6 deg / 1.9 deg settles within SAG_TOL_DEG in <= 2 rounds without warning."""
    bus = FakeFeetechBus({"shoulder_lift": -3.6, "elbow_flex": -1.9})
    robot = _make_mock_robot(bus)
    q_target = np.radians([-15.40, 41.63, -31.74, 80.11, 0.77, 0.0])
    robot.set_joint_positions(q_target)
    bus.written_goals.clear()

    robot.move_to_pose(q_target, duration_sec=0.04, dt=0.02)
    final_qpos = robot.get_joint_positions()
    per_joint_error_deg = np.abs(np.degrees(q_target[:5] - final_qpos[:5]))

    assert np.max(per_joint_error_deg) <= SAG_TOL_DEG
    assert len(bus.written_goals) >= 1
    last_write = bus.written_goals[-1]
    assert abs(last_write["shoulder_lift"] - (41.63 - 3.6)) < 0.1
    assert abs(last_write["elbow_flex"] - (-31.74 - 1.9)) < 0.1

    out, err = capsys.readouterr()
    assert "[SERVO] sag compensation round 1" in out
    assert "convergence timeout" not in err


def test_sag_exceeding_max_is_clipped_and_warns(
    fake_clock: FakeClock, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify sag exceeding MAX_SAG_COMP_DEG is clipped at 6 deg and emits a residual warning."""
    bus = FakeFeetechBus({"shoulder_lift": -8.0})
    robot = _make_mock_robot(bus)
    q_target = np.radians([-15.40, 41.63, -31.74, 80.11, 0.77, 0.0])
    robot.set_joint_positions(q_target)
    bus.written_goals.clear()

    robot.move_to_pose(q_target, duration_sec=0.04, dt=0.02)
    last_write = bus.written_goals[-1]
    applied_offset_deg = abs(last_write["shoulder_lift"] - 41.63)

    assert applied_offset_deg == pytest.approx(MAX_SAG_COMP_DEG, abs=0.01)

    out, err = capsys.readouterr()
    assert "[SERVO] sag compensation round 2" in out
    assert "convergence timeout after 2.0s" in err
    assert "shoulder_lift=2.0" in err
    assert "applied offset: shoulder_lift=-6.0" in err


def test_gripper_sag_is_never_compensated(fake_clock: FakeClock) -> None:
    """Verify gripper sag is excluded from compensation and its goal equals commanded target."""
    bus = FakeFeetechBus({"shoulder_lift": -3.6, "gripper": 10.0})
    robot = _make_mock_robot(bus)
    q_target = np.radians([-15.40, 41.63, -31.74, 80.11, 0.77, 0.0])
    robot.set_joint_positions(q_target)
    bus.written_goals.clear()

    robot.move_to_pose(q_target, duration_sec=0.04, dt=0.02)
    # Trajectory consists of 2 steps; subsequent writes are sag compensation
    compensation_writes = bus.written_goals[2:]
    for write_payload in compensation_writes:
        assert "gripper" not in write_payload


def test_contact_guard_blocks_correction_when_resting_on_floor(
    fake_clock: FakeClock, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify contact guard halts correction if floor holds TCP 10 mm above target z."""
    bus = FakeFeetechBus({"shoulder_lift": -3.6}, floor_hold_tcp_dz_mm=10.0)
    robot = _make_mock_robot(bus)
    q_target = np.radians([-15.40, 41.63, -31.74, 80.11, 0.77, 0.0])
    robot.set_joint_positions(q_target)
    bus.written_goals.clear()

    robot.move_to_pose(q_target, duration_sec=0.04, dt=0.02)
    compensation_writes = bus.written_goals[2:]
    assert len(compensation_writes) == 0

    out, _ = capsys.readouterr()
    assert "[SERVO] contact guard: present TCP z" in out
    assert "skipping sag correction" in out


def test_interrupt_mid_correction_raises_hardware_error(fake_clock: FakeClock) -> None:
    """Verify motion interruption during convergence wait raises HardwareError(robot_moved=True)."""
    bus = FakeFeetechBus({"shoulder_lift": -3.6})
    robot = _make_mock_robot(bus)
    q_target = np.radians([-15.40, 41.63, -31.74, 80.11, 0.77, 0.0])
    robot.set_joint_positions(q_target)

    call_count = 0

    def trigger_interrupt(seconds: float) -> None:
        nonlocal call_count
        call_count += 1
        fake_clock.sleep(seconds)
        if call_count >= 3:
            robot._stop_requested = True

    import robot.real_robot as rr

    rr.time.sleep = trigger_interrupt

    with pytest.raises(HardwareError) as exc_info:
        robot.move_to_pose(q_target, duration_sec=0.04, dt=0.02)

    assert exc_info.value.robot_moved is True
    assert "Motion interrupted" in str(exc_info.value)


def test_last_qpos_preserves_commanded_intent(fake_clock: FakeClock) -> None:
    """Verify _last_qpos retains commanded target angles rather than corrected servo goals."""
    bus = FakeFeetechBus({"shoulder_lift": -3.6, "elbow_flex": -1.9})
    robot = _make_mock_robot(bus)
    q_target = np.radians([-15.40, 41.63, -31.74, 80.11, 0.77, 0.0])
    robot.set_joint_positions(q_target)

    robot.move_to_pose(q_target, duration_sec=0.04, dt=0.02)
    assert np.allclose(robot._last_qpos, q_target)


def test_zero_convergence_timeout_returns_immediately(fake_clock: FakeClock) -> None:
    """Verify convergence_timeout_s <= 0 returns immediately without polling present positions."""
    bus = FakeFeetechBus({"shoulder_lift": -3.6})
    robot = _make_mock_robot(bus)
    robot.convergence_timeout_s = 0.0
    q_target = np.radians([-15.40, 41.63, -31.74, 80.11, 0.77, 0.0])
    robot.set_joint_positions(q_target)

    reads: list[str] = []
    original_sync_read = bus.sync_read

    def tracking_sync_read(register: str, normalize: bool = True) -> dict[str, float]:
        reads.append(register)
        return original_sync_read(register, normalize)

    bus.sync_read = tracking_sync_read

    robot.move_to_pose(q_target, duration_sec=0.04, dt=0.02)
    # move_to_pose performs exactly 1 read for start_qpos at start of interpolation
    assert len(reads) == 1


def test_grasp_pose_geometric_tcp_error_under_two_mm(fake_clock: FakeClock) -> None:
    """Verify forward kinematics TCP error is under 2 mm at the calibrated grasp reach pose."""
    bus = FakeFeetechBus({"shoulder_lift": -3.6, "elbow_flex": -1.9})
    robot = _make_mock_robot(bus)
    q_grasp_deg = np.array([-15.40, 41.63, -31.74, 80.11, 0.77, 0.0])
    q_grasp_rad = np.radians(q_grasp_deg)
    robot.set_joint_positions(q_grasp_rad)

    robot.move_to_pose(q_grasp_rad, duration_sec=0.04, dt=0.02)
    final_qpos = robot.get_joint_positions()

    ik = CartesianIK()
    tcp_cmd = ik.forward_kinematics(q_grasp_rad)
    tcp_final = ik.forward_kinematics(final_qpos)
    tcp_error_mm = float(np.linalg.norm(tcp_final - tcp_cmd) * 1000.0)

    # Documented geometric verification claim: TCP error must be < 2 mm
    assert tcp_error_mm < 2.0
    assert tcp_error_mm == pytest.approx(0.0, abs=0.01)
