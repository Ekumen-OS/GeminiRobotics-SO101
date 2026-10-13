"""RealSO101Robot interface to communicate with physical SO-101 arm via Feetech servos."""

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from robot.constants import JOINT_NAMES
from robot.contracts import GripperReading
from robot.errors import HardwareError
from robot.interface import RobotInterface
from robot.logger import log_with_tag
from robot.settings import get_settings

logger = logging.getLogger(__name__)

try:
    from lerobot.motors.feetech import FeetechMotorsBus
    from lerobot.motors.motors_bus import Motor, MotorCalibration, MotorNormMode

    LEROBOT_AVAILABLE = True
except ImportError:
    LEROBOT_AVAILABLE = False

# After a trajectory the arm joints must reach the goal within this many degrees...
CONVERGENCE_TOL_DEG = 1.5
# ...within this long (s), polled this often; otherwise a WARN naming the per-joint error is logged
# ("convergence timeout") and the move returns anyway: the servos may simply not reach the goal.
CONVERGENCE_TIMEOUT_S = 2.0
CONVERGENCE_POLL_S = 0.03

# Fields every motor entry must define with a concrete value for the file to be usable.
_REQUIRED_CALIB_FIELDS = ("id", "drive_mode", "homing_offset", "range_min", "range_max")


def resolve_calibration_path() -> Path | None:
    """Return the first existing LeRobot motor calibration file for the follower.

    Candidates come from settings: the LeRobot cache written by `lerobot-calibrate`
    first, then the in-repo `config/<robot_id>.json` backup.

    Returns:
        Path to an existing calibration file, or None if no candidate exists.
    """
    for candidate in get_settings().motor_calibration_candidates:
        if candidate.exists():
            return candidate
    return None


class RealSO101Robot(RobotInterface):
    """Wrapper class for physical SO-101 robot arm using Feetech STS3215 servos."""

    mode = "real"
    convergence_timeout_s: float = CONVERGENCE_TIMEOUT_S

    def __init__(
        self,
        port: str | None = None,
        baudrate: int = 1_000_000,
        calibration_path: str | Path | None = None,
        connect_on_init: bool = True,
    ):
        """Initialize connection to physical Feetech motors bus with LeRobot calibration.

        Args:
            port: Serial port device path (e.g., /dev/tty.usbmodem1201 or /dev/ttyUSB0).
                Defaults to get_settings().serial_port.
            baudrate: Motors bus communication baudrate (default: 1,000,000).
            calibration_path: Explicit path to a LeRobot motor calibration JSON file.
                Defaults to resolve_calibration_path().
            connect_on_init: Whether to connect to hardware during initialization.

        Raises:
            ImportError: If the `lerobot` package is not available.
            FileNotFoundError: If the specified LeRobot calibration file does not exist.
            RuntimeError: If loading the calibration file fails.
            HardwareError: If connection to the real robot hardware cannot be established.
        """
        self.port = port if port is not None else get_settings().serial_port
        self.baudrate = baudrate
        self.connected = False
        self.bus: Any = None
        self._bus_lock = threading.Lock()
        self._last_qpos = np.zeros(6, dtype=np.float32)
        self._stop_requested = False

        if calibration_path is None:
            calibration_path = resolve_calibration_path()
        self.calibration_path = Path(calibration_path) if calibration_path is not None else None

        if not LEROBOT_AVAILABLE:
            raise ImportError(
                "The 'lerobot' package is not installed or available, "
                "but is required for physical robot hardware control."
            )

        # Define Motor objects with MotorNormMode.DEGREES so LeRobot normalizes to degrees
        self.motors = {
            "shoulder_pan": Motor(id=1, model="sts3215", norm_mode=MotorNormMode.DEGREES),
            "shoulder_lift": Motor(id=2, model="sts3215", norm_mode=MotorNormMode.DEGREES),
            "elbow_flex": Motor(id=3, model="sts3215", norm_mode=MotorNormMode.DEGREES),
            "wrist_flex": Motor(id=4, model="sts3215", norm_mode=MotorNormMode.DEGREES),
            "wrist_roll": Motor(id=5, model="sts3215", norm_mode=MotorNormMode.DEGREES),
            "gripper": Motor(id=6, model="sts3215", norm_mode=MotorNormMode.DEGREES),
        }

        # Load calibration (required: the arm cannot be driven safely without it)
        self.calibration = None
        if self.calibration_path is None or not self.calibration_path.exists():
            searched = "\n  ".join(str(p) for p in get_settings().motor_calibration_candidates)
            raise FileNotFoundError(
                "No LeRobot motor calibration file found. Searched:\n  "
                f"{searched}\n"
                "Run `lerobot-calibrate --robot.type=so101_follower "
                f"--robot.id={get_settings().follower_robot_id}` or fill in the "
                "config/ template by hand. Robot calibration is required to launch "
                "tasks on real hardware."
            )

        try:
            calib_data = json.loads(self.calibration_path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            raise RuntimeError(
                f"Could not read calibration file {self.calibration_path}: {e}"
            ) from e

        unfilled = [
            f"{name}.{key}"
            for name, cfg in calib_data.items()
            if isinstance(cfg, dict)
            for key in _REQUIRED_CALIB_FIELDS
            if cfg.get(key) is None
        ]
        if unfilled:
            raise RuntimeError(
                f"Calibration file {self.calibration_path} is an unfilled template "
                f"(missing values: {', '.join(unfilled)}). Run `lerobot-calibrate "
                f"--robot.type=so101_follower --robot.id={get_settings().follower_robot_id}` "
                "or fill the file in by hand."
            )

        try:
            self.calibration = {name: MotorCalibration(**cfg) for name, cfg in calib_data.items()}
        except (TypeError, ValueError, KeyError) as e:
            raise RuntimeError(
                f"Could not load calibration file {self.calibration_path}: {e}"
            ) from e
        logger.info(f"Loaded LeRobot calibration profile from {self.calibration_path}")

        if connect_on_init:
            self.connect()
            self._last_qpos = self.get_joint_positions()

    def connect(self) -> None:
        """Establish physical connection to Feetech motors bus and enable torque.

        Raises:
            HardwareError: If connection to the real robot hardware cannot be established.
        """
        with self._bus_lock:
            try:
                self.bus = FeetechMotorsBus(
                    port=self.port,
                    motors=self.motors,
                    calibration=self.calibration,
                )
                self.bus.connect()
                self.bus.enable_torque()
                self.connected = True
                logger.info(f"Successfully connected to real SO-101 arm on port {self.port}.")
            except Exception as e:
                self.connected = False
                port_path = Path(self.port)
                err_details = [f"Could not connect to physical robot on port {self.port}: {e}"]
                if port_path.exists() and not (
                    os.access(port_path, os.R_OK) and os.access(port_path, os.W_OK)
                ):
                    err_details.append(
                        f"\n[PERMISSION DENIED] Port '{self.port}' exists but lacks read/write "
                        "permissions for the current user.\n"
                        "Grant access on Linux with:\n"
                        f"  1. Immediate fix: sudo chmod 666 {self.port}\n"
                        "  2. Permanent fix: sudo usermod -a -G dialout $USER "
                        "(then log out & back in)"
                    )
                raise HardwareError("\n".join(err_details)) from e

    def set_joint_positions(self, qpos: np.ndarray) -> None:
        """Command target joint positions (in radians) to physical motors using calibration.

        Args:
            qpos: Target joint angles in radians for 6 joints.

        Raises:
            HardwareError: If robot is not connected or bus write fails.
        """
        qpos = np.asarray(qpos, dtype=np.float32)

        if not self.connected or self.bus is None:
            raise HardwareError(
                f"Cannot set joint positions: robot is not connected on port {self.port}."
            )

        with self._bus_lock:
            joint_degrees = {}
            for i, joint_name in enumerate(JOINT_NAMES):
                deg = float(np.degrees(qpos[i]))
                joint_degrees[joint_name] = deg

            try:
                self.bus.sync_write("Goal_Position", joint_degrees, normalize=True)
            except Exception as e:
                logger.error(f"Error writing joint positions to bus: {e}")
                raise HardwareError(f"Failed to write joint positions to bus: {e}") from e

            # Update _last_qpos only after a verified successful write
            self._last_qpos = qpos.copy()

    def get_joint_positions(self) -> np.ndarray:
        """Read current joint positions from physical motor encoders in radians.

        Returns:
            A numpy array of 6 joint positions in radians.

        Raises:
            HardwareError: If robot is not connected or bus read fails.
        """
        if not self.connected or self.bus is None:
            raise HardwareError(
                f"Cannot read joint positions: robot is not connected on port {self.port}."
            )

        with self._bus_lock:
            try:
                deg_dict = self.bus.sync_read("Present_Position", normalize=True)
            except Exception as e:
                logger.error(f"Error reading present positions from bus: {e}")
                raise HardwareError(f"Failed to read present positions from bus: {e}") from e

            qpos = np.zeros(6, dtype=np.float32)
            for i, joint_name in enumerate(JOINT_NAMES):
                if joint_name in deg_dict:
                    qpos[i] = np.radians(deg_dict[joint_name])
                else:
                    raise HardwareError(
                        f"Missing joint '{joint_name}' in Present_Position read from bus."
                    )
            return qpos

    def disconnect(self) -> None:
        """Disconnect and release the serial bus."""
        if self.connected and self.bus is not None:
            try:
                self.bus.disconnect()
                logger.info("Disconnected from physical SO-101 robot arm.")
            except Exception as e:
                logger.error(f"Error disconnecting: {e}")
            finally:
                self.connected = False

    def interrupt(self) -> None:
        """Interrupt any ongoing physical movement."""
        self.interrupt_epoch += 1
        self._stop_requested = True

    def read_gripper_state(self) -> GripperReading:
        """Read measured gripper state from physical motor encoder and load registers.

        Returns:
            GripperReading with source='measured' if bus read succeeds, or
            source='unavailable' if disconnected or read fails. Never reports
            _last_qpos as measured.
        """
        if not self.connected or self.bus is None:
            return {
                "aperture_deg": float(np.degrees(self._last_qpos[5])),
                "load": None,
                "source": "unavailable",
            }

        with self._bus_lock:
            try:
                deg_dict = self.bus.sync_read("Present_Position", normalize=True)
                if "gripper" not in deg_dict:
                    return {
                        "aperture_deg": float(np.degrees(self._last_qpos[5])),
                        "load": None,
                        "source": "unavailable",
                    }
                aperture_deg = float(deg_dict["gripper"])
            except Exception as e:
                logger.error(f"Error reading gripper present position from bus: {e}")
                return {
                    "aperture_deg": float(np.degrees(self._last_qpos[5])),
                    "load": None,
                    "source": "unavailable",
                }

            load: float | None = None
            try:
                raw_load = self.bus.read("Present_Load", "gripper", normalize=False)
                if raw_load is not None:
                    load = float(raw_load)
            except Exception as e:
                logger.debug(f"Optional Present_Load read on gripper skipped/failed: {e}")
                load = None

            return {
                "aperture_deg": aperture_deg,
                "load": load,
                "source": "measured",
            }

    def move_to_pose(
        self,
        target_qpos: np.ndarray,
        duration_sec: float = 1.0,
        dt: float = 0.02,
        cancel_epoch: int | None = None,
    ) -> None:
        """Smoothly interpolate from current joint positions to target joint positions.

        Args:
            target_qpos: Target joint angles in radians.
            duration_sec: Duration of the interpolation trajectory in seconds.
            dt: Timestep duration in seconds.
            cancel_epoch: ``interrupt_epoch`` captured before a multi-segment motion started.

        Raises:
            HardwareError: If physical communication fails or motion is interrupted.
        """
        if cancel_epoch is not None and cancel_epoch != self.interrupt_epoch:
            raise HardwareError("Motion interrupted", robot_moved=False)
        self._stop_requested = False
        target_qpos = np.asarray(target_qpos, dtype=np.float32)
        start_qpos = self.get_joint_positions()
        num_steps = max(1, int(duration_sec / dt))

        steps_written = 0
        try:
            for step_idx in range(1, num_steps + 1):
                if self._stop_requested:
                    raise HardwareError("Motion interrupted", robot_moved=(steps_written > 0))
                t = step_idx / num_steps
                # Cosine interpolation for smooth acceleration/deceleration
                alpha = 0.5 * (1.0 - np.cos(np.pi * t))
                interp_qpos = (1.0 - alpha) * start_qpos + alpha * target_qpos
                try:
                    self.set_joint_positions(interp_qpos)
                except HardwareError as e:
                    e.robot_moved = step_idx > 1
                    raise
                steps_written += 1
                time.sleep(dt)
                if self._stop_requested:
                    raise HardwareError("Motion interrupted", robot_moved=(steps_written > 0))
            self._wait_for_convergence(target_qpos)
        finally:
            self._stop_requested = False

    def _wait_for_convergence(self, target_qpos: np.ndarray) -> None:
        """Wait until the arm joints have reached the goal (the servos lag the commanded sweep).

        The measurement frame that follows a move must show the arm where it ended up, so the
        move does not return while ``Present_Position`` is still travelling. The gripper is
        excluded: it legitimately stalls on a grasped object.

        Args:
            target_qpos: Goal joint angles in radians.

        Raises:
            HardwareError: If a position read fails or the motion is interrupted while waiting.
        """
        timeout = float(self.convergence_timeout_s)
        if timeout <= 0.0:
            return
        deadline = time.monotonic() + timeout
        while True:
            if self._stop_requested:
                raise HardwareError("Motion interrupted", robot_moved=True)
            present = self.get_joint_positions()
            error_deg = np.degrees(np.abs(present[:5] - target_qpos[:5]))
            if float(error_deg.max()) < CONVERGENCE_TOL_DEG:
                return
            if time.monotonic() >= deadline:
                per_joint = ", ".join(
                    f"{name}={err:.1f}"
                    for name, err in zip(JOINT_NAMES[:5], error_deg, strict=False)
                )
                log_with_tag(
                    logger,
                    "WARN",
                    f"convergence timeout after {timeout:.1f}s; per-joint error (deg): {per_joint}",
                )
                return
            time.sleep(CONVERGENCE_POLL_S)
