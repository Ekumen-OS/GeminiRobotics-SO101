"""RealSO101Robot interface to communicate with physical SO-101 arm via Feetech servos."""

import json
import logging
import os
import threading
from pathlib import Path

import numpy as np

from robot.constants import JOINT_NAMES
from robot.interface import RobotInterface
from robot.settings import get_settings

logger = logging.getLogger(__name__)

try:
    from lerobot.motors.feetech import FeetechMotorsBus
    from lerobot.motors.motors_bus import Motor, MotorCalibration, MotorNormMode

    LEROBOT_AVAILABLE = True
except ImportError:
    LEROBOT_AVAILABLE = False

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

    def __init__(
        self,
        port: str | None = None,
        baudrate: int = 1_000_000,
        calibration_path: str | Path | None = None,
    ):
        """Initialize connection to physical Feetech motors bus with LeRobot calibration.

        Args:
            port: Serial port device path (e.g., /dev/tty.usbmodem1201 or /dev/ttyUSB0).
                Defaults to get_settings().serial_port.
            baudrate: Motors bus communication baudrate (default: 1,000,000).
            calibration_path: Explicit path to a LeRobot motor calibration JSON file.
                Defaults to resolve_calibration_path().

        Raises:
            ImportError: If the `lerobot` package is not available.
            FileNotFoundError: If the specified LeRobot calibration file does not exist.
            RuntimeError: If loading the calibration file fails.
            ConnectionError: If connection to the real robot hardware cannot be established.
        """
        self.port = port if port is not None else get_settings().serial_port
        self.baudrate = baudrate
        self.connected = False
        self._bus_lock = threading.Lock()

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
        except Exception as e:
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
        except Exception as e:
            raise RuntimeError(
                f"Could not load calibration file {self.calibration_path}: {e}"
            ) from e
        logger.info(f"Loaded LeRobot calibration profile from {self.calibration_path}")

        try:
            self.bus = FeetechMotorsBus(
                port=self.port,
                motors=self.motors,
                calibration=self.calibration,
            )
            self.bus.connect()

            # Ensure torque is enabled so motors actually move
            self.bus.enable_torque()

            self.connected = True
            logger.info(f"Successfully connected to real SO-101 arm on port {self.port}.")
        except Exception as e:
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
            raise ConnectionError("\n".join(err_details)) from e

        self._last_qpos = np.zeros(6, dtype=np.float32)
        # Initialize _last_qpos with actual current positions if possible
        if self.connected:
            self._last_qpos = self.get_joint_positions()
        self._stop_requested = False

    def set_joint_positions(self, qpos: np.ndarray) -> None:
        """Command target joint positions (in radians) to physical motors using calibration.

        Args:
            qpos: Target joint angles in radians for 6 joints.
        """
        qpos = np.asarray(qpos, dtype=np.float32)
        self._last_qpos = qpos.copy()

        if not self.connected or self.bus is None:
            return

        # If calibration is loaded, convert radians to degrees for LeRobot normalized write
        with self._bus_lock:
            if self.calibration is not None:
                joint_degrees = {}
                for i, joint_name in enumerate(JOINT_NAMES):
                    deg = float(np.degrees(qpos[i]))
                    joint_degrees[joint_name] = deg

                try:
                    # Use sync_write to update all joint Goal_Positions cleanly
                    self.bus.sync_write("Goal_Position", joint_degrees, normalize=True)
                except Exception as e:
                    logger.error(f"Error writing joint positions to bus: {e}")
            else:
                # Fallback raw position calculation if no calibration profile
                TICKS_PER_RAD = 4096.0 / (2.0 * np.pi)
                CENTER_TICK = 2048
                joint_dict = {}
                for i, joint_name in enumerate(JOINT_NAMES):
                    tick = int(round(CENTER_TICK + qpos[i] * TICKS_PER_RAD))
                    joint_dict[joint_name] = tick

                try:
                    self.bus.sync_write("Goal_Position", joint_dict, normalize=False)
                except Exception as e:
                    logger.error(f"Error writing raw joint positions: {e}")

    def get_joint_positions(self) -> np.ndarray:
        """Read current joint positions from physical motor encoders in radians.

        Returns:
            A numpy array of 6 joint positions in radians.
        """
        if not self.connected or self.bus is None:
            return self._last_qpos.copy()

        with self._bus_lock:
            try:
                if self.calibration is not None:
                    deg_dict = self.bus.sync_read("Present_Position", normalize=True)
                    qpos = np.zeros(6, dtype=np.float32)
                    for i, joint_name in enumerate(JOINT_NAMES):
                        if joint_name in deg_dict:
                            qpos[i] = np.radians(deg_dict[joint_name])
                        else:
                            qpos[i] = self._last_qpos[i]
                    return qpos
                else:
                    TICKS_PER_RAD = 4096.0 / (2.0 * np.pi)
                    CENTER_TICK = 2048
                    tick_dict = self.bus.sync_read("Present_Position", normalize=False)
                    qpos = np.zeros(6, dtype=np.float32)
                    for i, joint_name in enumerate(JOINT_NAMES):
                        if joint_name in tick_dict:
                            qpos[i] = (tick_dict[joint_name] - CENTER_TICK) / TICKS_PER_RAD
                        else:
                            qpos[i] = self._last_qpos[i]
                    return qpos
            except Exception as e:
                logger.error(f"Error reading present positions from bus: {e}")
                return self._last_qpos.copy()

    def disconnect(self) -> None:
        """Disconnect and release the serial bus."""
        if self.connected and self.bus is not None:
            try:
                self.bus.disconnect()
                self.connected = False
                logger.info("Disconnected from physical SO-101 robot arm.")
            except Exception as e:
                logger.error(f"Error disconnecting: {e}")

    def interrupt(self) -> None:
        """Interrupt any ongoing physical movement."""
        self._stop_requested = True

    def move_to_pose(
        self,
        target_qpos: np.ndarray,
        duration_sec: float = 1.0,
        dt: float = 0.02,
    ) -> None:
        """Smoothly interpolate from current joint positions to target joint positions.

        Args:
            target_qpos: Target joint angles in radians.
            duration_sec: Duration of the interpolation trajectory in seconds.
            dt: Timestep duration in seconds.
        """
        import time

        target_qpos = np.asarray(target_qpos, dtype=np.float32)
        start_qpos = self.get_joint_positions()
        num_steps = max(1, int(duration_sec / dt))

        for step_idx in range(1, num_steps + 1):
            if self._stop_requested:
                break
            t = step_idx / num_steps
            # Cosine interpolation for smooth acceleration/deceleration
            alpha = 0.5 * (1.0 - np.cos(np.pi * t))
            interp_qpos = (1.0 - alpha) * start_qpos + alpha * target_qpos
            self.set_joint_positions(interp_qpos)
            time.sleep(dt)

        self._stop_requested = False
