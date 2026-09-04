"""Constants and configuration for the SO-101 robot arm."""

import os

import numpy as np
from dotenv import load_dotenv

load_dotenv()

# Preset target joint angles (in radians) for standard arm poses
PRESET_POSES: dict[str, np.ndarray] = {
    "HOME": np.array([-0.000, -1.840, 1.580, 1.273, 0.000, -0.733]),
    "MIDDLE": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "SWEEP_HOME": np.array([0.0, -1.358, 0.854, 0.962, 0.0, 0.0]),
}

JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

ACTUATOR_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

# Hardware joint limits in degrees (min, max)
JOINT_LIMITS = {
    "shoulder_pan": (-100.0, 100.0),
    "shoulder_lift": (-115.0, 115.0),
    "elbow_flex": (-110.0, 110.0),
    "wrist_flex": (-110.0, 110.0),
    "wrist_roll": (-180.0, 180.0),
    "gripper": (-75.0, 75.0),
}

# Serial communication configuration (can be overridden via ROBOT_PORT or SO101_PORT env var)
DEFAULT_SERIAL_PORT = os.environ.get(
    "ROBOT_PORT",
    os.environ.get("SO101_PORT", "/dev/tty.usbmodem5B415332861"),
)
