"""Constants and configuration for the SO-101 robot arm."""

import numpy as np

# Preset target joint angles (in radians) for standard arm poses
PRESET_POSES: dict[str, np.ndarray] = {
    "HOME": np.array([-0.000, -1.840, 1.580, 1.273, 0.000, -1.005]),
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
