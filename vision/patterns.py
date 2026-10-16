"""Placeholder frame generators for camera and simulation feeds."""

import cv2
import numpy as np


def generate_raw_sim_placeholder_pattern() -> np.ndarray:
    """Generate a clean, un-annotated placeholder frame for MuJoCo sim mode.

    Returns:
        RGB image frame as a NumPy uint8 array of shape (480, 640, 3).
    """
    height, width = 480, 640
    return np.full((height, width, 3), (20, 35, 45), dtype=np.uint8)


def generate_no_camera_pattern() -> np.ndarray:
    """Generate a static instruction frame when no cameras are detected.

    Returns:
        RGB image frame as a NumPy uint8 array of shape (480, 640, 3).
    """
    height, width = 480, 640
    frame = np.full((height, width, 3), (35, 30, 35), dtype=np.uint8)
    frame[0:60, :] = (60, 45, 55)

    cv2.putText(
        frame,
        "VLA STREAMING: HARDWARE DETECTION",
        (25, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (230, 220, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "No Hardware Cameras Detected",
        (40, 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (100, 140, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "Connect physical USB camera and click Refresh.",
        (40, 250),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    return frame
