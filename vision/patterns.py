"""Synthetic test patterns and placeholder frame generators for camera feeds."""

import time

import cv2
import numpy as np

from vision.camera import get_camera_name


def generate_raw_sim_placeholder_pattern() -> np.ndarray:
    """Generate a clean, un-annotated placeholder frame for MuJoCo sim mode.

    Returns:
        RGB image frame as a NumPy uint8 array of shape (480, 640, 3).
    """
    height, width = 480, 640
    return np.full((height, width, 3), (20, 35, 45), dtype=np.uint8)


def generate_mock_test_pattern(camera_index: int = -1) -> np.ndarray:
    """Generate an animated synthetic test pattern frame.

    Args:
        camera_index: Camera index being simulated. Defaults to -1.

    Returns:
        RGB image frame as a NumPy uint8 array of shape (480, 640, 3).
    """
    height, width = 480, 640
    t = time.time()
    x = np.linspace(0, 255, width)
    y = np.linspace(0, 255, height)
    xx, yy = np.meshgrid(x, y)
    frame = np.dstack(
        [
            (xx + (t * 50)) % 256,
            (yy + (t * 50)) % 256,
            np.full_like(xx, 128 + 127 * np.sin(t)),
        ]
    ).astype(np.uint8)

    cam_name = get_camera_name(camera_index)
    status_text = (
        f"{cam_name} (Index {camera_index}) Unavailable"
        if camera_index >= 0
        else "Synthetic Mock Test Feed Active"
    )
    cv2.putText(
        frame,
        status_text,
        (30, 220),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "Streaming Mock VLA Observation Feed",
        (30, 260),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (220, 220, 220),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Timestamp: {t:.2f}s",
        (30, 430),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 255, 200),
        1,
        cv2.LINE_AA,
    )
    return frame


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
        "1. Connect physical USB camera and click Refresh.",
        (40, 250),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "2. Or select Synthetic Mock Feed from dropdown.",
        (40, 290),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (200, 220, 220),
        1,
        cv2.LINE_AA,
    )
    return frame


def generate_loading_pattern(
    camera_index: int, status_message: str = "Connecting to source..."
) -> np.ndarray:
    """Generate a visual loading screen frame during stream initialization.

    Args:
        camera_index: Target camera index being connected.
        status_message: Informational status string displayed on the frame.

    Returns:
        RGB image frame as a NumPy uint8 array of shape (480, 640, 3).
    """
    height, width = 480, 640
    frame = np.full((height, width, 3), (30, 35, 45), dtype=np.uint8)
    frame[0:60, :] = (45, 55, 75)

    cv2.putText(
        frame,
        "VLA STREAMING: SWITCHING SOURCE",
        (25, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (230, 240, 255),
        2,
        cv2.LINE_AA,
    )

    cam_name = get_camera_name(camera_index)
    source_name = f"{cam_name} (Idx {camera_index})" if camera_index >= 0 else "Synthetic Mock Feed"
    cv2.putText(
        frame,
        f"Target Source: {source_name}",
        (40, 200),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Status: {status_message}",
        (40, 260),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (100, 220, 255),
        2,
        cv2.LINE_AA,
    )
    return frame


def generate_sim_placeholder_pattern() -> np.ndarray:
    """Generate an animated placeholder frame for MuJoCo sim offscreen rendering mode.

    Returns:
        RGB image frame as a NumPy uint8 array of shape (480, 640, 3).
    """
    height, width = 480, 640
    t = time.time()
    frame = np.full((height, width, 3), (20, 35, 45), dtype=np.uint8)
    frame[0:60, :] = (30, 75, 95)

    cv2.putText(
        frame,
        "GEMINI ROBOTICS SO-101 — MODE: SIMULATION",
        (20, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (220, 240, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "MuJoCo Offscreen RGB Renderer",
        (50, 200),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (100, 230, 180),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "Placeholder Feed — Available in Next Session",
        (50, 250),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Simulation Timestamp: {t:.2f}s",
        (50, 420),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (160, 200, 200),
        1,
        cv2.LINE_AA,
    )
    return frame
