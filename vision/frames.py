"""Dual frame buffers, frame presence checks, and JPEG encoders for raw/annotated streams."""

import threading

import cv2
import numpy as np

from robot.registry import get_active_robot
from vision.calibrator import get_calibrator
from vision.overlays import is_calibration_overlay_enabled, is_target_overlay_enabled
from vision.patterns import (
    generate_raw_sim_placeholder_pattern,
    generate_sim_placeholder_pattern,
)
from vision.sim_runtime import _SIM_RUNTIME, _robot_can_render
from vision.targets import draw_model_target_overlay

_global_latest_raw_frame: np.ndarray | None = None
_global_latest_annotated_frame: np.ndarray | None = None
_global_latest_frame: np.ndarray | None = None
_global_frame_lock = threading.Lock()


def _on_sim_frame(raw_frame: np.ndarray | None) -> None:
    """Callback for simulation runtime worker to publish rendered frames to global buffers."""
    global _global_latest_raw_frame, _global_latest_annotated_frame, _global_latest_frame
    with _global_frame_lock:
        _global_latest_raw_frame = raw_frame.copy() if raw_frame is not None else None
        _global_latest_annotated_frame = raw_frame.copy() if raw_frame is not None else None
        _global_latest_frame = raw_frame.copy() if raw_frame is not None else None


_SIM_RUNTIME.set_frame_sink(_on_sim_frame)


def has_published_frame() -> bool:
    """Return True once a capture has published at least one frame to the shared buffers."""
    with _global_frame_lock:
        return any(
            frame is not None
            for frame in (
                _global_latest_annotated_frame,
                _global_latest_frame,
                _global_latest_raw_frame,
            )
        )


def get_latest_raw_jpeg() -> bytes | None:
    """Get the most recent camera frame as a 100% pristine, un-annotated JPEG.

    Guaranteed free of UI bounding boxes, ArUco annotations, grids, crosshairs,
    or label badge artifacts for Gemini Multimodal VLA ingestion.

    Returns:
        JPEG-encoded bytes of the raw RGB frame, or None if unavailable.
    """
    with _global_frame_lock:
        frame = _global_latest_raw_frame.copy() if _global_latest_raw_frame is not None else None

    if frame is None:
        robot = get_active_robot()
        if _robot_can_render(robot):
            frame = _SIM_RUNTIME.get_latest_raw_frame(robot)

    if frame is None:
        frame = generate_raw_sim_placeholder_pattern()

    if frame is None:
        return None

    # Convert RGB to BGR for standard OpenCV JPEG compression
    bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    success, encoded = cv2.imencode(".jpg", bgr_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 98])
    if success:
        return encoded.tobytes()
    return None


def get_latest_jpeg(quality: int = 95) -> bytes | None:
    """Get the most recent camera frame from any active stream as a JPEG with UI overlays.

    Used by Web UI endpoints (/api/stream, /api/sim_stream) and preview viewers.

    Args:
        quality: JPEG quality (1-100). The model path uses ``get_latest_raw_jpeg`` instead.

    Returns:
        JPEG-encoded bytes of the annotated RGB frame, or None if unavailable.
    """
    with _global_frame_lock:
        frame = (
            _global_latest_annotated_frame.copy()
            if _global_latest_annotated_frame is not None
            else (
                _global_latest_frame.copy()
                if _global_latest_frame is not None
                else (
                    _global_latest_raw_frame.copy()
                    if _global_latest_raw_frame is not None
                    else None
                )
            )
        )

    if frame is None:
        robot = get_active_robot()
        if _robot_can_render(robot):
            frame = _SIM_RUNTIME.get_latest_frame(robot)

    if frame is None:
        frame = generate_sim_placeholder_pattern()

    if frame is not None:
        if is_calibration_overlay_enabled():
            calib = get_calibrator(mode="sim")
            if calib.is_calibrated:
                frame = calib.draw_workspace_overlay(frame, detected_markers={}, show_markers=False)
        if is_target_overlay_enabled():
            frame = draw_model_target_overlay(frame)

    bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    success, encoded = cv2.imencode(
        ".jpg", bgr_frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    )
    if success:
        return encoded.tobytes()
    return None
