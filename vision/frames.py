"""Dual frame buffers, frame presence checks, and JPEG encoders for raw/annotated streams."""

import threading
import time

import cv2
import numpy as np

from robot.registry import get_active_robot
from vision.calibrator import get_calibrator
from vision.overlays import (
    draw_workspace_overlay,
    is_calibration_overlay_enabled,
    is_target_overlay_enabled,
)
from vision.patterns import generate_raw_sim_placeholder_pattern
from vision.sim_runtime import _SIM_RUNTIME, _robot_can_render
from vision.targets import draw_model_target_overlay

_global_latest_raw_frame: np.ndarray | None = None
# time.monotonic() at which the raw frame was published (0.0 = never): lets a measurement wait
# for a frame captured *after* the arm finished moving instead of reading a stale one.
_global_latest_raw_ts: float = 0.0
_global_latest_annotated_frame: np.ndarray | None = None
_global_frame_lock = threading.Lock()


def _on_sim_frame(raw_frame: np.ndarray | None) -> None:
    """Callback for simulation runtime worker to publish rendered frames to global buffers."""
    global _global_latest_raw_frame, _global_latest_annotated_frame, _global_latest_raw_ts
    annotated: np.ndarray | None = None
    if raw_frame is not None:
        annotated = raw_frame.copy()
        if is_calibration_overlay_enabled():
            calib = get_calibrator(mode="sim")
            if calib.is_calibrated:
                annotated = draw_workspace_overlay(
                    calib, annotated, detected_markers={}, show_markers=False
                )
        if is_target_overlay_enabled():
            annotated = draw_model_target_overlay(annotated)

    with _global_frame_lock:
        _global_latest_raw_frame = raw_frame.copy() if raw_frame is not None else None
        _global_latest_raw_ts = time.monotonic()
        _global_latest_annotated_frame = annotated


_SIM_RUNTIME.set_frame_sink(_on_sim_frame)


def has_published_frame() -> bool:
    """Return True once a capture has published at least one frame to the shared buffers."""
    with _global_frame_lock:
        return any(
            frame is not None
            for frame in (
                _global_latest_annotated_frame,
                _global_latest_raw_frame,
            )
        )


def get_latest_raw_frame_ts() -> float:
    """Return the ``time.monotonic()`` publish time of the latest raw frame (0.0 if none)."""
    with _global_frame_lock:
        return _global_latest_raw_ts


def get_raw_jpeg_after(
    t_after: float, timeout_s: float = 0.5, poll_s: float = 0.02
) -> bytes | None:
    """Return a pristine raw JPEG published after ``t_after``, waiting up to ``timeout_s``.

    Used after a motion: a frame published before the arm arrived shows the arm somewhere else,
    so a measurement must wait for a newer one. With no capture or sim render thread publishing
    (nothing could ever arrive) the current frame is returned immediately instead of timing out.

    Args:
        t_after: ``time.monotonic()`` value the frame must be newer than (end of the motion).
        timeout_s: Longest wait for a newer frame.
        poll_s: Poll interval.

    Returns:
        JPEG bytes, or None if no newer frame arrived in time (the measurement is unavailable).
    """
    from vision.capture import capture_was_started, is_camera_alive

    publishing = _SIM_RUNTIME.is_running() or (capture_was_started() and is_camera_alive())
    if not publishing:
        return get_latest_raw_jpeg()
    deadline = time.monotonic() + timeout_s
    while True:
        if get_latest_raw_frame_ts() > t_after:
            return get_latest_raw_jpeg()
        if time.monotonic() >= deadline:
            return None
        time.sleep(poll_s)


def get_latest_raw_jpeg() -> bytes | None:
    """Get the most recent camera frame as a 100% pristine, un-annotated JPEG.

    Guaranteed free of UI bounding boxes, ArUco annotations, grids, crosshairs,
    or label badge artifacts for Gemini Multimodal VLA ingestion.

    Returns:
        JPEG-encoded bytes of the raw RGB frame, or None if unavailable.
    """
    with _global_frame_lock:
        frame = _global_latest_raw_frame.copy() if _global_latest_raw_frame is not None else None

    robot = get_active_robot()
    if not _robot_can_render(robot):
        from vision.capture import capture_was_started, is_camera_alive

        if frame is None:
            return None
        if capture_was_started() and not is_camera_alive():
            return None
    else:
        if frame is None:
            frame = _SIM_RUNTIME.get_latest_frame(robot)
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

    Used by the Web UI endpoint (/api/stream) and preview viewers.

    Args:
        quality: JPEG quality (1-100). The model path uses ``get_latest_raw_jpeg`` instead.

    Returns:
        JPEG-encoded bytes of the annotated RGB frame, or None if unavailable.
    """
    with _global_frame_lock:
        frame = (
            _global_latest_annotated_frame.copy()
            if _global_latest_annotated_frame is not None
            else (_global_latest_raw_frame.copy() if _global_latest_raw_frame is not None else None)
        )

    if frame is None:
        robot = get_active_robot()
        if _robot_can_render(robot):
            frame = _SIM_RUNTIME.get_latest_frame(robot)

    if frame is None:
        frame = generate_raw_sim_placeholder_pattern()

    bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    success, encoded = cv2.imencode(
        ".jpg", bgr_frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    )
    if success:
        return encoded.tobytes()
    return None
