"""Shared video capture thread manager and streaming loop."""

import atexit
import logging
import threading
import time

import cv2

import vision.frames as frames_mod
from robot.settings import get_settings
from vision.calibrator import get_calibrator
from vision.camera import (
    _PREVIOUS_STREAM_RELEASE_TIMEOUT_S,
    _open_capture,
    _wait_for_previous_stream_release,
    get_camera_name,
)
from vision.overlays import (
    draw_workspace_overlay,
    is_calibration_overlay_enabled,
    is_target_overlay_enabled,
)
from vision.patterns import generate_no_camera_pattern
from vision.targets import draw_model_target_overlay

logger = logging.getLogger(__name__)

# Synchronization lock and atomic token for instant, zero-latency video source transitions
_stream_lock = threading.Lock()
_current_stream_id = 0
# Set by the active stream loop once it has released its capture device. A new capture thread
# waits on the previous event before opening hardware so two streams never fight over one node.
_stream_done_event: threading.Event | None = None


def prepare_stream_switch() -> None:
    """Invalidate atomic stream session token to trigger immediate capture teardown."""
    global _current_stream_id
    _current_stream_id = time.time_ns()


def _publish_no_camera_state() -> None:
    """Publish a fail-closed raw buffer (None) and a NO CAMERA annotated frame for the UI."""
    no_cam = generate_no_camera_pattern()
    with frames_mod._global_frame_lock:
        frames_mod._global_latest_raw_frame = None
        frames_mod._global_latest_annotated_frame = no_cam.copy()


def stream_video(camera_index: int = 0) -> None:
    """Capture real-time video frames and publish them to the shared frame buffers.

    Takes over the global stream token (stopping any previous capture) and registers a
    completion event so the next capture can wait for this one to release its camera.

    Args:
        camera_index: Integer index of target hardware capture device.
    """
    global _current_stream_id, _stream_done_event
    my_stream_id = time.time_ns()
    my_done = threading.Event()
    with _stream_lock:
        prev_done = _stream_done_event
        _stream_done_event = my_done
        _current_stream_id = my_stream_id
    try:
        _stream_video_impl(camera_index, my_stream_id, prev_done)
    finally:
        my_done.set()


def _stream_video_impl(
    camera_index: int,
    my_stream_id: int,
    prev_done: threading.Event | None,
) -> None:
    """Blocking capture body for :func:`stream_video`; see that function for the contract."""
    settings = get_settings()
    frame_interval = 1.0 / settings.stream_fps
    target_width = settings.stream_width
    target_height = settings.stream_height
    target_fps = settings.stream_fps

    if camera_index < 0:
        _publish_no_camera_state()
        return

    _wait_for_previous_stream_release(prev_done)

    cap = None
    cam_name = get_camera_name(camera_index)

    with _stream_lock:
        if _current_stream_id != my_stream_id:
            return

        logger.info(f"Initializing {cam_name} (Index {camera_index})...")
        cap = _open_capture(camera_index)
        if not cap.isOpened():
            logger.warning(
                f"Failed to open {cam_name} (Index {camera_index}): device busy or "
                "not a capture node. Stopping camera capture."
            )
            _publish_no_camera_state()
            return

        # Enable MJPG hardware video codec to prevent USB firmware FPS throttling
        cap.set(
            cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc("M", "J", "P", "G"),  # type: ignore[attr-defined]  # cv2 stubs omit top-level VideoWriter_fourcc
        )
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)
        cap.set(cv2.CAP_PROP_FPS, target_fps)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        real_fps = cap.get(cv2.CAP_PROP_FPS)
        logger.info(
            f"Connected to {cam_name} at {w}x{h} (~{real_fps:.0f} FPS) "
            f"via {cap.getBackendName()} backend."
        )

    try:
        while True:
            if _current_stream_id != my_stream_id:
                logger.info(f"Stopping stream for camera {camera_index} due to source transition.")
                break

            start_time = time.time()

            ret, frame = cap.read()
            if not ret or frame is None:
                logger.warning(f"Read failed for index {camera_index}; stopping camera capture.")
                _publish_no_camera_state()
                break

            raw_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            with frames_mod._global_frame_lock:
                frames_mod._global_latest_raw_frame = raw_frame.copy()
                frames_mod._global_latest_raw_ts = time.monotonic()

            ui_frame = raw_frame.copy()

            calib = get_calibrator()
            detected = calib.detect_markers(ui_frame)
            if len(detected) >= 4:
                calib.update_homography(detected)
            if is_calibration_overlay_enabled():
                ui_frame = draw_workspace_overlay(calib, ui_frame, detected)

            if is_target_overlay_enabled():
                ui_frame = draw_model_target_overlay(ui_frame)

            with frames_mod._global_frame_lock:
                frames_mod._global_latest_annotated_frame = ui_frame.copy()

            elapsed = time.time() - start_time
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)
    finally:
        if cap is not None and cap.isOpened():
            cap.release()
            logger.info(f"Released camera resource for index {camera_index}.")


# --- Shared camera capture -------------------------------------------------------------------
# The browser reads frames from /api/stream (MJPEG over one long-lived response), so capture must
# not depend on any Gradio session. One background thread drives ``stream_video`` -- which
# publishes the raw and annotated frame buffers -- and every client reads those buffers. A second
# browser tab therefore shares the stream instead of stealing it.
_capture_lock = threading.Lock()
_capture_thread: threading.Thread | None = None
_capture_key: int | None = None


def is_camera_alive() -> bool:
    """Return True when the camera capture thread is actively running."""
    return _capture_alive()


def capture_was_started() -> bool:
    """Return True if a shared camera capture thread has been started."""
    return _capture_thread is not None


def _capture_alive() -> bool:
    """Return True when the shared capture thread is running."""
    return _capture_thread is not None and _capture_thread.is_alive()


def _run_capture(camera_index: int) -> None:
    """Run ``stream_video`` so it publishes the frame buffers until stopped or camera lost."""
    try:
        stream_video(camera_index=camera_index)
    except Exception:
        logger.exception(f"Camera capture for index {camera_index} stopped unexpectedly.")


def _start_capture_locked(camera_index: int) -> None:
    """Start a capture thread; caller holds ``_capture_lock``. Supersedes any previous one."""
    global _capture_thread, _capture_key
    prepare_stream_switch()  # makes a still-running previous capture exit and release its camera
    _capture_key = camera_index
    _capture_thread = threading.Thread(
        target=_run_capture,
        args=(camera_index,),
        name="camera-capture",
        daemon=True,
    )
    _capture_thread.start()


def stop_camera_capture(timeout_s: float = _PREVIOUS_STREAM_RELEASE_TIMEOUT_S) -> None:
    """Stop the shared capture thread and wait for it to release its camera.

    Registered with ``atexit``: a daemon thread still blocked inside a V4L2 read while the
    interpreter tears down aborts the process (glibc "FATAL: exception not rethrown").

    Args:
        timeout_s: Maximum time to wait for the capture thread to exit.
    """
    global _capture_thread, _capture_key
    with _capture_lock:
        thread = _capture_thread
        _capture_thread = None
        _capture_key = None
    if thread is None or not thread.is_alive():
        return
    prepare_stream_switch()
    thread.join(timeout=timeout_s)


atexit.register(stop_camera_capture)


def ensure_camera_capture(camera_index: int) -> int:
    """Start shared capture if it is not running; never steal it from a running capture.

    Called on every page load, so a second tab joins the existing stream.

    Args:
        camera_index: Camera to open when nothing is capturing yet.

    Returns:
        The camera index that is actually being captured.
    """
    with _capture_lock:
        if not _capture_alive() or _capture_key is None:
            _start_capture_locked(camera_index)
        return _capture_key if _capture_key is not None else camera_index


def switch_camera_capture(camera_index: int, force: bool = False) -> int:
    """Switch shared capture to ``camera_index`` (restart only when it changed or ``force``).

    Args:
        camera_index: Requested camera index (< 0 selects no camera).
        force: Restart even when this camera is already active (e.g. after a replug).

    Returns:
        The camera index being captured after the call.
    """
    with _capture_lock:
        if force or not _capture_alive() or _capture_key != camera_index:
            _start_capture_locked(camera_index)
        return camera_index
