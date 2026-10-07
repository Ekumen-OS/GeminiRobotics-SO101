"""Shared video capture thread manager and streaming generator loop."""

import atexit
import logging
import threading
import time
from typing import Any

import cv2

from robot.settings import get_settings
from vision import frames as frames_mod
from vision.calibrator import get_calibrator
from vision.camera import (
    _PREVIOUS_STREAM_RELEASE_TIMEOUT_S,
    _open_capture,
    _wait_for_previous_stream_release,
    get_camera_name,
)
from vision.overlays import is_calibration_overlay_enabled, is_target_overlay_enabled
from vision.patterns import (
    generate_loading_pattern,
    generate_mock_test_pattern,
    generate_no_camera_pattern,
    generate_raw_sim_placeholder_pattern,
)
from vision.sim_runtime import get_sim_video_frame
from vision.targets import draw_model_target_overlay

logger = logging.getLogger(__name__)

# Synchronization lock and atomic token for instant, zero-latency video source transitions
_stream_lock = threading.Lock()
_current_stream_id = 0
# Set by the active stream generator once it has released its capture device. A new generator
# waits on the previous event before opening hardware so two streams never fight over one node.
_stream_done_event: threading.Event | None = None


def prepare_stream_switch() -> None:
    """Invalidate atomic stream session token to trigger immediate generator teardown."""
    global _current_stream_id
    _current_stream_id = time.time_ns()


def stream_video(
    camera_index: int = 0,
    mode: str = "real",
    robot: Any | None = None,
):
    """Capture real-time video frames and yield them to Gradio over Server-Sent Events.

    Takes over the global stream token (stopping any previous generator) and registers a
    completion event so the next generator can wait for this one to release its camera.

    Args:
        camera_index: Integer index of target hardware capture device.
        mode: Operation mode ('real' or 'sim').
        robot: Active robot controller object used for MuJoCo rendering in sim mode.

    Yields:
        NumPy array of shape (H, W, 3) containing RGB video frames.
    """
    global _current_stream_id, _stream_done_event
    my_stream_id = time.time_ns()
    my_done = threading.Event()
    with _stream_lock:
        prev_done = _stream_done_event
        _stream_done_event = my_done
        _current_stream_id = my_stream_id
    try:
        yield from _stream_video_impl(camera_index, mode, robot, my_stream_id, prev_done)
    finally:
        my_done.set()


def _stream_video_impl(
    camera_index: int,
    mode: str,
    robot: Any | None,
    my_stream_id: int,
    prev_done: threading.Event | None,
):
    """Generator body for :func:`stream_video`; see that function for the contract."""
    settings = get_settings()
    frame_interval = 1.0 / settings.stream_fps
    target_width = settings.stream_width
    target_height = settings.stream_height
    target_fps = settings.stream_fps

    # Mode SIM: live MuJoCo render without hardware camera dependencies
    if mode == "sim":
        calib = get_calibrator(mode="sim")
        while _current_stream_id == my_stream_id:
            start = time.time()
            raw_frame = get_sim_video_frame(robot=robot)
            with frames_mod._global_frame_lock:
                frames_mod._global_latest_raw_frame = raw_frame.copy()

            ui_frame = raw_frame.copy()
            if is_calibration_overlay_enabled() and calib.is_calibrated:
                ui_frame = calib.draw_workspace_overlay(
                    ui_frame, detected_markers={}, show_markers=False
                )
            if is_target_overlay_enabled():
                ui_frame = draw_model_target_overlay(ui_frame)

            with frames_mod._global_frame_lock:
                frames_mod._global_latest_annotated_frame = ui_frame.copy()
                frames_mod._global_latest_frame = ui_frame.copy()

            yield ui_frame

            elapsed = time.time() - start
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)
        return

    # Mode REAL: Live UVC hardware streaming with OpenCV & MJPG codec
    if camera_index == -2:
        no_cam = generate_no_camera_pattern()
        with frames_mod._global_frame_lock:
            frames_mod._global_latest_raw_frame = generate_raw_sim_placeholder_pattern()
            frames_mod._global_latest_annotated_frame = no_cam.copy()
            frames_mod._global_latest_frame = no_cam.copy()
        yield no_cam
        return

    yield generate_loading_pattern(camera_index, "Acquiring camera stream lock...")

    camera_active = False
    cap = None
    cam_name = get_camera_name(camera_index)

    if camera_index >= 0:
        # Yield the status frame *before* taking the lock: a generator suspended while holding
        # the lock would deadlock every later stream if the client never resumed it.
        msg = f"Connecting to OpenCV index {camera_index}..."
        yield generate_loading_pattern(camera_index, msg)
        _wait_for_previous_stream_release(prev_done)

    with _stream_lock:
        if _current_stream_id != my_stream_id:
            return

        if camera_index >= 0:
            logger.info(f"Initializing {cam_name} (Index {camera_index})...")
            cap = _open_capture(camera_index)
            camera_active = cap.isOpened()

            if not camera_active:
                logger.warning(
                    f"Failed to open {cam_name} (Index {camera_index}): device busy or "
                    "not a capture node. Falling back to synthetic mock feed."
                )
            else:
                # Enable MJPG hardware video codec to prevent USB firmware FPS throttling
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
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
        else:
            logger.info("Synthetic mock video feed selected.")

    try:
        while True:
            if _current_stream_id != my_stream_id:
                logger.info(f"Stopping stream for camera {camera_index} due to source transition.")
                break

            start_time = time.time()

            if camera_active and cap is not None:
                ret, frame = cap.read()
                if not ret or frame is None:
                    logger.warning(f"Read failed for index {camera_index}; using fallback.")
                    camera_active = False
                    raw_frame = generate_mock_test_pattern(camera_index)
                else:
                    raw_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                raw_frame = generate_mock_test_pattern(camera_index)

            with frames_mod._global_frame_lock:
                frames_mod._global_latest_raw_frame = raw_frame.copy()

            ui_frame = raw_frame.copy()

            if camera_active:
                calib = get_calibrator()
                detected = calib.detect_markers(ui_frame)
                if len(detected) >= 4:
                    calib.update_homography(detected)
                if is_calibration_overlay_enabled():
                    ui_frame = calib.draw_workspace_overlay(ui_frame, detected)

            if is_target_overlay_enabled():
                ui_frame = draw_model_target_overlay(ui_frame)

            with frames_mod._global_frame_lock:
                frames_mod._global_latest_annotated_frame = ui_frame.copy()
                frames_mod._global_latest_frame = ui_frame.copy()

            yield ui_frame

            elapsed = time.time() - start_time
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)
    finally:
        if cap is not None and cap.isOpened():
            cap.release()
            logger.info(f"Released camera resource for index {camera_index}.")


# --- Shared camera capture -------------------------------------------------------------------
# The browser reads frames from /api/stream (MJPEG over one long-lived response), so capture must
# not depend on any Gradio session. One background thread drives ``stream_video`` -- which already
# publishes the raw and annotated frame buffers -- and every client reads those buffers. A second
# browser tab therefore shares the stream instead of stealing it.
_capture_lock = threading.Lock()
_capture_thread: threading.Thread | None = None
_capture_key: tuple[int, str] | None = None


def _capture_alive() -> bool:
    """Return True when the shared capture thread is running."""
    return _capture_thread is not None and _capture_thread.is_alive()


def _run_capture(camera_index: int, mode: str, robot: Any | None) -> None:
    """Drain ``stream_video`` so it keeps publishing the frame buffers; frames are discarded."""
    try:
        for _ in stream_video(camera_index=camera_index, mode=mode, robot=robot):
            pass
    except Exception:
        logger.exception(f"Camera capture for index {camera_index} stopped unexpectedly.")


def _start_capture_locked(camera_index: int, mode: str, robot: Any | None) -> None:
    """Start a capture thread; caller holds ``_capture_lock``. Supersedes any previous one."""
    global _capture_thread, _capture_key
    prepare_stream_switch()  # makes a still-running previous generator exit and release its camera
    _capture_key = (camera_index, mode)
    _capture_thread = threading.Thread(
        target=_run_capture,
        args=(camera_index, mode, robot),
        name="camera-capture",
        daemon=True,
    )
    _capture_thread.start()


def stop_camera_capture(timeout_s: float = _PREVIOUS_STREAM_RELEASE_TIMEOUT_S) -> None:
    """Stop the shared capture thread and wait for it to release its camera.

    Registered with ``atexit``: a daemon thread still blocked inside a V4L2 read while the
    interpreter tears down aborts the process (glibc "FATAL: exception not rethrown").

    Args:
        timeout_s: Maximum time to wait for the capture generator to exit.
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


def ensure_camera_capture(camera_index: int, mode: str, robot: Any | None = None) -> int:
    """Start shared capture if it is not running; never steal it from a running capture.

    Called on every page load, so a second tab joins the existing stream.

    Args:
        camera_index: Camera to open when nothing is capturing yet.
        mode: Operation mode ('real' or 'sim').
        robot: Optional active robot controller.

    Returns:
        The camera index that is actually being captured.
    """
    with _capture_lock:
        if not _capture_alive() or _capture_key is None:
            _start_capture_locked(camera_index, mode, robot)
        return _capture_key[0] if _capture_key is not None else camera_index


def switch_camera_capture(
    camera_index: int, mode: str, robot: Any | None = None, force: bool = False
) -> int:
    """Switch shared capture to ``camera_index`` (restart only when it changed or ``force``).

    Args:
        camera_index: Requested camera index (-1 selects the synthetic feed).
        mode: Operation mode ('real' or 'sim').
        robot: Optional active robot controller.
        force: Restart even when this camera is already active (e.g. after a replug).

    Returns:
        The camera index being captured after the call.
    """
    with _capture_lock:
        if force or not _capture_alive() or _capture_key != (camera_index, mode):
            _start_capture_locked(camera_index, mode, robot)
        return camera_index
