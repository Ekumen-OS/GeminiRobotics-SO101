"""Unified Vision Streamer handling OpenCV hardware cameras and multi-mode placeholders."""

import atexit
import logging
import sys
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from robot.calibration import ArucoWorkspaceCalibrator
from robot.interface import RobotInterface
from robot.settings import get_settings

logger = logging.getLogger(__name__)

# Synchronization lock and atomic token for instant, zero-latency video source transitions
_stream_lock = threading.Lock()
_current_stream_id = 0
_device_names: dict[int, str] = {}
# Set by the active stream generator once it has released its capture device. A new generator
# waits on the previous event before opening hardware so two streams never fight over one node.
_stream_done_event: threading.Event | None = None
_PREVIOUS_STREAM_RELEASE_TIMEOUT_S = 3.0
_CAPTURE_OPEN_ATTEMPTS = 8
_CAPTURE_OPEN_RETRY_DELAY_S = 0.25

# Linux V4L2 constants (linux/videodev2.h) used to classify device nodes without opening them
_VIDIOC_QUERYCAP = 0x80685600
_V4L2_CAP_VIDEO_CAPTURE = 0x00000001
_V4L2_CAP_VIDEO_CAPTURE_MPLANE = 0x00001000
_V4L2_CAP_META_CAPTURE = 0x00800000
_V4L2_CAP_DEVICE_CAPS = 0x80000000
_BUILTIN_CAMERA_KEYWORDS = ("integrated", "built-in", "builtin", "facetime", "internal")


def _preferred_capture_backend() -> int:
    """Return the OpenCV capture backend that maps an integer index to a stable device.

    On Linux ``cv2.VideoCapture(idx)`` tries V4L2 and, if ``/dev/video{idx}`` is busy or not a
    capture node, silently falls back to the FFmpeg backend. FFmpeg numbers devices in ``/dev``
    directory order, so "index 3" can open a completely different camera than ``/dev/video3``.
    Pinning V4L2 guarantees the index always means ``/dev/video{idx}``.
    """
    if sys.platform.startswith("linux"):
        return cv2.CAP_V4L2
    return cv2.CAP_ANY


def _open_capture(index: int, attempts: int = _CAPTURE_OPEN_ATTEMPTS) -> cv2.VideoCapture:
    """Open a hardware capture device, retrying briefly while the node is still busy.

    Args:
        index: Integer camera index (``/dev/video{index}`` on Linux).
        attempts: Maximum number of open attempts before giving up.

    Returns:
        The ``cv2.VideoCapture`` instance (check ``isOpened()`` for success).
    """
    backend = _preferred_capture_backend()
    cap = cv2.VideoCapture(index, backend)
    attempt = 1
    while not cap.isOpened() and attempt < attempts:
        cap.release()
        time.sleep(_CAPTURE_OPEN_RETRY_DELAY_S)
        cap = cv2.VideoCapture(index, backend)
        attempt += 1
    if cap.isOpened() and attempt > 1:
        logger.info(f"Camera index {index} opened after {attempt} attempts (device was busy).")
    return cap


def _wait_for_previous_stream_release(prev_done: threading.Event | None) -> None:
    """Block (bounded) until the previously active stream generator has released its camera."""
    if prev_done is None or prev_done.is_set():
        return
    if not prev_done.wait(timeout=_PREVIOUS_STREAM_RELEASE_TIMEOUT_S):
        logger.warning(
            "Previous video stream did not release its camera within "
            f"{_PREVIOUS_STREAM_RELEASE_TIMEOUT_S:.1f}s; attempting to open anyway."
        )


def _query_v4l2_device_caps(dev_node: Path) -> int | None:
    """Read the per-node V4L2 ``device_caps`` bitmask via ``VIDIOC_QUERYCAP``.

    Unlike opening the device with OpenCV, this ioctl succeeds even while another process (or
    the active stream) is streaming from the node, so enumeration never steals or drops a camera.

    Args:
        dev_node: Path to ``/dev/videoN``.

    Returns:
        The ``device_caps`` bitmask, or None when the node cannot be queried.
    """
    import fcntl
    import os
    import struct

    try:
        fd = os.open(str(dev_node), os.O_RDWR | os.O_NONBLOCK)
    except OSError:
        return None
    try:
        buf = bytearray(104)  # struct v4l2_capability
        fcntl.ioctl(fd, _VIDIOC_QUERYCAP, buf)
    except OSError:
        return None
    finally:
        os.close(fd)
    _driver, _card, _bus, _version, capabilities, device_caps = struct.unpack(
        "<16s32s32sIII", bytes(buf[:92])
    )
    if capabilities & _V4L2_CAP_DEVICE_CAPS:
        return device_caps
    return capabilities


def _is_v4l2_video_capture_node(dev_node: Path) -> bool | None:
    """Classify a V4L2 node: True for video capture, False for metadata/output, None if unknown."""
    caps = _query_v4l2_device_caps(dev_node)
    if caps is None:
        return None
    if caps & _V4L2_CAP_META_CAPTURE and not caps & _V4L2_CAP_VIDEO_CAPTURE:
        return False
    return bool(caps & (_V4L2_CAP_VIDEO_CAPTURE | _V4L2_CAP_VIDEO_CAPTURE_MPLANE))


def __getattr__(name: str) -> Any:
    """Dynamically resolve stream constants from central Settings."""
    if name == "STREAM_TARGET_FPS":
        return get_settings().stream_fps
    if name == "STREAM_FRAME_WIDTH":
        return get_settings().stream_width
    if name == "STREAM_FRAME_HEIGHT":
        return get_settings().stream_height
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


_global_latest_raw_frame: np.ndarray | None = None
_global_latest_annotated_frame: np.ndarray | None = None
_global_latest_frame: np.ndarray | None = None
_global_frame_lock = threading.Lock()

_calibrator_lock = threading.Lock()
_global_calibrator: ArucoWorkspaceCalibrator | None = None
_calibration_overlay_enabled: bool = False
_bounding_boxes_overlay_enabled: bool = False
_grid_overlay_enabled: bool = False
_crosshair_overlay_enabled: bool = False
_target_overlay_enabled: bool = True

TARGET_PALETTE_RGB: list[tuple[int, int, int]] = [
    (0, 225, 255),  # Vibrant Cyan
    (16, 185, 129),  # Emerald Green
    (245, 158, 11),  # Amber
    (168, 85, 247),  # Violet
    (244, 63, 94),  # Rose / Coral
    (56, 189, 248),  # Sky Blue
]

_model_targets: list[dict[str, Any]] = []
_model_target_point: tuple[float, float] | None = None
_model_target_label: str | None = None
_model_target_lock = threading.Lock()


def get_calibrator(mode: str = "real") -> ArucoWorkspaceCalibrator:
    """Return the shared ArucoWorkspaceCalibrator singleton instance.

    Args:
        mode: Operational mode ('real', 'sim', 'twin').

    Returns:
        The active ArucoWorkspaceCalibrator instance.
    """
    global _global_calibrator
    with _calibrator_lock:
        if _global_calibrator is None:
            if mode == "sim":
                from simulation.sim_calibration import init_sim_workspace_calibration

                _global_calibrator = init_sim_workspace_calibration()
            else:
                _global_calibrator = ArucoWorkspaceCalibrator(mode=mode)
        elif mode == "sim" and not _global_calibrator.is_calibrated:
            from simulation.sim_calibration import init_sim_workspace_calibration

            _global_calibrator = init_sim_workspace_calibration(calibrator=_global_calibrator)
        return _global_calibrator


def set_calibration_overlay(enabled: bool) -> None:
    """Enable or disable the ArUco calibration overlay on live camera feeds.

    Args:
        enabled: True to enable overlay drawing, False to disable.
    """
    global _calibration_overlay_enabled
    _calibration_overlay_enabled = bool(enabled)


def is_calibration_overlay_enabled() -> bool:
    """Check if the calibration overlay is currently enabled.

    Returns:
        True if overlay is active, False otherwise.
    """
    return _calibration_overlay_enabled


def set_bounding_boxes_overlay(enabled: bool) -> None:
    """Enable or disable the bounding boxes overlay on live camera feeds.

    Args:
        enabled: True to enable overlay drawing, False to disable.
    """
    global _bounding_boxes_overlay_enabled
    _bounding_boxes_overlay_enabled = bool(enabled)


def is_bounding_boxes_overlay_enabled() -> bool:
    """Check if the bounding boxes overlay is currently enabled.

    Returns:
        True if overlay is active, False otherwise.
    """
    return _bounding_boxes_overlay_enabled


def set_grid_overlay(enabled: bool) -> None:
    """Enable or disable the workspace grid overlay on live camera feeds.

    Args:
        enabled: True to enable overlay drawing, False to disable.
    """
    global _grid_overlay_enabled
    _grid_overlay_enabled = bool(enabled)


def is_grid_overlay_enabled() -> bool:
    """Check if the workspace grid overlay is currently enabled.

    Returns:
        True if overlay is active, False otherwise.
    """
    return _grid_overlay_enabled


def set_crosshair_overlay(enabled: bool) -> None:
    """Enable or disable the crosshair / center marker overlay on live camera feeds.

    Args:
        enabled: True to enable overlay drawing, False to disable.
    """
    global _crosshair_overlay_enabled
    _crosshair_overlay_enabled = bool(enabled)


def is_crosshair_overlay_enabled() -> bool:
    """Check if the crosshair overlay is currently enabled.

    Returns:
        True if overlay is active, False otherwise.
    """
    return _crosshair_overlay_enabled


def set_target_overlay_enabled(enabled: bool) -> None:
    """Enable or disable the model target overlay on live camera feeds.

    Args:
        enabled: True to enable overlay drawing, False to disable.
    """
    global _target_overlay_enabled
    _target_overlay_enabled = bool(enabled)


def is_target_overlay_enabled() -> bool:
    """Check if the model target overlay is currently enabled.

    Returns:
        True if overlay is active, False otherwise.
    """
    return _target_overlay_enabled


def set_model_targets(
    targets: list[dict[str, Any] | tuple[float, float] | list[float]] | None,
) -> None:
    """Set the active list of 2D/3D model targets/objects from the VLA model.

    Args:
        targets: List of target dicts or points, or None to clear.
    """
    global _model_targets, _model_target_point, _model_target_label
    with _model_target_lock:
        if not targets:
            _model_targets = []
            _model_target_point = None
            _model_target_label = None
            return

        parsed: list[dict[str, Any]] = []
        for item in targets:
            if isinstance(item, dict):
                pt = item.get("point")
                box = item.get("box_2d")
                if pt is None and box is not None and len(box) == 4:
                    pt = ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)
                if pt is not None and len(pt) >= 2:
                    parsed.append(
                        {
                            "point": (float(pt[0]), float(pt[1])),
                            "label": item.get("label"),
                            "box_2d": [float(v) for v in box] if box and len(box) == 4 else None,
                            "color": item.get("color"),
                        }
                    )
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                parsed.append(
                    {
                        "point": (float(item[0]), float(item[1])),
                        "label": None,
                        "box_2d": None,
                        "color": None,
                    }
                )
        _model_targets = parsed
        if _model_targets:
            _model_target_point = _model_targets[0]["point"]
            _model_target_label = _model_targets[0]["label"]
        else:
            _model_target_point = None
            _model_target_label = None


def add_model_target(
    point: tuple[float, float] | list[float],
    label: str | None = None,
    box_2d: list[float] | tuple[float, ...] | None = None,
    color: tuple[int, int, int] | None = None,
) -> None:
    """Add a target object to the active model targets list.

    Args:
        point: [y, x] normalized coordinates (0..1000).
        label: Optional description label.
        box_2d: Optional [ymin, xmin, ymax, xmax] normalized bounding box (0..1000).
        color: Optional RGB color tuple.
    """
    global _model_targets, _model_target_point, _model_target_label
    if point is None or len(point) < 2:
        return
    with _model_target_lock:
        target_dict: dict[str, Any] = {
            "point": (float(point[0]), float(point[1])),
            "label": label,
            "box_2d": [float(v) for v in box_2d] if box_2d and len(box_2d) == 4 else None,
            "color": color,
        }
        _model_targets.append(target_dict)
        if _model_target_point is None:
            _model_target_point = target_dict["point"]
            _model_target_label = target_dict["label"]


def get_model_targets() -> list[dict[str, Any]]:
    """Get the list of active model targets.

    Returns:
        List of target dictionaries.
    """
    with _model_target_lock:
        return [dict(t) for t in _model_targets]


def clear_model_targets() -> None:
    """Clear all active model targets."""
    set_model_targets(None)


def set_model_target_point(
    point: tuple[float, float] | list[float] | None,
    label: str | None = None,
) -> None:
    """Set the active primary 2D normalized target point [y, x] from the VLA model.

    Args:
        point: [y, x] coordinate in normalized range 0..1000, or None to clear.
        label: Optional text label to display beside the target marker.
    """
    if point is None or len(point) < 2:
        clear_model_targets()
    else:
        set_model_targets([{"point": point, "label": label}])


def clear_model_target_point() -> None:
    """Clear the active model target point overlay."""
    clear_model_targets()


def get_model_target_point() -> tuple[tuple[float, float] | None, str | None]:
    """Get the active primary model target point and label.

    Returns:
        Tuple of (target_point, target_label).
    """
    with _model_target_lock:
        if not _model_targets:
            return None, None
        return _model_targets[0]["point"], _model_targets[0]["label"]


def draw_model_target_overlay(frame: np.ndarray) -> np.ndarray:
    """Draw clean, minimalist bounding boxes and label badges for all detected model targets.

    Args:
        frame: Input image (RGB or BGR numpy array).

    Returns:
        Frame with minimalist target box overlay drawn for all active targets.
    """
    if not is_target_overlay_enabled():
        return frame

    targets = get_model_targets()
    if not targets:
        return frame

    h, w = frame.shape[:2]
    out = frame.copy()

    # Base design tokens
    color_dark_bg = (15, 23, 42)  # Dark slate pill background
    color_text = (248, 250, 252)  # Crisp white text
    font = cv2.FONT_HERSHEY_SIMPLEX

    for idx, tgt in enumerate(targets):
        norm_y, norm_x = tgt["point"]
        px = int(np.clip((norm_x / 1000.0) * w, 0, w - 1))
        py = int(np.clip((norm_y / 1000.0) * h, 0, h - 1))
        color = tgt.get("color") or TARGET_PALETTE_RGB[idx % len(TARGET_PALETTE_RGB)]

        # Determine bounding box bounds
        box_2d = tgt.get("box_2d")
        if box_2d is not None and len(box_2d) == 4:
            ymin, xmin, ymax, xmax = box_2d
            x1 = int(np.clip((xmin / 1000.0) * w, 0, w - 1))
            y1 = int(np.clip((ymin / 1000.0) * h, 0, h - 1))
            x2 = int(np.clip((xmax / 1000.0) * w, 0, w - 1))
            y2 = int(np.clip((ymax / 1000.0) * h, 0, h - 1))
            # Ensure proper min/max ordering
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1
        else:
            half_box = 28
            x1, y1 = max(2, px - half_box), max(2, py - half_box)
            x2, y2 = min(w - 3, px + half_box), min(h - 3, py + half_box)

        # 1. Subtle translucent fill (alpha 12%)
        overlay = out.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, 0.12, out, 0.88, 0, out)

        # 2. Crisp thin minimalist outline
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)

        # 3. Refined corner brackets
        bw, bh = x2 - x1, y2 - y1
        corner_len = min(7, max(3, bw // 4), max(3, bh // 4))
        # Top-Left
        cv2.line(out, (x1, y1), (x1 + corner_len, y1), (255, 255, 255), 2, cv2.LINE_AA)
        cv2.line(out, (x1, y1), (x1, y1 + corner_len), (255, 255, 255), 2, cv2.LINE_AA)
        # Top-Right
        cv2.line(out, (x2, y1), (x2 - corner_len, y1), (255, 255, 255), 2, cv2.LINE_AA)
        cv2.line(out, (x2, y1), (x2, y1 + corner_len), (255, 255, 255), 2, cv2.LINE_AA)
        # Bottom-Left
        cv2.line(out, (x1, y2), (x1 + corner_len, y2), (255, 255, 255), 2, cv2.LINE_AA)
        cv2.line(out, (x1, y2), (x1, y2 - corner_len), (255, 255, 255), 2, cv2.LINE_AA)
        # Bottom-Right
        cv2.line(out, (x2, y2), (x2 - corner_len, y2), (255, 255, 255), 2, cv2.LINE_AA)
        cv2.line(out, (x2, y2), (x2, y2 - corner_len), (255, 255, 255), 2, cv2.LINE_AA)

        # 4. Target centroid dot
        cv2.circle(out, (px, py), 3, color_dark_bg, -1, cv2.LINE_AA)
        cv2.circle(out, (px, py), 2, color, -1, cv2.LINE_AA)

        # 5. Floating Label Badge
        label = tgt.get("label")
        tag_text = label if label else f"TARGET #{idx + 1} [{int(norm_y)}, {int(norm_x)}]"
        font_scale = 0.38
        font_thickness = 1
        (t_w, t_h), _ = cv2.getTextSize(tag_text, font, font_scale, font_thickness)

        pad_x, pad_y = 6, 4
        dot_radius = 3
        dot_space = 8
        total_badge_w = t_w + dot_space + (pad_x * 2)
        total_badge_h = t_h + (pad_y * 2)

        bx1 = max(4, min(w - total_badge_w - 4, px - total_badge_w // 2))
        if y1 - total_badge_h - 6 > 4:
            by1 = y1 - total_badge_h - 6
        else:
            by1 = min(h - total_badge_h - 4, y2 + 6)
        bx2 = bx1 + total_badge_w
        by2 = by1 + total_badge_h

        # Semi-transparent dark pill background (alpha 85%)
        badge_layer = out.copy()
        cv2.rectangle(badge_layer, (bx1, by1), (bx2, by2), color_dark_bg, -1)
        cv2.addWeighted(badge_layer, 0.85, out, 0.15, 0, out)

        # Clean 1px border with target accent color
        cv2.rectangle(out, (bx1, by1), (bx2, by2), color, 1, cv2.LINE_AA)

        # Indicator dot & text
        dot_cx = bx1 + pad_x + dot_radius
        dot_cy = by1 + pad_y + (t_h // 2)
        cv2.circle(out, (dot_cx, dot_cy), dot_radius, color, -1, cv2.LINE_AA)

        text_x = bx1 + pad_x + dot_space
        text_y = by1 + pad_y + t_h
        cv2.putText(
            out,
            tag_text,
            (text_x, text_y),
            font,
            font_scale,
            color_text,
            font_thickness,
            cv2.LINE_AA,
        )

    return out


def get_calibration_status_summary() -> str:
    """Return a formatted string describing current calibration status for the UI.

    Returns:
        Human-readable calibration status string.
    """
    calib = get_calibrator()
    status = calib.get_calibration_status()
    if status["calibrated"]:
        err = status.get("reprojection_error")
        err_str = (
            f" (err: {err:.1f}mm)"
            if isinstance(err, (int, float)) and not isinstance(err, bool)
            else ""
        )
        return f"✅ Calibrated{err_str} — {status['markers_detected']}/4 markers visible"
    elif status["markers_detected"] > 0:
        return f"⚠️ Incomplete: {status['markers_detected']}/4 markers visible (need 4)"
    else:
        return "❌ Not calibrated (no markers detected in view)"


def save_workspace_calibration_file() -> str:
    """Save current workspace homography to disk and return status message.

    Returns:
        Status message string from the calibrator.
    """
    calib = get_calibrator()
    return calib.save_calibration()


def generate_raw_sim_placeholder_pattern() -> np.ndarray:
    """Generate a clean, un-annotated placeholder frame for MuJoCo sim mode.

    Returns:
        RGB image frame as a NumPy uint8 array of shape (480, 640, 3).
    """
    height, width = 480, 640
    return np.full((height, width, 3), (20, 35, 45), dtype=np.uint8)


def _robot_can_render(robot: Any | None) -> bool:
    """Return True only when ``robot`` really implements ``render_rgb_frame``.

    ``RobotInterface`` defines ``render_rgb_frame`` as a method that raises
    ``NotImplementedError`` (physical arms inherit it), so ``hasattr`` alone is true for every
    robot. Only a subclass that overrides it (simulation, twin) can render a frame.

    Args:
        robot: Active robot instance, or None.

    Returns:
        True when the robot overrides ``render_rgb_frame``.
    """
    if robot is None:
        return False
    render = getattr(type(robot), "render_rgb_frame", None)
    return render is not None and render is not RobotInterface.render_rgb_frame


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
        from ui.state import get_active_robot

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
        from ui.state import get_active_robot

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


class _SingletonSimRuntime:
    """Singleton worker that owns sim stepping + rendering for UI streaming.

    This avoids concurrent MuJoCo access across multiple Gradio callback threads.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._robot: Any | None = None
        self._latest_raw_frame: np.ndarray | None = None
        self._latest_frame: np.ndarray | None = None

    def _loop(self) -> None:
        global _global_latest_frame, _global_latest_raw_frame, _global_latest_annotated_frame
        settings = get_settings()
        frame_interval = 1.0 / settings.stream_fps
        stream_width = settings.stream_width
        stream_height = settings.stream_height

        while not self._stop_event.is_set():
            start = time.time()

            try:
                with self._lock:
                    robot = self._robot

                if robot is None:
                    raw_frame = cv2.resize(
                        generate_raw_sim_placeholder_pattern(),
                        (stream_width, stream_height),
                        interpolation=cv2.INTER_LINEAR,
                    )
                else:
                    sim_steps = _estimate_sim_steps_per_frame(
                        robot=robot,
                        frame_interval=frame_interval,
                    )
                    if hasattr(robot, "step"):
                        try:
                            robot.step(sim_steps)
                        except Exception:
                            pass

                    raw_frame = _sim_frame_with_fallback(
                        robot=robot,
                        width=stream_width,
                        height=stream_height,
                    )

                with self._frame_lock:
                    self._latest_raw_frame = raw_frame
                    self._latest_frame = raw_frame

                with _global_frame_lock:
                    _global_latest_raw_frame = raw_frame.copy() if raw_frame is not None else None
                    _global_latest_annotated_frame = (
                        raw_frame.copy() if raw_frame is not None else None
                    )
                    _global_latest_frame = raw_frame.copy() if raw_frame is not None else None
            except Exception:
                pass

            elapsed = time.time() - start
            if elapsed < frame_interval:
                self._stop_event.wait(timeout=max(0.0, frame_interval - elapsed))

    def ensure_started(self, robot: Any | None) -> None:
        """Start worker once and update active robot reference."""
        with self._lock:
            if self._robot != robot:
                self._robot = robot
                with self._frame_lock:
                    self._latest_raw_frame = None
                    self._latest_frame = None
            if self._thread is not None and self._thread.is_alive():
                return

            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._loop,
                name="sim-runtime",
                daemon=True,
            )
            self._thread.start()

    def get_latest_raw_frame(self, robot: Any | None) -> np.ndarray:
        """Return most recent un-annotated raw simulation frame, bootstrapping runtime if needed.

        Args:
            robot: Active simulation robot object.

        Returns:
            RGB frame as uint8 numpy array.
        """
        self.ensure_started(robot)

        with self._frame_lock:
            if self._latest_raw_frame is not None:
                return self._latest_raw_frame.copy()
            if self._latest_frame is not None:
                return self._latest_frame.copy()

        settings = get_settings()
        stream_width = settings.stream_width
        stream_height = settings.stream_height

        if _robot_can_render(robot):
            frame = _sim_frame_with_fallback(
                robot=robot,
                width=stream_width,
                height=stream_height,
            )
            if frame is not None:
                with self._frame_lock:
                    self._latest_raw_frame = frame
                    self._latest_frame = frame
                return frame.copy()

        # First frame race: return deterministic raw fallback until worker publishes one.
        return cv2.resize(
            generate_raw_sim_placeholder_pattern(),
            (stream_width, stream_height),
            interpolation=cv2.INTER_LINEAR,
        )

    def get_latest_frame(self, robot: Any | None) -> np.ndarray:
        """Return most recent frame, bootstrapping runtime if needed.

        Args:
            robot: Active simulation robot object.

        Returns:
            RGB frame as uint8 numpy array.
        """
        self.ensure_started(robot)

        with self._frame_lock:
            if self._latest_frame is not None:
                return self._latest_frame.copy()

        settings = get_settings()
        stream_width = settings.stream_width
        stream_height = settings.stream_height

        if _robot_can_render(robot):
            frame = _sim_frame_with_fallback(
                robot=robot,
                width=stream_width,
                height=stream_height,
            )
            if frame is not None:
                with self._frame_lock:
                    self._latest_raw_frame = frame
                    self._latest_frame = frame
                return frame.copy()

        # First frame race: return deterministic fallback until worker publishes one.
        return cv2.resize(
            generate_sim_placeholder_pattern(),
            (stream_width, stream_height),
            interpolation=cv2.INTER_LINEAR,
        )

    def get_latest_raw_jpeg(self, robot: Any | None = None) -> bytes | None:
        """Return the most recent un-annotated raw frame as JPEG encoded bytes.

        Args:
            robot: Active simulation robot object.

        Returns:
            Pristine JPEG bytes, or None if encoding fails.
        """
        frame = self.get_latest_raw_frame(robot)
        if frame is None:
            return None
        # Convert RGB to BGR for cv2.imencode to produce correct JPEG colors.
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        success, encoded = cv2.imencode(".jpg", bgr_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 98])
        if success:
            return encoded.tobytes()
        return None

    def get_latest_jpeg(self, robot: Any | None = None) -> bytes | None:
        """Return the most recent frame as JPEG encoded bytes."""
        frame = self.get_latest_frame(robot)
        if frame is None:
            return None
        # Convert RGB to BGR for cv2.imencode to produce correct JPEG colors.
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        success, encoded = cv2.imencode(".jpg", bgr_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if success:
            return encoded.tobytes()
        return None

    def stop(self) -> None:
        """Stop worker thread cleanly."""
        self._stop_event.set()
        with self._lock:
            thread = self._thread
            self._thread = None
            self._robot = None

        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)

        with self._frame_lock:
            self._latest_raw_frame = None
            self._latest_frame = None


_SIM_RUNTIME = _SingletonSimRuntime()
atexit.register(_SIM_RUNTIME.stop)


def _enumerate_cameras_macos() -> list[tuple[str, int]]:
    """Enumerate hardware camera devices on macOS via AVFoundation sorted by uniqueID.

    Returns:
        List of tuples containing formatted dropdown labels and device index integers.
    """
    choices: list[tuple[str, int]] = []
    try:
        import AVFoundation as av

        devices = av.AVCaptureDevice.devicesWithMediaType_(
            av.AVMediaTypeVideo
        ) + av.AVCaptureDevice.devicesWithMediaType_(av.AVMediaTypeMuxed)
        # Sort identically to OpenCV (cap_avfoundation.mm) by uniqueID for 1-to-1 matching
        devices = sorted(devices, key=lambda d: str(d.uniqueID()))
        for idx, device in enumerate(devices):
            name = device.localizedName()
            _device_names[idx] = name
            choices.append((f"{name} (Index {idx})", idx))
    except (ImportError, AttributeError, RuntimeError, OSError) as e:
        logger.info(f"AVFoundation enumeration unavailable ({e}). Fallback to active probe...")
        return _enumerate_cameras_fallback()
    return choices


def _enumerate_cameras_linux() -> list[tuple[str, int]]:
    """Enumerate physical video devices on Linux via Video4Linux (V4L2) sysfs entries.

    Returns:
        List of tuples containing formatted dropdown labels and device index integers.
    """
    choices: list[tuple[str, int]] = []
    try:
        v4l_dir = Path("/sys/class/video4linux")
        if v4l_dir.exists():
            for dev_path in sorted(
                v4l_dir.glob("video*"),
                key=lambda p: (
                    int(p.name.replace("video", ""))
                    if p.name.replace("video", "").isdigit()
                    else p.name
                ),
            ):
                idx_str = dev_path.name.replace("video", "")
                if idx_str.isdigit():
                    idx = int(idx_str)
                    name_file = dev_path / "name"
                    if name_file.exists():
                        name = name_file.read_text(encoding="utf-8", errors="ignore").strip()
                    else:
                        name = f"Linux V4L2 Device (video{idx})"
                    is_capture = _is_v4l2_video_capture_node(Path(f"/dev/video{idx}"))
                    if is_capture is None:
                        # Capability query unavailable: fall back to a single open probe.
                        cap = _open_capture(idx, attempts=1)
                        is_capture = cap.isOpened()
                        cap.release()
                    if is_capture:
                        _device_names[idx] = name
                        choices.append((f"{name} (Index {idx})", idx))
    except (OSError, RuntimeError, AttributeError, ValueError, PermissionError) as e:
        logger.info(f"Linux V4L2 sysfs enumeration error ({e}). Fallback to active probe...")
    if not choices:
        return _enumerate_cameras_fallback()
    return choices


def _enumerate_cameras_windows() -> list[tuple[str, int]]:
    """Enumerate video capture devices on Windows using active hardware probing.

    Returns:
        List of tuples containing formatted dropdown labels and device index integers.
    """
    return _enumerate_cameras_fallback()


def _enumerate_cameras_fallback() -> list[tuple[str, int]]:
    """Probe video capture hardware indices as a cross-platform fallback.

    Returns:
        List of tuples containing formatted dropdown labels and device index integers.
    """
    choices: list[tuple[str, int]] = []
    for idx in range(4):
        cap = _open_capture(idx, attempts=1)
        if cap.isOpened():
            name = f"Hardware Camera {idx}"
            _device_names[idx] = name
            choices.append((f"{name} (Index {idx})", idx))
        cap.release()
    return choices


def get_available_camera_choices() -> list[tuple[str, int]]:
    """Enumerate physical video capture devices across operating systems.

    Returns:
        List of tuples containing formatted dropdown labels and device index integers.
    """
    _device_names.clear()
    choices: list[tuple[str, int]] = []

    if sys.platform == "darwin":
        choices = _enumerate_cameras_macos()
    elif sys.platform.startswith("linux"):
        choices = _enumerate_cameras_linux()
    elif sys.platform == "win32":
        choices = _enumerate_cameras_windows()
    else:
        choices = _enumerate_cameras_fallback()

    if not choices:
        _device_names[-2] = "No cameras detected"
        choices.append(("🚫 No cameras detected (Connect device and click Refresh)", -2))

    choices.append(("Synthetic Mock Test Feed", -1))
    return choices


def get_default_camera_index(choices: list[tuple[str, int]] | None = None) -> int:
    """Select the preferred default camera index.

    Priority order:
        1. ``CAMERA_INDEX`` setting, when that index is currently available.
        2. ``CAMERA_NAME`` setting, matched case-insensitively against device labels.
        3. The first external camera (labels not mentioning an integrated/built-in webcam).
        4. Legacy fallback priority: index 2 -> 1 -> 0, then the first listed choice.

    Args:
        choices: Optional list of available camera tuples (label, index).

    Returns:
        The selected integer camera index.
    """
    if choices is None:
        choices = get_available_camera_choices()

    hardware = {c[1]: c[0] for c in choices if c[1] >= 0}
    settings = get_settings()

    if settings.camera_index is not None and settings.camera_index in hardware:
        return settings.camera_index

    if settings.camera_name:
        needle = settings.camera_name.strip().lower()
        for idx in sorted(hardware):
            if needle and needle in hardware[idx].lower():
                return idx
        logger.warning(f"No camera matching CAMERA_NAME='{settings.camera_name}' found.")

    for idx in sorted(hardware):
        label = hardware[idx].lower()
        if not any(keyword in label for keyword in _BUILTIN_CAMERA_KEYWORDS):
            return idx

    for preferred in [2, 1, 0]:
        if preferred in hardware:
            return preferred

    if choices:
        return choices[0][1]
    return 2


def get_camera_name(index: int) -> str:
    """Retrieve the localized hardware device name for a given camera index.

    Args:
        index: Integer camera index or sentinel status value (-1 for mock, -2 for none).

    Returns:
        Human-readable hardware camera name or status string.
    """
    if index == -2:
        return "No Cameras Detected"
    if index < 0:
        return "Synthetic Mock Test Feed"
    return _device_names.get(index, f"Hardware Camera Index {index}")


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


def generate_twin_placeholder_pattern() -> np.ndarray:
    """Generate a split-screen placeholder frame for Digital Twin comparison mode.

    Returns:
        RGB image frame as a NumPy uint8 array of shape (600, 640, 3).
    """
    height, width = 600, 640
    t = time.time()
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    # Top half: Simulation mirror
    frame[0 : height // 2, :, :] = (25, 35, 50)
    frame[0:40, :, :] = (40, 70, 100)
    cv2.putText(
        frame,
        "TOP: MUJOCO DIGITAL TWIN RENDER",
        (20, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (220, 240, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "[Sim Mirror Feed Placeholder — Next Session]",
        (40, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (150, 220, 255),
        2,
        cv2.LINE_AA,
    )

    # Divider line
    frame[height // 2 - 2 : height // 2 + 2, :, :] = (255, 255, 255)

    # Bottom half: Real physical webcam
    frame[height // 2 :, :, :] = (45, 30, 35)
    frame[height // 2 : height // 2 + 40, :, :] = (90, 45, 60)
    cv2.putText(
        frame,
        "BOTTOM: REAL HARDWARE CAMERA FEED",
        (20, height // 2 + 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 220, 230),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "[Physical Camera Feed — Active in Real Mode]",
        (40, height // 2 + 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 180, 190),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        f"Twin Synchronize Timestamp: {t:.2f}s",
        (20, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    return frame


def prepare_stream_switch() -> None:
    """Invalidate atomic stream session token to trigger immediate generator teardown."""
    global _current_stream_id
    _current_stream_id = time.time_ns()


def _estimate_sim_steps_per_frame(robot: Any | None, frame_interval: float) -> int:
    """Estimate MuJoCo physics steps needed per rendered frame.

    Args:
        robot: Active simulation robot object.
        frame_interval: Frame period in seconds.

    Returns:
        Number of simulation steps to execute for near-real-time playback.
    """
    if robot is None:
        return 1

    try:
        model = getattr(robot, "model", None)
        opt = getattr(model, "opt", None)
        timestep = getattr(opt, "timestep", None)
        if timestep is None:
            return 1
        ts_float = float(timestep)
        if ts_float <= 0.0:
            return 1
        return max(1, int(round(frame_interval / ts_float)))
    except Exception:
        return 1


def _sim_frame_with_fallback(
    robot: Any | None,
    width: int,
    height: int,
) -> np.ndarray:
    """Render simulation frame if available, else return a placeholder.

    Args:
        robot: Active robot object for current mode.
        width: Frame width.
        height: Frame height.

    Returns:
        RGB frame as uint8 numpy array.
    """
    if _robot_can_render(robot):
        try:
            return robot.render_rgb_frame(width=width, height=height)
        except Exception as err:
            logger.warning(f"Sim render failed. Falling back to placeholder. Cause: {err}")

    fallback = generate_sim_placeholder_pattern()
    return cv2.resize(fallback, (width, height), interpolation=cv2.INTER_LINEAR)


def _compose_twin_frame(top_rgb: np.ndarray, bottom_rgb: np.ndarray) -> np.ndarray:
    """Compose a twin frame from top simulation and bottom camera streams.

    Args:
        top_rgb: Simulation RGB frame.
        bottom_rgb: Camera RGB frame.

    Returns:
        Combined RGB frame with a thin separator line between views.
    """
    target_width = max(top_rgb.shape[1], bottom_rgb.shape[1])
    half_height = max(top_rgb.shape[0], bottom_rgb.shape[0])

    top_resized = cv2.resize(top_rgb, (target_width, half_height), interpolation=cv2.INTER_LINEAR)
    bottom_resized = cv2.resize(
        bottom_rgb, (target_width, half_height), interpolation=cv2.INTER_LINEAR
    )

    divider = np.full((2, target_width, 3), 255, dtype=np.uint8)
    return np.vstack([top_resized, divider, bottom_resized])


def get_sim_video_frame(robot: Any | None = None) -> np.ndarray:
    """Return a single live simulation frame for timer-driven UI updates.

    Args:
        robot: Active simulation robot object.

    Returns:
        RGB frame as uint8 numpy array.
    """
    return _SIM_RUNTIME.get_latest_frame(robot=robot)


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
        mode: Operation mode ('real', 'sim', or 'twin').
        robot: Active robot controller object used for MuJoCo rendering in sim/twin modes.

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
    global _global_latest_frame, _global_latest_raw_frame, _global_latest_annotated_frame

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
            with _global_frame_lock:
                _global_latest_raw_frame = raw_frame.copy()

            ui_frame = raw_frame.copy()
            if is_calibration_overlay_enabled() and calib.is_calibrated:
                ui_frame = calib.draw_workspace_overlay(
                    ui_frame, detected_markers={}, show_markers=False
                )
            if is_target_overlay_enabled():
                ui_frame = draw_model_target_overlay(ui_frame)

            with _global_frame_lock:
                _global_latest_annotated_frame = ui_frame.copy()
                _global_latest_frame = ui_frame.copy()

            yield ui_frame
            elapsed = time.time() - start
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)
        return

    # Mode TWIN: top half MuJoCo sim + bottom half real/mock camera feed
    if mode == "twin":
        if camera_index == -2:
            while _current_stream_id == my_stream_id:
                start = time.time()
                top_frame = _sim_frame_with_fallback(
                    robot=robot,
                    width=target_width,
                    height=target_height,
                )
                bottom_frame = cv2.resize(
                    generate_no_camera_pattern(),
                    (target_width, target_height),
                    interpolation=cv2.INTER_LINEAR,
                )
                with _global_frame_lock:
                    _global_latest_raw_frame = top_frame.copy()

                final_frame = _compose_twin_frame(top_frame, bottom_frame)
                with _global_frame_lock:
                    _global_latest_annotated_frame = final_frame.copy()
                    _global_latest_frame = final_frame.copy()

                yield final_frame
                elapsed = time.time() - start
                if elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)
            return

        cap = None
        camera_active = False
        if camera_index >= 0:
            _wait_for_previous_stream_release(prev_done)
        with _stream_lock:
            if _current_stream_id != my_stream_id:
                return

            cam_name = get_camera_name(camera_index)
            if camera_index >= 0:
                logger.info(f"Initializing twin bottom camera {cam_name} (Index {camera_index})...")
                cap = _open_capture(camera_index)
                camera_active = cap.isOpened()
                if camera_active:
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)
                    cap.set(cv2.CAP_PROP_FPS, target_fps)
                else:
                    logger.warning("Twin bottom camera unavailable. Using mock fallback.")

        try:
            while _current_stream_id == my_stream_id:
                start = time.time()
                top_frame = _sim_frame_with_fallback(
                    robot=robot,
                    width=target_width,
                    height=target_height,
                )

                if camera_active and cap is not None:
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        bottom_raw = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    else:
                        camera_active = False
                        logger.warning(
                            "Twin bottom camera read failed. Switching to mock fallback."
                        )
                        bottom_raw = generate_mock_test_pattern(camera_index)
                else:
                    bottom_raw = generate_mock_test_pattern(camera_index)

                bottom_raw = cv2.resize(
                    bottom_raw,
                    (target_width, target_height),
                    interpolation=cv2.INTER_LINEAR,
                )

                with _global_frame_lock:
                    _global_latest_raw_frame = bottom_raw.copy()

                bottom_ui = bottom_raw.copy()
                if camera_active:
                    calib = get_calibrator()
                    detected = calib.detect_markers(bottom_ui)
                    if len(detected) >= 4:
                        calib.update_homography(detected)
                    if is_calibration_overlay_enabled():
                        bottom_ui = calib.draw_workspace_overlay(bottom_ui, detected)

                if is_target_overlay_enabled():
                    bottom_ui = draw_model_target_overlay(bottom_ui)

                final_frame = _compose_twin_frame(top_frame, bottom_ui)
                with _global_frame_lock:
                    _global_latest_annotated_frame = final_frame.copy()
                    _global_latest_frame = final_frame.copy()

                yield final_frame

                elapsed = time.time() - start
                if elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)
        finally:
            if cap is not None and cap.isOpened():
                cap.release()
        return

    # Mode REAL: Live UVC hardware streaming with OpenCV & MJPG codec
    if camera_index == -2:
        no_cam = generate_no_camera_pattern()
        with _global_frame_lock:
            _global_latest_raw_frame = generate_raw_sim_placeholder_pattern()
            _global_latest_annotated_frame = no_cam.copy()
            _global_latest_frame = no_cam.copy()
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

            with _global_frame_lock:
                _global_latest_raw_frame = raw_frame.copy()

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

            with _global_frame_lock:
                _global_latest_annotated_frame = ui_frame.copy()
                _global_latest_frame = ui_frame.copy()

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


def ensure_camera_capture(camera_index: int, mode: str, robot: Any | None = None) -> int:
    """Start shared capture if it is not running; never steal it from a running capture.

    Called on every page load, so a second tab joins the existing stream.

    Args:
        camera_index: Camera to open when nothing is capturing yet.
        mode: Operation mode ('real' or 'twin').
        robot: Optional robot used by twin mode to compose frames.

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
        mode: Operation mode ('real' or 'twin').
        robot: Optional robot used by twin mode to compose frames.
        force: Restart even when this camera is already active (e.g. after a replug).

    Returns:
        The camera index being captured after the call.
    """
    with _capture_lock:
        if force or not _capture_alive() or _capture_key != (camera_index, mode):
            _start_capture_locked(camera_index, mode, robot)
        return camera_index
