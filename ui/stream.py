"""Unified Vision Streamer handling OpenCV hardware cameras and multi-mode placeholders."""

import atexit
import sys
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from robot.calibration import ArucoWorkspaceCalibrator

# Synchronization lock and atomic token for instant, zero-latency video source transitions
_stream_lock = threading.Lock()
_current_stream_id = 0
_device_names: dict[int, str] = {}
_sim_frame_lock = threading.Lock()
STREAM_TARGET_FPS = 30
STREAM_FRAME_WIDTH = 640
STREAM_FRAME_HEIGHT = 480

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
        err = status["reprojection_error"]
        err_str = f" (err: {err:.1f}mm)" if err is not None else ""
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


def get_latest_jpeg() -> bytes | None:
    """Get the most recent camera frame from any active stream as a JPEG."""
    with _global_frame_lock:
        frame = _global_latest_frame

    if frame is None:
        from ui.state import get_active_robot

        robot = get_active_robot()
        if robot is not None and hasattr(robot, "render_rgb_frame"):
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
    success, encoded = cv2.imencode(".jpg", bgr_frame)
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
        self._latest_frame: np.ndarray | None = None

    def _loop(self) -> None:
        global _global_latest_frame
        frame_interval = 1.0 / STREAM_TARGET_FPS

        while not self._stop_event.is_set():
            start = time.time()

            with self._lock:
                robot = self._robot

            if robot is None:
                frame = cv2.resize(
                    generate_sim_placeholder_pattern(),
                    (STREAM_FRAME_WIDTH, STREAM_FRAME_HEIGHT),
                    interpolation=cv2.INTER_LINEAR,
                )
            else:
                with _sim_frame_lock:
                    sim_steps = _estimate_sim_steps_per_frame(
                        robot=robot,
                        frame_interval=frame_interval,
                    )
                    if hasattr(robot, "step"):
                        robot.step(sim_steps)

                    frame = _sim_frame_with_fallback(
                        robot=robot,
                        width=STREAM_FRAME_WIDTH,
                        height=STREAM_FRAME_HEIGHT,
                    )

            with self._frame_lock:
                self._latest_frame = frame

            with _global_frame_lock:
                _global_latest_frame = frame.copy() if frame is not None else None

            elapsed = time.time() - start
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)

    def ensure_started(self, robot: Any | None) -> None:
        """Start worker once and update active robot reference."""
        with self._lock:
            self._robot = robot
            if self._thread is not None and self._thread.is_alive():
                return

            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._loop,
                name="sim-runtime",
                daemon=True,
            )
            self._thread.start()

    def get_latest_frame(self, robot: Any | None) -> np.ndarray:
        """Return most recent frame, bootstrapping runtime if needed."""
        self.ensure_started(robot)

        with self._frame_lock:
            if self._latest_frame is not None:
                return self._latest_frame.copy()

        # First frame race: return deterministic fallback until worker publishes one.
        return cv2.resize(
            generate_sim_placeholder_pattern(),
            (STREAM_FRAME_WIDTH, STREAM_FRAME_HEIGHT),
            interpolation=cv2.INTER_LINEAR,
        )

    def get_latest_jpeg(self, robot: Any | None = None) -> bytes | None:
        """Return the most recent frame as JPEG encoded bytes."""
        frame = self.get_latest_frame(robot)
        if frame is None:
            return None
        # Convert RGB to BGR for cv2.imencode to produce correct JPEG colors.
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        success, encoded = cv2.imencode(".jpg", bgr_frame)
        if success:
            return encoded.tobytes()
        return None

    def stop(self) -> None:
        """Stop worker thread cleanly."""
        with self._lock:
            thread = self._thread
            self._thread = None
            self._robot = None

        self._stop_event.set()
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)


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
        print(f"[INFO] AVFoundation enumeration unavailable ({e}). Fallback to active probe...")
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
                    cap = cv2.VideoCapture(idx)
                    if cap.isOpened():
                        _device_names[idx] = name
                        choices.append((f"{name} (Index {idx})", idx))
                        cap.release()
    except (OSError, RuntimeError, AttributeError, ValueError, PermissionError) as e:
        print(f"[INFO] Linux V4L2 sysfs enumeration error ({e}). Fallback to active probe...")
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
        cap = cv2.VideoCapture(idx)
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
    """Select the preferred default camera index (priority: 2 -> 1 -> 0).

    Args:
        choices: Optional list of available camera tuples (label, index).

    Returns:
        The selected integer camera index.
    """
    if choices is None:
        choices = get_available_camera_choices()

    valid_indices = {c[1] for c in choices if c[1] >= 0}
    for preferred in [2, 1, 0]:
        if preferred in valid_indices:
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

    model = getattr(robot, "model", None)
    timestep = getattr(getattr(model, "opt", None), "timestep", None)
    if timestep is None:
        return 1

    return max(1, int(round(frame_interval / float(timestep))))


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
    if robot is not None and hasattr(robot, "render_rgb_frame"):
        try:
            return robot.render_rgb_frame(width=width, height=height)
        except Exception as err:
            print(f"[WARN] Sim render failed. Falling back to placeholder. Cause: {err}")

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

    Args:
        camera_index: Integer index of target hardware capture device.
        mode: Operation mode ('real', 'sim', or 'twin').
        robot: Active robot controller object used for MuJoCo rendering in sim/twin modes.

    Yields:
        NumPy array of shape (H, W, 3) containing RGB video frames.
    """
    global _global_latest_frame
    global _current_stream_id
    my_stream_id = time.time_ns()
    _current_stream_id = my_stream_id

    frame_interval = 1.0 / STREAM_TARGET_FPS
    target_width = STREAM_FRAME_WIDTH
    target_height = STREAM_FRAME_HEIGHT

    # Mode SIM: live MuJoCo render without hardware camera dependencies
    if mode == "sim":
        calib = get_calibrator(mode="sim")
        while _current_stream_id == my_stream_id:
            start = time.time()
            frame = get_sim_video_frame(robot=robot)
            if is_calibration_overlay_enabled() and calib.is_calibrated:
                frame = calib.draw_workspace_overlay(frame, detected_markers={}, show_markers=False)
            if is_target_overlay_enabled():
                frame = draw_model_target_overlay(frame)
            with _global_frame_lock:
                _global_latest_frame = frame.copy()
            yield frame
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
                yield _compose_twin_frame(top_frame, bottom_frame)
                elapsed = time.time() - start
                if elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)
            return

        cap = None
        camera_active = False
        with _stream_lock:
            if _current_stream_id != my_stream_id:
                return

            cam_name = get_camera_name(camera_index)
            if camera_index >= 0:
                print(
                    f"[INFO] Initializing twin bottom camera {cam_name} (Index {camera_index})..."
                )
                cap = cv2.VideoCapture(camera_index)
                camera_active = cap.isOpened()
                if camera_active:
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)
                    cap.set(cv2.CAP_PROP_FPS, STREAM_TARGET_FPS)
                else:
                    print("[WARN] Twin bottom camera unavailable. Using mock fallback.")

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
                        bottom_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    else:
                        camera_active = False
                        print("[WARN] Twin bottom camera read failed. Switching to mock fallback.")
                        bottom_frame = generate_mock_test_pattern(camera_index)
                else:
                    bottom_frame = generate_mock_test_pattern(camera_index)

                bottom_frame = cv2.resize(
                    bottom_frame,
                    (target_width, target_height),
                    interpolation=cv2.INTER_LINEAR,
                )
                if camera_active:
                    calib = get_calibrator()
                    detected = calib.detect_markers(bottom_frame)
                    if len(detected) >= 4:
                        calib.update_homography(detected)
                    if is_calibration_overlay_enabled():
                        bottom_frame = calib.draw_workspace_overlay(bottom_frame, detected)

                if is_target_overlay_enabled():
                    bottom_frame = draw_model_target_overlay(bottom_frame)

                final_frame = _compose_twin_frame(top_frame, bottom_frame)
                with _global_frame_lock:
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
        yield generate_no_camera_pattern()
        return

    yield generate_loading_pattern(camera_index, "Acquiring camera stream lock...")

    camera_active = False
    cap = None

    with _stream_lock:
        if _current_stream_id != my_stream_id:
            return

        cam_name = get_camera_name(camera_index)

        if camera_index >= 0:
            msg = f"Connecting to OpenCV index {camera_index}..."
            yield generate_loading_pattern(camera_index, msg)
            print(f"[INFO] Initializing {cam_name} (Index {camera_index})...")
            cap = cv2.VideoCapture(camera_index)
            camera_active = cap.isOpened()

            if not camera_active:
                print(f"[WARN] Failed to open {cam_name}. Falling back to synthetic mock feed.")
            else:
                # Enable MJPG hardware video codec to prevent USB firmware FPS throttling
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)

                cap.set(cv2.CAP_PROP_FPS, STREAM_TARGET_FPS)
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                real_fps = cap.get(cv2.CAP_PROP_FPS)
                print(f"[INFO] Connected to {cam_name} at {w}x{h} (~{real_fps:.0f} FPS).")
        else:
            print("[INFO] Synthetic mock video feed selected.")

    try:
        while True:
            if _current_stream_id != my_stream_id:
                print(f"[INFO] Stopping stream for camera {camera_index} due to source transition.")
                break

            start_time = time.time()

            if camera_active and cap is not None:
                ret, frame = cap.read()
                if not ret or frame is None:
                    print(f"[WARN] Read failed for index {camera_index}; using fallback.")
                    camera_active = False
                    frame_rgb = generate_mock_test_pattern(camera_index)
                else:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = generate_mock_test_pattern(camera_index)

            if camera_active:
                calib = get_calibrator()
                detected = calib.detect_markers(frame_rgb)
                if len(detected) >= 4:
                    calib.update_homography(detected)
                if is_calibration_overlay_enabled():
                    frame_rgb = calib.draw_workspace_overlay(frame_rgb, detected)

            if is_target_overlay_enabled():
                frame_rgb = draw_model_target_overlay(frame_rgb)

            with _global_frame_lock:
                _global_latest_frame = frame_rgb.copy()

            yield frame_rgb

            elapsed = time.time() - start_time
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)
    finally:
        if cap is not None and cap.isOpened():
            cap.release()
            print(f"[INFO] Released camera resource for index {camera_index}.")
