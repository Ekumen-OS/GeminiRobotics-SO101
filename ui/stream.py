"""Unified Vision Streamer handling OpenCV hardware cameras and multi-mode placeholders."""

import sys
import threading
import time
import atexit
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# Synchronization lock and atomic token for instant, zero-latency video source transitions
_stream_lock = threading.Lock()
_current_stream_id = 0
_device_names: dict[int, str] = {}
_sim_frame_lock = threading.Lock()
STREAM_TARGET_FPS = 30
STREAM_FRAME_WIDTH = 640
STREAM_FRAME_HEIGHT = 480


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
    global _current_stream_id
    my_stream_id = time.time_ns()
    _current_stream_id = my_stream_id

    frame_interval = 1.0 / STREAM_TARGET_FPS
    target_width = STREAM_FRAME_WIDTH
    target_height = STREAM_FRAME_HEIGHT

    # Mode SIM: live MuJoCo render without hardware camera dependencies
    if mode == "sim":
        while _current_stream_id == my_stream_id:
            start = time.time()
            yield get_sim_video_frame(robot=robot)
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
                yield _compose_twin_frame(top_frame, bottom_frame)

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

            yield frame_rgb

            elapsed = time.time() - start_time
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)
    finally:
        if cap is not None and cap.isOpened():
            cap.release()
            print(f"[INFO] Released camera resource for index {camera_index}.")
