"""Camera discovery, V4L2 device classification, and OpenCV hardware capture acquisition."""

import logging
import sys
import threading
import time
from pathlib import Path
from typing import Any

import cv2

from robot.settings import get_settings

logger = logging.getLogger(__name__)

_device_names: dict[int, str] = {}
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
