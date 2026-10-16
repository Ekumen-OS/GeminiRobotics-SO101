"""Characterization tests for camera device selection and V4L2 node classification."""

import struct
from pathlib import Path
from unittest.mock import patch

import vision.camera as stream_mod
from robot.settings import Settings, set_settings


def test_get_default_camera_index_by_camera_index(restore_settings: None) -> None:
    """Verify explicit CAMERA_INDEX preference is honored when present."""
    set_settings(Settings.from_env(camera_index=5))
    choices = [("Integrated Webcam A", 0), ("External Cam", 5)]
    assert stream_mod.get_default_camera_index(choices) == 5

    # If configured index is not available in hardware choices, fall through
    set_settings(Settings.from_env(camera_index=99))
    assert stream_mod.get_default_camera_index(choices) == 5  # external over integrated


def test_get_default_camera_index_by_camera_name(restore_settings: None) -> None:
    """Verify CAMERA_NAME preference is matched case-insensitively."""
    set_settings(Settings.from_env(camera_name="brio"))
    choices = [("Integrated Camera", 0), ("Logitech BRIO 4K", 4)]
    assert stream_mod.get_default_camera_index(choices) == 4

    # Non-matching name warns and falls back
    set_settings(Settings.from_env(camera_name="nonexistent"))
    assert stream_mod.get_default_camera_index(choices) == 4  # external over integrated


def test_get_default_camera_index_external_over_integrated(restore_settings: None) -> None:
    """Verify external cameras take precedence over built-in webcams."""
    set_settings(Settings.from_env())
    choices = [
        ("Built-in FaceTime HD Camera", 0),
        ("Integrated Webcam", 1),
        ("Desk USB Camera", 3),
    ]
    assert stream_mod.get_default_camera_index(choices) == 3


def test_get_default_camera_index_legacy_fallback_priority(restore_settings: None) -> None:
    """Verify legacy 2 -> 1 -> 0 preference when only integrated cameras exist."""
    set_settings(Settings.from_env())
    choices = [("Internal Cam 0", 0), ("Builtin Cam 1", 1), ("Integrated Cam 2", 2)]
    assert stream_mod.get_default_camera_index(choices) == 2

    choices_10 = [("Internal Cam 0", 0), ("Builtin Cam 1", 1)]
    assert stream_mod.get_default_camera_index(choices_10) == 1

    choices_other = [("Internal Cam 7", 7)]
    assert stream_mod.get_default_camera_index(choices_other) == 7

    assert stream_mod.get_default_camera_index([]) == 2


def test_is_v4l2_video_capture_node() -> None:
    """Verify V4L2 device node classification based on capability flags."""
    dev = Path("/dev/video0")

    # None when query fails
    with patch("vision.camera._query_v4l2_device_caps", return_value=None):
        assert stream_mod._is_v4l2_video_capture_node(dev) is None

    # False for metadata capture node without video capture
    meta_only = stream_mod._V4L2_CAP_META_CAPTURE
    with patch("vision.camera._query_v4l2_device_caps", return_value=meta_only):
        assert stream_mod._is_v4l2_video_capture_node(dev) is False

    # True for standard video capture node
    video_cap = stream_mod._V4L2_CAP_VIDEO_CAPTURE
    with patch("vision.camera._query_v4l2_device_caps", return_value=video_cap):
        assert stream_mod._is_v4l2_video_capture_node(dev) is True

    # True for multiplanar video capture node
    mplane_cap = stream_mod._V4L2_CAP_VIDEO_CAPTURE_MPLANE
    with patch("vision.camera._query_v4l2_device_caps", return_value=mplane_cap):
        assert stream_mod._is_v4l2_video_capture_node(dev) is True

    # False for non-capture device
    with patch("vision.camera._query_v4l2_device_caps", return_value=0x00000002):
        assert stream_mod._is_v4l2_video_capture_node(dev) is False


def test_query_v4l2_device_caps_ioctl() -> None:
    """Verify _query_v4l2_device_caps unpacks struct v4l2_capability correctly."""
    dev = Path("/dev/video0")

    # OSError on open
    with patch("os.open", side_effect=OSError("Device not found")):
        assert stream_mod._query_v4l2_device_caps(dev) is None

    # Successful ioctl with device_caps flag
    def fake_ioctl_with_dev_caps(fd: int, req: int, buf: bytearray) -> int:
        packed = struct.pack(
            "<16s32s32sIII",
            b"driver",
            b"card",
            b"bus",
            1,
            stream_mod._V4L2_CAP_DEVICE_CAPS | 0x1,
            stream_mod._V4L2_CAP_VIDEO_CAPTURE,
        )
        buf[: len(packed)] = packed
        return 0

    with (
        patch("os.open", return_value=42),
        patch("fcntl.ioctl", side_effect=fake_ioctl_with_dev_caps),
        patch("os.close"),
    ):
        caps = stream_mod._query_v4l2_device_caps(dev)
        assert caps == stream_mod._V4L2_CAP_VIDEO_CAPTURE

    # Successful ioctl without device_caps flag (uses capabilities)
    def fake_ioctl_standard(fd: int, req: int, buf: bytearray) -> int:
        packed = struct.pack(
            "<16s32s32sIII",
            b"driver",
            b"card",
            b"bus",
            1,
            stream_mod._V4L2_CAP_VIDEO_CAPTURE,
            0,
        )
        buf[: len(packed)] = packed
        return 0

    with (
        patch("os.open", return_value=42),
        patch("fcntl.ioctl", side_effect=fake_ioctl_standard),
        patch("os.close"),
    ):
        caps = stream_mod._query_v4l2_device_caps(dev)
        assert caps == stream_mod._V4L2_CAP_VIDEO_CAPTURE
