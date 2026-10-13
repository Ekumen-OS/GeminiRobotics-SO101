"""Characterization tests for frame buffers, JPEG getters, and calibrator caching."""

import numpy as np
import pytest

import vision as stream_mod
import vision.calibrator as calib_mod
import vision.frames as frames_mod
import vision.sim_runtime as sim_rt_mod
from robot.calibration import ArucoWorkspaceCalibrator


@pytest.fixture(autouse=True)
def reset_buffers_and_calibrator():
    """Reset global frame buffers and calibrator singleton before and after test."""
    with calib_mod._calibrator_lock:
        calib_mod._global_calibrator = None
    with frames_mod._global_frame_lock:
        frames_mod._global_latest_raw_frame = None
        frames_mod._global_latest_annotated_frame = None
        frames_mod._global_latest_frame = None
    sim_rt_mod.stop_sim_runtime()
    yield
    with calib_mod._calibrator_lock:
        calib_mod._global_calibrator = None
    with frames_mod._global_frame_lock:
        frames_mod._global_latest_raw_frame = None
        frames_mod._global_latest_annotated_frame = None
        frames_mod._global_latest_frame = None
    sim_rt_mod.stop_sim_runtime()


def test_get_calibrator_real_mode_and_caching() -> None:
    """Verify real mode calibrator creation and singleton caching."""
    cal1 = stream_mod.get_calibrator(mode="real")
    assert isinstance(cal1, ArucoWorkspaceCalibrator)
    assert cal1.mode == "real"

    cal2 = stream_mod.get_calibrator(mode="real")
    assert cal1 is cal2


def test_get_calibrator_sim_mode() -> None:
    """Verify sim mode calibrator initialization."""
    cal_sim = stream_mod.get_calibrator(mode="sim")
    assert isinstance(cal_sim, ArucoWorkspaceCalibrator)
    assert cal_sim.is_calibrated


def test_has_published_frame_and_buffer_retrieval() -> None:
    """Verify has_published_frame accurately reflects frame buffer presence."""
    assert not stream_mod.has_published_frame()

    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    with frames_mod._global_frame_lock:
        frames_mod._global_latest_raw_frame = dummy.copy()
        frames_mod._global_latest_frame = dummy.copy()

    assert stream_mod.has_published_frame()

    raw_jpeg = stream_mod.get_latest_raw_jpeg()
    assert isinstance(raw_jpeg, bytes)
    assert raw_jpeg.startswith(b"\xff\xd8")

    annotated_jpeg = stream_mod.get_latest_jpeg()
    assert isinstance(annotated_jpeg, bytes)
    assert annotated_jpeg.startswith(b"\xff\xd8")


def test_raw_frame_publish_time_and_wait_for_newer_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    """get_raw_jpeg_after waits for a frame published after t_after and times out otherwise."""
    import threading
    import time

    dummy = np.full((480, 640, 3), 90, dtype=np.uint8)
    monkeypatch.setattr(frames_mod._SIM_RUNTIME, "is_running", lambda: True)
    with frames_mod._global_frame_lock:
        frames_mod._global_latest_raw_frame = dummy.copy()
        frames_mod._global_latest_raw_ts = time.monotonic()
    assert stream_mod.get_latest_raw_frame_ts() > 0.0

    # A frame published before t_after is stale: no newer frame arrives -> None after the timeout.
    t_after = time.monotonic()
    started = time.monotonic()
    assert stream_mod.get_raw_jpeg_after(t_after, timeout_s=0.1) is None
    assert time.monotonic() - started >= 0.1

    # A frame published while waiting is returned.
    def publish_later() -> None:
        time.sleep(0.05)
        frames_mod._on_sim_frame(dummy)

    thread = threading.Thread(target=publish_later)
    thread.start()
    jpeg = stream_mod.get_raw_jpeg_after(t_after, timeout_s=1.0)
    thread.join()
    assert jpeg is not None and jpeg.startswith(b"\xff\xd8")


def test_raw_jpeg_after_returns_immediately_when_nothing_publishes() -> None:
    """With no capture or sim render thread, waiting could never succeed: use the current frame."""
    import time

    dummy = np.full((480, 640, 3), 90, dtype=np.uint8)
    with frames_mod._global_frame_lock:
        frames_mod._global_latest_raw_frame = dummy.copy()
    started = time.monotonic()
    jpeg = stream_mod.get_raw_jpeg_after(time.monotonic() + 100.0, timeout_s=5.0)
    assert time.monotonic() - started < 1.0
    assert jpeg is None or jpeg.startswith(b"\xff\xd8")
