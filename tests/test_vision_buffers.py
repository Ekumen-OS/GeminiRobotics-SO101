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
