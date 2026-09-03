"""Tests for raw camera stream isolation and decoupled UI overlay rendering.

Verifies:
- Tier 1: get_latest_raw_jpeg basic retrieval, simulation runtime raw frames,
  raw vs annotated encoding.
- Tier 2: Color channel fidelity (RGB vs BGR), empty stream fallback,
  overlay toggle independence.
- Tier 3: High-concurrency thread safety between stream writer and parallel readers.
- Tier 4: Pixel-exact raw frame isolation when all UI overlays and model targets are active.
"""

import threading
import time
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

import ui.stream as stream_mod
from ui.state import get_active_robot, set_active_robot
from ui.stream import (
    _SIM_RUNTIME,
    _global_frame_lock,
    add_model_target,
    clear_model_targets,
    get_calibrator,
    get_latest_jpeg,
    get_latest_raw_jpeg,
    set_bounding_boxes_overlay,
    set_calibration_overlay,
    set_crosshair_overlay,
    set_grid_overlay,
    set_model_targets,
    set_target_overlay_enabled,
)


def _make_smooth_gradient_canvas() -> np.ndarray:
    """Create a pristine 480x640 RGB gradient canvas for deterministic compression testing."""
    y, x = np.mgrid[0:480, 0:640]
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    canvas[:, :, 0] = (x / 640.0 * 200).astype(np.uint8)
    canvas[:, :, 1] = (y / 480.0 * 200).astype(np.uint8)
    canvas[:, :, 2] = 100
    return canvas


@pytest.fixture(autouse=True)
def cleanup_stream_state():
    """Reset stream overlays, model targets, and frame buffers before and after tests."""
    # Reset overlays to defaults
    set_target_overlay_enabled(True)
    set_calibration_overlay(False)
    set_bounding_boxes_overlay(False)
    set_grid_overlay(False)
    set_crosshair_overlay(False)
    clear_model_targets()

    # Reset global frame buffers
    stream_mod._SIM_RUNTIME.stop()
    stream_mod._SIM_RUNTIME._latest_raw_frame = None
    stream_mod._SIM_RUNTIME._latest_frame = None
    stream_mod._SIM_RUNTIME._robot = None
    with _global_frame_lock:
        if hasattr(stream_mod, "_global_latest_raw_frame"):
            stream_mod._global_latest_raw_frame = None
        stream_mod._global_latest_frame = None
        if hasattr(stream_mod, "_global_latest_annotated_frame"):
            stream_mod._global_latest_annotated_frame = None

    orig_robot = get_active_robot()
    yield
    # Cleanup after test
    set_target_overlay_enabled(True)
    set_calibration_overlay(False)
    set_bounding_boxes_overlay(False)
    set_grid_overlay(False)
    set_crosshair_overlay(False)
    clear_model_targets()
    set_active_robot(orig_robot)
    stream_mod._SIM_RUNTIME.stop()
    stream_mod._SIM_RUNTIME._latest_raw_frame = None
    stream_mod._SIM_RUNTIME._latest_frame = None
    stream_mod._SIM_RUNTIME._robot = None
    with _global_frame_lock:
        if hasattr(stream_mod, "_global_latest_raw_frame"):
            stream_mod._global_latest_raw_frame = None
        stream_mod._global_latest_frame = None
        if hasattr(stream_mod, "_global_latest_annotated_frame"):
            stream_mod._global_latest_annotated_frame = None


# ==============================================================================
# TIER 1: CORE RAW FRAME EXTRACTION & ENCODING
# ==============================================================================


def test_get_latest_raw_jpeg_basic() -> None:
    """Verify get_latest_raw_jpeg() returns valid JPEG bytes matching input frame."""
    # 1. Create a pristine test RGB frame with known gradient pattern
    raw_canvas = _make_smooth_gradient_canvas()

    # 2. Inject raw canvas into global buffer
    with _global_frame_lock:
        if hasattr(stream_mod, "_global_latest_raw_frame"):
            stream_mod._global_latest_raw_frame = raw_canvas.copy()
        stream_mod._global_latest_frame = raw_canvas.copy()

    # 3. Retrieve raw JPEG bytes
    raw_jpeg = get_latest_raw_jpeg()
    assert raw_jpeg is not None
    assert isinstance(raw_jpeg, bytes)
    assert raw_jpeg.startswith(b"\xff\xd8")  # Valid JPEG SOI header

    # 4. Decode JPEG bytes to RGB and verify shape and fidelity
    decoded_bgr = cv2.imdecode(np.frombuffer(raw_jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded_bgr is not None
    decoded_rgb = cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB)
    assert decoded_rgb.shape == (480, 640, 3)

    # Max difference within smooth gradient lossy JPEG tolerance (<= 5)
    diff = np.abs(decoded_rgb.astype(int) - raw_canvas.astype(int))
    assert np.max(diff) <= 5
    assert np.mean(diff) <= 1.0


def test_get_latest_raw_jpeg_sim_runtime() -> None:
    """Verify _SingletonSimRuntime provides un-annotated raw frames and JPEGs."""
    mock_robot = MagicMock()
    sim_rgb = _make_smooth_gradient_canvas()
    mock_robot.render_rgb_frame.return_value = sim_rgb.copy()

    # Test get_latest_raw_frame on runtime if present
    if hasattr(_SIM_RUNTIME, "get_latest_raw_frame"):
        raw_frame = _SIM_RUNTIME.get_latest_raw_frame(mock_robot)
        assert raw_frame is not None
        assert raw_frame.shape == (480, 640, 3)

    if hasattr(_SIM_RUNTIME, "get_latest_raw_jpeg"):
        raw_jpeg = _SIM_RUNTIME.get_latest_raw_jpeg(mock_robot)
        assert raw_jpeg is not None
        assert isinstance(raw_jpeg, bytes)
        assert raw_jpeg.startswith(b"\xff\xd8")

        decoded = cv2.imdecode(np.frombuffer(raw_jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        assert decoded is not None
        decoded_rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
        diff = np.abs(decoded_rgb.astype(int) - sim_rgb.astype(int))
        assert np.max(diff) <= 5
        assert np.mean(diff) <= 1.0


def test_raw_vs_annotated_encoding() -> None:
    """Verify raw JPEG bytes are distinct from annotated JPEG bytes when overlays are active."""
    raw_canvas = np.full((480, 640, 3), 100, dtype=np.uint8)

    with _global_frame_lock:
        if hasattr(stream_mod, "_global_latest_raw_frame"):
            stream_mod._global_latest_raw_frame = raw_canvas.copy()
        stream_mod._global_latest_frame = raw_canvas.copy()

    # Enable target overlays and add prominent model targets
    set_target_overlay_enabled(True)
    set_model_targets(
        [
            {
                "point": (500.0, 500.0),
                "label": "TARGET OBJECT [500, 500]",
                "box_2d": [300, 300, 700, 700],
            }
        ]
    )

    raw_bytes = get_latest_raw_jpeg()
    annotated_bytes = get_latest_jpeg()

    assert raw_bytes is not None
    assert annotated_bytes is not None
    # Bytes must be distinct because annotated frame contains drawings/text
    assert raw_bytes != annotated_bytes

    raw_decoded = cv2.cvtColor(
        cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR),
        cv2.COLOR_BGR2RGB,
    )
    annotated_decoded = cv2.cvtColor(
        cv2.imdecode(np.frombuffer(annotated_bytes, np.uint8), cv2.IMREAD_COLOR),
        cv2.COLOR_BGR2RGB,
    )

    diff_raw = np.abs(raw_decoded.astype(int) - raw_canvas.astype(int))
    diff_annotated = np.abs(annotated_decoded.astype(int) - raw_canvas.astype(int))

    assert np.max(diff_raw) <= 8
    assert np.max(diff_annotated) > 50  # Overlays introduce high-contrast graphics


# ==============================================================================
# TIER 2: COLOR CHANNEL FIDELITY, FALLBACK & OVERLAY INDEPENDENCE
# ==============================================================================


def test_raw_stream_color_channel_fidelity() -> None:
    """Verify RGB channel ordering is preserved through encoding and not inverted to BGR."""
    # Create 4 distinct quadrants: Pure Red, Pure Green, Pure Blue, Pure Yellow
    quadrants = np.zeros((480, 640, 3), dtype=np.uint8)
    quadrants[0:240, 0:320] = [255, 0, 0]  # Top-Left: Pure Red
    quadrants[0:240, 320:640] = [0, 255, 0]  # Top-Right: Pure Green
    quadrants[240:480, 0:320] = [0, 0, 255]  # Bottom-Left: Pure Blue
    quadrants[240:480, 320:640] = [255, 255, 0]  # Bottom-Right: Pure Yellow

    with _global_frame_lock:
        if hasattr(stream_mod, "_global_latest_raw_frame"):
            stream_mod._global_latest_raw_frame = quadrants.copy()
        stream_mod._global_latest_frame = quadrants.copy()

    raw_jpeg = get_latest_raw_jpeg()
    assert raw_jpeg is not None

    decoded_bgr = cv2.imdecode(np.frombuffer(raw_jpeg, np.uint8), cv2.IMREAD_COLOR)
    decoded_rgb = cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB)

    # Sample centers of the 4 quadrants (away from boundary lines)
    tl_rgb = decoded_rgb[120, 160]
    tr_rgb = decoded_rgb[120, 480]
    bl_rgb = decoded_rgb[360, 160]
    br_rgb = decoded_rgb[360, 480]

    # Check Red quadrant
    assert tl_rgb[0] > 240 and tl_rgb[1] < 20 and tl_rgb[2] < 20, f"Red quadrant inverted: {tl_rgb}"
    # Check Green quadrant
    assert tr_rgb[0] < 20 and tr_rgb[1] > 240 and tr_rgb[2] < 20, (
        f"Green quadrant inverted: {tr_rgb}"
    )
    # Check Blue quadrant
    assert bl_rgb[0] < 20 and bl_rgb[1] < 20 and bl_rgb[2] > 240, (
        f"Blue quadrant inverted: {bl_rgb}"
    )
    # Check Yellow quadrant
    assert br_rgb[0] > 240 and br_rgb[1] > 240 and br_rgb[2] < 20, (
        f"Yellow quadrant inverted: {br_rgb}"
    )


def test_empty_stream_fallback() -> None:
    """Verify get_latest_raw_jpeg() returns valid placeholder JPEG when no frame is present."""
    with _global_frame_lock:
        if hasattr(stream_mod, "_global_latest_raw_frame"):
            stream_mod._global_latest_raw_frame = None
        stream_mod._global_latest_frame = None

    set_active_robot(None)

    raw_jpeg = get_latest_raw_jpeg()
    assert raw_jpeg is not None
    assert isinstance(raw_jpeg, bytes)
    assert raw_jpeg.startswith(b"\xff\xd8")

    decoded_bgr = cv2.imdecode(np.frombuffer(raw_jpeg, np.uint8), cv2.IMREAD_COLOR)
    assert decoded_bgr is not None
    assert decoded_bgr.shape == (480, 640, 3)


def test_overlay_toggle_independence() -> None:
    """Verify toggling UI overlays on/off has 0% effect on get_latest_raw_jpeg() output."""
    base_frame = np.full((480, 640, 3), 128, dtype=np.uint8)
    base_frame[150:200, 200:300] = [200, 100, 50]

    with _global_frame_lock:
        if hasattr(stream_mod, "_global_latest_raw_frame"):
            stream_mod._global_latest_raw_frame = base_frame.copy()
        stream_mod._global_latest_frame = base_frame.copy()

    # Get baseline raw JPEG with overlays OFF
    set_target_overlay_enabled(False)
    set_calibration_overlay(False)
    set_bounding_boxes_overlay(False)
    set_grid_overlay(False)
    set_crosshair_overlay(False)
    clear_model_targets()

    baseline_raw_jpeg = get_latest_raw_jpeg()
    assert baseline_raw_jpeg is not None

    # Toggle ALL overlays ON and add model targets
    set_target_overlay_enabled(True)
    set_calibration_overlay(True)
    set_bounding_boxes_overlay(True)
    set_grid_overlay(True)
    set_crosshair_overlay(True)
    add_model_target(
        point=(350.0, 420.0),
        label="TOGGLE TEST TARGET",
        box_2d=[300, 350, 450, 550],
    )

    overlay_active_raw_jpeg = get_latest_raw_jpeg()
    assert overlay_active_raw_jpeg is not None

    # Raw JPEG must be identical regardless of overlay toggle states
    assert overlay_active_raw_jpeg == baseline_raw_jpeg


# ==============================================================================
# TIER 3: CONCURRENCY & THREAD SAFETY
# ==============================================================================


def test_raw_stream_concurrency_thread_safety() -> None:
    """Simulate parallel generator and reader threads to assert zero race conditions or crashes."""
    stop_event = threading.Event()
    exceptions: list[Exception] = []

    def writer_loop() -> None:
        """Simulate video capture thread updating raw and annotated buffers."""
        try:
            step = 0
            while not stop_event.is_set():
                step += 1
                color_val = (step * 5) % 255
                frame = np.full((480, 640, 3), color_val, dtype=np.uint8)
                with _global_frame_lock:
                    if hasattr(stream_mod, "_global_latest_raw_frame"):
                        stream_mod._global_latest_raw_frame = frame.copy()
                    stream_mod._global_latest_frame = frame.copy()
                    if hasattr(stream_mod, "_global_latest_annotated_frame"):
                        stream_mod._global_latest_annotated_frame = frame.copy()
                time.sleep(0.005)
        except Exception as exc:
            exceptions.append(exc)

    def raw_reader_loop() -> None:
        """Simulate Gemini Live session fetching pristine raw frames."""
        try:
            for _ in range(50):
                if stop_event.is_set():
                    break
                jpeg = get_latest_raw_jpeg()
                assert jpeg is not None
                assert jpeg.startswith(b"\xff\xd8")
                decoded = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
                assert decoded is not None
                assert decoded.shape == (480, 640, 3)
                time.sleep(0.003)
        except Exception as exc:
            exceptions.append(exc)

    def ui_reader_loop() -> None:
        """Simulate Gradio UI MJPEG streaming fetching annotated frames."""
        try:
            for _ in range(50):
                if stop_event.is_set():
                    break
                jpeg = get_latest_jpeg()
                assert jpeg is not None
                assert jpeg.startswith(b"\xff\xd8")
                time.sleep(0.003)
        except Exception as exc:
            exceptions.append(exc)

    writer_thread = threading.Thread(target=writer_loop, daemon=True)
    raw_readers = [threading.Thread(target=raw_reader_loop, daemon=True) for _ in range(4)]
    ui_readers = [threading.Thread(target=ui_reader_loop, daemon=True) for _ in range(4)]

    writer_thread.start()
    for t in raw_readers + ui_readers:
        t.start()

    for t in raw_readers + ui_readers:
        t.join(timeout=5.0)

    stop_event.set()
    writer_thread.join(timeout=2.0)

    assert len(exceptions) == 0, f"Concurrent stream operations raised exceptions: {exceptions}"


# ==============================================================================
# TIER 4: PIXEL-EXACT RAW FRAME ISOLATION
# ==============================================================================


def test_pixel_exact_raw_frame_isolation_with_all_overlays_active() -> None:
    """Verify pixel-exact raw frame cleanliness when all UI overlays are rendered."""
    # 1. Create a pristine test canvas with continuous gradient features
    test_canvas = _make_smooth_gradient_canvas()

    # 2. Inject canvas into raw buffer
    with _global_frame_lock:
        if hasattr(stream_mod, "_global_latest_raw_frame"):
            stream_mod._global_latest_raw_frame = test_canvas.copy()
        stream_mod._global_latest_frame = test_canvas.copy()

    # 3. Enable ALL overlays simultaneously
    set_target_overlay_enabled(True)
    set_calibration_overlay(True)
    set_bounding_boxes_overlay(True)
    set_grid_overlay(True)
    set_crosshair_overlay(True)

    # Add 3 distinct model targets across the canvas
    set_model_targets(
        [
            {
                "point": (300.0, 200.0),
                "label": "RED CUBE [300, 200]",
                "box_2d": [250, 150, 350, 250],
            },
            {
                "point": (650.0, 400.0),
                "label": "BLUE PRISM [650, 400]",
                "box_2d": [600, 350, 700, 450],
            },
            {
                "point": (500.0, 500.0),
                "label": "GREEN CYLINDER [500, 500]",
                "box_2d": [450, 450, 550, 550],
            },
        ]
    )

    # Ensure calibrator is initialized
    calib = get_calibrator(mode="sim")
    assert calib.is_calibrated

    # 4. Fetch raw frame and annotated frame
    raw_jpeg = get_latest_raw_jpeg()
    annotated_jpeg = get_latest_jpeg()

    assert raw_jpeg is not None
    assert annotated_jpeg is not None

    # 5. Decode both images
    raw_bgr = cv2.imdecode(np.frombuffer(raw_jpeg, np.uint8), cv2.IMREAD_COLOR)
    raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)

    annotated_bgr = cv2.imdecode(np.frombuffer(annotated_jpeg, np.uint8), cv2.IMREAD_COLOR)
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    # 6. Verify raw frame error is within JPEG compression tolerance (max <= 5, mean < 1.0)
    diff_raw = np.abs(raw_rgb.astype(int) - test_canvas.astype(int))
    max_raw_diff = np.max(diff_raw)
    mean_raw_diff = np.mean(diff_raw)

    assert max_raw_diff <= 5, f"Raw frame altered by overlays! Max pixel diff: {max_raw_diff}"
    assert mean_raw_diff < 1.0, f"Raw frame noise higher than JPEG compression: {mean_raw_diff}"

    # 7. Verify annotated frame has large significant delta from overlays
    diff_annotated = np.abs(annotated_rgb.astype(int) - test_canvas.astype(int))
    max_annotated_diff = np.max(diff_annotated)
    altered_pixels = np.count_nonzero(diff_annotated > 30)

    assert max_annotated_diff > 100, (
        f"Annotated frame did not contain visible overlays: max diff {max_annotated_diff}"
    )
    assert altered_pixels > 1000, f"Too few overlay pixels rendered: {altered_pixels}"
