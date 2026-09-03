"""Adversarial stress testing for vision stream isolation, overlay decoupling, and concurrency."""

import threading
import time

import cv2
import numpy as np
import pytest

import ui.stream as stream_mod
from simulation.sim_robot import SO101Robot
from ui.state import get_active_robot, set_active_robot
from ui.stream import (
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


@pytest.fixture(autouse=True)
def stream_cleanup():
    """Reset stream state before and after each test."""
    set_target_overlay_enabled(True)
    set_calibration_overlay(False)
    set_bounding_boxes_overlay(False)
    set_grid_overlay(False)
    set_crosshair_overlay(False)
    clear_model_targets()
    stream_mod._SIM_RUNTIME.stop()
    stream_mod._SIM_RUNTIME._latest_raw_frame = None
    stream_mod._SIM_RUNTIME._latest_frame = None
    stream_mod._SIM_RUNTIME._robot = None
    with _global_frame_lock:
        stream_mod._global_latest_raw_frame = None
        stream_mod._global_latest_annotated_frame = None
        stream_mod._global_latest_frame = None
    orig_robot = get_active_robot()
    yield
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
        stream_mod._global_latest_raw_frame = None
        stream_mod._global_latest_annotated_frame = None
        stream_mod._global_latest_frame = None


# ==============================================================================
# 1. ZERO-LEAKAGE ADVERSARIAL STRESS TEST & ORACLES
# ==============================================================================


def test_adversarial_zero_leakage_dense_overlays_and_textures():
    """Stress-test zero overlay leakage under dense multi-overlay and textured inputs."""
    # 1. Generate high-frequency checkered and gradient pattern
    h, w = 480, 640
    y, x = np.mgrid[0:h, 0:w]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:, :, 0] = ((x // 32) % 2 * 100 + (y // 32) % 2 * 50).astype(np.uint8)
    canvas[:, :, 1] = ((x / w) * 220).astype(np.uint8)
    canvas[:, :, 2] = ((y / h) * 220).astype(np.uint8)

    # Reference pristine direct JPEG encoding
    ref_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    _, ref_enc = cv2.imencode(".jpg", ref_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 98])
    ref_bytes = ref_enc.tobytes()
    ref_decoded_rgb = cv2.cvtColor(cv2.imdecode(ref_enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

    # 2. Inject canvas into raw frame buffer
    with _global_frame_lock:
        stream_mod._global_latest_raw_frame = canvas.copy()
        stream_mod._global_latest_frame = canvas.copy()

    # 3. Activate ALL overlay flags
    set_target_overlay_enabled(True)
    set_calibration_overlay(True)
    set_bounding_boxes_overlay(True)
    set_grid_overlay(True)
    set_crosshair_overlay(True)

    # Ensure calibrator is active
    calib = get_calibrator(mode="sim")
    assert calib.is_calibrated

    # Add 20 dense targets across all regions of the frame
    dense_targets = []
    for i in range(20):
        ty = 50.0 + (i * 45.0) % 900.0
        tx = 50.0 + (i * 70.0) % 900.0
        box = [max(0, ty - 30), max(0, tx - 30), min(1000, ty + 30), min(1000, tx + 30)]
        dense_targets.append(
            {
                "point": (ty, tx),
                "label": f"DENSE_TARGET_{i:02d} [{int(ty)}, {int(tx)}]",
                "box_2d": box,
            }
        )
    set_model_targets(dense_targets)

    # 4. Fetch raw and annotated JPEGs repeatedly
    for _ in range(5):
        raw_jpeg = get_latest_raw_jpeg()
        annotated_jpeg = get_latest_jpeg()

        assert raw_jpeg is not None
        assert annotated_jpeg is not None
        assert raw_jpeg != annotated_jpeg

        # Byte-for-byte exact equality between raw JPEG and pure un-annotated reference JPEG
        assert raw_jpeg == ref_bytes, (
            "Raw JPEG byte stream differs from reference pristine compression!"
        )

        # Decode raw JPEG
        raw_bgr = cv2.imdecode(np.frombuffer(raw_jpeg, np.uint8), cv2.IMREAD_COLOR)
        assert raw_bgr is not None
        raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)

        # Pixel delta vs reference decoded JPEG must be identically ZERO everywhere
        diff_vs_ref = np.abs(raw_rgb.astype(int) - ref_decoded_rgb.astype(int))
        assert np.max(diff_vs_ref) == 0, f"Raw frame pixel altered: max diff {np.max(diff_vs_ref)}"

        # Decode annotated JPEG
        annotated_bgr = cv2.imdecode(np.frombuffer(annotated_jpeg, np.uint8), cv2.IMREAD_COLOR)
        assert annotated_bgr is not None
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        # Check annotated frame has massive overlay modifications
        diff_annotated = np.abs(annotated_rgb.astype(int) - canvas.astype(int))
        assert np.max(diff_annotated) > 150
        assert np.count_nonzero(diff_annotated > 30) > 5000


def test_adversarial_zero_leakage_full_screen_spanning_overlays():
    """Verify zero leakage when target bounding box spans the entire 1000x1000 viewport."""
    canvas = np.full((480, 640, 3), (70, 140, 210), dtype=np.uint8)
    with _global_frame_lock:
        stream_mod._global_latest_raw_frame = canvas.copy()

    # Full screen bounding box overlay
    set_target_overlay_enabled(True)
    set_model_targets(
        [
            {
                "point": (500.0, 500.0),
                "label": "FULL_SCREEN_TARGET [0, 0, 1000, 1000]",
                "box_2d": [0.0, 0.0, 1000.0, 1000.0],
                "color": (255, 0, 0),
            }
        ]
    )

    raw_jpeg = get_latest_raw_jpeg()
    annotated_jpeg = get_latest_jpeg()

    assert raw_jpeg is not None
    assert annotated_jpeg is not None
    assert raw_jpeg != annotated_jpeg

    raw_rgb = cv2.cvtColor(
        cv2.imdecode(np.frombuffer(raw_jpeg, np.uint8), cv2.IMREAD_COLOR),
        cv2.COLOR_BGR2RGB,
    )
    annotated_rgb = cv2.cvtColor(
        cv2.imdecode(np.frombuffer(annotated_jpeg, np.uint8), cv2.IMREAD_COLOR),
        cv2.COLOR_BGR2RGB,
    )

    # Raw frame is pristine
    diff_raw = np.abs(raw_rgb.astype(int) - canvas.astype(int))
    assert np.max(diff_raw) <= 5

    # Annotated frame has full screen translucent tint / border changes
    diff_ann = np.abs(annotated_rgb.astype(int) - canvas.astype(int))
    assert np.count_nonzero(diff_ann > 10) > 200000  # More than 200k pixels changed in UI


def test_adversarial_raw_frame_immutability_and_defensive_copying():
    """Assert that mutating frame arrays downstream cannot corrupt the buffer."""
    test_canvas = np.full((480, 640, 3), 150, dtype=np.uint8)
    with _global_frame_lock:
        stream_mod._global_latest_raw_frame = test_canvas.copy()

    # Fetch raw frame via get_latest_raw_jpeg
    raw_jpeg_1 = get_latest_raw_jpeg()
    assert raw_jpeg_1 is not None

    with _global_frame_lock:
        frame_ref = stream_mod._global_latest_raw_frame
        assert frame_ref is not None
        assert np.all(frame_ref == 150)

    # Calling get_latest_jpeg (which applies overlays) must NOT alter _global_latest_raw_frame
    set_target_overlay_enabled(True)
    add_model_target((500.0, 500.0), label="OVERLAY", box_2d=[200, 200, 800, 800])
    get_latest_jpeg()

    with _global_frame_lock:
        assert np.all(stream_mod._global_latest_raw_frame == 150), (
            "get_latest_jpeg mutated _global_latest_raw_frame in place!"
        )


# ==============================================================================
# 2. MULTITHREADED HIGH-FREQUENCY CONCURRENCY STRESS
# ==============================================================================


def test_adversarial_high_concurrency_dual_stream_hammer():
    """Hammer stream with 12 concurrent threads executing high-frequency read/write/toggle ops."""
    stop_event = threading.Event()
    errors: list[str] = []
    frame_counter = [0]
    raw_read_counter = [0]
    annotated_read_counter = [0]

    def producer_thread(thread_id: int):
        """Simulate high-rate video ingestion (up to 120 FPS)."""
        nonlocal errors
        step = 0
        try:
            while not stop_event.is_set():
                step += 1
                color = (step * 7) % 255
                # Create a frame with unique pattern per step
                f = np.full((480, 640, 3), color, dtype=np.uint8)
                cv2.circle(f, (step % 640, (step * 2) % 480), 20, (255 - color, 255, 0), -1)

                with _global_frame_lock:
                    stream_mod._global_latest_raw_frame = f.copy()
                    stream_mod._global_latest_annotated_frame = f.copy()
                    stream_mod._global_latest_frame = f.copy()
                    frame_counter[0] += 1

                time.sleep(0.003)
        except Exception as exc:
            errors.append(f"Producer {thread_id} failed: {exc}")

    def raw_consumer_thread(thread_id: int):
        """Simulate Gemini VLA frame grabber calling get_latest_raw_jpeg."""
        nonlocal errors
        try:
            while not stop_event.is_set():
                raw_bytes = get_latest_raw_jpeg()
                if raw_bytes is None:
                    errors.append(f"RawConsumer {thread_id}: get_latest_raw_jpeg returned None")
                    break
                if not raw_bytes.startswith(b"\xff\xd8"):
                    errors.append(f"RawConsumer {thread_id}: Invalid JPEG SOI header")
                    break
                if not raw_bytes.endswith(b"\xff\xd9"):
                    errors.append(f"RawConsumer {thread_id}: Incomplete JPEG EOI")
                    break
                decoded = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
                if decoded is None or decoded.shape != (480, 640, 3):
                    errors.append(f"RawConsumer {thread_id}: Decoded frame invalid shape")
                    break
                raw_read_counter[0] += 1
                time.sleep(0.002)
        except Exception as exc:
            errors.append(f"RawConsumer {thread_id} failed: {exc}")

    def annotated_consumer_thread(thread_id: int):
        """Simulate Gradio UI MJPEG streamer calling get_latest_jpeg."""
        nonlocal errors
        try:
            while not stop_event.is_set():
                ui_bytes = get_latest_jpeg()
                if ui_bytes is None:
                    errors.append(f"AnnotatedConsumer {thread_id}: get_latest_jpeg returned None")
                    break
                if not ui_bytes.startswith(b"\xff\xd8"):
                    errors.append(f"AnnotatedConsumer {thread_id}: Invalid JPEG SOI header")
                    break
                annotated_read_counter[0] += 1
                time.sleep(0.002)
        except Exception as exc:
            errors.append(f"AnnotatedConsumer {thread_id} failed: {exc}")

    def overlay_mutator_thread():
        """Simulate UI user toggling overlays and model parsing new targets."""
        nonlocal errors
        toggle = False
        try:
            while not stop_event.is_set():
                toggle = not toggle
                set_target_overlay_enabled(toggle)
                set_calibration_overlay(not toggle)
                set_bounding_boxes_overlay(toggle)
                set_grid_overlay(not toggle)
                set_crosshair_overlay(toggle)
                if toggle:
                    set_model_targets([{"point": (400.0, 500.0), "label": "HAMMER TARGET"}])
                else:
                    clear_model_targets()
                time.sleep(0.01)
        except Exception as exc:
            errors.append(f"OverlayMutator failed: {exc}")

    # Launch threads
    producers = [threading.Thread(target=producer_thread, args=(i,), daemon=True) for i in range(2)]
    raw_consumers = [
        threading.Thread(target=raw_consumer_thread, args=(i,), daemon=True) for i in range(4)
    ]
    ann_consumers = [
        threading.Thread(target=annotated_consumer_thread, args=(i,), daemon=True) for i in range(4)
    ]
    mutators = [threading.Thread(target=overlay_mutator_thread, daemon=True) for _ in range(2)]

    all_threads = producers + raw_consumers + ann_consumers + mutators
    for t in all_threads:
        t.start()

    # Run heavy concurrency for 3.0 seconds
    time.sleep(3.0)
    stop_event.set()

    for t in all_threads:
        t.join(timeout=3.0)

    assert len(errors) == 0, f"Concurrency errors detected: {errors}"
    assert frame_counter[0] > 50, f"Too few frames produced: {frame_counter[0]}"
    assert raw_read_counter[0] > 100, f"Too few raw reads: {raw_read_counter[0]}"
    assert annotated_read_counter[0] > 100, f"Too few annotated reads: {annotated_read_counter[0]}"


def test_adversarial_simulation_runtime_live_motion_raw_isolation():
    """Verify raw JPEG purity during active MuJoCo robot joint actuation."""
    sim_robot = SO101Robot(render_viewer=False)
    set_active_robot(sim_robot)

    # Enable all overlays
    set_target_overlay_enabled(True)
    set_calibration_overlay(True)
    set_model_targets([{"point": (450.0, 450.0), "label": "ROBOT_TARGET"}])

    # Move joints across several angles and verify raw vs annotated
    for q0 in [0.0, 0.3, -0.3]:
        sim_robot.set_joint_positions([q0, -1.0, 1.2, 0.5, 0.0, 0.0])
        sim_robot.step(5)

        raw_jpeg = get_latest_raw_jpeg()
        annotated_jpeg = get_latest_jpeg()

        assert raw_jpeg is not None
        assert annotated_jpeg is not None
        assert raw_jpeg != annotated_jpeg

        raw_rgb = cv2.cvtColor(
            cv2.imdecode(np.frombuffer(raw_jpeg, np.uint8), cv2.IMREAD_COLOR),
            cv2.COLOR_BGR2RGB,
        )
        # Verify shape
        assert raw_rgb.shape == (480, 640, 3)


# ==============================================================================
# 3. COLOR SPACE FIDELITY & CHANNEL ORDERING VERIFICATION
# ==============================================================================


def test_adversarial_color_space_fidelity_across_all_probes():
    """Verify exact color channel preservation (no RGB/BGR inversions) across all encoders."""
    # Define distinct pure color blocks in an 8-patch test card
    h, w = 480, 640
    test_card = np.zeros((h, w, 3), dtype=np.uint8)

    # 8 regions of 240x160
    colors_rgb = {
        "RED": (255, 0, 0),
        "GREEN": (0, 255, 0),
        "BLUE": (0, 0, 255),
        "CYAN": (0, 255, 255),
        "MAGENTA": (255, 0, 255),
        "YELLOW": (255, 255, 0),
        "WHITE": (255, 255, 255),
        "DARK_GRAY": (40, 40, 40),
    }

    patches = [
        ("RED", (0, 240, 0, 160)),
        ("GREEN", (0, 240, 160, 320)),
        ("BLUE", (0, 240, 320, 480)),
        ("CYAN", (0, 240, 480, 640)),
        ("MAGENTA", (240, 480, 0, 160)),
        ("YELLOW", (240, 480, 160, 320)),
        ("WHITE", (240, 480, 320, 480)),
        ("DARK_GRAY", (240, 480, 480, 640)),
    ]

    for name, (y1, y2, x1, x2) in patches:
        test_card[y1:y2, x1:x2] = colors_rgb[name]

    with _global_frame_lock:
        stream_mod._global_latest_raw_frame = test_card.copy()

    # Fetch raw JPEG
    raw_jpeg = get_latest_raw_jpeg()
    assert raw_jpeg is not None

    # Decode using cv2 (returns BGR) and convert to RGB
    decoded_bgr = cv2.imdecode(np.frombuffer(raw_jpeg, np.uint8), cv2.IMREAD_COLOR)
    assert decoded_bgr is not None
    decoded_rgb = cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB)

    for name, (y1, y2, x1, x2) in patches:
        # Sample patch center
        cy = (y1 + y2) // 2
        cx = (x1 + x2) // 2
        sampled = decoded_rgb[cy, cx]
        expected = colors_rgb[name]

        # Check fidelity
        diff = np.abs(sampled.astype(int) - np.array(expected, dtype=int))
        assert np.max(diff) <= 10, (
            f"Color distortion in {name}: expected RGB {expected}, "
            f"got RGB {tuple(sampled)} (diff {diff})"
        )


def test_adversarial_twin_frame_composition_color_fidelity():
    """Verify _compose_twin_frame maintains RGB alignment without channel swap."""
    top_rgb = np.full((240, 320, 3), (255, 0, 0), dtype=np.uint8)  # Top pure red
    bottom_rgb = np.full((240, 320, 3), (0, 0, 255), dtype=np.uint8)  # Bottom pure blue

    composed = stream_mod._compose_twin_frame(top_rgb, bottom_rgb)
    assert composed.shape == (482, 320, 3)

    # Top sample: must be Red
    top_sample = composed[100, 160]
    assert top_sample[0] == 255 and top_sample[1] == 0 and top_sample[2] == 0

    # Divider sample: must be white
    divider_sample = composed[240, 160]
    assert np.all(divider_sample == 255)

    # Bottom sample: must be Blue
    bottom_sample = composed[380, 160]
    assert bottom_sample[0] == 0 and bottom_sample[1] == 0 and bottom_sample[2] == 255
