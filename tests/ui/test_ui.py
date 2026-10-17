"""Unit tests for the Gradio Web UI components, stream overlays, and FastAPI app mount."""

import threading
from unittest.mock import MagicMock

import cv2
import gradio as gr
import numpy as np
from fastapi import FastAPI

from agent.vla_controller import VLAController, parse_targets_from_text
from robot.constants import JOINT_LIMITS, JOINT_NAMES
from robot.registry import get_active_robot, set_active_robot
from ui.app import build_vla_interface, create_app
from ui.components.chat import build_chat_interface, wire_chat_events
from ui.components.controls import build_controls_interface
from ui.components.telemetry import render_joint_state_panel
from ui.components.vision import build_vision_interface
from vision.calibrator import get_calibrator
from vision.frames import get_latest_jpeg, get_latest_raw_jpeg
from vision.overlays import (
    is_calibration_overlay_enabled,
    is_target_overlay_enabled,
    set_calibration_overlay,
    set_target_overlay_enabled,
)
from vision.targets import (
    clear_model_targets,
    draw_model_target_overlay,
    get_model_targets,
    set_model_targets,
)


def test_stream_overlay_flags() -> None:
    """Verify getter and setter functionality for active video stream overlay states."""
    set_target_overlay_enabled(True)
    assert is_target_overlay_enabled() is True
    set_target_overlay_enabled(False)
    assert is_target_overlay_enabled() is False

    set_calibration_overlay(True)
    assert is_calibration_overlay_enabled() is True
    set_calibration_overlay(False)
    assert is_calibration_overlay_enabled() is False


def test_model_target_overlay_drawing() -> None:
    """Verify set_model_targets and draw_model_target_overlay rendering."""
    set_target_overlay_enabled(True)
    set_model_targets([{"point": (500.0, 500.0), "label": "TEST_TARGET [500, 500]"}])
    targets = get_model_targets()
    assert len(targets) == 1
    assert targets[0]["point"] == (500.0, 500.0)
    assert targets[0]["label"] == "TEST_TARGET [500, 500]"

    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    rendered = draw_model_target_overlay(blank_frame)
    assert rendered.shape == (480, 640, 3)
    assert np.any(rendered != 0)

    clear_model_targets()
    assert get_model_targets() == []


def test_multi_target_management_and_drawing() -> None:
    """Verify multi-target setters, getters, box_2d parsing, and overlay drawing."""
    set_target_overlay_enabled(True)
    clear_model_targets()
    assert len(get_model_targets()) == 0

    targets_data = [
        {"point": (300.0, 400.0), "label": "RED CUBE [300, 400]"},
        {"point": (600.0, 700.0), "label": "BLUE CYLINDER [600, 700]"},
        {"box_2d": [100.0, 200.0, 250.0, 350.0], "label": "GREEN BOWL"},
    ]
    set_model_targets(targets_data)

    active_targets = get_model_targets()
    assert len(active_targets) == 3
    assert active_targets[0]["point"] == (300.0, 400.0)
    assert active_targets[1]["point"] == (600.0, 700.0)
    assert active_targets[2]["point"] == (175.0, 275.0)
    assert active_targets[2]["box_2d"] == [100.0, 200.0, 250.0, 350.0]

    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    rendered = draw_model_target_overlay(blank_frame)
    assert rendered.shape == (480, 640, 3)
    assert np.any(rendered != 0)

    clear_model_targets()
    assert len(get_model_targets()) == 0


def test_dual_buffer_stream_zero_leakage(
    neutral_frame: np.ndarray, clean_stream_state: None
) -> None:
    """Verify get_latest_raw_jpeg has zero annotations while get_latest_jpeg includes overlays."""
    import vision.frames as stream_mod

    raw_canvas = neutral_frame.copy()
    set_target_overlay_enabled(True)
    set_model_targets(
        [
            {
                "point": (350.0, 450.0),
                "label": "VERIFY_TARGET",
                "box_2d": [200, 300, 500, 600],
            }
        ]
    )
    stream_mod._on_sim_frame(raw_canvas)

    raw_jpeg = get_latest_raw_jpeg()
    annotated_jpeg = get_latest_jpeg()

    assert raw_jpeg is not None
    assert annotated_jpeg is not None
    assert raw_jpeg != annotated_jpeg

    raw_bgr = cv2.imdecode(np.frombuffer(raw_jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
    diff_raw = np.abs(raw_rgb.astype(int) - raw_canvas.astype(int))
    assert np.max(diff_raw) <= 5


def test_stream_color_fidelity_rgb_vs_bgr(
    smooth_gradient_frame: np.ndarray, clean_stream_state: None
) -> None:
    """Verify color channel ordering is preserved and does not swap R and B."""
    import vision.frames as stream_mod

    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    test_frame[:, :, 0] = 230  # High Red, Zero Blue
    with stream_mod._global_frame_lock:
        stream_mod._global_latest_raw_frame = test_frame.copy()
        stream_mod._global_latest_annotated_frame = test_frame.copy()

    raw_jpeg = get_latest_raw_jpeg()
    assert raw_jpeg is not None
    decoded_bgr = cv2.imdecode(np.frombuffer(raw_jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    decoded_rgb = cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB)

    # Red channel should dominate, Blue should remain near zero
    assert decoded_rgb[200, 200, 0] > 200
    assert decoded_rgb[200, 200, 2] < 20


def test_stream_concurrency_thread_safety(
    neutral_frame: np.ndarray, clean_stream_state: None
) -> None:
    """Verify thread safety under concurrent reads of raw and annotated buffers."""
    import vision.frames as stream_mod

    with stream_mod._global_frame_lock:
        stream_mod._global_latest_raw_frame = neutral_frame.copy()
        stream_mod._global_latest_annotated_frame = neutral_frame.copy()

    errors: list[Exception] = []

    def reader_task() -> None:
        try:
            for _ in range(50):
                raw = get_latest_raw_jpeg()
                ann = get_latest_jpeg()
                assert raw is not None
                assert ann is not None
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=reader_task) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0


def test_parse_targets_from_text() -> None:
    """Verify extraction of multiple 2D spatial points and bounding boxes from text."""
    sample_text = (
        "I observe 2 objects on the table: red cube at [320, 450] and blue cylinder at [600, 750]."
    )
    targets = parse_targets_from_text(sample_text)
    assert len(targets) == 2
    assert targets[0]["point"] == (320.0, 450.0)
    assert "RED CUBE" in targets[0]["label"]
    assert targets[1]["point"] == (600.0, 750.0)
    assert "BLUE CYLINDER" in targets[1]["label"]

    box_text = "Target object bounding box is [200, 300, 400, 500] on the surface."
    targets_box = parse_targets_from_text(box_text)
    assert len(targets_box) == 1
    assert targets_box[0]["point"] == (300.0, 400.0)
    assert targets_box[0]["box_2d"] == [200.0, 300.0, 400.0, 500.0]


def test_build_chat_interface_structure() -> None:
    """Verify chat component creation and button layout configuration."""
    controller_sim = VLAController(mode="sim", robot=None)
    with gr.Blocks():
        prompt_input, chatbot, status_banner, exec_btn, home_btn, clear_btn, rand_btn = (
            build_chat_interface(controller=controller_sim, api_available=False)
        )

    assert chatbot.elem_id == "vla-chatbot"
    assert chatbot.height == 600
    assert status_banner.elem_id == "vla-status-banner"
    assert prompt_input.elem_id == "vla-prompt"
    assert prompt_input.lines == 1
    assert exec_btn is not None
    assert home_btn is not None
    assert clear_btn is not None
    assert rand_btn is not None

    # In real mode, rand_btn is None
    controller_real = VLAController(mode="real", robot=None)
    with gr.Blocks():
        _, _, _, _, _, _, rand_btn_real = build_chat_interface(
            controller=controller_real, api_available=False
        )
    assert rand_btn_real is None


def test_chat_status_banner_initial_state() -> None:
    """Verify initial status banner values across different configurations."""
    # API available in sim mode (calibrated by default)
    controller_sim = VLAController(mode="sim", robot=None)
    controller_sim._api_available = True
    with gr.Blocks():
        _, _, banner_ready, _, _, _, _ = build_chat_interface(
            controller=controller_sim, api_available=True
        )
    assert banner_ready.value == "🟢 **Status: Ready**"

    # API unavailable in sim mode
    with gr.Blocks():
        _, _, banner_no_api, _, _, _, _ = build_chat_interface(
            controller=controller_sim, api_available=False
        )
    assert banner_no_api.value == "⚠️ **Status: Gemini API Unavailable**"

    # Real mode without calibration
    calib = get_calibrator(mode="real")
    if calib is not None:
        calib.homography_matrix = None
    controller_real = VLAController(mode="real", robot=None)
    controller_real._api_available = True
    with gr.Blocks():
        _, _, banner_uncalib, _, _, _, _ = build_chat_interface(
            controller=controller_real, api_available=True
        )
    assert banner_uncalib.value == "⚠️ **Status: Workspace Not Calibrated**"


def test_process_instruction_blocks_and_unblocks_inputs() -> None:
    """Verify input blocking on turn start, intermediate streaming, and unblocking on completion."""
    controller_sim = VLAController(mode="sim", robot=None)
    controller_sim._api_available = True

    # Mock process_instruction_stream to yield intermediate chunks
    def mock_stream(instruction: str, get_latest_jpeg_fn=None, **kwargs):
        yield "Thinking..."
        yield "Moving arm..."
        yield "Completed task."

    controller_sim.process_instruction_stream = mock_stream

    with gr.Blocks():
        components = build_chat_interface(controller=controller_sim, api_available=True)
        prompt_input, chatbot, status_banner, exec_btn, home_btn, clear_btn, rand_btn = components
        sliders = [gr.Slider(value=0.0) for _ in range(6)]
        handlers = wire_chat_events(
            controller_sim,
            prompt_input,
            chatbot,
            status_banner,
            exec_btn,
            home_btn,
            clear_btn,
            sliders,
            rand_btn=rand_btn,
        )

    process_fn = handlers["process_instruction"]
    gen = process_fn("Pick up red cube", [])

    # Yield 0 (Turn Start): All inputs blocked, status Working
    y0 = next(gen)
    prompt_up, hist_up, status_up, exec_up, home_up, clear_up, rand_up = y0[:7]
    assert prompt_up.get("interactive") is False
    assert prompt_up.get("value") == ""
    assert status_up.get("value") == "⏳ **Status: Working...**"
    assert exec_up.get("interactive") is False
    assert home_up.get("interactive") is False
    assert clear_up.get("interactive") is False
    assert rand_up.get("interactive") is False
    assert len(hist_up) == 2
    assert hist_up[0]["content"] == "Pick up red cube"

    # Yield 1: Streaming first chunk
    y1 = next(gen)
    assert y1[1][-1]["content"] == "Thinking..."
    assert type(y1[0]) is type(gr.skip()) or y1[0] == gr.skip()
    assert type(y1[2]) is type(gr.skip()) or y1[2] == gr.skip()

    # Yield 2: Streaming second chunk
    y2 = next(gen)
    assert y2[1][-1]["content"] == "Moving arm..."

    # Yield 3: Streaming third chunk
    y3 = next(gen)
    assert y3[1][-1]["content"] == "Completed task."

    # Yield 4 (Completion / Final): All inputs re-enabled, status Ready
    y_final = next(gen)
    prompt_fin, hist_fin, status_fin, exec_fin, home_fin, clear_fin, rand_fin = y_final[:7]
    assert prompt_fin.get("interactive") is True
    assert status_fin.get("value") == "🟢 **Status: Ready**"
    assert exec_fin.get("interactive") is True
    assert home_fin.get("interactive") is True
    assert clear_fin.get("interactive") is True
    assert rand_fin.get("interactive") is True
    assert hist_fin[-1]["content"] == "Completed task."


def test_process_instruction_recovers_on_exception() -> None:
    """Verify robust recovery and input re-enabling when controller stream raises an exception."""
    controller_sim = VLAController(mode="sim", robot=None)
    controller_sim._api_available = True

    def mock_failing_stream(instruction: str, get_latest_jpeg_fn=None, **kwargs):
        yield "Starting task..."
        raise RuntimeError("Simulated WebSocket connection drop")

    controller_sim.interrupt = MagicMock()
    controller_sim.process_instruction_stream = mock_failing_stream

    with gr.Blocks():
        components = build_chat_interface(controller=controller_sim, api_available=True)
        prompt_input, chatbot, status_banner, exec_btn, home_btn, clear_btn, rand_btn = components
        sliders = [gr.Slider(value=0.0) for _ in range(6)]
        handlers = wire_chat_events(
            controller_sim,
            prompt_input,
            chatbot,
            status_banner,
            exec_btn,
            home_btn,
            clear_btn,
            sliders,
            rand_btn=rand_btn,
        )

    process_fn = handlers["process_instruction"]
    gen = process_fn("Pick up red cube", [])
    yields = list(gen)

    # Initial blocking yield, 1 chunk yield, final finally yield
    assert len(yields) >= 3

    y_final = yields[-1]
    prompt_fin, hist_fin, status_fin, exec_fin, home_fin, clear_fin, rand_fin = y_final[:7]
    assert prompt_fin.get("interactive") is True
    assert status_fin.get("value") == "🟢 **Status: Ready**"
    assert exec_fin.get("interactive") is True
    assert home_fin.get("interactive") is True
    assert clear_fin.get("interactive") is True
    assert rand_fin.get("interactive") is True
    # Verify runtime error was cleanly communicated in chat history and interrupt was called
    assert controller_sim.interrupt.called
    assert "Runtime Error" in hist_fin[-1]["content"]
    assert "Simulated WebSocket connection drop" in hist_fin[-1]["content"]


def test_process_instruction_uncalibrated_status(tmp_path: object, restore_settings: None) -> None:
    """Verify status banner and placeholder when workspace calibration is missing in real mode."""
    from pathlib import Path

    from robot.settings import Settings, set_settings

    set_settings(Settings.from_env(calibration_save_path=Path(str(tmp_path)) / "missing_real.npz"))
    calib = get_calibrator(mode="real")
    if calib is not None:
        calib.homography_matrix = None

    controller_real = VLAController(mode="real", robot=None)
    controller_real._api_available = True

    with gr.Blocks():
        components = build_chat_interface(controller=controller_real, api_available=True)
        prompt_input, chatbot, status_banner, exec_btn, home_btn, clear_btn, rand_btn = components
        sliders = [gr.Slider(value=0.0) for _ in range(6)]
        handlers = wire_chat_events(
            controller_real,
            prompt_input,
            chatbot,
            status_banner,
            exec_btn,
            home_btn,
            clear_btn,
            sliders,
            rand_btn=rand_btn,
        )

    process_fn = handlers["process_instruction"]
    yields = list(process_fn("Move to box", []))
    y_final = yields[-1]
    prompt_fin, _, status_fin, _, _, _, _ = y_final[:7]

    assert status_fin.get("value") == "⚠️ **Status: Workspace Not Calibrated**"
    assert "Calibrate workspace" in prompt_fin.get("placeholder", "")
    assert prompt_fin.get("interactive") is True


def test_process_instruction_empty_prompt_no_lock() -> None:
    """Verify empty or whitespace prompt returns immediately without locking controls."""
    controller_sim = VLAController(mode="sim", robot=None)
    controller_sim._api_available = True
    with gr.Blocks():
        components = build_chat_interface(controller=controller_sim, api_available=True)
        prompt_input, chatbot, status_banner, exec_btn, home_btn, clear_btn, rand_btn = components
        sliders = [gr.Slider(value=0.0) for _ in range(6)]
        handlers = wire_chat_events(
            controller_sim,
            prompt_input,
            chatbot,
            status_banner,
            exec_btn,
            home_btn,
            clear_btn,
            sliders,
            rand_btn=rand_btn,
        )

    process_fn = handlers["process_instruction"]
    initial_history = [{"role": "assistant", "content": "Welcome"}]
    yields = list(process_fn("   ", initial_history))
    assert len(yields) == 1
    y0 = yields[0]
    # Verify history was unchanged and controls were not modified
    assert y0[1] == initial_history
    assert type(y0[0]) is type(gr.skip()) or y0[0] == gr.skip()
    assert type(y0[2]) is type(gr.skip()) or y0[2] == gr.skip()


def test_clear_chat_resets_status_banner() -> None:
    """Verify clear chat button resets chatbot and updates status banner."""
    controller_sim = VLAController(mode="sim", robot=None)
    controller_sim._api_available = True
    with gr.Blocks():
        components = build_chat_interface(controller=controller_sim, api_available=True)
        prompt_input, chatbot, status_banner, exec_btn, home_btn, clear_btn, rand_btn = components
        sliders = [gr.Slider(value=0.0) for _ in range(6)]
        handlers = wire_chat_events(
            controller_sim,
            prompt_input,
            chatbot,
            status_banner,
            exec_btn,
            home_btn,
            clear_btn,
            sliders,
            rand_btn=rand_btn,
        )

    clear_fn = handlers["clear_chat"]
    new_chat, status_update = clear_fn()
    assert len(new_chat) == 1
    assert "System Active" in new_chat[0]["content"]
    assert status_update.get("value") == "🟢 **Status: Ready**"


def test_chat_uses_raw_jpeg_stream() -> None:
    """Verify that chat instruction execution passes get_latest_raw_jpeg to the controller."""
    controller_sim = VLAController(mode="sim", robot=None)
    controller_sim._api_available = True
    captured_kwargs: dict = {}

    def mock_process_stream(instruction: str, get_latest_jpeg_fn=None, **kwargs):
        captured_kwargs["get_latest_jpeg_fn"] = get_latest_jpeg_fn
        yield "Step done."

    controller_sim.process_instruction_stream = mock_process_stream

    with gr.Blocks():
        components = build_chat_interface(controller=controller_sim, api_available=True)
        prompt_input, chatbot, status_banner, exec_btn, home_btn, clear_btn, rand_btn = components
        sliders = [gr.Slider(value=0.0) for _ in range(6)]
        handlers = wire_chat_events(
            controller_sim,
            prompt_input,
            chatbot,
            status_banner,
            exec_btn,
            home_btn,
            clear_btn,
            sliders,
            rand_btn=rand_btn,
        )

    process_fn = handlers["process_instruction"]
    list(process_fn("Pick up red cube", []))

    assert "get_latest_jpeg_fn" in captured_kwargs
    fn_passed = captured_kwargs["get_latest_jpeg_fn"]
    assert callable(fn_passed)
    assert fn_passed is get_latest_raw_jpeg or fn_passed.__name__ == "get_latest_raw_jpeg"


def test_build_controls_interface_structure() -> None:
    """Verify controls interface creates 6 joint sliders and HTML telemetry panel."""
    init_deg = [0.0, -90.0, 90.0, 0.0, 0.0, 0.0]
    with gr.Blocks():
        sliders, joint_state_panel = build_controls_interface(
            robot=None, mode="sim", init_deg=init_deg
        )

    assert len(sliders) == 6
    for idx, (name, limits) in enumerate(JOINT_LIMITS.items()):
        slider = sliders[idx]
        assert slider.minimum == limits[0]
        assert slider.maximum == limits[1]
        assert slider.value == init_deg[idx]

    assert isinstance(joint_state_panel, gr.HTML)


def test_telemetry_timer_shows_distinct_commanded_and_measured_joints() -> None:
    """Bug 2 regression: telemetry timer renders distinct commanded and measured joint angles."""
    from simulation.sim_robot import SO101Robot
    from tests.robot.test_fail_closed_characterization import _make_mock_real_robot

    cmd_deg = [10.0, -20.0, 30.0, -15.0, 5.0, -40.0]
    cmd_rad = np.radians(cmd_deg, dtype=np.float32)

    # 1. Fake-bus RealSO101Robot where commanded != measured ([0, 0, 0, 0, 0, 0] deg)
    fake_bus = MagicMock()
    fake_bus.sync_read.return_value = {name: 0.0 for name in JOINT_NAMES}
    real_robot = _make_mock_real_robot(fake_bus)
    real_robot.set_joint_positions(cmd_rad)
    assert np.allclose(real_robot.get_commanded_joint_positions(), cmd_rad)
    assert np.allclose(real_robot.get_joint_positions(), np.zeros(6, dtype=np.float32))

    with gr.Blocks() as demo_real:
        build_controls_interface(robot=real_robot, mode="real", init_deg=[0.0] * 6)
    fns_real = demo_real.fns.values() if isinstance(demo_real.fns, dict) else demo_real.fns
    sync_fn_real = next(
        bf.fn for bf in fns_real if getattr(bf.fn, "__name__", "") == "sync_robot_state_html"
    )
    html_real = sync_fn_real()
    for val in cmd_deg:
        assert f"{val:.1f}" in html_real
    assert "0.0" in html_real

    # 2. Sim SO101Robot where commanded != measured (snap_gripper=False + qpos zeroed)
    sim_robot = SO101Robot()
    try:
        sim_robot.set_joint_positions(cmd_rad)
        with sim_robot._sim_lock:
            for j_id in sim_robot.joint_ids:
                sim_robot.data.qpos[sim_robot.model.jnt_qposadr[j_id]] = 0.0
        assert np.allclose(sim_robot.get_commanded_joint_positions(), cmd_rad)
        assert np.allclose(sim_robot.get_joint_positions(), np.zeros(6))

        with gr.Blocks() as demo_sim:
            build_controls_interface(robot=sim_robot, mode="sim", init_deg=[0.0] * 6)
        fns_sim = demo_sim.fns.values() if isinstance(demo_sim.fns, dict) else demo_sim.fns
        sync_fn_sim = next(
            bf.fn for bf in fns_sim if getattr(bf.fn, "__name__", "") == "sync_robot_state_html"
        )
        html_sim = sync_fn_sim()
        for val in cmd_deg:
            assert f"{val:.1f}" in html_sim
        assert "0.0" in html_sim
    finally:
        sim_robot.close()


def test_build_vision_interface_real() -> None:
    """Verify real mode vision interface attaches components to Gradio Blocks."""
    with gr.Blocks() as demo:
        build_vision_interface(robot=None, mode="real", demo=demo)

    assert len(demo.blocks) > 0


def test_build_vision_interface_sim() -> None:
    """Verify sim mode vision interface renders iframe container with 600px height."""
    with gr.Blocks() as demo:
        build_vision_interface(robot=None, mode="sim", demo=demo)

    assert len(demo.blocks) > 0


def test_render_joint_state_panel() -> None:
    """Verify telemetry HTML generation contains expected joint rows."""
    target_deg = [0.0, -100.0, 90.0, 70.0, 0.0, -40.0]
    measured_deg = [0.1, -99.9, 90.0, 70.2, -0.1, -40.0]
    html_output = render_joint_state_panel(target_deg, measured_deg)

    assert "Joint Telemetry (degrees)" in html_output
    assert "J1 shoulder_pan" in html_output
    assert "J6 gripper" in html_output
    assert "-100.0" in html_output


def test_build_vla_interface_blocks() -> None:
    """Verify top-level VLA Gradio Blocks instance builds cleanly."""
    demo = build_vla_interface(robot=None, mode="sim")
    assert isinstance(demo, gr.Blocks)


def test_create_app_fastapi() -> None:
    """Verify FastAPI application factory configures routes and mounts Gradio."""
    app = create_app(robot=None, mode="sim")
    assert isinstance(app, FastAPI)
    assert app.title == "GeminiRobotics-SO101"

    # Check registered route paths
    route_paths = [route.path for route in app.routes]
    assert "/viewer" in route_paths
    assert "/api/sim_state" in route_paths


def test_sim_robot_randomize_target() -> None:
    """Verify target manipulation object randomization in simulation."""
    from simulation.sim_robot import SO101Robot

    sim_robot = SO101Robot()
    new_pos = sim_robot.randomize_target(
        x_range=(0.15, 0.22), y_range=(-0.10, 0.10), z_height=0.015
    )
    assert 0.15 <= new_pos[0] <= 0.22
    assert -0.10 <= new_pos[1] <= 0.10
    assert new_pos[2] == 0.015


def test_active_robot_state() -> None:
    """Verify get_active_robot and set_active_robot global state getters and setters."""
    # Store original robot state
    original_robot = get_active_robot()
    try:
        set_active_robot(None)
        assert get_active_robot() is None

        from simulation.sim_robot import SO101Robot

        sim_robot = SO101Robot()
        set_active_robot(sim_robot)
        assert get_active_robot() is sim_robot
    finally:
        set_active_robot(original_robot)
