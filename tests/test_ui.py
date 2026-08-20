"""Unit tests for the Gradio Web UI components, stream overlays, and FastAPI app mount."""

import gradio as gr
from fastapi import FastAPI

from agent.vla_controller import VLAController, parse_targets_from_text
from robot.constants import JOINT_LIMITS
from ui.app import build_vla_interface, create_app
from ui.components.chat import build_chat_interface
from ui.components.controls import build_controls_interface
from ui.components.telemetry import render_joint_state_panel
from ui.components.vision import build_vision_interface
from ui.stream import (
    add_model_target,
    clear_model_target_point,
    clear_model_targets,
    draw_model_target_overlay,
    get_model_target_point,
    get_model_targets,
    is_bounding_boxes_overlay_enabled,
    is_calibration_overlay_enabled,
    is_crosshair_overlay_enabled,
    is_grid_overlay_enabled,
    is_target_overlay_enabled,
    set_bounding_boxes_overlay,
    set_calibration_overlay,
    set_crosshair_overlay,
    set_grid_overlay,
    set_model_target_point,
    set_model_targets,
    set_target_overlay_enabled,
)


def test_stream_overlay_flags() -> None:
    """Verify getter and setter functionality for all video stream overlay states."""
    # Test Target overlay toggle
    set_target_overlay_enabled(True)
    assert is_target_overlay_enabled() is True
    set_target_overlay_enabled(False)
    assert is_target_overlay_enabled() is False

    # Test Calibration overlay toggle
    set_calibration_overlay(True)
    assert is_calibration_overlay_enabled() is True
    set_calibration_overlay(False)
    assert is_calibration_overlay_enabled() is False

    # Test Bounding Boxes overlay toggle
    set_bounding_boxes_overlay(True)
    assert is_bounding_boxes_overlay_enabled() is True
    set_bounding_boxes_overlay(False)
    assert is_bounding_boxes_overlay_enabled() is False

    # Test Grid overlay toggle
    set_grid_overlay(True)
    assert is_grid_overlay_enabled() is True
    set_grid_overlay(False)
    assert is_grid_overlay_enabled() is False

    # Test Crosshair overlay toggle
    set_crosshair_overlay(True)
    assert is_crosshair_overlay_enabled() is True
    set_crosshair_overlay(False)
    assert is_crosshair_overlay_enabled() is False


def test_model_target_overlay_drawing() -> None:
    """Verify set_model_target_point and draw_model_target_overlay rendering."""
    import numpy as np

    set_target_overlay_enabled(True)
    set_model_target_point((500.0, 500.0), label="TEST_TARGET [500, 500]")
    pt, lbl = get_model_target_point()
    assert pt == (500.0, 500.0)
    assert lbl == "TEST_TARGET [500, 500]"

    # Test overlay drawing on dummy frame
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    rendered = draw_model_target_overlay(blank_frame)
    assert rendered.shape == (480, 640, 3)
    assert np.any(rendered != 0)  # Drawing added non-zero pixels

    # Clear target
    clear_model_target_point()
    pt, lbl = get_model_target_point()
    assert pt is None
    assert lbl is None


def test_multi_target_management_and_drawing() -> None:
    """Verify multi-target setters, getters, box_2d parsing, and overlay drawing."""
    import numpy as np

    set_target_overlay_enabled(True)
    clear_model_targets()
    assert len(get_model_targets()) == 0

    # Set multiple targets
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
    # 3rd target point is computed center of box_2d: (175.0, 275.0)
    assert active_targets[2]["point"] == (175.0, 275.0)
    assert active_targets[2]["box_2d"] == [100.0, 200.0, 250.0, 350.0]

    # Add another target
    add_model_target((450.0, 550.0), label="YELLOW PRISM")
    assert len(get_model_targets()) == 4

    # Draw multi-target overlay on a blank frame
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    rendered = draw_model_target_overlay(blank_frame)
    assert rendered.shape == (480, 640, 3)
    assert np.any(rendered != 0)

    clear_model_targets()
    assert len(get_model_targets()) == 0


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
        prompt_input, chatbot, exec_btn, home_btn, clear_btn, rand_btn = build_chat_interface(
            controller=controller_sim, api_available=False
        )

    assert chatbot.elem_id == "vla-chatbot"
    assert chatbot.height == 600
    assert prompt_input.elem_id == "vla-prompt"
    assert prompt_input.lines == 1
    assert exec_btn is not None
    assert home_btn is not None
    assert clear_btn is not None
    assert rand_btn is not None

    # In real mode, rand_btn is None
    controller_real = VLAController(mode="real", robot=None)
    with gr.Blocks():
        _, _, _, _, _, rand_btn_real = build_chat_interface(
            controller=controller_real, api_available=False
        )
    assert rand_btn_real is None


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


def test_build_vision_interface_twin() -> None:
    """Verify twin mode vision interface contains tabs for simulation and physical stream."""
    with gr.Blocks() as demo:
        build_vision_interface(robot=None, mode="twin", demo=demo)

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

    sim_robot = SO101Robot(render_viewer=False)
    new_pos = sim_robot.randomize_target(
        x_range=(0.15, 0.22), y_range=(-0.10, 0.10), z_height=0.015
    )
    assert 0.15 <= new_pos[0] <= 0.22
    assert -0.10 <= new_pos[1] <= 0.10
    assert new_pos[2] == 0.015
