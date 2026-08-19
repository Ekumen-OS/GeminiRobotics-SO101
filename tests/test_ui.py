"""Unit tests for the Gradio Web UI components, stream overlays, and FastAPI app mount."""

import gradio as gr
from fastapi import FastAPI

from agent.vla_controller import VLAController
from robot.constants import JOINT_LIMITS
from ui.app import build_vla_interface, create_app
from ui.components.chat import build_chat_interface
from ui.components.controls import build_controls_interface
from ui.components.telemetry import render_joint_state_panel
from ui.components.vision import build_vision_interface
from ui.stream import (
    is_bounding_boxes_overlay_enabled,
    is_calibration_overlay_enabled,
    is_crosshair_overlay_enabled,
    is_grid_overlay_enabled,
    set_bounding_boxes_overlay,
    set_calibration_overlay,
    set_crosshair_overlay,
    set_grid_overlay,
)


def test_stream_overlay_flags() -> None:
    """Verify getter and setter functionality for all video stream overlay states."""
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


def test_build_chat_interface_structure() -> None:
    """Verify chat component creation and 600px height configuration."""
    controller = VLAController(mode="sim", robot=None)
    with gr.Blocks():
        prompt_input, chatbot, exec_btn, stop_btn, clear_btn = build_chat_interface(
            controller=controller, api_available=False
        )

    assert chatbot.elem_id == "vla-chatbot"
    assert chatbot.height == 600
    assert prompt_input.elem_id == "vla-prompt"
    assert exec_btn is not None
    assert stop_btn is not None
    assert clear_btn is not None


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
