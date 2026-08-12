"""Gradio Web UI package for GeminiRobotics-SO101 interactive operational mode."""

from ui.app import build_vla_interface, launch_web_ui
from ui.stream import (
    get_available_camera_choices,
    get_default_camera_index,
    prepare_stream_switch,
    stream_video,
)

__all__ = [
    "build_vla_interface",
    "get_available_camera_choices",
    "get_default_camera_index",
    "launch_web_ui",
    "prepare_stream_switch",
    "stream_video",
]
