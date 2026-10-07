"""Vision subsystem: camera capture, frame buffers, overlays, and sim runtime."""

import os
import sys

# MuJoCo picks its OpenGL backend when `mujoco` is first imported, so this must run before any
# `import mujoco`. On headless Linux (e.g. an SSH session) the default GLFW backend has no display
# and every offscreen render fails with "gladLoadGL error"; EGL renders offscreen instead.
if (
    sys.platform.startswith("linux")
    and "DISPLAY" not in os.environ
    and "WAYLAND_DISPLAY" not in os.environ
):
    os.environ.setdefault("MUJOCO_GL", "egl")

from vision.calibrator import (
    get_calibration_status_summary,
    get_calibrator,
    save_workspace_calibration_file,
)
from vision.camera import (
    get_available_camera_choices,
    get_camera_name,
    get_default_camera_index,
)
from vision.capture import (
    ensure_camera_capture,
    prepare_stream_switch,
    stop_camera_capture,
    stream_video,
    switch_camera_capture,
)
from vision.frames import (
    get_latest_jpeg,
    get_latest_raw_jpeg,
    has_published_frame,
)
from vision.overlays import (
    is_bounding_boxes_overlay_enabled,
    is_calibration_overlay_enabled,
    is_crosshair_overlay_enabled,
    is_grid_overlay_enabled,
    is_target_overlay_enabled,
    set_bounding_boxes_overlay,
    set_calibration_overlay,
    set_crosshair_overlay,
    set_grid_overlay,
    set_target_overlay_enabled,
)
from vision.sim_runtime import (
    ensure_sim_runtime_started,
    get_sim_video_frame,
    stop_sim_runtime,
)
from vision.targets import (
    add_model_target,
    clear_model_target_point,
    clear_model_targets,
    draw_model_target_overlay,
    get_model_target_point,
    get_model_targets,
    set_model_target_point,
    set_model_targets,
)

__all__ = [
    "add_model_target",
    "clear_model_target_point",
    "clear_model_targets",
    "draw_model_target_overlay",
    "ensure_camera_capture",
    "ensure_sim_runtime_started",
    "get_available_camera_choices",
    "get_calibration_status_summary",
    "get_calibrator",
    "get_camera_name",
    "get_default_camera_index",
    "get_latest_jpeg",
    "get_latest_raw_jpeg",
    "get_model_target_point",
    "get_model_targets",
    "get_sim_video_frame",
    "has_published_frame",
    "is_bounding_boxes_overlay_enabled",
    "is_calibration_overlay_enabled",
    "is_crosshair_overlay_enabled",
    "is_grid_overlay_enabled",
    "is_target_overlay_enabled",
    "prepare_stream_switch",
    "save_workspace_calibration_file",
    "set_bounding_boxes_overlay",
    "set_calibration_overlay",
    "set_crosshair_overlay",
    "set_grid_overlay",
    "set_model_target_point",
    "set_model_targets",
    "set_target_overlay_enabled",
    "stop_camera_capture",
    "stop_sim_runtime",
    "stream_video",
    "switch_camera_capture",
]
