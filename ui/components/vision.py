"""Vision component for video streaming and camera selection."""

import gradio as gr

from robot.interface import RobotInterface
from vision.calibrator import get_calibration_status_summary
from vision.camera import get_available_camera_choices, get_default_camera_index
from vision.capture import ensure_camera_capture, switch_camera_capture
from vision.overlays import set_calibration_overlay, set_target_overlay_enabled

CAMERA_FEED_HTML = (
    '<img src="/api/stream" alt="LIVE Observation Stream" '
    'style="width: 100%; height: 100%; object-fit: contain; border-radius: 8px;" />'
)


def _build_overlay_toggles() -> None:
    """Render the shared model-target and calibration overlay checkboxes and wire their events."""
    with gr.Row():
        target_chk = gr.Checkbox(
            label="🎯 Show Model Targets",
            value=True,
            scale=1,
        )
        calib_chk = gr.Checkbox(
            label="📐 Show Calibration Overlay",
            value=False,
            scale=1,
        )

    target_chk.change(
        fn=set_target_overlay_enabled,
        inputs=[target_chk],
        outputs=None,
    )
    calib_chk.change(
        fn=set_calibration_overlay,
        inputs=[calib_chk],
        outputs=None,
    )


def build_vision_interface(
    robot: RobotInterface | None, mode: str, demo: gr.Blocks
) -> tuple[gr.Button | None, gr.Textbox | None, gr.Timer | None]:
    """Build the vision stream interface within the current Gradio context.

    Args:
        robot: Initialized robot hardware or simulation controller instance.
        mode: Execution operational mode ('real' or 'sim').
        demo: The parent Gradio Blocks app to attach load events to.

    Returns:
        Tuple of (save_calib_btn, calib_status_box, calib_timer).
    """
    viewer_iframe = (
        '<div style="width: 100%; height: 600px; border-radius: 12px; overflow: hidden; '
        "border: 1px solid #334155; position: relative; background: #0b0f19; "
        'box-shadow: 0 8px 30px rgba(0,0,0,0.5);">'
        '<iframe src="/viewer" style="width: 100%; height: 100%; border: none;" '
        'allow="accelerometer; autoplay; encrypted-media; gyroscope; '
        'picture-in-picture"></iframe>'
        "</div>"
    )

    if mode == "sim":
        gr.HTML(value=viewer_iframe)

        with gr.Accordion("🤖 Canonical VLA Vision (MuJoCo Offscreen Observation)", open=False):
            mjpeg_viewer = (
                '<div style="width: 100%; height: 480px; border-radius: 8px; overflow: hidden; '
                "border: 1px solid #334155; position: relative; background: #0b0f19; "
                'display: flex; align-items: center; justify-content: center;">'
                '<img src="/api/stream" '
                'style="width: 100%; height: 100%; object-fit: contain; border-radius: 8px;" '
                'alt="MuJoCo Offscreen Camera Feed (workspace_cam)" />'
                "</div>"
            )
            gr.HTML(value=mjpeg_viewer)
            _build_overlay_toggles()
        return None, None, None

    # Real mode
    gr.HTML(value=CAMERA_FEED_HTML, elem_id="vision-feed")
    _build_overlay_toggles()

    initial_choices = get_available_camera_choices()
    initial_value = get_default_camera_index(initial_choices)

    with gr.Row(equal_height=True, elem_classes=["camera-control-row"]):
        camera_dropdown = gr.Dropdown(
            choices=initial_choices,
            value=initial_value,
            label="Select Video Source Device",
            interactive=True,
            scale=3,
            elem_classes=["camera-dropdown"],
        )
        refresh_button = gr.Button(
            "🔄 Refresh Cameras",
            variant="secondary",
            scale=1,
            elem_classes=["camera-refresh-btn"],
        )

    with gr.Accordion("🎯 ArUco Workspace Calibration", open=False):
        with gr.Row():
            calib_status = gr.Textbox(
                label="Calibration Status",
                value=get_calibration_status_summary(),
                interactive=False,
                scale=3,
            )
            save_calib_btn = gr.Button(
                "💾 Save Calibration",
                variant="secondary",
                scale=1,
            )

        calib_timer = gr.Timer(value=1.5, active=True)
        calib_timer.tick(
            fn=get_calibration_status_summary,
            inputs=None,
            outputs=[calib_status],
        )

    def refresh_camera_list(current_val: int) -> gr.Dropdown:
        updated = get_available_camera_choices()
        valid = {c[1] for c in updated}
        new_val = current_val
        if (current_val == -2 and updated[0][1] != -2) or current_val not in valid:
            new_val = updated[0][1]
        return gr.Dropdown(choices=updated, value=new_val)

    def restart_real_cam(cam_idx: int) -> int:
        return switch_camera_capture(cam_idx, force=True)

    # Capture is shared by every browser tab: page load joins the running capture (and shows
    # which camera it is using) instead of restarting it.
    demo.load(
        fn=ensure_camera_capture,
        inputs=[camera_dropdown],
        outputs=[camera_dropdown],
    )

    refresh_button.click(
        fn=refresh_camera_list,
        inputs=[camera_dropdown],
        outputs=[camera_dropdown],
    ).then(
        fn=restart_real_cam,
        inputs=[camera_dropdown],
        outputs=None,
    )

    camera_dropdown.change(
        fn=switch_camera_capture,
        inputs=[camera_dropdown],
        outputs=None,
    )
    return save_calib_btn, calib_status, calib_timer
