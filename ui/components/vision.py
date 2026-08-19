"""Vision component for video streaming and camera selection."""

import gradio as gr

from robot.interface import RobotInterface
from ui.stream import (
    get_available_camera_choices,
    get_calibration_status_summary,
    get_default_camera_index,
    prepare_stream_switch,
    save_workspace_calibration_file,
    set_bounding_boxes_overlay,
    set_calibration_overlay,
    set_crosshair_overlay,
    set_grid_overlay,
    stream_video,
)


def build_vision_interface(robot: RobotInterface | None, mode: str, demo: gr.Blocks) -> None:
    """Build the vision stream interface within the current Gradio context.

    Args:
        robot: Initialized robot hardware or simulation controller instance.
        mode: Execution operational mode ('real', 'sim', or 'twin').
        demo: The parent Gradio Blocks app to attach load events to.
    """
    viewer_iframe = (
        '<div style="width: 100%; height: 600px; border-radius: 12px; overflow: hidden; '
        "border: 1px solid #334155; position: relative; background: #0b0f19; "
        'box-shadow: 0 8px 30px rgba(0,0,0,0.5);">'
        '<iframe src="/ui_assets/sim_viewer.html" style="width: 100%; height: 100%; border: none;" '
        'allow="accelerometer; autoplay; encrypted-media; gyroscope; '
        'picture-in-picture"></iframe>'
        "</div>"
    )

    if mode == "sim":
        gr.HTML(value=viewer_iframe)

        with gr.Accordion("🤖 Canonical VLA Vision (MuJoCo Offscreen Observation)", open=False):
            video_feed = gr.Image(
                label="AI Model Observation Stream",
                show_label=False,
                interactive=False,
                elem_id="vision-feed",
            )
            mode_state = gr.State(value=mode)
            camera_dropdown = gr.State(value=-1)

            def stream_sim_obs(cam_idx: int, m: str):
                yield from stream_video(camera_index=cam_idx, mode=m, robot=robot)

            demo.load(
                fn=stream_sim_obs,
                inputs=[camera_dropdown, mode_state],
                outputs=[video_feed],
                concurrency_limit=None,
            )
        return

    elif mode == "twin":
        with gr.Tabs():
            with gr.Tab("🎮 3D Digital Twin Simulation"):
                gr.HTML(value=viewer_iframe)

            with gr.Tab("📹 Physical Camera Stream"):
                video_feed = gr.Image(
                    label="LIVE Observation Stream",
                    show_label=True,
                    interactive=False,
                    elem_id="vision-feed",
                )
                with gr.Row():
                    calib_chk_twin = gr.Checkbox(
                        label="Show Calibration Overlay",
                        value=False,
                        scale=1,
                    )
                    bbox_chk_twin = gr.Checkbox(
                        label="Show Bounding Boxes",
                        value=False,
                        scale=1,
                    )
                    grid_chk_twin = gr.Checkbox(
                        label="Show Grid Overlay",
                        value=False,
                        scale=1,
                    )
                    crosshair_chk_twin = gr.Checkbox(
                        label="Show Crosshair",
                        value=False,
                        scale=1,
                    )

                calib_chk_twin.change(
                    fn=set_calibration_overlay,
                    inputs=[calib_chk_twin],
                    outputs=None,
                )
                bbox_chk_twin.change(
                    fn=set_bounding_boxes_overlay,
                    inputs=[bbox_chk_twin],
                    outputs=None,
                )
                grid_chk_twin.change(
                    fn=set_grid_overlay,
                    inputs=[grid_chk_twin],
                    outputs=None,
                )
                crosshair_chk_twin.change(
                    fn=set_crosshair_overlay,
                    inputs=[crosshair_chk_twin],
                    outputs=None,
                )

                initial_choices = get_available_camera_choices()
                initial_value = get_default_camera_index(initial_choices)
                with gr.Row():
                    camera_dropdown = gr.Dropdown(
                        choices=initial_choices,
                        value=initial_value,
                        label="Select Video Source Device",
                        interactive=True,
                        scale=3,
                    )
                    refresh_button = gr.Button("🔄 Refresh Cameras", variant="secondary", scale=1)
                mode_state = gr.State(value=mode)

                with gr.Accordion("🎯 ArUco Workspace Calibration", open=False):
                    with gr.Row():
                        calib_status_twin = gr.Textbox(
                            label="Calibration Status",
                            value=get_calibration_status_summary(),
                            interactive=False,
                            scale=3,
                        )
                        save_calib_btn_twin = gr.Button(
                            "💾 Save Calibration",
                            variant="secondary",
                            scale=1,
                        )

                    save_calib_btn_twin.click(
                        fn=save_workspace_calibration_file,
                        inputs=None,
                        outputs=[calib_status_twin],
                    )

                    calib_timer_twin = gr.Timer(value=1.5, active=True)
                    calib_timer_twin.tick(
                        fn=get_calibration_status_summary,
                        inputs=None,
                        outputs=[calib_status_twin],
                    )

                def refresh_camera_list_twin(current_val: int) -> gr.Dropdown:
                    updated = get_available_camera_choices()
                    valid = {c[1] for c in updated}
                    new_val = current_val
                    if (current_val == -2 and updated[0][1] != -2) or current_val not in valid:
                        new_val = updated[0][1]
                    return gr.Dropdown(choices=updated, value=new_val)

                def stream_twin_cam(cam_idx: int, m: str):
                    yield from stream_video(camera_index=cam_idx, mode=m, robot=robot)

                refresh_button.click(
                    fn=refresh_camera_list_twin,
                    inputs=[camera_dropdown],
                    outputs=[camera_dropdown],
                ).then(
                    fn=stream_twin_cam,
                    inputs=[camera_dropdown, mode_state],
                    outputs=[video_feed],
                    concurrency_limit=None,
                )

                demo.load(
                    fn=stream_twin_cam,
                    inputs=[camera_dropdown, mode_state],
                    outputs=[video_feed],
                    concurrency_limit=None,
                )

                switch_evt = camera_dropdown.change(
                    fn=prepare_stream_switch,
                    inputs=None,
                    outputs=None,
                )
                switch_evt.then(
                    fn=stream_twin_cam,
                    inputs=[camera_dropdown, mode_state],
                    outputs=[video_feed],
                    concurrency_limit=None,
                )
        return

    # Real mode
    video_feed = gr.Image(
        label="LIVE Observation Stream",
        show_label=True,
        interactive=False,
        elem_id="vision-feed",
    )
    with gr.Row():
        calib_chk = gr.Checkbox(
            label="Show Calibration Overlay",
            value=False,
            scale=1,
        )
        bbox_chk = gr.Checkbox(
            label="Show Bounding Boxes",
            value=False,
            scale=1,
        )
        grid_chk = gr.Checkbox(
            label="Show Grid Overlay",
            value=False,
            scale=1,
        )
        crosshair_chk = gr.Checkbox(
            label="Show Crosshair",
            value=False,
            scale=1,
        )

    calib_chk.change(
        fn=set_calibration_overlay,
        inputs=[calib_chk],
        outputs=None,
    )
    bbox_chk.change(
        fn=set_bounding_boxes_overlay,
        inputs=[bbox_chk],
        outputs=None,
    )
    grid_chk.change(
        fn=set_grid_overlay,
        inputs=[grid_chk],
        outputs=None,
    )
    crosshair_chk.change(
        fn=set_crosshair_overlay,
        inputs=[crosshair_chk],
        outputs=None,
    )

    initial_choices = get_available_camera_choices()
    initial_value = get_default_camera_index(initial_choices)

    with gr.Row():
        camera_dropdown = gr.Dropdown(
            choices=initial_choices,
            value=initial_value,
            label="Select Video Source Device",
            interactive=True,
            scale=3,
        )
        refresh_button = gr.Button(
            "🔄 Refresh Cameras",
            variant="secondary",
            scale=1,
        )
        mode_state = gr.State(value=mode)

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

        save_calib_btn.click(
            fn=save_workspace_calibration_file,
            inputs=None,
            outputs=[calib_status],
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

    def stream_real_cam(cam_idx: int, m: str):
        yield from stream_video(camera_index=cam_idx, mode=m, robot=robot)

    refresh_event = refresh_button.click(
        fn=refresh_camera_list,
        inputs=[camera_dropdown],
        outputs=[camera_dropdown],
    )
    refresh_event.then(
        fn=stream_real_cam,
        inputs=[camera_dropdown, mode_state],
        outputs=[video_feed],
        concurrency_limit=None,
    )

    demo.load(
        fn=stream_real_cam,
        inputs=[camera_dropdown, mode_state],
        outputs=[video_feed],
        concurrency_limit=None,
    )

    switch_evt = camera_dropdown.change(
        fn=prepare_stream_switch,
        inputs=None,
        outputs=None,
    )
    switch_evt.then(
        fn=stream_real_cam,
        inputs=[camera_dropdown, mode_state],
        outputs=[video_feed],
        concurrency_limit=None,
    )
