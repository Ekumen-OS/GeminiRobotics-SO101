"""Vision component for video streaming and camera selection."""

import gradio as gr

from robot.interface import RobotInterface
from ui.stream import (
    get_available_camera_choices,
    prepare_stream_switch,
    stream_video,
)


def build_vision_interface(robot: RobotInterface | None, mode: str, demo: gr.Blocks) -> None:
    """Build the vision stream interface within the current Gradio context.

    Args:
        robot: Initialized robot hardware or simulation controller instance.
        mode: Execution operational mode ('real', 'sim', or 'twin').
        demo: The parent Gradio Blocks app to attach load events to.
    """
    video_feed = gr.Image(
        label="LIVE Observation Stream",
        show_label=True,
        interactive=False,
    )

    initial_choices = get_available_camera_choices()
    initial_value = initial_choices[0][1] if initial_choices else 0
    show_camera_source_controls = mode != "sim"

    def stream_video_with_robot(
        camera_index: int,
        mode_value: str,
    ):
        yield from stream_video(
            camera_index=camera_index,
            mode=mode_value,
            robot=robot,
        )

    with gr.Row():
        camera_dropdown = gr.Dropdown(
            choices=initial_choices,
            value=initial_value,
            label="Select Video Source Device",
            interactive=show_camera_source_controls,
            visible=show_camera_source_controls,
            scale=3,
        )
        refresh_button = gr.Button(
            "🔄 Refresh Cameras",
            variant="secondary",
            scale=1,
            visible=show_camera_source_controls,
        )
        mode_state = gr.State(value=mode)

    def refresh_camera_list(current_val: int) -> gr.Dropdown:
        updated = get_available_camera_choices()
        valid = {c[1] for c in updated}
        new_val = current_val
        if (current_val == -2 and updated[0][1] != -2) or current_val not in valid:
            new_val = updated[0][1]
        return gr.Dropdown(choices=updated, value=new_val)

    refresh_event = refresh_button.click(
        fn=refresh_camera_list,
        inputs=[camera_dropdown],
        outputs=[camera_dropdown],
    )

    if mode == "sim":
        demo.load(
            fn=stream_video_with_robot,
            inputs=[camera_dropdown, mode_state],
            outputs=[video_feed],
            concurrency_limit=None,
        )
    else:
        refresh_event.then(
            fn=stream_video_with_robot,
            inputs=[camera_dropdown, mode_state],
            outputs=[video_feed],
            concurrency_limit=None,
        )

        demo.load(
            fn=stream_video_with_robot,
            inputs=[camera_dropdown, mode_state],
            outputs=[video_feed],
            concurrency_limit=None,
        )

        # Two-Stage Instant Switching Pattern over SSE generator
        for ctrl in [camera_dropdown]:
            switch_evt = ctrl.change(
                fn=prepare_stream_switch,
                inputs=None,
                outputs=None,
            )
            switch_evt.then(
                fn=stream_video_with_robot,
                inputs=[camera_dropdown, mode_state],
                outputs=[video_feed],
                concurrency_limit=None,
            )
