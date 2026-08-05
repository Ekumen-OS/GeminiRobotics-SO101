"""Main Gradio Web UI constructor for GeminiRobotics-SO101 VLA interaction."""

from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np

from robot.constants import JOINT_NAMES, PRESET_POSES
from ui.stream import get_available_camera_choices, prepare_stream_switch, stream_video

CUSTOM_CSS = """
#vla-chatbot {
    font-size: 13px !important;
}
#vla-chatbot .message {
    font-size: 13px !important;
    line-height: 1.4 !important;
    padding: 8px 12px !important;
}
#vla-chatbot p {
    font-size: 13px !important;
    margin: 0 !important;
}
#vla-prompt textarea {
    font-size: 13px !important;
}
"""


def get_cached_or_hub_theme(theme_id: str = "YTheme/TehnoX") -> gr.Theme:
    """Retrieve theme from local JSON cache if available, else download and cache locally.

    Args:
        theme_id: Hugging Face Hub theme identifier string.

    Returns:
        Loaded Gradio Theme instance.
    """
    cache_dir = Path(__file__).parent / "theme_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe_name = theme_id.replace("/", "_") + ".json"
    cache_file = cache_dir / safe_name

    if cache_file.exists():
        try:
            print(f"[INFO] Loading cached Gradio theme from {cache_file}...")
            return gr.Theme.load(str(cache_file))
        except (AttributeError, ValueError, OSError, RuntimeError) as e:
            print(f"[WARN] Failed loading cached theme ({e}). Downloading from Hub...")

    print(f"[INFO] Downloading theme '{theme_id}' from Gradio Hub and saving to cache...")
    theme = gr.Theme.from_hub(theme_id)
    try:
        if hasattr(theme, "dump"):
            theme.dump(str(cache_file))
            print(f"[INFO] Cached theme locally to {cache_file}.")
    except (OSError, RuntimeError, AttributeError) as e:
        print(f"[WARN] Could not save local theme cache ({e}).")

    return theme


def build_vla_interface(robot: Any | None = None, mode: str = "real") -> gr.Blocks:
    """Construct the Unified Gradio VLA interface for real robot hardware or sim placeholders.

    Args:
        robot: Initialized robot hardware or simulation controller instance.
        mode: Execution operational mode ('real', 'sim', or 'twin').

    Returns:
        Gradio Blocks application demo instance.
    """
    # Acquire initial joint states in radians and convert to degrees for user UI
    init_qpos = PRESET_POSES["HOME"].copy()
    if robot is not None and hasattr(robot, "get_joint_positions"):
        try:
            curr = robot.get_joint_positions()
            if len(curr) == len(JOINT_NAMES):
                init_qpos = curr
        except Exception:
            pass

    init_deg = [round(float(np.degrees(v)), 1) for v in init_qpos]

    with gr.Blocks(title=f"GeminiRobotics SO-101 VLA [{mode.upper()}]") as demo:
        gr.Markdown(
            "### 🦾 GeminiRobotics SO-101 — Unified VLA Interface | "
            f"**Operational Mode: `{mode.upper()}`**"
        )

        # -------------------------------------------------------------------------
        # TOP ROW: VISION STREAM ON LEFT | VLA CHAT ON RIGHT
        # -------------------------------------------------------------------------
        with gr.Row():
            # LEFT COLUMN: LIVE VISION FEED (Scale=7)
            with gr.Column(scale=7):
                video_feed = gr.Image(
                    label="LIVE Observation Stream",
                    show_label=True,
                    interactive=False,
                )

                initial_choices = get_available_camera_choices()
                initial_value = initial_choices[0][1] if initial_choices else 0

                with gr.Row():
                    camera_dropdown = gr.Dropdown(
                        choices=initial_choices,
                        value=initial_value,
                        label="Select Video Source Device",
                        interactive=True,
                        scale=3,
                    )
                    resolution_dropdown = gr.Dropdown(
                        choices=[
                            "1600x1200 (4:3 HD - Uncropped)",
                            "1920x1080 (16:9 Full HD)",
                            "3264x2448 (4:3 Max Sensor)",
                            "1280x720 (16:9 HD)",
                            "640x480 (4:3 SD)",
                        ],
                        value="1600x1200 (4:3 HD - Uncropped)",
                        label="Capture Aspect Ratio",
                        interactive=True,
                        scale=2,
                    )
                    fps_slider = gr.Slider(
                        minimum=5,
                        maximum=60,
                        step=5,
                        value=30,
                        label="Target Frame Rate (FPS)",
                        interactive=True,
                        scale=2,
                    )
                    refresh_button = gr.Button("🔄 Refresh Cameras", variant="secondary", scale=1)
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
                refresh_event.then(
                    fn=stream_video,
                    inputs=[camera_dropdown, fps_slider, resolution_dropdown, mode_state],
                    outputs=[video_feed],
                    concurrency_limit=None,
                )

                demo.load(
                    fn=stream_video,
                    inputs=[camera_dropdown, fps_slider, resolution_dropdown, mode_state],
                    outputs=[video_feed],
                    concurrency_limit=None,
                )

                # Two-Stage Instant Switching Pattern over SSE generator
                for ctrl in [camera_dropdown, resolution_dropdown, fps_slider]:
                    switch_evt = ctrl.change(fn=prepare_stream_switch, inputs=None, outputs=None)
                    switch_evt.then(
                        fn=stream_video,
                        inputs=[camera_dropdown, fps_slider, resolution_dropdown, mode_state],
                        outputs=[video_feed],
                        concurrency_limit=None,
                    )

            # RIGHT COLUMN: MULTIMODAL VLA CHAT INTERFACE (Scale=5)
            with gr.Column(scale=5):
                chatbot = gr.Chatbot(
                    label="Gemini Robotics Assistant (gemini-robotics-er)",
                    height=440,
                    elem_id="vla-chatbot",
                    value=[
                        {
                            "role": "assistant",
                            "content": (
                                f"🤖 **System Active**: Ready in `{mode.upper()}` mode. "
                                "Type commands below to invoke spatial reasoning."
                            ),
                        }
                    ],
                )
                prompt_input = gr.Textbox(
                    label="Instruction Prompt",
                    placeholder='e.g., "Pick up red cube and place in sorting basket..."',
                    lines=2,
                    interactive=True,
                    elem_id="vla-prompt",
                )
                with gr.Row():
                    exec_btn = gr.Button("🚀 Execute Instruction", variant="primary", scale=2)
                    clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary", scale=1)

                def vla_placeholder_response(msg: str, history: list) -> tuple[str, list]:
                    if not msg.strip():
                        return "", history
                    history = history or []
                    response = (
                        f"⚡ **[gemini-robotics-er]**: Received action command *"
                        f'"{msg}"* for mode `{mode.upper()}`. Live trajectory generation over '
                        "visual observations will execute in upcoming inference pipeline."
                    )
                    history.append({"role": "user", "content": msg})
                    history.append({"role": "assistant", "content": response})
                    return "", history

                exec_btn.click(
                    fn=vla_placeholder_response,
                    inputs=[prompt_input, chatbot],
                    outputs=[prompt_input, chatbot],
                )
                prompt_input.submit(
                    fn=vla_placeholder_response,
                    inputs=[prompt_input, chatbot],
                    outputs=[prompt_input, chatbot],
                )
                clear_btn.click(lambda: [], inputs=None, outputs=[chatbot])

        # -------------------------------------------------------------------------
        # BOTTOM SECTION: FULL-WIDTH ROBOT MOVEMENT CONTROL SLIDER WIDGET (DEGREES)
        # -------------------------------------------------------------------------
        with gr.Group():
            gr.Markdown("## 🎮 Robot Movement Control")
            with gr.Row():
                s_pan = gr.Slider(
                    minimum=-100.0,
                    maximum=100.0,
                    step=0.5,
                    value=init_deg[0],
                    label="Shoulder Pan (°)",
                )
                s_lift = gr.Slider(
                    minimum=-115.0,
                    maximum=115.0,
                    step=0.5,
                    value=init_deg[1],
                    label="Shoulder Lift (°)",
                )
                e_flex = gr.Slider(
                    minimum=-110.0,
                    maximum=110.0,
                    step=0.5,
                    value=init_deg[2],
                    label="Elbow Flex (°)",
                )
            with gr.Row():
                w_flex = gr.Slider(
                    minimum=-110.0,
                    maximum=110.0,
                    step=0.5,
                    value=init_deg[3],
                    label="Wrist Flex (°)",
                )
                w_roll = gr.Slider(
                    minimum=-180.0,
                    maximum=180.0,
                    step=0.5,
                    value=init_deg[4],
                    label="Wrist Roll (°)",
                )
                grip = gr.Slider(
                    minimum=-75.0,
                    maximum=75.0,
                    step=0.5,
                    value=init_deg[5],
                    label="Gripper (°)",
                )

            with gr.Row():
                status_display = gr.Textbox(
                    label="Actuator Bus Status",
                    value=f"Initial joints holding pose: {init_deg}°",
                    interactive=False,
                    scale=4,
                )
                reset_btn = gr.Button("🏠 Reset to HOME Pose", variant="secondary", scale=1)
                stop_btn = gr.Button("🛑 EMERGENCY STOP / PARK", variant="stop", scale=1)

            sliders = [s_pan, s_lift, e_flex, w_flex, w_roll, grip]

            def apply_joint_angles(
                j1: float, j2: float, j3: float, j4: float, j5: float, j6: float
            ) -> str:
                target_deg = [j1, j2, j3, j4, j5, j6]
                target_qpos = np.radians(target_deg, dtype=np.float32)
                if robot is not None:
                    if hasattr(robot, "set_joint_positions"):
                        robot.set_joint_positions(target_qpos)
                    if hasattr(robot, "step"):
                        robot.step(1)
                return f"⚡ Command routed to servos: {[round(val, 1) for val in target_deg]}°"

            for s in sliders:
                s.change(
                    fn=apply_joint_angles,
                    inputs=sliders,
                    outputs=[status_display],
                )

            def reset_to_home():
                home_q = PRESET_POSES["HOME"]
                if robot is not None:
                    if hasattr(robot, "reset"):
                        robot.reset("HOME")
                    elif hasattr(robot, "set_joint_positions"):
                        robot.set_joint_positions(home_q)
                    if hasattr(robot, "step"):
                        robot.step(1)
                home_deg = [round(float(np.degrees(x)), 1) for x in home_q]
                msg = f"🏠 Reset arm servos to HOME pose: {home_deg}°"
                return [*home_deg, msg]

            reset_btn.click(
                fn=reset_to_home,
                inputs=None,
                outputs=[*sliders, status_display],
            )

            def emergency_stop():
                home_q = PRESET_POSES["HOME"]
                if robot is not None and hasattr(robot, "set_joint_positions"):
                    robot.set_joint_positions(home_q)
                    if hasattr(robot, "step"):
                        robot.step(1)
                home_deg = [round(float(np.degrees(x)), 1) for x in home_q]
                msg = "🛑 EMERGENCY STOP: Robot motion paused & parked to safe stance."
                return [*home_deg, msg]

            stop_btn.click(
                fn=emergency_stop,
                inputs=None,
                outputs=[*sliders, status_display],
            )

    return demo


def launch_web_ui(
    robot: Any | None = None,
    mode: str = "real",
    server_name: str = "0.0.0.0",
    server_port: int = 7860,
) -> None:
    """Launch the Gradio VLA Web UI server.

    Args:
        robot: Initialized robot controller instance.
        mode: Target execution operational mode ('real', 'sim', 'twin').
        server_name: Host binding address (default: 0.0.0.0).
        server_port: Port number for web application (default: 7860).
    """
    theme = get_cached_or_hub_theme("YTheme/TehnoX")
    demo = build_vla_interface(robot=robot, mode=mode)
    print("=" * 60)
    print(f"Starting GeminiRobotics SO-101 Unified Gradio Web UI [{mode.upper()} MODE]...")
    print(f"Open browser at http://localhost:{server_port}/")
    print("=" * 60)
    demo.launch(server_name=server_name, server_port=server_port, theme=theme, css=CUSTOM_CSS)
