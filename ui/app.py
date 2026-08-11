"""Main Gradio Web UI constructor for GeminiRobotics-SO101 VLA interaction."""

from pathlib import Path

import gradio as gr
import numpy as np

from robot.constants import JOINT_NAMES, PRESET_POSES
from robot.interface import RobotInterface
from ui.components.chat import build_chat_interface
from ui.components.controls import build_controls_interface
from ui.components.vision import build_vision_interface

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


def build_vla_interface(robot: RobotInterface | None = None, mode: str = "real") -> gr.Blocks:
    """Construct the Unified Gradio VLA interface for real robot or MuJoCo simulation.

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

        with gr.Row():
            with gr.Column(scale=7):
                build_vision_interface(robot, mode, demo)

            with gr.Column(scale=5):
                build_chat_interface(mode)

        with gr.Group():
            build_controls_interface(robot, mode, init_deg)

    return demo


def launch_web_ui(
    robot: RobotInterface | None = None,
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
