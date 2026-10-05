"""Main Gradio Web UI constructor for GeminiRobotics-SO101 VLA interaction."""

import json
import logging
from pathlib import Path

import gradio as gr
import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from agent.vla_controller import VLAController
from robot.constants import JOINT_NAMES, PRESET_POSES
from robot.interface import RobotInterface
from robot.settings import get_settings
from ui.components.chat import build_chat_interface, wire_chat_events
from ui.components.controls import build_controls_interface
from ui.components.vision import build_vision_interface
from ui.state import get_robot_state_dict, set_active_robot

logger = logging.getLogger(__name__)

CUSTOM_CSS = """
#vla-status-banner {
    padding: 6px 12px !important;
    border-radius: 8px !important;
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    font-size: 13px !important;
    margin-bottom: 6px !important;
}
#vla-status-banner p {
    margin: 0 !important;
    line-height: 1.4 !important;
    font-size: 13px !important;
}
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
#vision-feed {
    height: 600px !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: center !important;
}
#vision-feed .image-container {
    height: 100% !important;
    max-height: 600px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}
#vision-feed img {
    max-height: 570px !important;
    width: 100% !important;
    height: 100% !important;
    object-fit: contain !important;
    border-radius: 8px !important;
}
.camera-control-row {
    display: flex !important;
    flex-direction: row !important;
    align-items: stretch !important;
    gap: 8px !important;
}
.camera-control-row > .form,
.camera-control-row > div:first-child {
    flex: 3 1 0% !important;
    margin: 0 !important;
}
.camera-control-row > button,
.camera-control-row .camera-refresh-btn {
    flex: 1 1 auto !important;
    min-width: 170px !important;
    max-width: 220px !important;
    height: auto !important;
    min-height: 100% !important;
    align-self: stretch !important;
    margin: 0 !important;
    padding: 0 16px !important;
    white-space: nowrap !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    box-sizing: border-box !important;
}
.chat-action-row {
    display: flex !important;
    align-items: stretch !important;
    gap: 8px !important;
}
.chat-action-row > div {
    flex: 1 1 0% !important;
    display: flex !important;
}
.chat-action-row button,
.chat-action-btn {
    height: 46px !important;
    min-height: 46px !important;
    max-height: 46px !important;
    padding: 0 12px !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    white-space: nowrap !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    box-sizing: border-box !important;
    width: 100% !important;
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
            logger.info(f"Loading cached Gradio theme from {cache_file}...")
            return gr.Theme.load(str(cache_file))
        except (AttributeError, ValueError, OSError, RuntimeError) as e:
            logger.warning(f"Failed loading cached theme ({e}). Downloading from Hub...")

    logger.info(f"Downloading theme '{theme_id}' from Gradio Hub and saving to cache...")
    theme = gr.Theme.from_hub(theme_id)
    try:
        if hasattr(theme, "dump"):
            theme.dump(str(cache_file))
            logger.info(f"Cached theme locally to {cache_file}.")
    except (OSError, RuntimeError, AttributeError) as e:
        logger.warning(f"Could not save local theme cache ({e}).")

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
        gr.Markdown(f"## 🦾  GeminiRobotics SO-101 | **Mode: `{mode.upper()}`**")

        vla_controller = VLAController(mode=mode, robot=robot)

        with gr.Row():
            with gr.Column(scale=7):
                save_calib_btn, calib_status, calib_timer = build_vision_interface(
                    robot, mode, demo
                )

            with gr.Column(scale=5):
                prompt_input, chatbot, status_banner, exec_btn, home_btn, clear_btn, rand_btn = (
                    build_chat_interface(
                        vla_controller, api_available=vla_controller.is_api_available
                    )
                )

        sliders, joint_state_panel = build_controls_interface(robot, mode, init_deg)

        # Wire up chat events now that sliders exist in the layout
        wire_chat_events(
            vla_controller,
            prompt_input,
            chatbot,
            status_banner,
            exec_btn,
            home_btn,
            clear_btn,
            sliders,
            rand_btn=rand_btn,
        )

        ready_placeholder = (
            'e.g., "Pick up red cube and place in sorting basket..." (Press Enter to Send)'
        )

        # Live synchronization between workspace calibration and Chat interface
        if save_calib_btn is not None and calib_status is not None:

            def on_save_calibration(history: list) -> tuple:
                from ui.stream import get_calibrator, save_workspace_calibration_file

                msg = save_workspace_calibration_file()
                calib = get_calibrator(mode=mode)
                if calib.is_calibrated:
                    ready_status = "🟢 **Status: Ready**"
                    prompt_upd = gr.update(
                        placeholder=ready_placeholder,
                        interactive=vla_controller.is_api_available,
                    )
                    exec_upd = gr.update(interactive=vla_controller.is_api_available)
                    new_hist = list(history) if history else []
                    if new_hist and "Workspace Not Calibrated" in new_hist[0].get("content", ""):
                        new_hist[0]["content"] = (
                            f"🤖 **System Active**: Ready in `{mode.upper()}` mode with "
                            "calibrated workspace. Type commands below to invoke spatial reasoning."
                        )
                    return msg, gr.update(value=ready_status), prompt_upd, exec_upd, new_hist
                return msg, gr.skip(), gr.skip(), gr.skip(), gr.skip()

            save_calib_btn.click(
                fn=on_save_calibration,
                inputs=[chatbot],
                outputs=[calib_status, status_banner, prompt_input, exec_btn, chatbot],
            )

        if calib_timer is not None:

            def sync_calibration_timer(history: list, current_status: str) -> tuple:
                from ui.stream import get_calibrator

                calib = get_calibrator(mode=mode)
                if calib.is_calibrated and "Workspace Not Calibrated" in (current_status or ""):
                    ready_status = "🟢 **Status: Ready**"
                    prompt_upd = gr.update(
                        placeholder=ready_placeholder,
                        interactive=vla_controller.is_api_available,
                    )
                    exec_upd = gr.update(interactive=vla_controller.is_api_available)
                    new_hist = list(history) if history else []
                    if new_hist and "Workspace Not Calibrated" in new_hist[0].get("content", ""):
                        new_hist[0]["content"] = (
                            f"🤖 **System Active**: Ready in `{mode.upper()}` mode with "
                            "calibrated workspace. Type commands below to invoke spatial reasoning."
                        )
                    return (
                        gr.update(value=ready_status),
                        prompt_upd,
                        exec_upd,
                        new_hist,
                    )
                return gr.skip(), gr.skip(), gr.skip(), gr.skip()

            calib_timer.tick(
                fn=sync_calibration_timer,
                inputs=[chatbot, status_banner],
                outputs=[status_banner, prompt_input, exec_btn, chatbot],
            )

    return demo


def create_app(robot: RobotInterface | None = None, mode: str = "real") -> FastAPI:
    """Create FastAPI application with mounted Gradio UI, 3D assets, and simulation endpoints.

    Args:
        robot: Initialized robot controller instance.
        mode: Target execution operational mode ('real', 'sim', 'twin').

    Returns:
        Configured FastAPI application instance.
    """
    set_active_robot(robot)
    if mode == "sim":
        from simulation.sim_calibration import init_sim_workspace_calibration
        from ui.stream import _SIM_RUNTIME, get_calibrator

        calib = get_calibrator(mode="sim")
        init_sim_workspace_calibration(calibrator=calib)
        if robot is not None:
            _SIM_RUNTIME.ensure_started(robot)

    app = FastAPI(title="GeminiRobotics-SO101", docs_url=None, redoc_url=None)

    project_root = Path(__file__).resolve().parent.parent
    sim_assets_dir = project_root / "simulation" / "assets"
    ui_assets_dir = Path(__file__).resolve().parent / "assets"
    aruco_markers_dir = project_root / "assets" / "aruco_markers"

    if sim_assets_dir.exists():
        app.mount("/sim_assets", StaticFiles(directory=str(sim_assets_dir)), name="sim_assets")

    if ui_assets_dir.exists():
        app.mount("/ui_assets", StaticFiles(directory=str(ui_assets_dir)), name="ui_assets")

    if aruco_markers_dir.exists():
        app.mount(
            "/aruco_markers", StaticFiles(directory=str(aruco_markers_dir)), name="aruco_markers"
        )

    @app.get("/viewer")
    async def get_viewer():
        viewer_path = ui_assets_dir / "sim_viewer.html"
        if viewer_path.exists():
            return FileResponse(
                str(viewer_path),
                media_type="text/html",
                headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
            )
        return JSONResponse({"error": "sim_viewer.html not found"}, status_code=404)

    @app.get("/api/sim_state")
    async def get_sim_state():
        return JSONResponse(get_robot_state_dict())

    @app.get("/api/workspace_config")
    async def get_workspace_config():
        cfg_path = project_root / "config" / "workspace_calibration.json"
        if cfg_path.exists():
            try:
                return JSONResponse(json.loads(cfg_path.read_text()))
            except Exception:
                pass
        return JSONResponse({})

    @app.get("/api/sim_stream")
    async def get_sim_stream():
        import asyncio

        from ui.stream import get_latest_jpeg

        async def frame_generator():
            while True:
                jpg = get_latest_jpeg()
                if jpg is not None:
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
                await asyncio.sleep(0.033)

        return StreamingResponse(
            frame_generator(),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    theme = get_cached_or_hub_theme("YTheme/TehnoX")
    demo = build_vla_interface(robot=robot, mode=mode)
    app = gr.mount_gradio_app(app, demo, path="/", theme=theme, css=CUSTOM_CSS)
    return app


def launch_web_ui(
    robot: RobotInterface | None = None,
    mode: str = "real",
    server_name: str | None = None,
    server_port: int | None = None,
) -> None:
    """Launch the Gradio VLA Web UI server.

    Args:
        robot: Initialized robot controller instance.
        mode: Target execution operational mode ('real', 'sim', 'twin').
        server_name: Host binding address (default: from settings).
        server_port: Port number for web application (default: from settings).
    """
    settings = get_settings()
    if server_name is None:
        server_name = settings.server_host
    if server_port is None:
        server_port = settings.server_port
    app = create_app(robot=robot, mode=mode)
    print("=" * 60)
    print(f"Starting GeminiRobotics SO-101 Unified Gradio Web UI [{mode.upper()} MODE]...")
    print(f"Open browser at http://localhost:{server_port}/")
    print("=" * 60)
    config = uvicorn.Config(
        app,
        host=server_name,
        port=server_port,
        log_level="warning",
        timeout_graceful_shutdown=1,
    )
    server = uvicorn.Server(config)
    server.run()
