"""Main Gradio Web UI constructor for GeminiRobotics-SO101 VLA interaction."""

import json
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
from ui.components.chat import build_chat_interface, wire_chat_events
from ui.components.controls import build_controls_interface
from ui.components.vision import build_vision_interface
from ui.state import get_robot_state_dict, set_active_robot

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
        gr.Markdown(f"## 🦾  GeminiRobotics SO-101 | **Mode: `{mode.upper()}`**")

        vla_controller = VLAController(mode=mode, robot=robot)

        with gr.Row():
            with gr.Column(scale=7):
                build_vision_interface(robot, mode, demo)

            with gr.Column(scale=5):
                prompt_input, chatbot, exec_btn, home_btn, clear_btn, rand_btn = (
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
            exec_btn,
            home_btn,
            clear_btn,
            sliders,
            rand_btn=rand_btn,
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
