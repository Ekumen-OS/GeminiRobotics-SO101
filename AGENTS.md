# GeminiRobotics-SO101 Agent Guidelines

## 1. Context Sources

- **GeminiRobotics-SO101 Arm**: 6-DOF robotic arm setup integrated with Gemini Robotics models (`gemini-robotics-er-2-preview` / `gemini-robotics-er-2-streaming-preview`). Check [README.md](README.md) for an overview, [robot/](robot/) for core robot definitions (kinematics, hardware driver, digital twin, constants, math theory, ArUco workspace calibration in [`calibration.py`](robot/calibration.py), and Cartesian inverse kinematics in [`cartesian_ik.py`](robot/cartesian_ik.py)), [agent/](agent/) for Gemini Live API WebSocket session management ([`live_session.py`](agent/live_session.py)), primitive tool declarations and adapters ([`tools.py`](agent/tools.py)), and model-driven closed-loop VLA orchestration with post-action observation frame injection and background reasoning streaming ([`vla_controller.py`](agent/vla_controller.py)), [simulation/sim_robot.py](simulation/sim_robot.py) for MuJoCo simulation wrappers, [simulation/sim_calibration.py](simulation/sim_calibration.py) for synthetic sim homography calibration, [simulation/env.py](simulation/env.py) for RL environment specs, [simulation/test_movement.py](simulation/test_movement.py) for motion validation, [ui/](ui/) for the Unified FastAPI + Gradio 6.0 VLA Web UI (interactive 3D Three.js simulation viewer, OpenCV Motion JPEG streaming with live continuous ArUco calibration tracking, minimalist target bounding box overlay, cross-platform hardware camera discovery (Linux V4L2 & macOS AVFoundation), TehnoX dark theme, thread-safe `RobotStateManager`, and degree-tuned servo sliders), [tests/](tests/) for automated unit test suites ([`test_tools.py`](tests/test_tools.py), [`test_vla_controller.py`](tests/test_vla_controller.py), [`test_calibration.py`](tests/test_calibration.py), [`test_ui.py`](tests/test_ui.py)), [test_live.py](test_live.py) for Live API testing, and [main.py](main.py) for command entry points.
- **MuJoCo & Warp**: See MuJoCo documentation and [simulation/simulate.py](simulation/simulate.py) for standard simulation loops. For GPU/CPU batched simulation, reference [simulation/warp_env.py](simulation/warp_env.py), `google-deepmind/mujoco_warp`, and NVIDIA Warp docs.

## 2. Environment (`uv`)

- **Activate Virtualenv**: `source .venv/bin/activate`
- **Run Commands**: Prefer executing application servers, tests, scripts, and linters via `uv run` (e.g., `uv run python main.py` to start the Web UI server at `http://localhost:7860/`, `uv run ruff check .`, `uv run python main.py --task check_environment`, `uv run python main.py --task test_simulation`, `uv run python test_live.py`).
- **Sync Dependencies**: `uv sync`
- **Server Lifecycle and Port Management**: Before launching the Web UI server (default port `7860`), ALWAYS verify that the port is free (`lsof -ti :7860`). If it is not available, kill the occupying process (`lsof -ti :7860 | xargs kill -9`) to avoid binding collisions or stale code caching. Start the server, test/inspect changes, and **ALWAYS terminate/kill the server process immediately when done** so the port is free for the user to run and use the application.

## 3. Formatting, Linting (`ruff`) and testing

- Always inspect and auto-format code using `ruff` as configured in [pyproject.toml](pyproject.toml):
  - Check & fix lint: `uv run ruff check --fix .`
  - Format code: `uv run ruff format .`
- **Docstrings**: All modules, classes, and functions must use **Google-styled docstrings** (`Args:`, `Returns:`, `Raises:`). Max line length is **100 chars**.
- In order to test Web changes with Gradio. Check if you have available Chrome Tools to interact with the browser and perform actions to test changes.
- After all you changes, ensure that all `.md` files are updated.

## 4. Git Restrictions

> [!IMPORTANT]
> **NEVER execute `git add` or `git commit`.** The user assumes full responsibility for staging and committing changes. Read-only exploratory commands (`git status`, `git diff`, `git log`) are allowed.

## 5. External Repo Info (`gh`)

- You are authorized and encouraged to use the GitHub CLI (`gh`) to query external repositories, docs, issues, and PRs (e.g., `gh repo view google-deepmind/mujoco_warp`).
