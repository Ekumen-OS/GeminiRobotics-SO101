# GeminiRobotics-SO101 Agent Guidelines

## 1. Context Sources

- **System Architecture & Testing Specification**: All agents working on this codebase MUST consult [ARCHITECTURE.md](ARCHITECTURE.md) as the authoritative single source of truth for system dataflows, subsystem interface contracts (calibration, perception metrics, tool response payloads), feature catalog (F1–F12), and the 4-tier E2E testing methodology.
- **GeminiRobotics-SO101 Arm**: 6-DOF robotic arm setup integrated with Gemini Robotics models (`gemini-robotics-er-2-preview` / `gemini-robotics-er-2-streaming-preview`). Check [README.md](README.md) for an overview, [robot/](robot/) for core robot definitions (kinematics, hardware driver, digital twin, constants, math theory, ArUco workspace calibration in [`calibration.py`](robot/calibration.py), programmatic perception metrics in [`perception.py`](robot/perception.py), and Cartesian inverse kinematics in [`cartesian_ik.py`](robot/cartesian_ik.py)), [agent/](agent/) for Gemini Live API WebSocket session management ([`live_session.py`](agent/live_session.py)), primitive tool declarations and adapters ([`tools.py`](agent/tools.py)), model-driven closed-loop VLA orchestration with strict "Move -> Look -> Correct" visual verification loop, post-action observation frame injection with vision settling delay, background reasoning streaming, and ISO-8601 millisecond-precision verbose diagnostic terminal logging ([`vla_controller.py`](agent/vla_controller.py)), [simulation/sim_robot.py](simulation/sim_robot.py) for MuJoCo simulation wrappers, [simulation/sim_calibration.py](simulation/sim_calibration.py) for synthetic sim homography calibration, [simulation/env.py](simulation/env.py) for RL environment specs, [simulation/test_movement.py](simulation/test_movement.py) for motion validation, [ui/](ui/) for the Unified FastAPI + Gradio 6.0 VLA Web UI (interactive 3D Three.js simulation viewer, OpenCV Motion JPEG streaming with live continuous ArUco calibration tracking, minimalist target bounding box overlay, native AVFoundation macOS camera discovery, TehnoX dark theme, thread-safe `RobotStateManager` with `get_active_robot()`, UI input blocking with dynamic `🟢 Status: Ready` / `⏳ Status: Working...` banner in [`components/chat.py`](ui/components/chat.py), and degree-tuned servo sliders), [tests/](tests/) for automated unit and E2E test suites ([`test_e2e_spatial_grounding.py`](tests/test_e2e_spatial_grounding.py), [`test_e2e_perception_guardrails.py`](tests/test_e2e_perception_guardrails.py), [`test_tools.py`](tests/test_tools.py), [`test_vla_controller.py`](tests/test_vla_controller.py), [`test_calibration.py`](tests/test_calibration.py), [`test_cartesian_ik.py`](tests/test_cartesian_ik.py), [`test_perception.py`](tests/test_perception.py), [`test_ui.py`](tests/test_ui.py)), [test_live.py](test_live.py) for Live API testing, and [main.py](main.py) for command entry points.
- **MuJoCo & Warp**: See MuJoCo documentation and [simulation/simulate.py](simulation/simulate.py) for standard simulation loops. For GPU/CPU batched simulation, reference [simulation/warp_env.py](simulation/warp_env.py), `google-deepmind/mujoco_warp`, and NVIDIA Warp docs.

## 2. Environment (`uv`)

- **Activate Virtualenv**: At start of every command execution, make sure to run `source .venv/bin/activate`
- **Run Commands**: Prefer executing application servers, tests, scripts, and linters via `uv run` (e.g., `uv run python main.py` to start the Web UI server at `http://localhost:7860/`, `uv run ruff check .`, `uv run python main.py --task check_environment`, `uv run python main.py --task test_simulation`, `uv run python test_live.py`).
- **Sync Dependencies**: `uv sync`
- **Server Lifecycle and Port Management**: Before launching the Web UI server (default port `7860`), ALWAYS verify that the port is free (`lsof -ti :7860`). If it is not available, kill the occupying process (`lsof -ti :7860 | xargs kill -9`) to avoid binding collisions or stale code caching. Start the server, test/inspect changes, and **ALWAYS terminate/kill the server process immediately when done** so the port is free for the user to run and use the application.

## 3. Formatting, Linting (`ruff`), Testing & Logging

- **Formatting & Linting**:
  - Always inspect and auto-format code using `ruff` as configured in [pyproject.toml](pyproject.toml):
    - Check & fix lint: `uv run ruff check --fix .`
    - Format code: `uv run ruff format .`
  - **Docstrings**: All modules, classes, and functions must use **Google-styled docstrings** (`Args:`, `Returns:`, `Raises:`). Max line length is **100 chars**.
  - After all changes, ensure that all `.md` files are updated.

- **Testing Guidelines**:
  - Run the full test suite with `uv run pytest tests/ -v`.
  - All new features and bug fixes must have corresponding unit or integration tests in `tests/` (`test_ui.py`, `test_vla_controller.py`, `test_tools.py`, `test_calibration.py`, `test_cartesian_ik.py`, `test_perception.py`, etc.).
  - Mock Gemini Live WebSocket sessions and external hardware appropriately so test suites execute fast, reliably, and deterministically without network dependencies.
  - Confirm 100% pass rate before committing or reporting milestone completion.

- **Diagnostic Terminal Logging**:
  - Use the standard ISO-8601 millisecond-precision format:
    `[YYYY-MM-DD HH:MM:SS.mmm] [MODULE] [TAG] message`
  - **Stream Routing**:
    - Route informational lifecycle events, turn dispatches, tool arguments (`call.args`), and execution returns (`result_dict`) to `sys.stdout`.
    - Route all errors, warnings, IK divergences, and WebSocket connection issues to `sys.stderr`.
  - **Stream Separation**: Do NOT leak raw JSON payloads, IK solver iterations, or debug traces into the Gradio Chatbot viewport. Keep the UI focused on conversational narration and markdown thought traces (`> 💭 *...*`).

- **UI Component Conventions**:
  - **Interactivity State Management**: Always block user inputs (`prompt_input`, `exec_btn`, `home_btn`, `clear_btn`, `rand_btn`) with `gr.update(interactive=False)` and set the status banner to `⏳ Status: Working...` during active model execution turns.
  - **Guaranteed Cleanup**: Wrap execution in `try...finally` blocks to guarantee all controls are restored to `gr.update(interactive=True)` and banner to `🟢 Status: Ready` upon completion or error.
  - **Component Event Parity**: Ensure event generator return tuples match exact component counts across modes (12 updates in `real` mode, 13 updates in `sim`/`twin` mode with `rand_btn`).
  - **Thread Safety**: Access and modify global robot instances exclusively through thread-safe helpers in `ui/state.py` (`get_active_robot()`, `set_active_robot()`, `get_robot_state_dict()`).
  - To test Web changes with Gradio, use Chrome DevTools if available.

## 4. Git Restrictions

> [!IMPORTANT]
> **NEVER execute `git add` or `git commit`.** The user assumes full responsibility for staging and committing changes. Read-only exploratory commands (`git status`, `git diff`, `git log`) are allowed.

## 5. External Repo Info (`gh`)

- You are authorized and encouraged to use the GitHub CLI (`gh`) to query external repositories, docs, issues, and PRs (e.g., `gh repo view google-deepmind/mujoco_warp`).
