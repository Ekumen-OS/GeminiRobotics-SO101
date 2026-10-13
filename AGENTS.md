# GeminiRobotics-SO101 Agent Guidelines

## 1. Context Sources

- **System Architecture & Testing Specification**: All agents working on this codebase MUST consult [ARCHITECTURE.md](ARCHITECTURE.md) as the authoritative single source of truth for system dataflows, subsystem interface contracts (calibration, perception metrics, tool response payloads), feature catalog (F1–F14), and the 4-tier E2E testing methodology.
- **Layered Dependency Architecture**: The codebase strictly enforces bottom-to-top layered dependency: `robot < simulation < vision < agent < ui`, with operator tooling in `tools/` importing layers below it. No layer may import from a layer above it, and cycle-dodging local imports are strictly prohibited. Enforced locally via `uv run lint-imports` and `pre-commit`.
- **Package Layout** (see [ARCHITECTURE.md §6](ARCHITECTURE.md#6-repository-layout--file-responsibilities) for file-level detail):
  - `robot/`: Hardware driver, Cartesian IK, calibration, camera model, and contracts.
  - `simulation/`: MuJoCo simulation wrappers, synthetic calibration, and physics tests.
  - `vision/`: Camera capture, pristine/annotated frame buffers, target blackboard, and jaw-tag locator.
  - `agent/`: Gemini Live API session, tool execution, pick loop, and VLA controller.
  - `tools/`: Operator utilities including printable calibration sheets and interactive setup wizard.
  - `ui/`: Unified FastAPI and Gradio 6.0 web application with 3D Three.js simulation viewer.
  - `tests/`: Automated unit, characterization, and end-to-end verification suites organized by subject.
- **MuJoCo & Warp**: See MuJoCo documentation and [simulation/simulate.py](simulation/simulate.py) for standard simulation loops. For GPU/CPU batched simulation, reference [simulation/warp_env.py](simulation/warp_env.py), `google-deepmind/mujoco_warp`, and NVIDIA Warp docs.

## 2. Environment (`uv`)

- **Activate Virtualenv**: At start of every command execution, make sure to run `source .venv/bin/activate`
- **Run Commands**: Prefer executing application servers, tests, scripts, and linters via `uv run` (e.g., `uv run python main.py` to start the Web UI server at `http://localhost:7860/`, `uv run ruff check .`, `uv run python main.py --task check_environment`, `uv run python main.py --task test_simulation`, `uv run python test_live.py`).
- **Sync Dependencies**: `uv sync --all-extras`
- **Server Lifecycle and Port Management**: Before launching the Web UI server (default port `7860`), ALWAYS verify that the port is free (`lsof -ti :7860`). If it is not available, kill the occupying process (`lsof -ti :7860 | xargs kill -9`) to avoid binding collisions or stale code caching. Start the server, test/inspect changes, and **ALWAYS terminate/kill the server process immediately when done** so the port is free for the user to run and use the application.

## 3. Formatting, Linting (`ruff`), Testing & Logging

- **Formatting & Linting**:
  - Always inspect and auto-format code using `ruff` as configured in [pyproject.toml](pyproject.toml):
    - Check & fix lint: `uv run ruff check --fix .`
    - Format code: `uv run ruff format .`
  - **Docstrings**: All modules, classes, and functions must use **Google-styled docstrings** (`Args:`, `Returns:`, `Raises:`). Max line length is **100 chars**.
  - After all changes, ensure that all `.md` files are updated.

- **Testing Guidelines**:
  - Run the full test suite with `uv run pytest tests/ -v` (physics-heavy scenarios marked `slow` are excluded by default; run them with `uv run pytest -m slow -s`).
  - Run test coverage across core modules: `uv run pytest tests/ -q --cov=robot --cov=vision --cov=agent --cov-report=term-missing` (enforces `--cov-fail-under=78`).
  - **Pre-push Hook**: The pre-push hook runs `uv run pytest -q -m "" --cov=robot --cov=vision --cov=agent` (executing all tests, slow ones included, with the 78% coverage gate, taking ~70 s).
  - All new features and bug fixes must have corresponding unit or integration tests in `tests/` (`tests/ui/test_ui.py`, `tests/agent/test_vla_controller.py`, `tests/agent/test_tools.py`, `tests/robot/test_calibration.py`, `tests/robot/test_cartesian_ik.py`, `tests/robot/test_perception.py`, etc.).
  - Mock Gemini Live WebSocket sessions and external hardware appropriately so test suites execute fast, reliably, and deterministically without network dependencies.
  - Confirm 100% pass rate before committing or reporting milestone completion.

- **Typed Errors & Exception Rules**:
  - Code must raise specific taxonomy errors from `robot/errors.py` (`RoboticsError`, `HardwareError`, `CalibrationUnavailable`, `PerceptionUnavailable`, `ToolArgumentError`) and `agent/errors.py` (`SessionError`).
  - Broad catches (`except Exception`) are strictly restricted to designated safety boundaries:
    - `agent/tools.py`: `execute_tool_call` outer boundary (maps to typed `ToolErrorResponse`, halts robot on `HardwareError`).
    - `agent/vla_controller.py`: `_process_instruction_async` outer catch (halts robot and emits error).
    - `agent/live_session.py`: `disconnect()` cleanup.
    - `robot/logger.py`: `emit()` standard logging boundary.
    - `simulation/sim_robot.py`: `__del__` resource cleanup.
    - `vision/sim_runtime.py` and `vision/capture.py`: long-lived worker thread loops (logging with `logger.exception`).
    - `simulation/check_env.py`: diagnostic environment checks.
    - `ui/components/chat.py`: stream consumer (must call `controller.interrupt()` before appending error).
  - All other sites must catch specific exception tuples (e.g., `(json.JSONDecodeError, OSError)`, `(KeyError, ValueError)`). Never swallow failures silently with `pass` without honest telemetry.

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
  - **Component Event Parity**: Ensure event generator return tuples match exact component counts across modes (12 updates in `real` mode, 13 updates in `sim` mode with `rand_btn`).
  - **Thread Safety**: Access and modify global robot instances exclusively through thread-safe helpers in `robot/registry.py` (`get_active_robot()`, `set_active_robot()`) or delegated via `ui/state.py`.
  - To test Web changes with Gradio, use Chrome DevTools if available.

## 4. Git Restrictions

> [!IMPORTANT]
> **NEVER execute `git add` or `git commit`.** The user assumes full responsibility for staging and committing changes. Read-only exploratory commands (`git status`, `git diff`, `git log`) are allowed.

## 5. External Repo Info (`gh`)

- You are authorized and encouraged to use the GitHub CLI (`gh`) to query external repositories, docs, issues, and PRs (e.g., `gh repo view google-deepmind/mujoco_warp`).
