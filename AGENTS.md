# GeminiRobotics-SO101 Agent Guidelines

## 1. Context Sources
- **GeminiRobotics-SO101 Arm**: 6-DOF robotic arm setup integrated with Gemini Robotics models. Check [README.md](README.md) for overview, [robot/](robot/) for core robot definitions (kinematics, hardware driver, digital twin, constants, and math theory), [simulation/sim_robot.py](simulation/sim_robot.py) for MuJoCo simulation wrappers, [simulation/env.py](simulation/env.py) for RL environment specs, [simulation/test_movement.py](simulation/test_movement.py) for motion validation, and [main.py](main.py) for entry points.
- **MuJoCo & Warp**: See MuJoCo documentation and [simulation/simulate.py](simulation/simulate.py) for standard simulation loops. For GPU/CPU batched simulation, reference [simulation/warp_env.py](simulation/warp_env.py), `google-deepmind/mujoco_warp`, and NVIDIA Warp docs.

## 2. Environment (`uv`)
- **Activate Virtualenv**: `source .venv/bin/activate`
- **Run Commands**: Prefer executing tests, scripts, and linters via `uv run` (e.g., `uv run ruff check .`, `uv run python main.py --task check_environment`, `uv run python main.py --task test_simulation`).
- **Sync Dependencies**: `uv sync`

## 3. Formatting & Linting (`ruff`)
- Always inspect and auto-format code using `ruff` as configured in [pyproject.toml](pyproject.toml):
  - Check & fix lint: `uv run ruff check --fix .`
  - Format code: `uv run ruff format .`
- **Docstrings**: All modules, classes, and functions must use **Google-styled docstrings** (`Args:`, `Returns:`, `Raises:`). Max line length is **100 chars**.

## 4. Git Restrictions
> [!IMPORTANT]
> **NEVER execute `git add` or `git commit`.** The user assumes full responsibility for staging and committing changes. Read-only exploratory commands (`git status`, `git diff`, `git log`) are allowed.

## 5. External Repo Info (`gh`)
- You are authorized and encouraged to use the GitHub CLI (`gh`) to query external repositories, docs, issues, and PRs (e.g., `gh repo view google-deepmind/mujoco_warp`).
