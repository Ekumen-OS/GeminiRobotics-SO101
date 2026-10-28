# Contributing to GeminiRobotics-SO101

Thank you for contributing! You don't need an SO-101 arm to help: simulation mode (`--mode sim`) and the full test suite both run without any physical robot. Hardware is only needed if you want to test changes on the real arm.

## Setup

Install dependencies with [`uv`](https://docs.astral.sh/uv/) and register both `pre-commit` and `pre-push` git hooks:

```bash
uv sync --extra ui
uv run pre-commit install --hook-type pre-commit --hook-type pre-push
```

(Add `--extra hw` if you also test on a physical SO-101 arm.)

## Architecture & Style Rules

- **Layered Dependency Architecture**: The codebase strictly enforces `robot < simulation < vision < agent < ui`, with operator utilities in `tools/` importing only layers below it. No module may import from a layer above it. See [ARCHITECTURE.md](ARCHITECTURE.md) for full details.
- **Formatting & Docstrings**: Code is formatted and linted with `ruff` at a **100-character** line limit using **Google-style docstrings** (`Args:`, `Returns:`, `Raises:`).
- **Tests**: Every bug fix or feature needs unit or integration tests under `tests/`. Physics-heavy scenarios use the `@pytest.mark.slow` marker (excluded by default in `uv run pytest`, included with `-m slow` or `-m ""`).

## Checks to Run Before Opening a PR

Run the same checks as the git hooks:

```bash
uv run ruff check . && uv run ruff format --check .
uv run mypy
uv run lint-imports
uv run pytest -q -m "" --cov=robot --cov=vision --cov=agent --cov-fail-under=78
```

## Pull Requests

Fill out all applicable sections of [`.github/pull_request_template.md`](.github/pull_request_template.md), including the test commands you ran and any setup impact.
