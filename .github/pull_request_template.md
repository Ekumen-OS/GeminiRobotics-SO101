## Description

<!-- What this PR does and why, in 2–4 sentences. Link the issue or prompt it comes from. -->

## Changes

<!-- One bullet per change. Link files or lines where it helps, e.g. [`agent/tools.py`](agent/tools.py). -->

-

## Testing

<!-- What was run, and what it showed. Delete the lines that do not apply. -->

- [ ] `uv run ruff check . && uv run ruff format --check .`
- [ ] `uv run mypy` and `uv run lint-imports`
- [ ] `uv run pytest -q -m "" --cov=robot --cov=vision --cov=agent --cov-fail-under=78` (slow tests included)
- [ ] Sim run (`--mode sim`, harness or a scripted pick):
- [ ] Real-arm run (`--mode real`): what was done and what was measured, in mm
- [ ] Screenshots / logs for UI or behaviour changes

## Setup impact

<!-- Anything a user must do after pulling: new or removed .env keys, re-calibration,
     re-printed tags, regenerated files. Write "None" if nothing. -->

None
