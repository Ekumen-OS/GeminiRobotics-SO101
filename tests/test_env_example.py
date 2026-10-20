"""Guard tests ensuring .env.example remains complete, truthful, and in sync with Settings."""

import dataclasses
import re
from pathlib import Path

from robot.settings import _CONFIG_SPEC, Settings, _get_platform_serial_port, _parse_xyz

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ENV_EXAMPLE_PATH = _PROJECT_ROOT / ".env.example"


def _parse_env_example() -> list[dict[str, str]]:
    """Parse commented variable definitions and their descriptions from .env.example.

    Returns:
        List of dicts with keys 'name', 'value', 'description', and 'line_num'.
    """
    lines = _ENV_EXAMPLE_PATH.read_text(encoding="utf-8").splitlines()
    entries: list[dict[str, str]] = []
    pattern = re.compile(r"^#\s*([A-Z0-9_]+)=(.*)$")

    for i, line in enumerate(lines):
        match = pattern.match(line.strip())
        if match:
            var_name = match.group(1)
            raw_val = match.group(2).strip()
            desc = ""
            if i > 0 and lines[i - 1].strip().startswith("#"):
                desc = lines[i - 1].strip().lstrip("#").strip()
            entries.append(
                {
                    "name": var_name,
                    "value": raw_val,
                    "description": desc,
                    "line_num": str(i + 1),
                }
            )
    return entries


def test_env_example_variable_set_matches_config_spec() -> None:
    """The set of variables in .env.example must exactly equal the set in _CONFIG_SPEC."""
    entries = _parse_env_example()
    parsed_names = [e["name"] for e in entries]

    # Every env var appears exactly once
    assert len(parsed_names) == len(set(parsed_names)), (
        f"Duplicate variables found in .env.example: "
        f"{[n for n in parsed_names if parsed_names.count(n) > 1]}"
    )

    spec_names = {env_var for _, env_var, _ in _CONFIG_SPEC}
    assert set(parsed_names) == spec_names, (
        f"Mismatch between .env.example and _CONFIG_SPEC. "
        f"Missing from .env.example: {spec_names - set(parsed_names)}. "
        f"Unexpected in .env.example: {set(parsed_names) - spec_names}."
    )


def test_env_example_documented_defaults_match_dataclass() -> None:
    """Each documented default in .env.example must equal the Settings dataclass default."""
    entries = {e["name"]: e for e in _parse_env_example()}
    dataclass_fields = {f.name: f for f in dataclasses.fields(Settings)}

    for field_name, env_var, parser in _CONFIG_SPEC:
        entry = entries[env_var]
        raw_val = entry["value"]
        desc = entry["description"]
        f = dataclass_fields[field_name]
        expected_default = (
            f.default_factory() if f.default_factory is not dataclasses.MISSING else f.default
        )

        # 1. Check description documents the default
        assert "(default:" in desc, f"Description for {env_var} must document '(default: ...)'"

        # 2. Check value matches dataclass default
        if expected_default is None:
            assert raw_val == "", (
                f"Variable {env_var} has default None, so its commented assignment must be empty, "
                f"got {raw_val!r}"
            )
            assert "(default: None)" in desc or "(default: unset)" in desc, (
                f"Description for {env_var} should state default is None/unset, got: {desc!r}"
            )
        elif parser is _parse_xyz:
            assert raw_val != "", f"Expected non-empty default for {env_var}"
            parsed_xyz = _parse_xyz(raw_val)
            assert parsed_xyz == expected_default, (
                f"Default mismatch for {env_var}: parsed {parsed_xyz} "
                f"!= dataclass {expected_default}"
            )
        elif parser is Path:
            assert raw_val != "", f"Expected non-empty default for {env_var}"
            resolved_path = _PROJECT_ROOT / raw_val
            assert resolved_path == expected_default, (
                f"Path mismatch for {env_var}: {resolved_path} != {expected_default}"
            )
        elif parser in (int, float):
            assert raw_val != "", f"Expected non-empty default for {env_var}"
            assert parser(raw_val) == expected_default, (
                f"Value mismatch for {env_var}: {parser(raw_val)} != {expected_default}"
            )
        elif parser is str:
            assert raw_val != "", f"Expected non-empty default for {env_var}"
            if env_var == "ROBOT_PORT":
                assert raw_val == _get_platform_serial_port()
            else:
                assert raw_val == expected_default, (
                    f"String mismatch for {env_var}: {raw_val!r} != {expected_default!r}"
                )


def test_env_example_grouping() -> None:
    """Check that .env.example contains all required section headers in order."""
    content = _ENV_EXAMPLE_PATH.read_text(encoding="utf-8")
    assert "Required:" in content
    assert "Real arm:" in content
    assert "Camera:" in content
    assert "Advanced:" in content

    # Check order of sections
    req_pos = content.find("Required:")
    real_pos = content.find("Real arm:")
    cam_pos = content.find("Camera:")
    adv_pos = content.find("Advanced:")
    assert 0 < req_pos < real_pos < cam_pos < adv_pos
