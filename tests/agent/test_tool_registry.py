"""Behavioural tests for TOOL_REGISTRY, ROBOT_TOOL_DECLARATIONS, and system prompt."""

import re

import pytest

from agent.tools import ROBOT_TOOL_DECLARATIONS, TOOL_REGISTRY
from agent.vla_controller import load_system_prompt


def test_tool_declarations_match_registry_and_schema() -> None:
    """Verify ROBOT_TOOL_DECLARATIONS matches TOOL_REGISTRY keys and has valid OBJECT schemas."""
    decl_names = [d["name"] for d in ROBOT_TOOL_DECLARATIONS]
    assert set(decl_names) == set(TOOL_REGISTRY.keys())
    assert len(decl_names) == len(TOOL_REGISTRY)

    for decl in ROBOT_TOOL_DECLARATIONS:
        assert isinstance(decl.get("description"), str) and decl["description"].strip()
        params = decl.get("parameters")
        assert isinstance(params, dict)
        assert params.get("type") == "OBJECT"
        properties = params.get("properties")
        assert isinstance(properties, dict)
        required = params.get("required", [])
        assert isinstance(required, list)
        for req_key in required:
            assert req_key in properties, (
                f"Tool {decl['name']!r} requires {req_key!r} which is missing from properties"
            )


def test_blocking_tools_expected_set() -> None:
    """Verify the set of blocking tools in TOOL_REGISTRY matches the expected motion tools."""
    blocking_tools = {name for name, spec in TOOL_REGISTRY.items() if spec.blocking}
    assert blocking_tools == {
        "close_gripper",
        "move_to_pose",
        "move_to_preset",
        "move_to_spatial_point",
        "open_gripper",
    }


@pytest.mark.parametrize("mode", ["sim", "real"])
def test_system_prompt_renders_all_tools_and_placeholders(mode: str) -> None:
    """Verify load_system_prompt includes every non-empty prompt_line and leaves no placeholders."""
    prompt = load_system_prompt(mode)
    for name, spec in TOOL_REGISTRY.items():
        if spec.prompt_line:
            assert f"- {spec.prompt_line}" in prompt, (
                f"Prompt in mode {mode!r} is missing prompt_line for {name!r}"
            )

    unrendered = re.findall(r"\{[a-zA-Z_][a-zA-Z0-9_]*\}", prompt)
    assert unrendered == []


def test_system_prompt_sim_vs_real_differs_only_in_mode_section() -> None:
    """Verify sim and real prompts differ only in the mode-specific header and section."""
    sim_prompt = load_system_prompt("sim")
    real_prompt = load_system_prompt("real")
    assert sim_prompt != real_prompt

    sim_lines = sim_prompt.splitlines()
    real_lines = real_prompt.splitlines()
    assert sim_lines[0] == (
        "You are the Gemini Robotics Assistant, an intelligent, conversational, and "
        "autonomous 6-DOF robot arm orchestrator operating in 'SIM' mode."
    )
    assert real_lines[0] == (
        "You are the Gemini Robotics Assistant, an intelligent, conversational, and "
        "autonomous 6-DOF robot arm orchestrator operating in 'REAL' mode."
    )

    common_marker = "### Available Tools:"
    assert common_marker in sim_prompt and common_marker in real_prompt
    assert sim_prompt.split(common_marker, 1)[1] == real_prompt.split(common_marker, 1)[1]
