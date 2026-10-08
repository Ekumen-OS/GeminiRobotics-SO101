"""Characterization snapshot tests to lock in prompt, tools, and response assembly."""

import json
from pathlib import Path

from agent.tools import _BLOCKING_TOOLS, ROBOT_TOOL_DECLARATIONS
from agent.vla_controller import VLAController, load_system_prompt
from tests.conftest import (
    MockRobot,
    fake_live_session,
    text_message,
    thought_message,
    tool_call_message,
)

SNAPSHOT_DIR = Path(__file__).parent / "snapshots"


def test_system_prompt_snapshots() -> None:
    """Verify load_system_prompt renders identically to baseline snapshots."""
    sim_prompt = load_system_prompt("sim")
    real_prompt = load_system_prompt("real")

    expected_sim = (SNAPSHOT_DIR / "system_prompt_sim.md").read_text(encoding="utf-8")
    expected_real = (SNAPSHOT_DIR / "system_prompt_real.md").read_text(encoding="utf-8")

    assert sim_prompt == expected_sim
    assert real_prompt == expected_real


def test_tool_declarations_snapshot() -> None:
    """Verify ROBOT_TOOL_DECLARATIONS matches the baseline declarations snapshot."""
    actual_json = json.dumps(ROBOT_TOOL_DECLARATIONS, indent=2, sort_keys=True)
    expected_json = (SNAPSHOT_DIR / "robot_tool_declarations.json").read_text(encoding="utf-8")

    assert actual_json == expected_json


def test_blocking_tools_snapshot() -> None:
    """Verify _BLOCKING_TOOLS matches the baseline set snapshot."""
    actual_json = json.dumps(sorted(list(_BLOCKING_TOOLS)), indent=2)
    expected_json = (SNAPSHOT_DIR / "blocking_tools.json").read_text(encoding="utf-8")

    assert actual_json == expected_json


def test_response_assembly_snapshot() -> None:
    """Verify progressive response streaming assembly against baseline chunks."""
    robot = MockRobot()
    controller = VLAController(mode="sim", robot=robot)
    controller._api_available = True

    messages = [
        thought_message("Thinking step 1..."),
        thought_message(" and step 2."),
        text_message("I will check the joints."),
        tool_call_message("get_joint_positions", {}, "call_001"),
        thought_message("Evaluating joints."),
        RuntimeError("Simulated connection failure"),
    ]

    controller._session = fake_live_session(messages)
    actual_chunks = list(controller.process_instruction_stream("Inspect joints"))

    expected_chunks = json.loads(
        (SNAPSHOT_DIR / "assembled_response_chunks.json").read_text(encoding="utf-8")
    )

    assert actual_chunks == expected_chunks


def test_dynamic_registry_tool_in_prompt_and_declarations() -> None:
    """Verify adding a ToolSpec to TOOL_REGISTRY reflects in prompt and declarations."""
    from agent.tools import ToolSpec, register_tool, unregister_tool

    custom_spec = ToolSpec(
        name="custom_inspect_fixture",
        description="Inspect custom fixture state.",
        parameters={"type": "OBJECT", "properties": {}},
        blocking=False,
        handler=lambda ctx, args: {"status": "success"},
        prompt_line="`custom_inspect_fixture()`: Inspect custom fixture state.",
    )
    try:
        register_tool(custom_spec)
        assert any(d["name"] == "custom_inspect_fixture" for d in ROBOT_TOOL_DECLARATIONS)
        prompt = load_system_prompt("sim")
        assert "- `custom_inspect_fixture()`: Inspect custom fixture state." in prompt
    finally:
        unregister_tool("custom_inspect_fixture")
