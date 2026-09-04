"""Adversarial stress-test suite for Visual Verification Loop (Milestone 2/3).

Empirically tests:
1. System prompt rules, formulas, and anti-batching constraints across all modes.
2. Tool response visual_hint fields across all motion, gripper, and preset tools.
3. Coordinate parser (parse_targets_from_text) resilience against delta/error tokens,
   malformed text, boundary limits, and fuzzed inputs.
4. Multi-turn closed-loop visual verification simulation and error recovery.
"""

import random
import string
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from google.genai import types

from agent.tools import (
    ROBOT_TOOL_DECLARATIONS,
    execute_tool_call,
)
from agent.vla_controller import VLAController, parse_targets_from_text
from robot.interface import RobotInterface


class AdversarialMockRobot(RobotInterface):
    """Mock robot for adversarial testing."""

    def __init__(self) -> None:
        """Initialize mock robot."""
        self.qpos = np.zeros(6, dtype=np.float32)
        self.move_history: list[tuple[np.ndarray, float]] = []
        self._stop_requested = False

    def set_joint_positions(self, qpos: np.ndarray) -> None:
        """Set joint positions."""
        self.qpos = np.asarray(qpos, dtype=np.float32)

    def get_joint_positions(self) -> np.ndarray:
        """Get current joint positions."""
        return self.qpos.copy()

    def move_to_pose(
        self, target_qpos: np.ndarray, duration_sec: float = 1.0, dt: float = 0.002
    ) -> None:
        """Simulate moving to pose."""
        self.qpos = np.asarray(target_qpos, dtype=np.float32)
        self.move_history.append((self.qpos.copy(), duration_sec))

    def interrupt(self) -> None:
        """Set stop requested flag."""
        self._stop_requested = True


class DummyCall:
    """Mock tool call object."""

    def __init__(self, name: str, args: dict | None = None, call_id: str = "adv_call_1") -> None:
        """Initialize dummy call.

        Args:
            name: Tool name.
            args: Tool arguments dictionary.
            call_id: Unique call identifier.
        """
        self.name = name
        self.args = args or {}
        self.id = call_id


# ==============================================================================
# 1. SYSTEM PROMPT ADVERSARIAL & COMPLIANCE TESTS
# ==============================================================================


@pytest.mark.parametrize("mode", ["sim", "real", "twin"])
def test_system_prompt_mandatory_rules_across_all_modes(mode: str) -> None:
    """Verify that all operational modes include complete Move->Look->Correct rules."""
    controller = VLAController(mode=mode)
    prompt = controller.get_history()[0]["content"]

    # 1. Operational Mode tag
    assert f"'{mode.upper()}' mode" in prompt

    # 2. Strict Visual Verification Loop header
    assert '### Strict "Move -> Look -> Correct" Visual Verification Loop (MANDATORY):' in prompt
    assert "Open-loop action chaining without verification is strictly forbidden." in prompt

    # 3. Phase 1: MOVE
    assert "1. **MOVE (Single Primitive Action)**:" in prompt
    assert (
        "Execute exactly ONE physical action per turn (`move_to_preset`, "
        "`move_to_spatial_point`, `open_gripper`, or `close_gripper`)." in prompt
    )
    assert "NEVER batch multiple physical commands together in a single turn." in prompt

    # 4. Phase 2: LOOK
    assert "2. **LOOK (Mandatory Visual Observation Pause)**:" in prompt
    assert (
        "fresh post-action camera frame is injected into context along with the tool "
        "execution response." in prompt
    )
    assert "You MUST pause and visually analyze this fresh image." in prompt
    assert "Gripper Position" in prompt
    assert "Target Position" in prompt
    assert "Relative Position Error" in prompt
    assert "[dy, dx] = [target_y - gripper_y, target_x - gripper_x]" in prompt

    # 5. Phase 3: CORRECT
    assert "3. **CORRECT (Alignment Verification or Adjustment)**:" in prompt
    assert "If Misaligned (|dy| > 25 or |dx| > 25)" in prompt
    assert "If Aligned (|dy| <= 25 and |dx| <= 25)" in prompt
    assert "hover height before attempting to descend" in prompt

    # 6. Conversational Narration Rules
    assert "### Conversational Interaction & Narration Rules (MANDATORY):" in prompt
    assert "ALWAYS speak to the user! Never execute tool actions in complete silence." in prompt
    assert "state the relative position error [dy, dx]" in prompt

    # 7. Spatial Grounding Rules
    assert "### CRITICAL Spatial Grounding Rule:" in prompt
    assert "BOTTOM CONTACT CENTER" in prompt
    assert "DO NOT target top faces or upper edges to avoid perspective parallax error." in prompt

    # 8. Complete Step-by-Step Workflow
    assert "1. **Clear Occlusion & Observe**:" in prompt
    assert "2. **Pre-Grasp Hover & Open**:" in prompt
    assert "3. **Visual Verification & Error Check (Hover)**:" in prompt
    assert "4. **Descend & Grasp**:" in prompt
    assert "5. **Lift & Grasp Inspection**:" in prompt
    assert "6. **Transport to Destination Hover**:" in prompt
    assert "7. **Release & Retract**:" in prompt
    assert "8. **Confirm Completion**:" in prompt

    # 9. Ensure deprecated/forbidden tools are NOT in prompt
    assert "execute_spav_pick_and_place" not in prompt


def test_system_prompt_sim_specific_section() -> None:
    """Verify sim mode includes virtual calibration details while real/twin modes omit it."""
    controller_sim = VLAController(mode="sim")
    prompt_sim = controller_sim.get_history()[0]["content"]
    assert "In SIM mode, the workspace is pre-calibrated" in prompt_sim

    controller_real = VLAController(mode="real")
    prompt_real = controller_real.get_history()[0]["content"]
    assert "In SIM mode, the workspace is pre-calibrated" not in prompt_real

    controller_twin = VLAController(mode="twin")
    prompt_twin = controller_twin.get_history()[0]["content"]
    assert "In SIM mode, the workspace is pre-calibrated" not in prompt_twin


# ==============================================================================
# 2. TOOL RESPONSE VISUAL HINT STRESS TESTS
# ==============================================================================


@pytest.mark.asyncio
async def test_move_to_spatial_point_visual_hints_height_thresholds() -> None:
    """Stress test move_to_spatial_point visual_hint for various z heights."""
    robot = AdversarialMockRobot()
    mock_calib = MagicMock()
    mock_calib.is_calibrated = True
    mock_calib.gemini_to_robot_xyz.return_value = np.array([0.2, 0.0, 0.05])

    with patch("ui.stream.get_calibrator", return_value=mock_calib):
        # Hover heights (>= 3.0 cm)
        hover_cases = [3.0, 3.1, 5.0, 10.0, 15.0, 20.0, 25.0]  # 25 clamps to 20
        for z in hover_cases:
            call = DummyCall(
                "move_to_spatial_point",
                {"point": [500, 500], "z_height_cm": z, "approach": "top_down"},
            )
            resp = await execute_tool_call(call, robot=robot)
            assert resp.get("status") == "success"
            assert "visual_hint" in resp
            hint = resp["visual_hint"]
            assert "hover position (z>=3cm)" in hint
            assert "Visual Verification Loop" in hint
            assert "[dy, dx]" in hint
            assert "correct or confirm alignment before descending" in hint

        # Contact heights (< 3.0 cm)
        contact_cases = [0.0, 0.5, 1.0, 2.0, 2.9, -5.0]  # -5 clamps to 0.0
        for z in contact_cases:
            call = DummyCall(
                "move_to_spatial_point",
                {"point": [500, 500], "z_height_cm": z, "approach": "top_down"},
            )
            resp = await execute_tool_call(call, robot=robot)
            assert resp.get("status") == "success"
            assert "visual_hint" in resp
            hint = resp["visual_hint"]
            assert "contact height" in hint
            assert "Visually verify gripper jaws surround the target contact center" in hint


@pytest.mark.asyncio
async def test_gripper_tools_visual_hints() -> None:
    """Verify open_gripper and close_gripper provide actionable visual verification instructions."""
    robot = AdversarialMockRobot()

    # Open gripper
    resp_open = await execute_tool_call(DummyCall("open_gripper"), robot=robot)
    assert resp_open.get("status") == "success"
    assert "visual_hint" in resp_open
    assert "Gripper opened" in resp_open["visual_hint"]
    assert "visually verify object release and clearance" in resp_open["visual_hint"]

    # Close gripper
    resp_close = await execute_tool_call(DummyCall("close_gripper"), robot=robot)
    assert resp_close.get("status") == "success"
    assert "visual_hint" in resp_close
    assert "Gripper closed" in resp_close["visual_hint"]
    assert "Lift arm to hover (z_height_cm=5.0)" in resp_close["visual_hint"]
    assert "confirm if object is held between jaws or still on table" in resp_close["visual_hint"]


@pytest.mark.asyncio
async def test_move_to_preset_visual_hints() -> None:
    """Verify move_to_preset HOME contains visual inspection guidance for targeting."""
    robot = AdversarialMockRobot()

    # HOME pose
    resp_home = await execute_tool_call(
        DummyCall("move_to_preset", {"pose_name": "HOME"}), robot=robot
    )
    assert resp_home.get("status") == "success"
    assert "visual_hint" in resp_home
    assert "Arm is at HOME pose with clear workspace view" in resp_home["visual_hint"]
    assert "locate target object coordinates [y, x]" in resp_home["visual_hint"]

    # MIDDLE pose (valid execution, visual hint not mandated for middle)
    resp_mid = await execute_tool_call(
        DummyCall("move_to_preset", {"pose_name": "MIDDLE"}), robot=robot
    )
    assert resp_mid.get("status") == "success"


# ==============================================================================
# 3. COORDINATE PARSER ADVERSARIAL & FUZZING TESTS
# ==============================================================================


def test_parse_targets_delta_and_error_tokens() -> None:
    """Ensure relative error offsets [dy, dx] and signed tokens are NOT parsed as targets."""
    adversarial_texts = [
        "Relative error offset is [+10, -5]. Correcting position.",
        "Calculated [dy, dx] = [-20, +30].",
        "Offsets: [-15, -25] and [+5, +12].",
        "Alignment check: [dy=0, dx=0] - perfect alignment.",
        "Error vector [dy=+30, dx=-15] exceeds threshold.",
        "Observed delta [+0, -0].",
        "Relative offset [ -10 , +20 ].",
        "Position delta: [+100, -50].",
        "Offset is [-5, 0].",
        "Offset is [0, -5].",
    ]

    for text in adversarial_texts:
        targets = parse_targets_from_text(text)
        # Signed numbers (+10, -5) or expressions (dy=0) must NOT be extracted as 2D targets
        assert targets == [], f"Failed on text '{text}': got unexpected targets {targets}"


def test_parse_targets_valid_coordinates() -> None:
    """Verify standard points and bounding boxes parse correctly with labels."""
    # 2D Point
    t1 = parse_targets_from_text("I see the target at [650, 420].")
    assert len(t1) == 1
    assert t1[0]["point"] == (650.0, 420.0)
    assert t1[0]["box_2d"] is None

    # Bounding Box 2D
    t2 = parse_targets_from_text("Red cube detected: [200, 300, 400, 500].")
    assert len(t2) == 1
    assert t2[0]["box_2d"] == [200.0, 300.0, 400.0, 500.0]
    assert t2[0]["point"] == (300.0, 400.0)
    assert "RED CUBE" in t2[0]["label"]

    # Multiple targets
    text_multi = "Target A is at [100, 200] and Target B is at [700, 800]."
    t3 = parse_targets_from_text(text_multi)
    assert len(t3) == 2
    assert t3[0]["point"] == (100.0, 200.0)
    assert t3[1]["point"] == (700.0, 800.0)


def test_parse_targets_boundary_values() -> None:
    """Test boundary conditions [0, 0] to [1000, 1000] and out-of-range rejection."""
    # Min/max valid
    t_min = parse_targets_from_text("[0, 0]")
    assert len(t_min) == 1
    assert t_min[0]["point"] == (0.0, 0.0)

    t_max = parse_targets_from_text("[1000, 1000]")
    assert len(t_max) == 1
    assert t_max[0]["point"] == (1000.0, 1000.0)

    # Out of range values should be rejected
    assert parse_targets_from_text("[1001, 500]") == []
    assert parse_targets_from_text("[500, 2000]") == []
    assert parse_targets_from_text("[0, 0, 1000, 1001]") == []


def test_parse_targets_malformed_inputs() -> None:
    """Test edge cases with empty strings, malformed numbers, and weird syntax."""
    assert parse_targets_from_text("") == []
    assert parse_targets_from_text(None) == []
    assert parse_targets_from_text("[]") == []
    assert parse_targets_from_text("[100]") == []  # 1 number
    assert parse_targets_from_text("[100, 200, 300]") == []  # 3 numbers
    assert parse_targets_from_text("[10, 20, 30, 40, 50]") == []  # 5 numbers
    assert parse_targets_from_text("[abc, def]") == []
    assert parse_targets_from_text("[,]") == []
    assert parse_targets_from_text("[  ]") == []


def test_parse_targets_fuzzer() -> None:
    """Fuzz parse_targets_from_text with 1000 randomized pseudo-model outputs."""
    random.seed(42)

    chars = string.ascii_letters + string.digits + " \t\n!@#$%^&*()_+-=[]{}|;':\",.<>?/~`"

    for _ in range(1000):
        # Generate random text length 10 to 200
        length = random.randint(10, 200)
        random_str = "".join(random.choice(chars) for _ in range(length))

        # Occasionally inject structured fragments
        injection_type = random.choice(["delta", "point", "box", "garbage", "none"])
        if injection_type == "delta":
            dy = random.randint(-100, 100)
            dx = random.randint(-100, 100)
            prefix_y = "+" if dy > 0 else ""
            prefix_x = "+" if dx > 0 else ""
            random_str += f" relative error [{prefix_y}{dy}, {prefix_x}{dx}] "
        elif injection_type == "point":
            y = random.randint(0, 1000)
            x = random.randint(0, 1000)
            random_str += f" spotted at [{y}, {x}] "
        elif injection_type == "box":
            coords = [random.randint(0, 1000) for _ in range(4)]
            random_str += f" box [{coords[0]}, {coords[1]}, {coords[2]}, {coords[3]}] "

        # Fuzz call: MUST NEVER RAISE EXCEPTION
        result = parse_targets_from_text(random_str)
        assert isinstance(result, list)
        for tgt in result:
            assert "point" in tgt
            assert "label" in tgt
            assert "box_2d" in tgt
            y, x = tgt["point"]
            assert 0 <= y <= 1000
            assert 0 <= x <= 1000


# ==============================================================================
# 4. CLOSED-LOOP VISUAL VERIFICATION WORKFLOW SIMULATION
# ==============================================================================


@pytest.mark.asyncio
async def test_full_visual_verification_loop_orchestration() -> None:
    """Simulate a complete 5-turn closed-loop pick sequence verifying frame injections and hints."""
    robot = AdversarialMockRobot()
    controller = VLAController(mode="sim", robot=robot)
    controller._api_available = True

    mock_session = MagicMock()
    mock_session.is_connected = True
    mock_session.connect = AsyncMock()
    mock_session.send_instruction = AsyncMock()
    mock_session.send_tool_response = AsyncMock()
    mock_session.send_frame = AsyncMock()
    mock_session.send_text = AsyncMock()

    mock_calib = MagicMock()
    mock_calib.is_calibrated = True
    mock_calib.gemini_to_robot_xyz.return_value = np.array([0.18, 0.05, 0.05])

    # Sequence of turns received from Gemini Live API
    turns = [
        # Turn 1: Open gripper
        types.LiveServerMessage(
            tool_call=types.LiveServerToolCall(
                function_calls=[types.FunctionCall(name="open_gripper", args={}, id="step_1_open")]
            )
        ),
        # Turn 2: Move to initial hover
        types.LiveServerMessage(
            tool_call=types.LiveServerToolCall(
                function_calls=[
                    types.FunctionCall(
                        name="move_to_spatial_point",
                        args={"point": [600, 400], "z_height_cm": 5.0, "approach": "top_down"},
                        id="step_2_hover",
                    )
                ]
            )
        ),
        # Turn 3: Corrective hover move after observing dy=+30, dx=-15
        types.LiveServerMessage(
            tool_call=types.LiveServerToolCall(
                function_calls=[
                    types.FunctionCall(
                        name="move_to_spatial_point",
                        args={"point": [630, 385], "z_height_cm": 5.0, "approach": "top_down"},
                        id="step_3_correct",
                    )
                ]
            )
        ),
        # Turn 4: Descend to contact
        types.LiveServerMessage(
            tool_call=types.LiveServerToolCall(
                function_calls=[
                    types.FunctionCall(
                        name="move_to_spatial_point",
                        args={"point": [630, 385], "z_height_cm": 0.0, "approach": "top_down"},
                        id="step_4_descend",
                    )
                ]
            )
        ),
        # Turn 5: Grasp
        types.LiveServerMessage(
            tool_call=types.LiveServerToolCall(
                function_calls=[
                    types.FunctionCall(name="close_gripper", args={}, id="step_5_grasp")
                ]
            )
        ),
    ]

    async def mock_receive_sequence():
        for t in turns:
            yield t
        yield types.LiveServerMessage(server_content=types.LiveServerContent(turn_complete=True))

    mock_session.receive = mock_receive_sequence
    controller._session = mock_session

    fake_frame = b"\xff\xd8\xff\xe0mock_frame_data"

    with patch("ui.stream.get_calibrator", return_value=mock_calib):
        chunks = list(
            controller.process_instruction_stream(
                "Pick up object with visual verification",
                get_latest_jpeg_fn=lambda: fake_frame,
            )
        )

    # 1. Verify all 5 tool responses were dispatched
    assert mock_session.send_tool_response.call_count == 5

    # 2. Verify post-action frames were injected for each blocking tool (5 times)
    assert mock_session.send_frame.call_count >= 5

    # 3. Verify history and stream output
    output_text = "".join(chunks)
    assert "open_gripper" in output_text
    assert "move_to_spatial_point" in output_text
    assert "close_gripper" in output_text


@pytest.mark.asyncio
async def test_tool_declarations_anti_batching_properties() -> None:
    """Verify tool declarations have proper BLOCKING flags to prevent unsynchronized execution."""
    blocking_tool_names = {
        "move_to_pose",
        "set_joint_positions",
        "open_gripper",
        "close_gripper",
        "move_to_preset",
        "move_to_spatial_point",
    }

    for decl in ROBOT_TOOL_DECLARATIONS:
        if decl["name"] in blocking_tool_names:
            assert decl.get("behavior") == "BLOCKING"
        elif decl["name"] == "send_message":
            assert decl.get("behavior") != "BLOCKING"


@pytest.mark.asyncio
async def test_grasp_failure_and_retry_loop() -> None:
    """Simulate a missed grasp detected post-lift triggering an automatic retry sequence."""
    robot = AdversarialMockRobot()
    controller = VLAController(mode="sim", robot=robot)
    controller._api_available = True

    mock_session = MagicMock()
    mock_session.is_connected = True
    mock_session.connect = AsyncMock()
    mock_session.send_instruction = AsyncMock()
    mock_session.send_tool_response = AsyncMock()
    mock_session.send_frame = AsyncMock()
    mock_session.send_text = AsyncMock()

    mock_calib = MagicMock()
    mock_calib.is_calibrated = True
    mock_calib.gemini_to_robot_xyz.return_value = np.array([0.18, 0.05, 0.05])

    # Sequence of turns simulating:
    # 1. Open gripper
    # 2. Hover above initial target [600, 400]
    # 3. Descend to contact [600, 400] z=0.0
    # 4. Close gripper
    # 5. Lift arm to z=5.0
    # [Post-lift inspection: model detects object still on table at [605, 395]]
    # 6. Retry 1: Open gripper
    # 7. Retry 1: Hover above target [605, 395] z=5.0
    # 8. Retry 1: Descend to contact [605, 395] z=0.0
    # 9. Retry 1: Close gripper
    # 10. Retry 1: Lift arm to z=5.0
    # 11. Grasp verified! Send message "Grasp confirmed on retry 1."
    turns = [
        types.LiveServerMessage(
            tool_call=types.LiveServerToolCall(
                function_calls=[types.FunctionCall(name="open_gripper", args={}, id="step_1_open")]
            )
        ),
        types.LiveServerMessage(
            tool_call=types.LiveServerToolCall(
                function_calls=[
                    types.FunctionCall(
                        name="move_to_spatial_point",
                        args={"point": [600, 400], "z_height_cm": 5.0, "approach": "top_down"},
                        id="step_2_hover",
                    )
                ]
            )
        ),
        types.LiveServerMessage(
            tool_call=types.LiveServerToolCall(
                function_calls=[
                    types.FunctionCall(
                        name="move_to_spatial_point",
                        args={"point": [600, 400], "z_height_cm": 0.0, "approach": "top_down"},
                        id="step_3_descend",
                    )
                ]
            )
        ),
        types.LiveServerMessage(
            tool_call=types.LiveServerToolCall(
                function_calls=[
                    types.FunctionCall(name="close_gripper", args={}, id="step_4_grasp")
                ]
            )
        ),
        types.LiveServerMessage(
            tool_call=types.LiveServerToolCall(
                function_calls=[
                    types.FunctionCall(
                        name="move_to_spatial_point",
                        args={"point": [600, 400], "z_height_cm": 5.0, "approach": "top_down"},
                        id="step_5_lift_check",
                    )
                ]
            )
        ),
        # Model detects missed grasp and opens gripper to retry
        types.LiveServerMessage(
            server_content=types.LiveServerContent(
                model_turn=types.Content(
                    parts=[
                        types.Part(
                            text="> 💭 *Post-lift verification: Gripper empty, object remains on "
                            "table at [605, 395]. Initiating retry 1 of 3.*"
                        )
                    ]
                )
            ),
            tool_call=types.LiveServerToolCall(
                function_calls=[
                    types.FunctionCall(name="open_gripper", args={}, id="step_6_retry_open")
                ]
            ),
        ),
        types.LiveServerMessage(
            tool_call=types.LiveServerToolCall(
                function_calls=[
                    types.FunctionCall(
                        name="move_to_spatial_point",
                        args={"point": [605, 395], "z_height_cm": 5.0, "approach": "top_down"},
                        id="step_7_retry_hover",
                    )
                ]
            )
        ),
        types.LiveServerMessage(
            tool_call=types.LiveServerToolCall(
                function_calls=[
                    types.FunctionCall(
                        name="move_to_spatial_point",
                        args={"point": [605, 395], "z_height_cm": 0.0, "approach": "top_down"},
                        id="step_8_retry_descend",
                    )
                ]
            )
        ),
        types.LiveServerMessage(
            tool_call=types.LiveServerToolCall(
                function_calls=[
                    types.FunctionCall(name="close_gripper", args={}, id="step_9_retry_close")
                ]
            )
        ),
        types.LiveServerMessage(
            tool_call=types.LiveServerToolCall(
                function_calls=[
                    types.FunctionCall(
                        name="move_to_spatial_point",
                        args={"point": [605, 395], "z_height_cm": 5.0, "approach": "top_down"},
                        id="step_10_retry_lift",
                    )
                ]
            )
        ),
        types.LiveServerMessage(
            server_content=types.LiveServerContent(
                model_turn=types.Content(
                    parts=[
                        types.Part(
                            text="> 💭 *Post-lift verification: Object confirmed elevated in jaws. "
                            "Task successful.*"
                        )
                    ]
                )
            ),
            tool_call=types.LiveServerToolCall(
                function_calls=[
                    types.FunctionCall(
                        name="send_message",
                        args={"message": "Target grasped successfully on retry 1."},
                        id="step_11_done",
                    )
                ]
            ),
        ),
    ]

    async def mock_receive_sequence():
        for t in turns:
            yield t
        yield types.LiveServerMessage(server_content=types.LiveServerContent(turn_complete=True))

    mock_session.receive = mock_receive_sequence
    controller._session = mock_session

    fake_frame = b"\xff\xd8\xff\xe0mock_frame_data"

    with patch("ui.stream.get_calibrator", return_value=mock_calib):
        chunks = list(
            controller.process_instruction_stream(
                "Pick up object with automatic grasp retry",
                get_latest_jpeg_fn=lambda: fake_frame,
            )
        )

    # Verify all 11 tool responses were dispatched
    assert mock_session.send_tool_response.call_count == 11

    # Verify post-action frames were injected for each blocking tool (10 blocking tools)
    assert mock_session.send_frame.call_count >= 10

    output_text = "".join(chunks)
    assert "Initiating retry 1 of 3" in output_text
    assert "Target grasped successfully on retry 1" in output_text


@pytest.mark.asyncio
async def test_max_retry_limit_exhaustion() -> None:
    """Simulate 3 consecutive missed grasps leading to retry exhaustion and user escalation."""
    robot = AdversarialMockRobot()
    controller = VLAController(mode="sim", robot=robot)
    controller._api_available = True

    mock_session = MagicMock()
    mock_session.is_connected = True
    mock_session.connect = AsyncMock()
    mock_session.send_instruction = AsyncMock()
    mock_session.send_tool_response = AsyncMock()
    mock_session.send_frame = AsyncMock()
    mock_session.send_text = AsyncMock()

    mock_calib = MagicMock()
    mock_calib.is_calibrated = True
    mock_calib.gemini_to_robot_xyz.return_value = np.array([0.18, 0.05, 0.05])

    # Construct 3 failed attempts:
    # Attempt 1: hover -> descend -> close -> lift
    # Attempt 2: open -> hover -> descend -> close -> lift
    # Attempt 3: open -> hover -> descend -> close -> lift
    # Final: send_message("Max retries reached. Requesting assistance.")
    turns = []

    # Attempt 1
    turns.extend(
        [
            types.LiveServerMessage(
                tool_call=types.LiveServerToolCall(
                    function_calls=[types.FunctionCall(name="open_gripper", args={}, id="a1_open")]
                )
            ),
            types.LiveServerMessage(
                tool_call=types.LiveServerToolCall(
                    function_calls=[
                        types.FunctionCall(
                            name="move_to_spatial_point",
                            args={"point": [600, 400], "z_height_cm": 5.0, "approach": "top_down"},
                            id="a1_hover",
                        )
                    ]
                )
            ),
            types.LiveServerMessage(
                tool_call=types.LiveServerToolCall(
                    function_calls=[
                        types.FunctionCall(
                            name="move_to_spatial_point",
                            args={"point": [600, 400], "z_height_cm": 0.0, "approach": "top_down"},
                            id="a1_descend",
                        )
                    ]
                )
            ),
            types.LiveServerMessage(
                tool_call=types.LiveServerToolCall(
                    function_calls=[
                        types.FunctionCall(name="close_gripper", args={}, id="a1_close")
                    ]
                )
            ),
            types.LiveServerMessage(
                tool_call=types.LiveServerToolCall(
                    function_calls=[
                        types.FunctionCall(
                            name="move_to_spatial_point",
                            args={"point": [600, 400], "z_height_cm": 5.0, "approach": "top_down"},
                            id="a1_lift",
                        )
                    ]
                )
            ),
        ]
    )

    # Attempt 2
    turns.extend(
        [
            types.LiveServerMessage(
                server_content=types.LiveServerContent(
                    model_turn=types.Content(
                        parts=[
                            types.Part(
                                text="> 💭 *Grasp missed (attempt 1). "
                                "Retrying grasp (attempt 2 of 3).*"
                            )
                        ]
                    )
                ),
                tool_call=types.LiveServerToolCall(
                    function_calls=[types.FunctionCall(name="open_gripper", args={}, id="a2_open")]
                ),
            ),
            types.LiveServerMessage(
                tool_call=types.LiveServerToolCall(
                    function_calls=[
                        types.FunctionCall(
                            name="move_to_spatial_point",
                            args={"point": [600, 400], "z_height_cm": 5.0, "approach": "top_down"},
                            id="a2_hover",
                        )
                    ]
                )
            ),
            types.LiveServerMessage(
                tool_call=types.LiveServerToolCall(
                    function_calls=[
                        types.FunctionCall(
                            name="move_to_spatial_point",
                            args={"point": [600, 400], "z_height_cm": 0.0, "approach": "top_down"},
                            id="a2_descend",
                        )
                    ]
                )
            ),
            types.LiveServerMessage(
                tool_call=types.LiveServerToolCall(
                    function_calls=[
                        types.FunctionCall(name="close_gripper", args={}, id="a2_close")
                    ]
                )
            ),
            types.LiveServerMessage(
                tool_call=types.LiveServerToolCall(
                    function_calls=[
                        types.FunctionCall(
                            name="move_to_spatial_point",
                            args={"point": [600, 400], "z_height_cm": 5.0, "approach": "top_down"},
                            id="a2_lift",
                        )
                    ]
                )
            ),
        ]
    )

    # Attempt 3
    turns.extend(
        [
            types.LiveServerMessage(
                server_content=types.LiveServerContent(
                    model_turn=types.Content(
                        parts=[
                            types.Part(
                                text="> 💭 *Grasp missed (attempt 2). "
                                "Retrying grasp (attempt 3 of 3).*"
                            )
                        ]
                    )
                ),
                tool_call=types.LiveServerToolCall(
                    function_calls=[types.FunctionCall(name="open_gripper", args={}, id="a3_open")]
                ),
            ),
            types.LiveServerMessage(
                tool_call=types.LiveServerToolCall(
                    function_calls=[
                        types.FunctionCall(
                            name="move_to_spatial_point",
                            args={"point": [600, 400], "z_height_cm": 5.0, "approach": "top_down"},
                            id="a3_hover",
                        )
                    ]
                )
            ),
            types.LiveServerMessage(
                tool_call=types.LiveServerToolCall(
                    function_calls=[
                        types.FunctionCall(
                            name="move_to_spatial_point",
                            args={"point": [600, 400], "z_height_cm": 0.0, "approach": "top_down"},
                            id="a3_descend",
                        )
                    ]
                )
            ),
            types.LiveServerMessage(
                tool_call=types.LiveServerToolCall(
                    function_calls=[
                        types.FunctionCall(name="close_gripper", args={}, id="a3_close")
                    ]
                )
            ),
            types.LiveServerMessage(
                tool_call=types.LiveServerToolCall(
                    function_calls=[
                        types.FunctionCall(
                            name="move_to_spatial_point",
                            args={"point": [600, 400], "z_height_cm": 5.0, "approach": "top_down"},
                            id="a3_lift",
                        )
                    ]
                )
            ),
        ]
    )

    # Escalation
    turns.append(
        types.LiveServerMessage(
            server_content=types.LiveServerContent(
                model_turn=types.Content(
                    parts=[
                        types.Part(
                            text="> 💭 *Grasp failed after 3 attempts. "
                            "Halting and requesting user intervention.*"
                        )
                    ]
                )
            ),
            tool_call=types.LiveServerToolCall(
                function_calls=[
                    types.FunctionCall(
                        name="send_message",
                        args={
                            "message": "Failed to grasp target object after 3 attempts. "
                            "Please reposition object."
                        },
                        id="escalation_msg",
                    )
                ]
            ),
        )
    )

    async def mock_receive_sequence():
        for t in turns:
            yield t
        yield types.LiveServerMessage(server_content=types.LiveServerContent(turn_complete=True))

    mock_session.receive = mock_receive_sequence
    controller._session = mock_session

    fake_frame = b"\xff\xd8\xff\xe0mock_frame_data"

    with patch("ui.stream.get_calibrator", return_value=mock_calib):
        chunks = list(
            controller.process_instruction_stream(
                "Pick up object with retry exhaustion test",
                get_latest_jpeg_fn=lambda: fake_frame,
            )
        )

    # Verify all 16 tool calls were dispatched
    assert mock_session.send_tool_response.call_count == 16

    # Verify post-action frames delivered for the 15 blocking tool calls
    assert mock_session.send_frame.call_count >= 15

    output_text = "".join(chunks)
    assert "attempt 3 of 3" in output_text
    assert "Failed to grasp target object after 3 attempts" in output_text
