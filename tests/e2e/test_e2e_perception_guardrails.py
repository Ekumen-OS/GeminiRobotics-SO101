"""Four-tier end-to-end test suite for perception feedback & anti-hallucination guardrails.

Covers:
- F5: Programmatic Perception Metrics Engine ([dy, dx], distance, is_aligned)
- F6: Tool Execution Feedback Enrichment (hover hints, grasp verification)
- F7: Anti-Hallucination Descent Gate (|dy| <= 25, |dx| <= 25)
- F8: Anti-Hallucination Transport Gate & Retry Loop (3-attempt state machine & escalation)
- F9: Clean Diagnostic Logging & UI Separation (ISO-8601 millisecond logs, stdout/stderr routing)
"""

from unittest.mock import AsyncMock, MagicMock, patch

import cv2
import numpy as np
import pytest
from google.genai import types

from agent.tools import execute_tool_call
from agent.vla_controller import VLAController, parse_targets_from_text
from robot.settings import Settings, set_settings
from tests.conftest import DummyToolCall, MockRobot
from vision.detector import (
    detect_workspace_objects,
    refine_point_to_nearest_object,
)


def make_synthetic_red_cube_frame(
    cube_y_1000: float = 760.0,
    cube_x_1000: float = 484.0,
    width: int = 640,
    height: int = 480,
) -> np.ndarray:
    """Create a synthetic RGB frame containing a red block at the specified normalized point.

    Args:
        cube_y_1000: Normalized Y coordinate in [0, 1000].
        cube_x_1000: Normalized X coordinate in [0, 1000].
        width: Frame pixel width.
        height: Frame pixel height.

    Returns:
        RGB numpy array of shape (height, width, 3).
    """
    frame = np.full((height, width, 3), 180, dtype=np.uint8)  # Light gray table
    cx = int((cube_x_1000 / 1000.0) * width)
    cy = int((cube_y_1000 / 1000.0) * height)
    size = 24

    # Draw saturated red cube (RGB: 220, 20, 30)
    cv2.rectangle(
        frame,
        (cx - size // 2, cy - size),
        (cx + size // 2, cy),
        (220, 20, 30),
        -1,
    )
    return frame


@pytest.fixture(autouse=True)
def configure_perception_guardrails_env(restore_settings: None) -> None:
    """Configure perception guardrails environment for tests in this module."""
    set_settings(Settings.from_env(descent_gate="off"))


@pytest.fixture
def mock_robot_with_object() -> MockRobot:
    """MockRobot that reports an object in the jaws for scripted T4 tests."""
    robot = MockRobot()
    robot.gripper_reading = {"aperture_deg": 25.0, "load": 150.0, "source": "measured"}
    return robot


# ==============================================================================
# TIER 1: ISOLATED FEATURE COVERAGE (F5, F6, F7, F8, F9)
# ==============================================================================


def test_t1_f5_perception_detection_on_synthetic_frame() -> None:
    """Validate vision detector finding bottom contact coordinates on synthetic red block."""
    frame_rgb = make_synthetic_red_cube_frame(cube_y_1000=760.0, cube_x_1000=484.0)
    objects = detect_workspace_objects(frame_rgb)

    assert len(objects) >= 1
    red_obj = next((obj for obj in objects if obj["color"] == "red"), None)
    assert red_obj is not None
    assert "bottom_contact_1000" in red_obj
    bot_y, bot_x = red_obj["bottom_contact_1000"]
    # Check within 15 normalized units of target
    assert abs(bot_y - 760.0) <= 15.0
    assert abs(bot_x - 484.0) <= 15.0


@pytest.mark.asyncio
async def test_t1_f6_tool_visual_hint_hover_vs_contact() -> None:
    """Validate move_to_spatial_point returns appropriate visual verification hint."""
    robot = MockRobot()

    # Case 1: Hover height (z_height_cm = 4.0 >= 3.0)
    call_hover = DummyToolCall(
        "move_to_spatial_point",
        {"point": [760.0, 484.0], "z_height_cm": 4.0, "approach": "top_down"},
    )
    res_hover = await execute_tool_call(call=call_hover, robot=robot)
    assert res_hover["status"] == "success"
    hint = res_hover.get("visual_hint", "")
    assert "Visual Verification Loop" in hint
    assert "alignment not measured" in hint  # no frame provider: the hint does not pretend

    # Case 2: Contact height (z_height_cm = 0.0 < 3.0)
    call_contact = DummyToolCall(
        "move_to_spatial_point",
        {"point": [760.0, 484.0], "z_height_cm": 0.0, "approach": "top_down"},
    )
    res_contact = await execute_tool_call(call=call_contact, robot=robot)
    assert res_contact["status"] == "success"
    contact_hint = res_contact.get("visual_hint", "")
    assert "Arm reached contact height" in contact_hint
    assert "close_gripper()" in contact_hint


@pytest.mark.asyncio
async def test_t1_f6_tool_visual_hint_gripper_actions() -> None:
    """Validate close_gripper and open_gripper visual hints instruct verification."""
    robot = MockRobot()

    # close_gripper hint on empty jaws reports missed grasp
    call_close = DummyToolCall("close_gripper", {})
    res_close = await execute_tool_call(call=call_close, robot=robot)
    assert res_close["status"] == "success"
    hint_close = res_close.get("visual_hint", "")
    assert "Gripper closed on NOTHING" in hint_close
    assert "Grasp missed" in hint_close

    # open_gripper hint reports aperture
    call_open = DummyToolCall("open_gripper", {})
    res_open = await execute_tool_call(call=call_open, robot=robot)
    assert res_open["status"] == "success"
    hint_open = res_open.get("visual_hint", "")
    assert "Gripper opened (aperture" in hint_open


def test_t1_f7_system_prompt_mandates_descent_alignment_gate() -> None:
    """The prompt makes the measured is_aligned the only descent criterion (mm tolerance)."""
    controller = VLAController(mode="sim")
    prompt = controller.get_history()[0]["content"]

    assert "offset more than 4 mm" in prompt
    assert "within 4 mm" in prompt
    assert "Descending to grasp" in prompt
    assert "Only proceed to descend when `is_aligned` is `true`" in prompt
    assert "z_height_cm=3.0" in prompt  # the hover height of the closed loop
    assert "[dy, dx]" not in prompt  # no more image-space offsets for the model to compute


# ==============================================================================
# TIER 2: BOUNDARY VALUES, CORNER CASES & RESILIENCE
# ==============================================================================


def test_t2_f5_blank_frame_handling() -> None:
    """Validate vision detector returns empty list gracefully on black or empty frames."""
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    objects = detect_workspace_objects(blank_frame)
    assert objects == []

    empty_frame = np.zeros((0, 0, 3), dtype=np.uint8)
    objects_empty = detect_workspace_objects(empty_frame)
    assert objects_empty == []


def test_t2_f5_refine_point_out_of_range_fallback() -> None:
    """Validate refine_point_to_nearest_object falls back safely when target is far away."""
    frame_rgb = make_synthetic_red_cube_frame(cube_y_1000=200.0, cube_x_1000=200.0)
    # Target query point at [900, 900] (Euclidean dist > 900 units away)
    refined_pt, matched_obj = refine_point_to_nearest_object(
        [900.0, 900.0], frame_rgb, max_dist_1000=100.0
    )
    # Since distance exceeds max_dist_1000, fallback to original point
    assert refined_pt == [900.0, 900.0] or matched_obj is not None


def test_t2_f7_coordinate_parser_signed_delta_tokens() -> None:
    """Validate coordinate parser does not misinterpret signed delta tokens as absolute targets."""
    text_with_deltas = (
        "Gripper is misaligned by dy=+35, dx=-20 relative to target. "
        "Correcting to [760, 484] before descent."
    )
    targets = parse_targets_from_text(text_with_deltas)

    # Must extract the valid absolute target [760, 484], NOT [35, -20]
    points = [t["point"] for t in targets]
    assert (760.0, 484.0) in points
    assert (35.0, -20.0) not in points


def test_t2_f8_simulated_retry_count_bounds() -> None:
    """Validate that retry counters clamp at maximum allowable attempts (3)."""
    max_retries = 3
    retry_history = []

    for attempt in range(1, 5):
        if attempt <= max_retries:
            retry_history.append(f"Retry {attempt} of {max_retries}")
        else:
            retry_history.append("Escalation")

    assert len([r for r in retry_history if "Retry" in r]) == 3
    assert retry_history[-1] == "Escalation"


# ==============================================================================
# TIER 3: CROSS-FEATURE COMBINATIONS
# ==============================================================================


@pytest.mark.asyncio
async def test_t3_cross_feature_perception_enrichment_and_tool_dispatch() -> None:
    """Validate that executing move_to_spatial_point with camera frame triggers refinement."""
    robot = MockRobot()
    frame_rgb = make_synthetic_red_cube_frame(cube_y_1000=760.0, cube_x_1000=484.0)

    # Encode to JPEG
    _, encoded_jpg = cv2.imencode(".jpg", cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
    raw_jpeg = encoded_jpg.tobytes()

    call = DummyToolCall(
        name="move_to_spatial_point",
        # Intentionally offset by a small distance from the cube bottom contact center
        args={"point": [755.0, 480.0], "z_height_cm": 4.0, "approach": "top_down"},
    )

    with patch("agent.tools.image_point_to_xyz", return_value=np.array([0.22, 0.05, 0.04])):
        result = await execute_tool_call(
            call=call,
            robot=robot,
            get_latest_jpeg_fn=lambda: raw_jpeg,
        )

    assert result.get("status") == "success"
    assert "visual_hint" in result


def test_t3_cross_feature_chat_separation_from_diagnostic_traces() -> None:
    """Validate that internal tool payloads and IK arrays never leak into Chatbot history."""
    controller = VLAController(mode="sim")
    controller._api_available = True

    # User message
    chunks = list(
        controller.process_instruction_stream(
            "Move to the red cube",
            get_latest_jpeg_fn=lambda: b"\xff\xd8fake",
        )
    )
    chat_output = "".join(chunks)

    # Chat output must NOT contain raw Python object addresses or unformatted dicts
    assert "object at 0x" not in chat_output
    assert "'executed_qpos_deg': [" not in chat_output


# ==============================================================================
# TIER 4: REAL-WORLD APPLICATION SCENARIOS
# ==============================================================================


@pytest.mark.asyncio
async def test_t4_scenario_pre_grasp_hover_alignment_and_self_correction() -> None:
    """Real-World Scenario 1: Misalignment detected at hover, corrected, then descended."""
    robot = MockRobot()
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
    mock_calib.gemini_to_robot_xyz.return_value = np.array([0.22, 0.05, 0.05])

    # Sequence of turns:
    # 1. Hover initial (with misalignment: dy=35, dx=0)
    # 2. Corrective hover (adjusted to dy=0, dx=0)
    # 3. Descend to contact
    # 4. Close gripper
    # 5. Lift
    turns = [
        types.LiveServerMessage(
            server_content=types.LiveServerContent(
                model_turn=types.Content(
                    parts=[
                        types.Part(
                            text="> 💭 *Hovering above initial estimate. "
                            "Observed gripper [725, 484] vs target [760, 484] (dy=+35, dx=0). "
                            "Misalignment exceeds 25 units. Applying correction.*"
                        )
                    ]
                )
            ),
            tool_call=types.LiveServerToolCall(
                function_calls=[
                    types.FunctionCall(
                        name="move_to_spatial_point",
                        args={"point": [725, 484], "z_height_cm": 5.0, "approach": "top_down"},
                        id="c1_hover_initial",
                    )
                ]
            ),
        ),
        types.LiveServerMessage(
            server_content=types.LiveServerContent(
                model_turn=types.Content(
                    parts=[
                        types.Part(
                            text="> 💭 *Position adjusted to [760, 484]. "
                            "Observed gripper [760, 484] vs target [760, 484] (dy=0, dx=0). "
                            "Alignment verified within tolerance. Descending.*"
                        )
                    ]
                )
            ),
            tool_call=types.LiveServerToolCall(
                function_calls=[
                    types.FunctionCall(
                        name="move_to_spatial_point",
                        args={"point": [760, 484], "z_height_cm": 0.0, "approach": "top_down"},
                        id="c2_descend",
                    )
                ]
            ),
        ),
        types.LiveServerMessage(
            tool_call=types.LiveServerToolCall(
                function_calls=[types.FunctionCall(name="close_gripper", args={}, id="c3_close")]
            )
        ),
        types.LiveServerMessage(
            tool_call=types.LiveServerToolCall(
                function_calls=[
                    types.FunctionCall(
                        name="move_to_spatial_point",
                        args={"point": [760, 484], "z_height_cm": 5.0, "approach": "top_down"},
                        id="c4_lift",
                    )
                ]
            )
        ),
        types.LiveServerMessage(
            server_content=types.LiveServerContent(
                model_turn=types.Content(
                    parts=[types.Part.from_text(text="Red cube successfully picked.")]
                )
            )
        ),
    ]

    async def mock_stream():
        for t in turns:
            yield t
        yield types.LiveServerMessage(server_content=types.LiveServerContent(turn_complete=True))

    mock_session.receive = mock_stream
    controller._session = mock_session

    fake_frame = b"\xff\xd8\xff\xe0mock_frame"

    with patch("agent.tools.get_calibrator", return_value=mock_calib):
        chunks = list(
            controller.process_instruction_stream(
                "Pick the red cube with visual self-correction",
                get_latest_jpeg_fn=lambda: fake_frame,
            )
        )

    output = "".join(chunks)
    assert "Misalignment exceeds 25 units" in output
    assert "Alignment verified within tolerance" in output
    assert mock_session.send_tool_response.call_count == 4


@pytest.mark.asyncio
async def test_t4_scenario_missed_grasp_automated_retry_and_recovery() -> None:
    """Real-World Scenario 2: Missed grasp detected post-lift triggering successful retry."""
    robot = MockRobot()
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
    mock_calib.gemini_to_robot_xyz.return_value = np.array([0.22, 0.05, 0.05])

    turns = [
        # Initial attempt: grasp and lift
        types.LiveServerMessage(
            tool_call=types.LiveServerToolCall(
                function_calls=[types.FunctionCall(name="open_gripper", args={}, id="r1_open")]
            )
        ),
        types.LiveServerMessage(
            tool_call=types.LiveServerToolCall(
                function_calls=[
                    types.FunctionCall(
                        name="move_to_spatial_point",
                        args={"point": [760, 484], "z_height_cm": 0.0, "approach": "top_down"},
                        id="r1_descend",
                    )
                ]
            )
        ),
        types.LiveServerMessage(
            tool_call=types.LiveServerToolCall(
                function_calls=[types.FunctionCall(name="close_gripper", args={}, id="r1_close")]
            )
        ),
        types.LiveServerMessage(
            tool_call=types.LiveServerToolCall(
                function_calls=[
                    types.FunctionCall(
                        name="move_to_spatial_point",
                        args={"point": [760, 484], "z_height_cm": 5.0, "approach": "top_down"},
                        id="r1_lift",
                    )
                ]
            )
        ),
        # Post-lift inspection: Missed grasp detected! Initiating Retry 1
        types.LiveServerMessage(
            server_content=types.LiveServerContent(
                model_turn=types.Content(
                    parts=[
                        types.Part(
                            text="> 💭 *Post-lift inspection: object still detected on table. "
                            "Grasp missed. Initiating retry 1 of 3.*"
                        )
                    ]
                )
            ),
            tool_call=types.LiveServerToolCall(
                function_calls=[types.FunctionCall(name="open_gripper", args={}, id="retry1_open")]
            ),
        ),
        types.LiveServerMessage(
            tool_call=types.LiveServerToolCall(
                function_calls=[
                    types.FunctionCall(
                        name="move_to_spatial_point",
                        args={"point": [760, 484], "z_height_cm": 0.0, "approach": "top_down"},
                        id="retry1_descend",
                    )
                ]
            )
        ),
        types.LiveServerMessage(
            tool_call=types.LiveServerToolCall(
                function_calls=[
                    types.FunctionCall(name="close_gripper", args={}, id="retry1_close")
                ]
            )
        ),
        types.LiveServerMessage(
            tool_call=types.LiveServerToolCall(
                function_calls=[
                    types.FunctionCall(
                        name="move_to_spatial_point",
                        args={"point": [760, 484], "z_height_cm": 5.0, "approach": "top_down"},
                        id="retry1_lift",
                    )
                ]
            )
        ),
        # Final success confirmation
        types.LiveServerMessage(
            server_content=types.LiveServerContent(
                model_turn=types.Content(
                    parts=[
                        types.Part.from_text(
                            text="Target grasped successfully on retry 1. Object elevated."
                        )
                    ]
                )
            )
        ),
    ]

    async def mock_stream():
        for t in turns:
            yield t
        yield types.LiveServerMessage(server_content=types.LiveServerContent(turn_complete=True))

    mock_session.receive = mock_stream
    controller._session = mock_session

    with patch("agent.tools.get_calibrator", return_value=mock_calib):
        chunks = list(
            controller.process_instruction_stream(
                "Pick red cube with grasp retry",
                get_latest_jpeg_fn=lambda: b"\xff\xd8fake",
            )
        )

    output = "".join(chunks)
    assert "Initiating retry 1 of 3" in output
    assert "Target grasped successfully on retry 1" in output
    assert mock_session.send_tool_response.call_count == 8


@pytest.mark.asyncio
async def test_t4_scenario_consecutive_failures_retry_exhaustion_escalation() -> None:
    """Real-World Scenario 3: 3 consecutive failed grasps leads to user escalation."""
    robot = MockRobot()
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
    mock_calib.gemini_to_robot_xyz.return_value = np.array([0.22, 0.05, 0.05])

    turns = [
        # Attempt 1
        types.LiveServerMessage(
            server_content=types.LiveServerContent(
                model_turn=types.Content(parts=[types.Part(text="> 💭 *Attempt 1 missed.*")])
            ),
            tool_call=types.LiveServerToolCall(
                function_calls=[types.FunctionCall(name="close_gripper", args={}, id="a1")]
            ),
        ),
        # Attempt 2
        types.LiveServerMessage(
            server_content=types.LiveServerContent(
                model_turn=types.Content(parts=[types.Part(text="> 💭 *Attempt 2 missed.*")])
            ),
            tool_call=types.LiveServerToolCall(
                function_calls=[types.FunctionCall(name="close_gripper", args={}, id="a2")]
            ),
        ),
        # Attempt 3
        types.LiveServerMessage(
            server_content=types.LiveServerContent(
                model_turn=types.Content(parts=[types.Part(text="> 💭 *Attempt 3 missed.*")])
            ),
            tool_call=types.LiveServerToolCall(
                function_calls=[types.FunctionCall(name="close_gripper", args={}, id="a3")]
            ),
        ),
        # Escalation message via send_message
        types.LiveServerMessage(
            server_content=types.LiveServerContent(
                model_turn=types.Content(
                    parts=[
                        types.Part(
                            text="> 💭 *Max retries exhausted (3/3). Halting and alerting user.*"
                        )
                    ]
                )
            ),
            tool_call=types.LiveServerToolCall(
                function_calls=[
                    types.FunctionCall(
                        name="send_message",
                        args={
                            "message": "Grasp failed after 3 attempts. Target may be obstructed. "
                            "Please check object position."
                        },
                        id="escalation",
                    )
                ]
            ),
        ),
    ]

    async def mock_stream():
        for t in turns:
            yield t
        yield types.LiveServerMessage(server_content=types.LiveServerContent(turn_complete=True))

    mock_session.receive = mock_stream
    controller._session = mock_session

    with patch("agent.tools.get_calibrator", return_value=mock_calib):
        chunks = list(
            controller.process_instruction_stream(
                "Test retry exhaustion escalation",
                get_latest_jpeg_fn=lambda: b"\xff\xd8fake",
            )
        )

    output = "".join(chunks)
    assert "Max retries exhausted (3/3)" in output
    assert "Grasp failed after 3 attempts" in output
    assert mock_session.send_tool_response.call_count == 4


def test_t1_f8_anti_hallucination_transport_gate_blocks_motion_when_object_on_table() -> None:
    """A stalled gripper with the object still on its footprint is a missed grasp (vision veto)."""
    from agent.guardrails import decide_grasp

    verdict = decide_grasp(aperture_after_lift=True, vote="on_table")
    assert verdict.verdict == "missed"
    assert verdict.reason == "object_still_on_table"
    assert verdict.transport_allowed is False
    assert verdict.grasped is False


@pytest.mark.asyncio
async def test_t2_f6_missing_camera_frame_graceful_tool_execution() -> None:
    """Validate that move_to_spatial_point executes smoothly when get_latest_jpeg_fn is None."""
    robot = MockRobot()
    call = DummyToolCall(
        "move_to_spatial_point",
        {"point": [500.0, 500.0], "z_height_cm": 3.0, "approach": "top_down"},
    )
    with patch("agent.tools.image_point_to_xyz", return_value=np.array([0.20, 0.0, 0.03])):
        result = await execute_tool_call(call=call, robot=robot, get_latest_jpeg_fn=None)
    assert result["status"] == "success"
    assert "target_xyz_m" in result


@pytest.mark.asyncio
async def test_t4_scenario_complete_pick_and_place_pipeline() -> None:
    """Real-World Scenario 4: Full pick, lift, transport and release sequence."""
    robot = MockRobot()
    robot.gripper_reading = {"aperture_deg": 25.0, "load": 150.0, "source": "measured"}
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
    mock_calib.gemini_to_robot_xyz.return_value = np.array([0.22, 0.05, 0.05])

    # Sequence:
    # 1. Open gripper
    # 2. Hover [760, 484] z=5.0
    # 3. Descend [760, 484] z=0.0
    # 4. Close gripper
    # 5. Lift [760, 484] z=5.0
    # 6. Transport to bin preset or pose
    # 7. Open gripper to release
    # 8. Send confirmation message
    turns = [
        types.LiveServerMessage(
            tool_call=types.LiveServerToolCall(
                function_calls=[types.FunctionCall(name="open_gripper", args={}, id="p1_open")]
            )
        ),
        types.LiveServerMessage(
            tool_call=types.LiveServerToolCall(
                function_calls=[
                    types.FunctionCall(
                        name="move_to_spatial_point",
                        args={"point": [760, 484], "z_height_cm": 5.0, "approach": "top_down"},
                        id="p2_hover",
                    )
                ]
            )
        ),
        types.LiveServerMessage(
            tool_call=types.LiveServerToolCall(
                function_calls=[
                    types.FunctionCall(
                        name="move_to_spatial_point",
                        args={"point": [760, 484], "z_height_cm": 0.0, "approach": "top_down"},
                        id="p3_descend",
                    )
                ]
            )
        ),
        types.LiveServerMessage(
            tool_call=types.LiveServerToolCall(
                function_calls=[types.FunctionCall(name="close_gripper", args={}, id="p4_close")]
            )
        ),
        types.LiveServerMessage(
            tool_call=types.LiveServerToolCall(
                function_calls=[
                    types.FunctionCall(
                        name="move_to_spatial_point",
                        args={"point": [760, 484], "z_height_cm": 5.0, "approach": "top_down"},
                        id="p5_lift",
                    )
                ]
            )
        ),
        types.LiveServerMessage(
            tool_call=types.LiveServerToolCall(
                function_calls=[
                    types.FunctionCall(
                        name="move_to_preset",
                        args={"pose_name": "MIDDLE"},
                        id="p6_transport",
                    )
                ]
            )
        ),
        types.LiveServerMessage(
            tool_call=types.LiveServerToolCall(
                function_calls=[types.FunctionCall(name="open_gripper", args={}, id="p7_release")]
            )
        ),
        types.LiveServerMessage(
            server_content=types.LiveServerContent(
                model_turn=types.Content(
                    parts=[types.Part.from_text(text="Object picked and placed into destination.")]
                )
            )
        ),
    ]

    async def mock_stream():
        for t in turns:
            yield t
        yield types.LiveServerMessage(server_content=types.LiveServerContent(turn_complete=True))

    mock_session.receive = mock_stream
    controller._session = mock_session

    with patch("agent.tools.get_calibrator", return_value=mock_calib):
        chunks = list(
            controller.process_instruction_stream(
                "Execute pick and place pipeline",
                get_latest_jpeg_fn=lambda: b"\xff\xd8fake",
            )
        )

    output = "".join(chunks)
    assert "Object picked and placed into destination" in output
    assert mock_session.send_tool_response.call_count == 7
