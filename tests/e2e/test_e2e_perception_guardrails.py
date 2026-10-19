"""End-to-end test suite for perception feedback & anti-hallucination guardrails.

Covers:
- F5: Programmatic Perception Metrics Engine ([dy, dx], distance, is_aligned)
- F6: Tool Execution Feedback Enrichment (hover hints, grasp verification)
- F7: Anti-Hallucination Descent Gate
- F8: Anti-Hallucination Transport Gate & Retry Loop (3-attempt state machine & escalation)
- F9: Clean Diagnostic Logging & UI Separation (ISO-8601 millisecond logs, stdout/stderr routing)
"""

from unittest.mock import AsyncMock, MagicMock, patch

import cv2
import numpy as np
import pytest
from google.genai import types

from agent.tools import execute_tool_call
from agent.vla_controller import VLAController
from robot.settings import Settings, set_settings
from tests.conftest import DummyToolCall, MockRobot


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
# TIER 1: ISOLATED FEATURE COVERAGE (F6, F8)
# ==============================================================================


@pytest.mark.asyncio
async def test_t1_f6_tool_visual_hint_hover_vs_contact() -> None:
    """Validate move_to_spatial_point returns appropriate visual verification hint."""
    robot = MockRobot()

    # Case 1: Hover height (z_height_cm = 4.0 >= 3.0)
    call_hover = DummyToolCall(
        "move_to_spatial_point",
        {"point": [760.0, 484.0], "z_height_cm": 4.0},
    )
    res_hover = await execute_tool_call(call=call_hover, robot=robot)
    assert res_hover["status"] == "success"
    hint = res_hover.get("visual_hint", "")
    assert "Visual Verification Loop" in hint
    assert "alignment not measured" in hint  # no frame provider: the hint does not pretend

    # Case 2: Contact height (z_height_cm = 0.0 < 3.0)
    call_contact = DummyToolCall(
        "move_to_spatial_point",
        {"point": [760.0, 484.0], "z_height_cm": 0.0},
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


# ==============================================================================
# TIER 3: CROSS-FEATURE COMBINATIONS
# ==============================================================================


@pytest.mark.asyncio
async def test_t3_cross_feature_perception_enrichment_and_tool_dispatch() -> None:
    """Validate that executing move_to_spatial_point with camera frame triggers refinement."""
    from simulation.sim_calibration import init_sim_workspace_calibration
    from vision.calibrator import get_calibrator
    from vision.detector import refine_point_to_nearest_object

    calib = get_calibrator(mode="sim")
    init_sim_workspace_calibration(calibrator=calib)

    robot = MockRobot()
    frame_rgb = make_synthetic_red_cube_frame(cube_y_1000=760.0, cube_x_1000=484.0)

    # Encode to JPEG and decode identically to _decode_frame in agent/tools.py
    _, encoded_jpg = cv2.imencode(".jpg", cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
    raw_jpeg = encoded_jpg.tobytes()
    decoded_rgb = cv2.cvtColor(
        cv2.imdecode(np.frombuffer(raw_jpeg, np.uint8), cv2.IMREAD_COLOR),
        cv2.COLOR_BGR2RGB,
    )

    expected_pt, matched_obj = refine_point_to_nearest_object(
        [755.0, 480.0], decoded_rgb, calibrator=calib
    )
    assert matched_obj is not None
    assert matched_obj["color"] == "red"
    assert abs(expected_pt[0] - 760.0) <= 25.0
    assert abs(expected_pt[1] - 484.0) <= 25.0

    call = DummyToolCall(
        name="move_to_spatial_point",
        # Intentionally offset by a small distance from the cube bottom contact center
        args={"point": [755.0, 480.0], "z_height_cm": 4.0},
    )

    with patch("agent.tools.image_point_to_xyz", return_value=np.array([0.22, 0.05, 0.04])):
        result = await execute_tool_call(
            call=call,
            robot=robot,
            mode="sim",
            get_latest_jpeg_fn=lambda: raw_jpeg,
        )

    assert result.get("status") == "success"
    assert "visual_hint" in result
    assert result.get("target_xy_mm") == pytest.approx(list(matched_obj["centre_xy_mm"]), abs=0.5)


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
                        args={"point": [725, 484], "z_height_cm": 5.0},
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
                        args={"point": [760, 484], "z_height_cm": 0.0},
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
                        args={"point": [760, 484], "z_height_cm": 5.0},
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

    with patch("vision.calibrator.get_calibrator", return_value=mock_calib):
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
                        args={"point": [760, 484], "z_height_cm": 0.0},
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
                        args={"point": [760, 484], "z_height_cm": 5.0},
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
                        args={"point": [760, 484], "z_height_cm": 0.0},
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
                        args={"point": [760, 484], "z_height_cm": 5.0},
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

    with patch("vision.calibrator.get_calibrator", return_value=mock_calib):
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
        # Escalation message in plain text
        types.LiveServerMessage(
            server_content=types.LiveServerContent(
                model_turn=types.Content(
                    parts=[
                        types.Part(
                            text="> 💭 *Max retries exhausted (3/3). Halting and alerting user.*",
                            thought=True,
                        ),
                        types.Part.from_text(
                            text="Grasp failed after 3 attempts. Target may be obstructed. "
                            "Please check object position."
                        ),
                    ]
                )
            ),
        ),
    ]

    async def mock_stream():
        for t in turns:
            yield t
        yield types.LiveServerMessage(server_content=types.LiveServerContent(turn_complete=True))

    mock_session.receive = mock_stream
    controller._session = mock_session

    with patch("vision.calibrator.get_calibrator", return_value=mock_calib):
        chunks = list(
            controller.process_instruction_stream(
                "Test retry exhaustion escalation",
                get_latest_jpeg_fn=lambda: b"\xff\xd8fake",
            )
        )

    output = "".join(chunks)
    assert "Max retries exhausted (3/3)" in output
    assert "Grasp failed after 3 attempts" in output
    assert mock_session.send_tool_response.call_count == 3


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
        {"point": [500.0, 500.0], "z_height_cm": 3.0},
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
                        args={"point": [760, 484], "z_height_cm": 5.0},
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
                        args={"point": [760, 484], "z_height_cm": 0.0},
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
                        args={"point": [760, 484], "z_height_cm": 5.0},
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

    with patch("vision.calibrator.get_calibrator", return_value=mock_calib):
        chunks = list(
            controller.process_instruction_stream(
                "Execute pick and place pipeline",
                get_latest_jpeg_fn=lambda: b"\xff\xd8fake",
            )
        )

    output = "".join(chunks)
    assert "Object picked and placed into destination" in output
    assert mock_session.send_tool_response.call_count == 7
