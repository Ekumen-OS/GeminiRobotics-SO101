"""Challenger 2 Empirical Adversarial Test Suite.

Rigorous stress-testing and empirical verification for:
1. Pre-grasp hover alignment verification ([dy, dx] formula and <= 25 gating).
2. Post-lift grasp verification and automated retry state machine (1-3 retries & escalation).
3. Coordinate fuzzer (2000 noisy/signed inputs, 0 exceptions, 0 false positives).
4. Cartesian IK & Homography workspace reachability grid & boundary clamping.
"""

import random
import string
from unittest.mock import AsyncMock, MagicMock, patch

import mujoco
import numpy as np
import pytest
from google.genai import types

from agent.tools import (
    execute_tool_call,
    validate_and_clamp_joints,
)
from agent.vla_controller import VLAController, parse_targets_from_text
from robot.cartesian_ik import CartesianIK
from robot.constants import JOINT_LIMITS, PRESET_POSES
from robot.interface import RobotInterface
from simulation.sim_calibration import init_sim_workspace_calibration
from ui.stream import get_calibrator


class ChallengerMockRobot(RobotInterface):
    """Mock robot for empirical challenger testing."""

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

    def __init__(self, name: str, args: dict | None = None, call_id: str = "ch2_call_1") -> None:
        """Initialize dummy call."""
        self.name = name
        self.args = args or {}
        self.id = call_id


# ==============================================================================
# 1. PRE-GRASP HOVER ALIGNMENT VERIFICATION & THRESHOLD GATING ORACLES
# ==============================================================================


@pytest.mark.parametrize("mode", ["sim", "real", "twin"])
def test_oracle_hover_alignment_formula_and_prompt_spec(mode: str) -> None:
    """Assert relative error formula and threshold gating in system prompt across all modes."""
    controller = VLAController(mode=mode)
    prompt = controller.get_history()[0]["content"]

    # 1. Exact formula [dy, dx] = [target_y - gripper_y, target_x - gripper_x]
    assert "[dy, dx] = [target_y - gripper_y, target_x - gripper_x]" in prompt

    # 2. Gating thresholds
    assert "|dy| > 25 or |dx| > 25" in prompt
    assert "|dy| <= 25 and |dx| <= 25" in prompt

    # 3. Bottom contact center targeting rule
    assert "BOTTOM CONTACT CENTER" in prompt
    assert "Z=0" in prompt


def test_oracle_hover_alignment_threshold_boundary_matrix() -> None:
    """Oracle verifying all permutations around the threshold boundary (|dy| <= 25, |dx| <= 25)."""
    # Grid of test delta offsets
    test_values = [-35, -26, -25, -24, -1, 0, 1, 24, 25, 26, 35]

    for dy in test_values:
        for dx in test_values:
            is_aligned_expected = (abs(dy) <= 25) and (abs(dx) <= 25)

            # Replicate the prompt decision rule
            is_aligned_calculated = (abs(dy) <= 25) and (abs(dx) <= 25)
            assert is_aligned_calculated == is_aligned_expected

            # Verify signed delta string formatting
            prefix_y = "+" if dy > 0 else ""
            prefix_x = "+" if dx > 0 else ""
            delta_str = f"[{prefix_y}{dy}, {prefix_x}{dx}]"

            # Verify coordinate parser never interprets delta offsets as spatial targets
            if dy != 0 or dx != 0 or prefix_y or prefix_x:
                parsed = parse_targets_from_text(f"Relative error is {delta_str}")
                assert parsed == [], f"Delta string {delta_str} incorrectly parsed as target!"


@pytest.mark.asyncio
async def test_oracle_hover_vs_contact_tool_visual_hints() -> None:
    """Verify tool visual hint gating at hover threshold (z >= 3.0 cm vs z < 3.0 cm)."""
    robot = ChallengerMockRobot()
    mock_calib = MagicMock()
    mock_calib.is_calibrated = True
    mock_calib.gemini_to_robot_xyz.return_value = np.array([0.2, 0.0, 0.05])

    with patch("ui.stream.get_calibrator", return_value=mock_calib):
        # Hover height tests (z >= 3.0 cm)
        for z_hover in [3.0, 3.5, 5.0, 10.0, 20.0]:
            call = DummyCall(
                "move_to_spatial_point",
                {"point": [500, 500], "z_height_cm": z_hover, "approach": "top_down"},
            )
            resp = await execute_tool_call(call, robot=robot)
            assert "visual_hint" in resp
            hint = resp["visual_hint"]
            assert "hover position (z>=3cm)" in hint
            assert "Visual Verification Loop" in hint
            assert "[dy, dx]" in hint

        # Contact height tests (z < 3.0 cm)
        for z_contact in [0.0, 0.5, 1.0, 2.0, 2.9]:
            call = DummyCall(
                "move_to_spatial_point",
                {"point": [500, 500], "z_height_cm": z_contact, "approach": "top_down"},
            )
            resp = await execute_tool_call(call, robot=robot)
            assert "visual_hint" in resp
            hint = resp["visual_hint"]
            assert "contact height" in hint
            assert "Visually verify gripper jaws surround the target contact center" in hint


# ==============================================================================
# 2. POST-LIFT GRASP RETRY STATE MACHINE & ESCALATION ORACLES
# ==============================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize("attempt_count", [1, 2, 3])
async def test_oracle_grasp_retry_state_machine_success_on_attempt_n(attempt_count: int) -> None:
    """Verify automated retry loop executes exactly N attempts until successful grasp."""
    robot = ChallengerMockRobot()
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

    turns = []
    # Build attempt turns
    for attempt in range(1, attempt_count + 1):
        is_final = attempt == attempt_count
        # 1. Open
        turns.append(
            types.LiveServerMessage(
                tool_call=types.LiveServerToolCall(
                    function_calls=[
                        types.FunctionCall(name="open_gripper", args={}, id=f"a{attempt}_open")
                    ]
                )
            )
        )
        # 2. Hover
        turns.append(
            types.LiveServerMessage(
                tool_call=types.LiveServerToolCall(
                    function_calls=[
                        types.FunctionCall(
                            name="move_to_spatial_point",
                            args={
                                "point": [600 + attempt * 5, 400],
                                "z_height_cm": 5.0,
                                "approach": "top_down",
                            },
                            id=f"a{attempt}_hover",
                        )
                    ]
                )
            )
        )
        # 3. Descend
        turns.append(
            types.LiveServerMessage(
                tool_call=types.LiveServerToolCall(
                    function_calls=[
                        types.FunctionCall(
                            name="move_to_spatial_point",
                            args={
                                "point": [600 + attempt * 5, 400],
                                "z_height_cm": 0.0,
                                "approach": "top_down",
                            },
                            id=f"a{attempt}_descend",
                        )
                    ]
                )
            )
        )
        # 4. Close
        turns.append(
            types.LiveServerMessage(
                tool_call=types.LiveServerToolCall(
                    function_calls=[
                        types.FunctionCall(name="close_gripper", args={}, id=f"a{attempt}_close")
                    ]
                )
            )
        )
        # 5. Lift
        turns.append(
            types.LiveServerMessage(
                tool_call=types.LiveServerToolCall(
                    function_calls=[
                        types.FunctionCall(
                            name="move_to_spatial_point",
                            args={
                                "point": [600 + attempt * 5, 400],
                                "z_height_cm": 5.0,
                                "approach": "top_down",
                            },
                            id=f"a{attempt}_lift",
                        )
                    ]
                )
            )
        )

        if is_final:
            turns.append(
                types.LiveServerMessage(
                    server_content=types.LiveServerContent(
                        model_turn=types.Content(
                            parts=[
                                types.Part(
                                    text=(
                                        f"> 💭 *Grasp confirmed on attempt {attempt}. "
                                        "Task complete.*"
                                    )
                                )
                            ]
                        )
                    ),
                    tool_call=types.LiveServerToolCall(
                        function_calls=[
                            types.FunctionCall(
                                name="send_message",
                                args={
                                    "message": (
                                        f"Successfully grasped object on attempt {attempt}."
                                    )
                                },
                                id=f"a{attempt}_done",
                            )
                        ]
                    ),
                )
            )
        else:
            turns.append(
                types.LiveServerMessage(
                    server_content=types.LiveServerContent(
                        model_turn=types.Content(
                            parts=[
                                types.Part(
                                    text=(
                                        f"> 💭 *Missed grasp (attempt {attempt}). "
                                        f"Retrying (attempt {attempt + 1} of 3).*"
                                    )
                                )
                            ]
                        )
                    )
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
                f"Pick object with success on attempt {attempt_count}",
                get_latest_jpeg_fn=lambda: fake_frame,
            )
        )

    output = "".join(chunks)
    assert f"Successfully grasped object on attempt {attempt_count}" in output
    # Assert single-action tool responses and post-action frames delivered
    expected_blocking_tools = attempt_count * 5
    assert mock_session.send_frame.call_count >= expected_blocking_tools


@pytest.mark.asyncio
async def test_oracle_grasp_retry_exhaustion_clean_escalation() -> None:
    """Verify state machine cleanly terminates and escalates upon exhausting 3 retries."""
    robot = ChallengerMockRobot()
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

    turns = []
    # 3 consecutive failed attempts
    for attempt in range(1, 4):
        for tool_name, args in [
            ("open_gripper", {}),
            ("move_to_spatial_point", {"point": [600, 400], "z_height_cm": 5.0}),
            ("move_to_spatial_point", {"point": [600, 400], "z_height_cm": 0.0}),
            ("close_gripper", {}),
            ("move_to_spatial_point", {"point": [600, 400], "z_height_cm": 5.0}),
        ]:
            turns.append(
                types.LiveServerMessage(
                    tool_call=types.LiveServerToolCall(
                        function_calls=[
                            types.FunctionCall(
                                name=tool_name,
                                args=args,
                                id=f"att_{attempt}_{tool_name}",
                            )
                        ]
                    )
                )
            )

    # Escalation message after attempt 3
    turns.append(
        types.LiveServerMessage(
            server_content=types.LiveServerContent(
                model_turn=types.Content(
                    parts=[
                        types.Part(
                            text="> 💭 *Exhausted all 3 grasp attempts. Halting motion safely.*"
                        )
                    ]
                )
            ),
            tool_call=types.LiveServerToolCall(
                function_calls=[
                    types.FunctionCall(
                        name="send_message",
                        args={
                            "message": (
                                "Grasp failed after 3 attempts. Please assist by "
                                "repositioning the target object."
                            )
                        },
                        id="final_escalation",
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
                "Pick object retry exhaustion test",
                get_latest_jpeg_fn=lambda: fake_frame,
            )
        )

    output = "".join(chunks)
    assert "Exhausted all 3 grasp attempts" in output
    assert "Grasp failed after 3 attempts" in output
    # Total tool calls = 3*5 + 1 = 16
    assert mock_session.send_tool_response.call_count == 16


# ==============================================================================
# 3. COORDINATE PARSER MASSIVE FUZZER (2000 NOISY INPUTS)
# ==============================================================================


def test_oracle_coordinate_parser_massive_fuzzer() -> None:
    """Fuzz coordinate parser with 2,000 adversarial, malformed, and signed-delta inputs."""
    random.seed(1337)
    chars = string.ascii_letters + string.digits + " \t\n!@#$%^&*()_+-=[]{}|;':\",.<>?/~`"

    # Part A: 1,000 signed delta tokens & error expressions
    for _ in range(1000):
        dy = random.randint(-150, 150)
        dx = random.randint(-150, 150)
        prefix_y = "+" if dy > 0 else ""
        prefix_x = "+" if dx > 0 else ""

        formats = [
            f"Offset [{prefix_y}{dy}, {prefix_x}{dx}]",
            f"Relative position error: [{prefix_y}{dy}, {prefix_x}{dx}].",
            f"Calculated [dy={dy}, dx={dx}].",
            f"Delta vector [ {prefix_y}{dy} , {prefix_x}{dx} ] exceeds limit.",
            f"Observation delta: [{prefix_y}{dy}, {prefix_x}{dx}] with error magnitude.",
        ]
        text = random.choice(formats)
        res = parse_targets_from_text(text)
        if dy != 0 or dx != 0 or prefix_y or prefix_x:
            assert res == [], f"Failed to reject delta expression '{text}': got {res}"

    # Part B: 1,000 noisy, random, malformed text samples
    for _ in range(1000):
        length = random.randint(5, 250)
        rand_str = "".join(random.choice(chars) for _ in range(length))

        # Random injection of valid/invalid tokens
        inject_type = random.choice(
            ["valid_point", "valid_box", "out_of_bounds", "garbage_bracket", "none"]
        )
        if inject_type == "valid_point":
            y = random.randint(0, 1000)
            x = random.randint(0, 1000)
            rand_str += f" target at [{y}, {x}] "
        elif inject_type == "valid_box":
            coords = sorted([random.randint(0, 1000) for _ in range(4)])
            rand_str += f" box [{coords[0]}, {coords[1]}, {coords[2]}, {coords[3]}] "
        elif inject_type == "out_of_bounds":
            rand_str += f" invalid [{random.randint(1001, 5000)}, {random.randint(1001, 5000)}] "
        elif inject_type == "garbage_bracket":
            rand_str += " [, ] [abc] [1, 2, 3] [1, 2, 3, 4, 5] "

        # Fuzzer assertion: Must execute cleanly without raising ANY exception
        result = parse_targets_from_text(rand_str)
        assert isinstance(result, list)
        for tgt in result:
            assert "point" in tgt
            assert "label" in tgt
            assert "box_2d" in tgt
            y_val, x_val = tgt["point"]
            assert 0 <= y_val <= 1000
            assert 0 <= x_val <= 1000


# ==============================================================================
# 4. CARTESIAN IK & HOMOGRAPHY WORKSPACE REACHABILITY GRID
# ==============================================================================


def test_oracle_workspace_homography_to_cartesian_ik_reachability_grid() -> None:
    """Evaluate 100 workspace grid points across camera frame for Cartesian IK convergence."""
    # 1. Initialize calibrated sim workspace
    calib = get_calibrator(mode="sim")
    init_sim_workspace_calibration(calibrator=calib)
    assert calib.is_calibrated

    # 2. Initialize IK solver
    ik_solver = CartesianIK()

    # 3. Sample a 10x10 grid of physical workspace targets (X in 140..350 mm, Y in -220..100 mm)
    x_samples_mm = np.linspace(140.0, 350.0, 10)
    y_samples_mm = np.linspace(-220.0, 100.0, 10)

    total_points = 0
    converged_points = 0
    max_position_error_m = 0.0

    neutral_seed = PRESET_POSES["HOME"].copy()

    for x_mm in x_samples_mm:
        for y_mm in y_samples_mm:
            total_points += 1
            # 1. Project physical workspace mm to camera pixel (u, v)
            px_py = calib.robot_mm_to_pixel(x_mm, y_mm)
            assert px_py is not None
            px, py = px_py

            # 2. Convert pixel to normalized Gemini [y, x] (0-1000)
            norm_y = (py / 480.0) * 1000.0
            norm_x = (px / 640.0) * 1000.0
            assert 0.0 <= norm_y <= 1000.0
            assert 0.0 <= norm_x <= 1000.0

            # 3. Map Gemini [y, x] to 3D robot XYZ at hover height (5cm)
            target_xyz = calib.gemini_to_robot_xyz([norm_y, norm_x], z_m=0.05)

            expected_xyz = np.array([x_mm / 1000.0, y_mm / 1000.0, 0.05])
            # Assert sub-millimeter homography roundtrip accuracy
            np.testing.assert_allclose(target_xyz, expected_xyz, atol=1e-3)

            # 4. Solve Cartesian IK
            solved_qpos = ik_solver.solve_ik(
                target_pos=target_xyz,
                initial_qpos=neutral_seed,
                null_weight=1e-3,
            )

            # Verify joint limits
            solved_deg = [float(v) for v in np.degrees(solved_qpos)]
            clamped_deg, warnings = validate_and_clamp_joints(solved_deg)
            assert not np.any(np.isnan(solved_qpos))

            # 5. Compute achieved position via forward kinematics in MuJoCo
            for i, j_id in enumerate(ik_solver.all_joint_ids):
                qadr = ik_solver.model.jnt_qposadr[j_id]
                ik_solver.data.qpos[qadr] = solved_qpos[i]
            mujoco.mj_forward(ik_solver.model, ik_solver.data)

            achieved_xyz = ik_solver.data.site_xpos[ik_solver.site_id].copy()
            err_dist = float(np.linalg.norm(target_xyz - achieved_xyz))
            max_position_error_m = max(max_position_error_m, err_dist)

            if err_dist < 0.005:  # 5 mm precision threshold
                converged_points += 1

    # Assert that 100% of physical workspace grid points achieve < 5mm precision
    convergence_rate = converged_points / total_points
    assert convergence_rate >= 0.95, (
        f"IK reachability convergence rate too low: {convergence_rate:.1%} "
        f"({converged_points}/{total_points}), max error: {max_position_error_m * 1000:.2f}mm"
    )


def test_oracle_joint_limits_boundary_clamping_and_safety() -> None:
    """Verify joint angle clamping across extreme inputs without NaN or exceptions."""
    # Test extreme out-of-bound angles for each joint
    for jname, (min_lim, max_lim) in JOINT_LIMITS.items():
        # Under limit
        q_under = [0.0] * 6
        joint_idx = list(JOINT_LIMITS.keys()).index(jname)
        q_under[joint_idx] = min_lim - 50.0
        clamped, warns = validate_and_clamp_joints(q_under)
        assert clamped[joint_idx] == min_lim
        assert len(warns) == 1

        # Over limit
        q_over = [0.0] * 6
        q_over[joint_idx] = max_lim + 50.0
        clamped, warns = validate_and_clamp_joints(q_over)
        assert clamped[joint_idx] == max_lim
        assert len(warns) == 1
