"""Robot tool declarations and execution adapters for Gemini Live API and SPAV pipeline."""

import asyncio
from collections.abc import Callable
from typing import Any

import numpy as np

from robot.cartesian_ik import CartesianIK
from robot.constants import JOINT_LIMITS, PRESET_POSES
from robot.interface import RobotInterface

# Gripper limits derived from JOINT_LIMITS in degrees
GRIPPER_OPEN_DEG = JOINT_LIMITS["gripper"][1]
GRIPPER_CLOSE_DEG = JOINT_LIMITS["gripper"][0]

ROBOT_TOOL_DECLARATIONS = [
    {
        "name": "move_to_pose",
        "description": "Smoothly interpolate motion to a 6-DOF joint target.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "qpos": {
                    "type": "ARRAY",
                    "description": "Target joint angles in degrees for all 6 joints.",
                    "items": {"type": "NUMBER"},
                },
                "duration_sec": {
                    "type": "NUMBER",
                    "description": "Duration of the movement in seconds.",
                },
            },
            "required": ["qpos", "duration_sec"],
        },
    },
    {
        "name": "set_joint_positions",
        "description": "Instant or snap joint position command to 6-DOF joint target.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "qpos": {
                    "type": "ARRAY",
                    "description": "Target joint angles in degrees for all 6 joints.",
                    "items": {"type": "NUMBER"},
                },
            },
            "required": ["qpos"],
        },
    },
    {
        "name": "get_joint_positions",
        "description": "Read current joint state (informational, non-blocking). Returns degrees.",
        "parameters": {
            "type": "OBJECT",
            "properties": {},
        },
    },
    {
        "name": "open_gripper",
        "description": "Convenience wrapper to open the gripper fully.",
        "parameters": {
            "type": "OBJECT",
            "properties": {},
        },
    },
    {
        "name": "close_gripper",
        "description": "Convenience wrapper to close the gripper fully.",
        "parameters": {
            "type": "OBJECT",
            "properties": {},
        },
    },
    {
        "name": "move_to_preset",
        "description": "Move the robot to a predefined preset pose (e.g., 'HOME', 'MIDDLE').",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "pose_name": {
                    "type": "STRING",
                    "description": "Name of the preset pose.",
                },
            },
            "required": ["pose_name"],
        },
    },
    {
        "name": "get_end_effector_pose",
        "description": (
            "Read current end-effector XYZ position and rotation matrix. "
            "Available in sim or twin mode."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {},
        },
    },
    {
        "name": "move_to_spatial_point",
        "description": (
            "Move the robot end-effector to a 2D point visible in the camera feed. "
            "Provide a normalized [y, x] coordinate (0-1000) from the camera image. "
            "The system will map this to 3D workspace coordinates using ArUco calibration "
            "or geometric bounds and use inverse kinematics to reach the target."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "point": {
                    "type": "ARRAY",
                    "description": "Normalized [y, x] image coordinate (0-1000) of the target.",
                    "items": {"type": "NUMBER"},
                },
            },
            "required": ["point"],
        },
    },
    {
        "name": "execute_spav_pick_and_place",
        "description": (
            "Execute a closed-loop pick-and-place operation using the SPAV "
            "(Sense-Plan-Act-Verify) state machine pipeline with Gemini ER 2. "
            "Captures workspace snapshot, detects target object, generates calibrated "
            "minimum-jerk trajectory, executes grasp, and verifies success with "
            "automatic self-correction retry."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "instruction": {
                    "type": "STRING",
                    "description": "Natural language pick-and-place instruction.",
                },
            },
            "required": ["instruction"],
        },
    },
]

# Set BLOCKING behavior for motion tools
_BLOCKING_TOOLS = {
    "move_to_pose",
    "set_joint_positions",
    "open_gripper",
    "close_gripper",
    "move_to_preset",
    "move_to_spatial_point",
    "execute_spav_pick_and_place",
}
for decl in ROBOT_TOOL_DECLARATIONS:
    if decl["name"] in _BLOCKING_TOOLS:
        decl["behavior"] = "BLOCKING"

# Lazy-initialized IK solver (avoids loading MuJoCo XML until first spatial call)
_ik_solver: CartesianIK | None = None


def _get_ik_solver() -> CartesianIK:
    """Return a lazily-initialized CartesianIK solver singleton.

    Returns:
        The shared CartesianIK solver instance.
    """
    global _ik_solver
    if _ik_solver is None:
        _ik_solver = CartesianIK()
    return _ik_solver


DEFAULT_HOVER_Z = 0.03  # 3 cm above table


def image_point_to_xyz(
    point: list[float] | np.ndarray,
    hover_z: float = DEFAULT_HOVER_Z,
    img_shape: tuple[int, int] = (480, 640),
) -> np.ndarray:
    """Map a normalized [y, x] image coordinate to a 3D robot-frame position.

    Requires an active ArUco workspace calibration homography.

    Args:
        point: Normalized [y, x] coordinate where each value is in 0-1000.
        hover_z: Height above the table surface in metres.
        img_shape: Frame dimensions as (height, width).

    Returns:
        A (3,) numpy array [X, Y, Z] in the robot base frame (metres).

    Raises:
        RuntimeError: If the workspace is not calibrated.
    """
    from ui.stream import get_calibrator

    calib = get_calibrator()
    if calib is None or not calib.is_calibrated:
        raise RuntimeError(
            "Workspace is not calibrated. Please ensure 4 ArUco markers are in view and "
            "save calibration under the ArUco Workspace Calibration panel before executing "
            "spatial commands."
        )

    return calib.gemini_to_robot_xyz(point, img_shape=img_shape, z_m=hover_z)


def validate_and_clamp_joints(qpos_deg: list[float]) -> tuple[list[float], list[str]]:
    """Validate joint angles against JOINT_LIMITS and clamp if necessary.

    Args:
        qpos_deg: Joint angles in degrees.

    Returns:
        A tuple of (clamped_qpos_deg, warnings).
    """
    joint_names = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    ]
    clamped_qpos_deg = []
    warnings = []
    for i, angle in enumerate(qpos_deg):
        if i < len(joint_names):
            jname = joint_names[i]
            min_val, max_val = JOINT_LIMITS[jname]
            clamped_val = min(max(angle, min_val), max_val)
            if clamped_val != angle:
                warnings.append(
                    f"{jname} was clamped from {angle:.1f}° to {clamped_val:.1f}° (limit exceeded)"
                )
            clamped_qpos_deg.append(clamped_val)
        else:
            clamped_qpos_deg.append(angle)
    return clamped_qpos_deg, warnings


async def execute_tool_call(
    call: Any,
    robot: RobotInterface | None,
    get_latest_jpeg_fn: Callable[[], bytes | None] | None = None,
    status_callback: Callable[[str], None] | None = None,
) -> dict:
    """Dispatch tool call to the RobotInterface and SPAV state machine asynchronously.

    Args:
        call: The tool call object with name and args.
        robot: The initialized RobotInterface instance.
        get_latest_jpeg_fn: Optional callable returning latest camera frame JPEG bytes.
        status_callback: Optional callable for streaming status messages to UI.

    Returns:
        A dictionary containing the response result to be sent back to Gemini.
    """
    if robot is None:
        return {"error": "RobotInterface is not initialized or injected."}

    name = call.name
    args = call.args if hasattr(call, "args") else {}

    try:
        if name == "move_to_pose":
            qpos_deg = args.get("qpos", [])
            duration_sec = args.get("duration_sec", 1.0)
            if len(qpos_deg) != 6:
                return {"error": f"move_to_pose expects 6 joint angles, got {len(qpos_deg)}"}
            clamped_qpos, warnings = validate_and_clamp_joints(qpos_deg)
            qpos_rad = np.radians(clamped_qpos)
            robot.move_to_pose(qpos_rad, duration_sec=duration_sec)
            response = {"status": "success", "executed_qpos_deg": clamped_qpos}
            if warnings:
                response["warnings"] = warnings
            return response

        elif name == "set_joint_positions":
            qpos_deg = args.get("qpos", [])
            if len(qpos_deg) != 6:
                return {"error": f"set_joint_positions expects 6 joint angles, got {len(qpos_deg)}"}
            clamped_qpos, warnings = validate_and_clamp_joints(qpos_deg)
            qpos_rad = np.radians(clamped_qpos)
            robot.set_joint_positions(qpos_rad)
            response = {"status": "success", "executed_qpos_deg": clamped_qpos}
            if warnings:
                response["warnings"] = warnings
            return response

        elif name == "get_joint_positions":
            qpos_rad = robot.get_joint_positions()
            qpos_deg = [float(v) for v in np.degrees(qpos_rad)]
            return {"qpos_deg": qpos_deg}

        elif name == "open_gripper":
            qpos_rad = robot.get_joint_positions()
            qpos_deg = [float(v) for v in np.degrees(qpos_rad)]
            qpos_deg[5] = float(GRIPPER_OPEN_DEG)
            clamped_qpos, warnings = validate_and_clamp_joints(qpos_deg)
            robot.move_to_pose(np.radians(clamped_qpos), duration_sec=0.5)
            response = {"status": "success", "executed_qpos_deg": clamped_qpos}
            if warnings:
                response["warnings"] = warnings
            return response

        elif name == "close_gripper":
            qpos_rad = robot.get_joint_positions()
            qpos_deg = [float(v) for v in np.degrees(qpos_rad)]
            qpos_deg[5] = float(GRIPPER_CLOSE_DEG)
            clamped_qpos, warnings = validate_and_clamp_joints(qpos_deg)
            robot.move_to_pose(np.radians(clamped_qpos), duration_sec=0.5)
            response = {"status": "success", "executed_qpos_deg": clamped_qpos}
            if warnings:
                response["warnings"] = warnings
            return response

        elif name == "move_to_preset":
            pose_name = args.get("pose_name", "HOME").upper()
            if pose_name not in PRESET_POSES:
                available_poses = list(PRESET_POSES.keys())
                return {"error": f"Unknown preset pose: {pose_name}. Available: {available_poses}"}
            target_rad = PRESET_POSES[pose_name]
            robot.move_to_pose(target_rad, duration_sec=1.0)
            target_deg = [float(v) for v in np.degrees(target_rad)]
            return {"status": "success", "pose_name": pose_name, "executed_qpos_deg": target_deg}

        elif name == "get_end_effector_pose":
            if not hasattr(robot, "get_end_effector_pose"):
                return {"error": "End effector pose is not available in real physical mode."}
            pos, mat = robot.get_end_effector_pose()
            return {
                "position": pos.tolist(),
                "rotation_matrix": mat.tolist(),
            }

        elif name == "move_to_spatial_point":
            point = args.get("point", [])
            if len(point) != 2:
                return {
                    "error": f"move_to_spatial_point expects [y, x] (2 values), got {len(point)}"
                }
            hover_z = DEFAULT_HOVER_Z
            target_xyz = image_point_to_xyz(point, hover_z=hover_z)
            print(
                f"[Tools] move_to_spatial_point: image [{point[0]:.0f}, {point[1]:.0f}] "
                f"→ XYZ [{target_xyz[0]:.3f}, {target_xyz[1]:.3f}, {target_xyz[2]:.3f}]"
            )

            # Use current joints as IK seed for smooth continuation
            current_rad = robot.get_joint_positions()
            ik = _get_ik_solver()
            solved_qpos = ik.solve_ik(
                target_pos=target_xyz,
                initial_qpos=current_rad,
                null_weight=1e-3,
            )
            # Retain current gripper joint angle (joint 5)
            if len(current_rad) >= 6:
                solved_qpos[5] = current_rad[5]

            solved_deg = [float(v) for v in np.degrees(solved_qpos)]
            clamped_qpos, warnings = validate_and_clamp_joints(solved_deg)
            qpos_rad = np.radians(clamped_qpos)
            robot.move_to_pose(qpos_rad, duration_sec=1.5)
            response = {
                "status": "success",
                "target_xyz_m": target_xyz.tolist(),
                "executed_qpos_deg": clamped_qpos,
            }
            if warnings:
                response["warnings"] = warnings
            return response

        elif name == "execute_spav_pick_and_place":
            instruction = args.get("instruction", "")
            if not instruction:
                return {"error": "execute_spav_pick_and_place requires an 'instruction' string."}

            from agent.spav import SPAVStateMachine
            from ui.stream import get_calibrator, get_latest_jpeg

            jpeg_fn = get_latest_jpeg_fn if get_latest_jpeg_fn is not None else get_latest_jpeg
            spav = SPAVStateMachine(robot=robot, calibrator=get_calibrator())

            res = await spav.execute_spav(
                instruction=instruction,
                get_latest_jpeg_fn=jpeg_fn,
                status_callback=status_callback,
            )

            return {
                "status": "success" if res.success else "failed",
                "message": res.final_message,
                "pick_label": res.pick_label,
                "place_label": res.place_label,
                "attempts": res.attempts,
            }

        else:
            return {"error": f"Unknown tool called: {name}"}

    except Exception as e:
        return {"error": f"Exception during execution of {name}: {e}"}


def execute_tool_call_sync(
    call: Any,
    robot: RobotInterface | None,
    get_latest_jpeg_fn: Callable[[], bytes | None] | None = None,
    status_callback: Callable[[str], None] | None = None,
) -> dict:
    """Synchronous convenience wrapper for execute_tool_call."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(
                asyncio.run,
                execute_tool_call(
                    call=call,
                    robot=robot,
                    get_latest_jpeg_fn=get_latest_jpeg_fn,
                    status_callback=status_callback,
                ),
            ).result()
    else:
        return asyncio.run(
            execute_tool_call(
                call=call,
                robot=robot,
                get_latest_jpeg_fn=get_latest_jpeg_fn,
                status_callback=status_callback,
            )
        )
