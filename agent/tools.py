"""Robot tool declarations and execution adapters for Gemini Live API."""

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
            "The system will map this to 3D workspace coordinates and use inverse "
            "kinematics to reach the target. Use this tool when you can SEE an object "
            "in the camera and want to reach toward it."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "point": {
                    "type": "ARRAY",
                    "description": ("Normalized [y, x] image coordinate (0-1000) of the target."),
                    "items": {"type": "NUMBER"},
                },
            },
            "required": ["point"],
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


# --- Workspace calibration constants ---
# These map normalized image coordinates (0-1000) to robot-base-frame metres.
# Tuned for the Seeed Studio desk mat camera angle.
# Image top (y=0) → far from base (0.38m), Image bottom (y=1000) → near base (0.12m).
# Image left (x=0) → robot's left (+0.28m), Image right (x=1000) → robot's right (-0.20m).
WORKSPACE_X_RANGE = (0.38, 0.12)  # (far, near) in metres along robot X
WORKSPACE_Y_RANGE = (0.28, -0.20)  # (left, right) in metres along robot Y
DEFAULT_HOVER_Z = 0.03  # 3 cm above table


def image_point_to_xyz(
    point: list[float],
    hover_z: float = DEFAULT_HOVER_Z,
) -> np.ndarray:
    """Map a normalized [y, x] image coordinate to a 3D robot-frame position.

    In the workspace camera view:
    - Normalized y (0 = top, 1000 = bottom) maps to Robot X (0.36m far -> 0.12m near).
    - Normalized x (0 = left, 1000 = right) maps to Robot Y (+0.20m left -> -0.20m right).

    Args:
        point: Normalized [y, x] coordinate where each value is in 0-1000.
        hover_z: Height above the table surface in metres.

    Returns:
        A (3,) numpy array [X, Y, Z] in the robot base frame (metres).
    """
    img_y, img_x = point[0] / 1000.0, point[1] / 1000.0

    # Image Y (top to bottom) -> Robot X (far to near)
    robot_x = WORKSPACE_X_RANGE[0] + img_y * (WORKSPACE_X_RANGE[1] - WORKSPACE_X_RANGE[0])

    # Image X (left to right) -> Robot Y (left to right)
    robot_y = WORKSPACE_Y_RANGE[0] + img_x * (WORKSPACE_Y_RANGE[1] - WORKSPACE_Y_RANGE[0])

    return np.array([robot_x, robot_y, hover_z])


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


def execute_tool_call(call, robot: RobotInterface | None) -> dict:
    """Dispatch tool call to the RobotInterface.

    Args:
        call: The tool call object with name and args.
        robot: The initialized RobotInterface instance.

    Returns:
        A dictionary containing the response result to be sent back to Gemini.
    """
    if robot is None:
        return {"error": "RobotInterface is not initialized or injected."}

    name = call.name
    args = call.args

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
            # Always use default hover height to prevent out-of-reach IK targets
            # that cause hardware torque-off from over-extension.
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

        else:
            return {"error": f"Unknown tool called: {name}"}

    except Exception as e:
        return {"error": f"Exception during execution of {name}: {e}"}
