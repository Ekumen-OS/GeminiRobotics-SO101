"""Core robot module containing kinematics, hardware drivers, and constants for SO-101."""

from robot.calibration import (
    ArucoWorkspaceCalibrator,
    generate_aruco_markers,
    generate_calibration_sheet,
    setup_workspace_interactive,
)
from robot.cartesian_ik import CartesianIK
from robot.constants import ACTUATOR_NAMES, JOINT_NAMES, PRESET_POSES
from robot.kinematics import dh_transform, forward_kinematics, plot_robot
from robot.real_robot import RealSO101Robot
from robot.trajectory import (
    TrajectoryPhase,
    TrajectoryPlanner,
    quintic_minimum_jerk,
    quintic_minimum_jerk_derivative,
    quintic_minimum_jerk_second_derivative,
)

__all__ = [
    "ACTUATOR_NAMES",
    "JOINT_NAMES",
    "PRESET_POSES",
    "ArucoWorkspaceCalibrator",
    "CartesianIK",
    "RealSO101Robot",
    "TrajectoryPhase",
    "TrajectoryPlanner",
    "dh_transform",
    "forward_kinematics",
    "generate_aruco_markers",
    "generate_calibration_sheet",
    "plot_robot",
    "quintic_minimum_jerk",
    "quintic_minimum_jerk_derivative",
    "quintic_minimum_jerk_second_derivative",
    "setup_workspace_interactive",
]
