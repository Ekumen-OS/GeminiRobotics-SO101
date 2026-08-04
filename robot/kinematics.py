"""Forward kinematics solver and visualization for the SO-101 robot arm."""

import matplotlib.pyplot as plt
import numpy as np
from spatialmath import SE3

# Estimated DH Parameters for SO-101 (Subject to mechanical calibration)
# [theta_offset, d (Z offset), a (X offset), alpha (X twist)]
DH_PARAMS = [
    [0, 35, 0, np.pi / 2],  # Joint 1: Pan
    [-np.pi / 2, 0, 85, 0],  # Joint 2: Shoulder
    [0, 0, 85, 0],  # Joint 3: Elbow
    [0, 0, 70, np.pi / 2],  # Joint 4: Wrist Flex
    [0, 50, 0, np.pi / 2],  # Joint 5: Wrist Roll
    [0, 50, 0, 0],  # Joint 6: Gripper (Effectively offset for end effector)
]


def dh_transform(theta: float, d: float, a: float, alpha: float) -> np.ndarray:
    """Compute the DH transformation matrix for a single link.

    Args:
        theta: Joint angle in radians.
        d: Link offset along Z-axis in mm.
        a: Link length along X-axis in mm.
        alpha: Link twist angle in radians.

    Returns:
        4x4 homogeneous transformation matrix as a NumPy array.
    """
    return np.array(
        [
            [
                np.cos(theta),
                -np.sin(theta) * np.cos(alpha),
                np.sin(theta) * np.sin(alpha),
                a * np.cos(theta),
            ],
            [
                np.sin(theta),
                np.cos(theta) * np.cos(alpha),
                -np.cos(theta) * np.sin(alpha),
                a * np.sin(theta),
            ],
            [0, np.sin(alpha), np.cos(alpha), d],
            [0, 0, 0, 1],
        ]
    )


def forward_kinematics(
    joint_angles: list[float] | None = None,
) -> tuple[SE3, np.ndarray]:
    """Computes the Forward Kinematics for the SO-101 Robot Arm.

    Args:
        joint_angles: A list of 6 joint angles in radians. Defaults to all zeros.

    Returns:
        A tuple (T, points) containing the SE3 transformation matrix of the
        end effector and an Nx3 NumPy array of joint 3D positions for plotting.
    """
    if joint_angles is None:
        joint_angles = [0, 0, 0, 0, 0, 0]

    # Apply theta offset from DH table
    angles = [joint_angles[i] + DH_PARAMS[i][0] for i in range(6)]

    T = SE3()  # Start with identity matrix
    points = [[0, 0, 0]]  # Base coordinate

    for i in range(6):
        d = DH_PARAMS[i][1]
        a = DH_PARAMS[i][2]
        alpha = DH_PARAMS[i][3]
        theta = angles[i]

        # We manually compute the DH matrix as SE3
        link_T = SE3(dh_transform(theta, d, a, alpha))
        T = T * link_T

        # Save position of this joint for plotting
        pos = T.t
        points.append([pos[0], pos[1], pos[2]])

    return T, np.array(points)


def plot_robot(points: np.ndarray) -> None:
    """Plots a wireframe model of the robot configuration.

    Args:
        points: Nx3 array of 3D coordinates representing the robot joints.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3rd")

    xs = points[:, 0]
    ys = points[:, 1]
    zs = points[:, 2]

    ax.plot(xs, ys, zs, marker="o", color="blue", linewidth=4, markersize=8)
    ax.plot([0], [0], [0], marker="s", color="red", markersize=10)  # Base

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")

    # Set equal aspect ratio
    max_range = (
        np.array([xs.max() - xs.min(), ys.max() - ys.min(), zs.max() - zs.min()]).max() / 2.0
    )
    mid_x = (xs.max() + xs.min()) * 0.5
    mid_y = (ys.max() + ys.min()) * 0.5
    mid_z = (zs.max() + zs.min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.title("GeminiRobotics-SO101 FK Visualization")
    plt.show()


if __name__ == "__main__":
    print("Testing GeminiRobotics-SO101 Forward Kinematics\n")

    # Define some arbitrary test angles (in radians)
    test_angles = [0.0, np.radians(45), np.radians(-90), 0.0, 0.0, 0.0]

    print(f"Input Joint Angles (rad): {test_angles}")
    T_end_effector, joint_points = forward_kinematics(test_angles)

    print("\nEnd Effector Transformation Matrix:")
    print(T_end_effector)

    print("\nEnd Effector Position (x,y,z in mm):")
    print(T_end_effector.t)
