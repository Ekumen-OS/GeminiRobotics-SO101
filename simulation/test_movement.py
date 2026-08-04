"""Movement validation script to test kinematics, MuJoCo simulation, and line trajectories."""

from robot.constants import PRESET_POSES
from simulation.line_trajectory import LineTrajectoryGenerator
from simulation.sim_robot import SO101Robot


def test_robot_movement() -> None:
    """Verify linear Cartesian sweeps along X, Y, and Z axes in MuJoCo simulation."""
    print("Initializing MuJoCo Simulation Environment for Movement Verification...")
    robot = SO101Robot()

    print("Resetting robot to HOME pose...")
    robot.reset("HOME")

    print("Transitioning to SWEEP_HOME pose before starting line sweeps...")
    robot.move_to_pose(PRESET_POSES["SWEEP_HOME"], duration_sec=0.5, dt=0.01)

    trajectory_gen = LineTrajectoryGenerator(robot=robot)

    # Execute short validation sweeps across all three Cartesian axes
    for axis in ["x", "y", "z"]:
        print(f"\n--- Testing Line Sweep along [{axis.upper()}] axis ---")
        trajectory_gen.execute_line_sweep(
            axis=axis,
            distance=0.08,
            num_points=16,
            step_delay=0.0,
        )

    print("\nReturning robot to HOME pose...")
    robot.move_to_pose(PRESET_POSES["HOME"], duration_sec=0.5, dt=0.01)
    print("\nRobot Movement & Kinematics Verification Test PASSED!")


if __name__ == "__main__":
    test_robot_movement()
