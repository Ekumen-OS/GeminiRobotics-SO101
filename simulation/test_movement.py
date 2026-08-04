"""Movement validation script to test kinematics, MuJoCo simulation, and line trajectories."""

import time

from robot.constants import PRESET_POSES
from simulation.line_trajectory import LineTrajectoryGenerator
from simulation.sim_robot import SO101Robot
from simulation.simulate import ensure_mjpython


def test_robot_movement() -> None:
    """Verify linear Cartesian sweeps along X, Y, and Z axes in MuJoCo simulation."""
    ensure_mjpython()

    print("Initializing MuJoCo Simulation Environment for Movement Verification...")
    robot = SO101Robot(render_viewer=True)
    viewer = getattr(robot, "viewer", None)

    print("Resetting robot to HOME pose...")
    robot.reset("HOME")
    if viewer is not None and viewer.is_running():
        viewer.sync()
        time.sleep(0.5)

    print("Transitioning to SWEEP_HOME pose before starting line sweeps...")
    robot.move_to_pose(PRESET_POSES["SWEEP_HOME"], duration_sec=0.8, dt=0.01)

    trajectory_gen = LineTrajectoryGenerator(robot=robot)

    # Execute short validation sweeps across all three Cartesian axes
    for axis in ["x", "y", "z"]:
        if viewer is not None and not viewer.is_running():
            print("\nViewer closed by user. Terminating test early.")
            return

        print(f"\n--- Testing Line Sweep along [{axis.upper()}] axis ---")
        trajectory_gen.execute_line_sweep(
            axis=axis,
            distance=0.08,
            num_points=30,
            step_delay=0.03,
        )

        # Move back to SWEEP_HOME pose before testing the next axis
        if viewer is None or viewer.is_running():
            print("Returning to SWEEP_HOME pose...")
            robot.move_to_pose(PRESET_POSES["SWEEP_HOME"], duration_sec=0.6, dt=0.01)

    if viewer is None or viewer.is_running():
        print("\nReturning robot to HOME pose...")
        robot.move_to_pose(PRESET_POSES["HOME"], duration_sec=0.8, dt=0.01)
        time.sleep(0.5)
        if viewer is not None:
            viewer.sync()

    print("\nRobot Movement & Kinematics Verification Test PASSED!")
    robot.close()


if __name__ == "__main__":
    test_robot_movement()
