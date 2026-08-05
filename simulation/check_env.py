"""Environment diagnostics and system check script."""

import sys
import time

import numpy as np


def check_environment() -> None:
    """Run diagnostics and system checks for simulation and physical dependencies."""
    print("=" * 60)
    print("GeminiRobotics-SO101 Simulation & Hardware Environment Diagnostic")
    print("=" * 60)

    # 1. Python & System
    print(f"Python Version: {sys.version.split()[0]}")

    # 2. MuJoCo Simulation Check
    try:
        import mujoco

        from simulation.sim_robot import SO101Robot

        robot = SO101Robot(render_viewer=False)
        print(f"[OK] MuJoCo library version {mujoco.__version__} loaded.")
        msg = f"[OK] SO-101 MJCF parsed ({robot.model.nq} DOFs, {len(robot.actuator_ids)} acts)."
        print(msg)

        # Physics Step Benchmark
        start_time = time.perf_counter()
        steps = 1000
        for _ in range(steps):
            robot.step(1)
        elapsed = time.perf_counter() - start_time
        fps = steps / elapsed
        print(f"[OK] MuJoCo Physics Step FPS: {fps:.1f} steps/sec")
        robot.close()

    except Exception as e:
        print(f"[FAIL] MuJoCo environment check failed: {e}")

    # 3. Inverse Kinematics Check
    try:
        from robot.cartesian_ik import CartesianIK

        ik = CartesianIK()
        target = np.array([0.18, 0.05, 0.15])
        qpos_ik = ik.solve_ik(target_pos=target)
        print(f"[OK] Inverse Kinematics solved target {target} -> qpos: {np.round(qpos_ik, 3)}")
    except Exception as e:
        print(f"[FAIL] Inverse Kinematics check failed: {e}")

    # 4. LeRobot Hardware Bus Check
    try:
        from robot.real_robot import LEROBOT_AVAILABLE

        if LEROBOT_AVAILABLE:
            print("[OK] LeRobot package and FeetechMotorsBus import verified.")
        else:
            print(
                "[INFO] LeRobot package not detected. Physical robot control will be unavailable."
            )
    except Exception as e:
        print(f"[INFO] Real robot hardware driver check: {e}")

    print("=" * 60)
    print("All diagnostic checks completed.")
    print("=" * 60)


if __name__ == "__main__":
    check_environment()
