"""TwinSO101Robot interface — mirror MuJoCo simulation state directly to physical SO-101 arm."""

import time

import numpy as np

from robot.real_robot import RealSO101Robot
from simulation.sim_robot import SO101Robot


class TwinSO101Robot:
    """Wrapper that synchronizes a MuJoCo simulation instance with a physical SO-101 robot arm.

    Every joint position command or simulation step automatically streams the resulting
    simulated joint angles to the physical robot servos in real time (Digital Twin mode).
    """

    def __init__(
        self,
        port: str = "/dev/tty.usbmodem1201",
        xml_path=None,
    ):
        """Initialize digital twin simulation + real robot controller.

        Args:
            port: Serial port device path for physical SO-101 arm.
            xml_path: Optional path to MuJoCo scene XML file.
        """
        print("Initializing Digital Twin Mode (MuJoCo Simulation + Real Hardware Mirror)...")
        self.sim = SO101Robot(xml_path=xml_path)
        self.real = RealSO101Robot(port=port)

        # Expose MuJoCo model & data for viewer compatibility
        self.model = self.sim.model
        self.data = self.sim.data
        self.viewer = None

    def reset(self, pose: str = "HOME") -> None:
        """Reset both simulation and real robot to a specified pose.

        Args:
            pose: Preset pose name ('HOME', 'REACH', 'PICK', 'STOW').
        """
        self.sim.reset(pose)
        qpos = self.sim.get_joint_positions()
        self.real.set_joint_positions(qpos)

    def set_joint_positions(self, qpos: np.ndarray) -> None:
        """Command target joint positions to both MuJoCo simulation and physical robot.

        Args:
            qpos: Target joint angles in radians for 6 joints.
        """
        self.sim.set_joint_positions(qpos)
        self.real.set_joint_positions(qpos)

    def get_joint_positions(self) -> np.ndarray:
        """Read current joint positions from simulation (or real arm if connected).

        Returns:
            A numpy array of 6 joint positions in radians.
        """
        return self.sim.get_joint_positions()

    def get_end_effector_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Get end-effector position and orientation from MuJoCo physics engine."""
        return self.sim.get_end_effector_pose()

    def step(self, num_steps: int = 1) -> None:
        """Advance physics in MuJoCo simulation and mirror joint state to physical arm."""
        self.sim.step(num_steps)
        qpos = self.sim.get_joint_positions()
        self.real.set_joint_positions(qpos)

    def move_to_pose(
        self,
        target_qpos: np.ndarray,
        duration_sec: float = 1.0,
        dt: float = 0.002,
    ) -> None:
        """Smoothly interpolate joint positions in simulation while mirroring live to real robot."""
        start_qpos = self.sim.get_joint_positions()
        num_steps = int(duration_sec / dt)

        for step_idx in range(1, num_steps + 1):
            if hasattr(self, "viewer") and self.viewer is not None and not self.viewer.is_running():
                t = step_idx / num_steps
                alpha = 0.5 * (1.0 - np.cos(np.pi * t))
                interp_qpos = (1.0 - alpha) * start_qpos + alpha * target_qpos
                self.set_joint_positions(interp_qpos)
                self.sim.step(1)
                time.sleep(dt)
                continue

            t = step_idx / num_steps
            alpha = 0.5 * (1.0 - np.cos(np.pi * t))
            interp_qpos = (1.0 - alpha) * start_qpos + alpha * target_qpos
            self.set_joint_positions(interp_qpos)
            self.sim.step(1)

            if hasattr(self, "viewer") and self.viewer is not None:
                self.viewer.sync()

            time.sleep(dt)

        self.sim.step(50)
        qpos = self.sim.get_joint_positions()
        self.real.set_joint_positions(qpos)
        if hasattr(self, "viewer") and self.viewer is not None:
            self.viewer.sync()

    def disconnect(self) -> None:
        """Disconnect physical serial bus."""
        self.real.disconnect()
