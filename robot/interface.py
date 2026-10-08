"""Abstract Base Class defining the contract for all robot models."""

from abc import ABC, abstractmethod

import numpy as np

from robot.contracts import GripperReading


class RobotInterface(ABC):
    """Abstract Base Class for SO-101 robot controllers (Physical, Simulated).

    Attributes:
        mode: Operational mode this controller represents: ``"real"`` for a physical arm or
            ``"sim"`` for a MuJoCo simulation. Consumers (tool adapters, calibrator routing)
            read this instead of duck-typing on rendering methods.
    """

    mode: str = "real"

    @abstractmethod
    def set_joint_positions(self, qpos: np.ndarray) -> None:
        """Command target joint positions (in radians).

        Args:
            qpos: Array of target joint angles.
        """
        pass

    @abstractmethod
    def get_joint_positions(self) -> np.ndarray:
        """Read current joint positions (in radians).

        Returns:
            Array of current joint angles.
        """
        pass

    def step(self, num_steps: int = 1) -> None:
        """Advance the physics simulation (no-op for purely physical robots).

        Args:
            num_steps: Number of simulation steps to execute.
        """
        pass

    def move_to_pose(
        self,
        target_qpos: np.ndarray,
        duration_sec: float = 1.0,
        dt: float = 0.002,
    ) -> None:
        """Interpolate smoothly to target positions.

        Args:
            target_qpos: Array of target joint angles.
            duration_sec: Duration in seconds.
            dt: Timestep.
        """
        pass

    def render_rgb_frame(self, width: int = 640, height: int = 480) -> np.ndarray:
        """Render a simulation frame (no-op for purely physical robots).

        Args:
            width: Frame width.
            height: Frame height.

        Returns:
            RGB image as numpy array.
        """
        raise NotImplementedError("This robot instance does not support rgb frame rendering.")

    def disconnect(self) -> None:
        """Release hardware or simulation resources."""
        pass

    def interrupt(self) -> None:
        """Interrupt any ongoing physical movement or blocking operations."""
        pass

    def read_gripper_state(self) -> GripperReading:
        """Read the current state of the gripper joint.

        The base implementation derives aperture from the 6th joint position
        (index 5) in degrees, with load=None and source='commanded'. Subclasses
        with hardware or physics sensor feedback override this method with
        measured values.

        Returns:
            A GripperReading dict containing aperture_deg, load, and source.
        """
        qpos = self.get_joint_positions()
        aperture_deg = float(np.degrees(qpos[5])) if len(qpos) > 5 else 0.0
        return {
            "aperture_deg": aperture_deg,
            "load": None,
            "source": "commanded",
        }
