"""SO101Robot helper controller class for MuJoCo simulation."""

import time
from pathlib import Path

import mujoco
import numpy as np

from robot.constants import ACTUATOR_NAMES, JOINT_NAMES, PRESET_POSES


class SO101Robot:
    """High-level Python wrapper for controlling the SO-101 robot arm in MuJoCo.

    Attributes:
        model: The MuJoCo model instance.
        data: The MuJoCo data instance associated with the model.
        joint_ids: List of MuJoCo joint IDs corresponding to the robot joints.
        actuator_ids: List of MuJoCo actuator IDs corresponding to the robot actuators.
        site_id: MuJoCo site ID for the end effector or gripper frame.
    """

    def __init__(
        self,
        xml_path: str | Path | None = None,
        model: mujoco.MjModel | None = None,
        data: mujoco.MjData | None = None,
    ):
        """Initialize the SO-101 robot controller wrapper.

        Args:
            xml_path: Optional file path to the MJCF XML model. Used if model is None.
            model: Optional existing preloaded MuJoCo model instance.
            data: Optional existing MuJoCo data instance corresponding to the model.
        """
        if model is not None and data is not None:
            self.model = model
            self.data = data
        else:
            if xml_path is None:
                default_scene = Path(__file__).parent / "assets" / "scene.xml"
                xml_path = default_scene
            self.model = mujoco.MjModel.from_xml_path(str(xml_path))
            self.data = mujoco.MjData(self.model)

        self.joint_ids = [self.model.joint(name).id for name in JOINT_NAMES]
        self.actuator_ids = [self.model.actuator(name).id for name in ACTUATOR_NAMES]
        try:
            self.site_id = self.model.site("gripperframe").id
        except KeyError:
            self.site_id = self.model.site("end_effector").id

    def reset(self, pose: str = "HOME") -> None:
        """Reset the robot to a specified preset pose.

        Args:
            pose: Name of the target preset pose (e.g., 'HOME', 'REACH', 'PICK', 'STOW').
        """
        mujoco.mj_resetData(self.model, self.data)
        target_qpos = PRESET_POSES.get(pose.upper(), PRESET_POSES["HOME"])
        for i, j_id in enumerate(self.joint_ids):
            qpos_adr = self.model.jnt_qposadr[j_id]
            self.data.qpos[qpos_adr] = target_qpos[i]
        self.set_joint_positions(target_qpos)
        mujoco.mj_forward(self.model, self.data)
        self.step(50)

    def set_joint_positions(self, qpos: np.ndarray) -> None:
        """Set target joint position control targets.

        Args:
            qpos: Target joint angles as a numpy array of shape (6,).
        """
        assert len(qpos) == len(self.actuator_ids), f"Expected {len(self.actuator_ids)} values"
        for i, act_id in enumerate(self.actuator_ids):
            self.data.ctrl[act_id] = qpos[i]

    def get_joint_positions(self) -> np.ndarray:
        """Get current joint position angles.

        Returns:
            A numpy array of current joint positions in radians.
        """
        pos_list = []
        for j_id in self.joint_ids:
            qpos_adr = self.model.jnt_qposadr[j_id]
            pos_list.append(self.data.qpos[qpos_adr])
        return np.array(pos_list)

    def get_joint_velocities(self) -> np.ndarray:
        """Get current joint velocities.

        Returns:
            A numpy array of current joint velocities in radians per second.
        """
        vel_list = []
        for j_id in self.joint_ids:
            dof_adr = self.model.jnt_dofadr[j_id]
            vel_list.append(self.data.qvel[dof_adr])
        return np.array(vel_list)

    def get_end_effector_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Return end-effector position and orientation rotation matrix.

        Returns:
            A tuple containing the end-effector 3D position array (x, y, z) and
            the 3x3 rotation matrix.
        """
        pos = self.data.site_xpos[self.site_id].copy()
        mat = self.data.site_xmat[self.site_id].reshape(3, 3).copy()
        return pos, mat

    def step(self, num_steps: int = 1) -> None:
        """Advance physics simulation by N steps.

        Args:
            num_steps: Number of simulation sub-steps to execute.
        """
        for _ in range(num_steps):
            mujoco.mj_step(self.model, self.data)

    def move_to_pose(
        self,
        target_qpos: np.ndarray,
        duration_sec: float = 1.0,
        dt: float = 0.002,
    ) -> None:
        """Smoothly interpolate from current joint positions to target joint positions.

        Args:
            target_qpos: Target joint angles in radians.
            duration_sec: Duration of the interpolation trajectory in seconds.
            dt: Physics timestep duration in seconds.
        """
        start_qpos = self.get_joint_positions()
        num_steps = int(duration_sec / dt)

        for step_idx in range(1, num_steps + 1):
            if hasattr(self, "viewer") and self.viewer is not None and not self.viewer.is_running():
                t = step_idx / num_steps
                alpha = 0.5 * (1.0 - np.cos(np.pi * t))
                interp_qpos = (1.0 - alpha) * start_qpos + alpha * target_qpos
                self.set_joint_positions(interp_qpos)
                self.step(1)
                time.sleep(dt)
                continue

            t = step_idx / num_steps
            alpha = 0.5 * (1.0 - np.cos(np.pi * t))
            interp_qpos = (1.0 - alpha) * start_qpos + alpha * target_qpos
            self.set_joint_positions(interp_qpos)
            self.step(1)
            if hasattr(self, "viewer") and self.viewer is not None:
                self.viewer.sync()
            time.sleep(dt)

        # Allow position actuators to settle
        self.step(100)
        if hasattr(self, "viewer") and self.viewer is not None:
            self.viewer.sync()
