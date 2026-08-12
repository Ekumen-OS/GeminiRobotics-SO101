"""SO101Robot helper controller class for MuJoCo simulation."""

import threading
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from robot.constants import ACTUATOR_NAMES, JOINT_NAMES, PRESET_POSES
from robot.interface import RobotInterface


class SO101Robot(RobotInterface):
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
        render_viewer: bool = True,
    ):
        """Initialize the SO-101 robot controller wrapper.

        Args:
            xml_path: Optional file path to the MJCF XML model. Used if model is None.
            model: Optional existing preloaded MuJoCo model instance.
            data: Optional existing MuJoCo data instance corresponding to the model.
            render_viewer: Whether to launch a passive 3D MuJoCo viewer window by default.
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

        self.viewer = None
        self._renderer = None
        self._renderer_size: tuple[int, int] | None = None
        # MuJoCo Python bindings are not safe for concurrent model/data access.
        # Serialize all sim state read/write + rendering through one lock.
        self._sim_lock = threading.RLock()
        self._renderer_lock = threading.Lock()
        self._stop_requested = False
        self._offscreen_warned = False
        if render_viewer:
            try:
                from simulation.simulate import ensure_mjpython

                ensure_mjpython()

                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
                self.viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
            except Exception as err:
                print(f"Notice: Could not launch MuJoCo viewer ({err}). Running in headless mode.")
                self.viewer = None

    def reset(self, pose: str = "HOME") -> None:
        """Reset the robot to a specified preset pose.

        Args:
            pose: Name of the target preset pose (e.g., 'HOME', 'REACH', 'PICK', 'STOW').
        """
        with self._sim_lock:
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
        with self._sim_lock:
            assert len(qpos) == len(self.actuator_ids), f"Expected {len(self.actuator_ids)} values"
            for i, act_id in enumerate(self.actuator_ids):
                self.data.ctrl[act_id] = qpos[i]

    def get_joint_positions(self) -> np.ndarray:
        """Get current joint position angles.

        Returns:
            A numpy array of current joint positions in radians.
        """
        with self._sim_lock:
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
        with self._sim_lock:
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
        with self._sim_lock:
            pos = self.data.site_xpos[self.site_id].copy()
            mat = self.data.site_xmat[self.site_id].reshape(3, 3).copy()
            return pos, mat

    def step(self, num_steps: int = 1) -> None:
        """Advance physics simulation by N steps.

        Args:
            num_steps: Number of simulation sub-steps to execute.
        """
        with self._sim_lock:
            for _ in range(num_steps):
                mujoco.mj_step(self.model, self.data)
            if hasattr(self, "viewer") and self.viewer is not None and self.viewer.is_running():
                self.viewer.sync()

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
        target_qpos = np.asarray(target_qpos, dtype=np.float32)
        start_qpos = self.get_joint_positions()
        num_steps = int(duration_sec / dt)
        sim_steps = max(1, int(round(dt / self.model.opt.timestep)))

        for step_idx in range(1, num_steps + 1):
            if self._stop_requested:
                break
            t = step_idx / num_steps
            alpha = 0.5 * (1.0 - np.cos(np.pi * t))
            interp_qpos = (1.0 - alpha) * start_qpos + alpha * target_qpos
            self.set_joint_positions(interp_qpos)
            self.step(sim_steps)
            if hasattr(self, "viewer") and self.viewer is not None and self.viewer.is_running():
                self.viewer.sync()
            time.sleep(dt)

        if not self._stop_requested:
            # Allow position actuators a short smooth settling phase
            for _ in range(10):
                self.step(sim_steps)
                if hasattr(self, "viewer") and self.viewer is not None and self.viewer.is_running():
                    self.viewer.sync()
                time.sleep(dt)
        self._stop_requested = False

    def render_rgb_frame(self, width: int = 640, height: int = 480) -> np.ndarray:
        """Render the current MuJoCo simulation state as an RGB image.

        Args:
            width: Frame width in pixels.
            height: Frame height in pixels.

        Returns:
            RGB image with shape (height, width, 3) and dtype uint8.

        Raises:
            RuntimeError: If MuJoCo offscreen rendering fails.
        """
        req_width = max(1, int(width))
        req_height = max(1, int(height))

        max_width = int(self.model.vis.global_.offwidth)
        max_height = int(self.model.vis.global_.offheight)
        width = min(req_width, max_width)
        height = min(req_height, max_height)

        if (width != req_width or height != req_height) and not self._offscreen_warned:
            print(
                "[WARN] Requested sim render resolution "
                f"{req_width}x{req_height} exceeds MuJoCo offscreen limit "
                f"{max_width}x{max_height}. Clamping to supported size."
            )
            self._offscreen_warned = True

        with self._sim_lock:
            with self._renderer_lock:
                target_size = (width, height)
                if self._renderer is None or self._renderer_size != target_size:
                    if self._renderer is not None:
                        self._renderer.close()
                    self._renderer = mujoco.Renderer(self.model, height=height, width=width)
                    self._renderer_size = target_size

                try:
                    self._renderer.update_scene(self.data)
                    frame = self._renderer.render()
                except Exception as err:
                    raise RuntimeError(f"MuJoCo offscreen render failed: {err}") from err

        return np.asarray(frame, dtype=np.uint8)

    def _close_renderer(self) -> None:
        """Release MuJoCo renderer resources if currently allocated."""
        with self._sim_lock:
            with self._renderer_lock:
                if self._renderer is not None:
                    self._renderer.close()
                    self._renderer = None
                    self._renderer_size = None
                    self._offscreen_warned = False

    def close(self) -> None:
        """Close the MuJoCo simulation viewer if currently open."""
        with self._sim_lock:
            self._close_renderer()
            if hasattr(self, "viewer") and self.viewer is not None:
                if self.viewer.is_running():
                    self.viewer.close()
                self.viewer = None

    def interrupt(self) -> None:
        """Interrupt any ongoing physical movement."""
        self._stop_requested = True

    def __del__(self) -> None:
        """Cleanup resources on object deletion."""
        try:
            self.close()
        except Exception:
            pass
