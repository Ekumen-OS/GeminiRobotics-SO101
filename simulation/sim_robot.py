"""SO101Robot helper controller class for MuJoCo simulation."""

import json
import os
import sys
import threading
import time
from pathlib import Path

# On headless Linux systems without an active display server, default to EGL offscreen rendering
if (
    sys.platform.startswith("linux")
    and "DISPLAY" not in os.environ
    and "WAYLAND_DISPLAY" not in os.environ
):
    os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import mujoco.viewer
import numpy as np

from robot.constants import ACTUATOR_NAMES, JOINT_NAMES, PRESET_POSES
from robot.interface import RobotInterface

DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent / "config" / "workspace_calibration.json"
)


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
        render_viewer: bool = False,
    ) -> None:
        """Initialize SO101Robot with MuJoCo model, data, and joint/actuator references.

        Args:
            xml_path: Relative or absolute path to the MuJoCo XML scene file.
            model: Optional preloaded MuJoCo model.
            data: Optional preloaded MuJoCo data.
            render_viewer: Whether to launch the passive visualizer.
        """
        if model is not None and data is not None:
            self.model = model
            self.data = data
            self.xml_path = str(xml_path) if xml_path else "simulation/assets/scene.xml"
        else:
            default_scene = Path(__file__).parent / "assets" / "scene.xml"
            scene_to_load = str(xml_path) if xml_path else str(default_scene)
            self.xml_path = scene_to_load
            self.model = mujoco.MjModel.from_xml_path(scene_to_load)
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

        # Dynamically synchronize marker positions and workspace bounds from JSON config
        self.apply_workspace_calibration_config()

        if render_viewer:
            try:
                from simulation.simulate import ensure_mjpython

                ensure_mjpython()

                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
                self.viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
            except Exception as err:
                print(f"Notice: Could not launch MuJoCo viewer ({err}). Running in headless mode.")
                self.viewer = None

    def apply_workspace_calibration_config(self, config_path: str | Path | None = None) -> None:
        """Dynamically update MuJoCo scene marker positions and boundaries from JSON config.

        Args:
            config_path: Path to workspace calibration JSON.
        """
        cfg_file = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        if not cfg_file.exists():
            return

        try:
            cfg = json.loads(cfg_file.read_text())
            corners = cfg.get("workspace_corners_mm", {})
            tl = corners.get("top_left", {"x": 380.0, "y": 130.0})
            tr = corners.get("top_right", {"x": 380.0, "y": -250.0})
            br = corners.get("bottom_right", {"x": 35.0, "y": -250.0})
            bl = corners.get("bottom_left", {"x": 35.0, "y": 130.0})

            x_tl, y_tl = tl["x"] / 1000.0, tl["y"] / 1000.0
            x_tr, y_tr = tr["x"] / 1000.0, tr["y"] / 1000.0
            x_br, y_br = br["x"] / 1000.0, br["y"] / 1000.0
            x_bl, y_bl = bl["x"] / 1000.0, bl["y"] / 1000.0

            marker_half = cfg.get("marker_size_mm", 30.0) / 2000.0

            with self._sim_lock:
                marker_geom_map = {
                    "aruco_tl_0": (x_tl, y_tl),
                    "aruco_tr_1": (x_tr, y_tr),
                    "aruco_br_2": (x_br, y_br),
                    "aruco_bl_3": (x_bl, y_bl),
                }
                for geom_name, (gx, gy) in marker_geom_map.items():
                    gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
                    if gid >= 0:
                        self.model.geom_pos[gid] = [gx, gy, 0.0008]
                        self.model.geom_size[gid] = [marker_half, marker_half, 0.0005]

                top_gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "bound_top")
                if top_gid >= 0:
                    top_cy = (y_tl + y_tr) / 2.0
                    top_len_y = abs(y_tl - y_tr) / 2.0
                    self.model.geom_pos[top_gid] = [x_tl, top_cy, 0.0005]
                    self.model.geom_size[top_gid] = [0.0015, top_len_y, 0.0005]

                bottom_gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "bound_bottom")
                if bottom_gid >= 0:
                    bot_cy = (y_bl + y_br) / 2.0
                    bot_len_y = abs(y_bl - y_br) / 2.0
                    self.model.geom_pos[bottom_gid] = [x_bl, bot_cy, 0.0005]
                    self.model.geom_size[bottom_gid] = [0.0015, bot_len_y, 0.0005]

                left_gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "bound_left")
                if left_gid >= 0:
                    left_cx = (x_tl + x_bl) / 2.0
                    left_len_x = abs(x_tl - x_bl) / 2.0
                    self.model.geom_pos[left_gid] = [left_cx, y_tl, 0.0005]
                    self.model.geom_size[left_gid] = [left_len_x, 0.0015, 0.0005]

                right_gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "bound_right")
                if right_gid >= 0:
                    right_cx = (x_tr + x_br) / 2.0
                    right_len_x = abs(x_tr - x_br) / 2.0
                    self.model.geom_pos[right_gid] = [right_cx, y_tr, 0.0005]
                    self.model.geom_size[right_gid] = [right_len_x, 0.0015, 0.0005]

                mujoco.mj_forward(self.model, self.data)
        except Exception as e:
            print(f"[SO101Robot] Warning updating workspace geoms from config: {e}")

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

    def randomize_target(
        self,
        x_range: tuple[float, float] = (0.13, 0.25),
        y_range: tuple[float, float] = (-0.12, 0.12),
        z_height: float = 0.015,
    ) -> np.ndarray:
        """Randomize the target manipulation object position on the tabletop surface.

        Args:
            x_range: Min and max X coordinates in meters (forward from robot base).
            y_range: Min and max Y coordinates in meters (left/right span).
            z_height: Resting height of the object above ground plane in meters.

        Returns:
            New [x, y, z] position as numpy array.
        """
        with self._sim_lock:
            try:
                j_id = self.model.joint("target_joint").id
                qpos_adr = self.model.jnt_qposadr[j_id]
                qvel_adr = self.model.jnt_dofadr[j_id]

                new_x = float(np.random.uniform(x_range[0], x_range[1]))
                new_y = float(np.random.uniform(y_range[0], y_range[1]))
                new_pos = np.array([new_x, new_y, z_height], dtype=np.float64)

                self.data.qpos[qpos_adr : qpos_adr + 3] = new_pos
                self.data.qpos[qpos_adr + 3 : qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]
                self.data.qvel[qvel_adr : qvel_adr + 6] = 0.0

                mujoco.mj_forward(self.model, self.data)
                if hasattr(self, "viewer") and self.viewer is not None and self.viewer.is_running():
                    self.viewer.sync()
                return new_pos
            except (KeyError, AttributeError) as e:
                print(f"[SO101Robot] Could not randomize target: {e}")
                return np.zeros(3)

    def set_joint_positions(self, qpos: np.ndarray) -> None:
        """Set target joint position control targets and snap joints in sim.

        Args:
            qpos: Target joint angles as a numpy array of shape (6,).
        """
        with self._sim_lock:
            assert len(qpos) == len(self.actuator_ids), f"Expected {len(self.actuator_ids)} values"
            for i, act_id in enumerate(self.actuator_ids):
                self.data.ctrl[act_id] = qpos[i]
            for i, j_id in enumerate(self.joint_ids):
                qpos_adr = self.model.jnt_qposadr[j_id]
                self.data.qpos[qpos_adr] = qpos[i]
            mujoco.mj_forward(self.model, self.data)
            if hasattr(self, "viewer") and self.viewer is not None and self.viewer.is_running():
                self.viewer.sync()

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

    def render_rgb_frame(
        self, width: int = 640, height: int = 480, camera: str | int | None = "workspace_cam"
    ) -> np.ndarray:
        """Render the current MuJoCo simulation state as an RGB image.

        Args:
            width: Frame width in pixels.
            height: Frame height in pixels.
            camera: Named camera identifier or camera index (default: 'workspace_cam').

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

        cam_to_use: str | int | None = camera
        if isinstance(camera, str):
            cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
            if cam_id < 0:
                cam_to_use = None

        with self._renderer_lock:
            with self._sim_lock:
                target_size = (width, height)
                if self._renderer is None or self._renderer_size != target_size:
                    if self._renderer is not None:
                        self._renderer.close()
                    self._renderer = mujoco.Renderer(self.model, height=height, width=width)
                    self._renderer_size = target_size

                if cam_to_use is not None:
                    self._renderer.update_scene(self.data, camera=cam_to_use)
                else:
                    self._renderer.update_scene(self.data)

            try:
                frame = self._renderer.render()
            except Exception as err:
                raise RuntimeError(f"MuJoCo offscreen render failed: {err}") from err

        return np.asarray(frame, dtype=np.uint8)

    def _close_renderer(self) -> None:
        """Release MuJoCo renderer resources if currently allocated."""
        with self._renderer_lock:
            with self._sim_lock:
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
