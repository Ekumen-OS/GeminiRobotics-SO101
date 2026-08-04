"""SO-101 MuJoCo Warp GPU/CPU Batched Simulation Environment Interface."""

from pathlib import Path

import numpy as np

try:
    import mujoco
    import mujoco_warp as mjw
    import warp as wp

    WARP_AVAILABLE = True
except ImportError:
    WARP_AVAILABLE = False


class SO101WarpEnv:
    """Parallel batched simulation environment for SO-101 using MuJoCo Warp (`mujoco_warp`)."""

    def __init__(
        self,
        num_envs: int = 128,
        xml_path: str | None = None,
        device: str = "cpu",
    ):
        """Initialize batched parallel Warp environment.

        Args:
            num_envs: Number of parallel simulation environments.
            xml_path: Path to scene XML file.
            device: Compute device ('cpu' or 'cuda').
        """
        self.num_envs = num_envs
        self.device = device

        if xml_path is None:
            xml_path = str(Path(__file__).parent / "assets" / "scene.xml")

        self.xml_path = xml_path

        if not WARP_AVAILABLE:
            print("Notice: 'mujoco_warp' or 'warp' package not found in current environment.")
            self.mjw_model = None
            self.mjw_data = None
            return

        print(f"Initializing MuJoCo Warp with {num_envs} parallel environments...")
        wp.init()
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.mjw_model = mjw.put_model(self.model)
        self.mjw_data = mjw.put_data(self.model, self.data, nworld=num_envs)

    def reset(self) -> np.ndarray:
        """Reset all parallel environments and return observations.

        Returns:
            Observation matrix of shape (num_envs, 18).
        """
        if not WARP_AVAILABLE or self.mjw_model is None:
            return np.zeros((self.num_envs, 18), dtype=np.float32)

        mjw.reset_data(self.mjw_model, self.mjw_data)
        mjw.forward(self.mjw_model, self.mjw_data)
        return self.get_observations()

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Step all parallel environments forward in physics simulation.

        Args:
            actions: Action matrix of shape (num_envs, 6).

        Returns:
            Tuple of (observations, rewards, dones).
        """
        if not WARP_AVAILABLE or self.mjw_model is None:
            rewards = np.zeros(self.num_envs, dtype=np.float32)
            dones = np.zeros(self.num_envs, dtype=bool)
            obs = np.zeros((self.num_envs, 18), dtype=np.float32)
            return obs, rewards, dones

        actions_wp = wp.from_numpy(actions, dtype=wp.float32, device=self.mjw_data.ctrl.device)
        wp.copy(self.mjw_data.ctrl, actions_wp)

        mjw.step(self.mjw_model, self.mjw_data)

        obs = self.get_observations()
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        return obs, rewards, dones

    def get_observations(self) -> np.ndarray:
        """Extract observation state vector from parallel GPU/CPU buffers.

        Returns:
            NumPy array of observations of shape (num_envs, 18).
        """
        if not WARP_AVAILABLE or self.mjw_data is None:
            return np.zeros((self.num_envs, 18), dtype=np.float32)

        qpos_np = self.mjw_data.qpos.numpy()
        qvel_np = self.mjw_data.qvel.numpy()
        return np.hstack([qpos_np, qvel_np])
