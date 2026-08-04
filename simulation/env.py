"""SO-101 Reach Gymnasium Environment."""

from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from simulation.sim_robot import SO101Robot


class SO101ReachEnv(gym.Env):
    """Gymnasium environment for target reaching with SO-101 6-DOF arm."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        xml_path: str | None = None,
        render_mode: str | None = None,
        max_episode_steps: int = 200,
    ):
        """Initialize SO-101 reach Gymnasium environment.

        Args:
            xml_path: Path to scene MJCF XML file.
            render_mode: Render mode ('human' or 'rgb_array').
            max_episode_steps: Maximum step count per episode.
        """
        super().__init__()
        if xml_path is None:
            xml_path = str(Path(__file__).parent / "assets" / "scene.xml")

        self.robot = SO101Robot(xml_path=xml_path)
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.current_step = 0

        # Action: 6 delta joint position commands
        self.action_space = spaces.Box(low=-0.05, high=0.05, shape=(6,), dtype=np.float32)

        # Observation: joint_pos (6) + joint_vel (6) + ee_pos (3) + target_pos (3)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(18,), dtype=np.float32)

    def _get_obs(self) -> np.ndarray:
        qpos = self.robot.get_joint_positions()
        qvel = self.robot.get_joint_velocities()
        ee_pos, _ = self.robot.get_end_effector_pose()
        target_pos = self.robot.data.body("target").xpos.copy()
        return np.concatenate([qpos, qvel, ee_pos, target_pos]).astype(np.float32)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset environment state and randomize target position.

        Args:
            seed: Random seed for environment initialization.
            options: Additional options dictionary.

        Returns:
            Tuple of (initial_observation, info_dict).
        """
        super().reset(seed=seed)
        self.current_step = 0

        self.robot.reset("HOME")

        if seed is not None:
            np.random.seed(seed)
        rand_offset = np.random.uniform([-0.03, -0.04, 0.0], [0.03, 0.04, 0.0])
        base_target = np.array([0.18, 0.0, 0.015])
        self.robot.data.body("target").xpos[:] = base_target + rand_offset

        mujoco.mj_forward(self.robot.model, self.robot.data)
        obs = self._get_obs()
        return obs, {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Apply action step to the environment.

        Args:
            action: Array of 6 delta joint commands.

        Returns:
            Tuple of (observation, reward, terminated, truncated, info_dict).
        """
        self.current_step += 1

        current_qpos = self.robot.get_joint_positions()
        target_qpos = current_qpos + action
        self.robot.set_joint_positions(target_qpos)

        self.robot.step(num_steps=10)

        obs = self._get_obs()
        ee_pos = obs[12:15]
        target_pos = obs[15:18]

        dist = float(np.linalg.norm(ee_pos - target_pos))
        reward = -dist - 0.01 * float(np.linalg.norm(action))

        success = dist < 0.03
        if success:
            reward += 10.0

        terminated = success
        truncated = self.current_step >= self.max_episode_steps

        info = {"distance": dist, "success": success}
        return obs, reward, terminated, truncated, info
