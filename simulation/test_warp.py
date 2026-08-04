"""Test MuJoCo Warp parallel batched environment."""

import numpy as np

from simulation.warp_env import SO101WarpEnv


def test_warp() -> None:
    """Test parallel environment simulation with MuJoCo Warp."""
    num_envs = 64
    print(f"Testing MuJoCo Warp with {num_envs} parallel environments...")
    env = SO101WarpEnv(num_envs=num_envs)

    obs = env.reset()
    print(f"Initial observations shape: {obs.shape}")

    actions = np.random.uniform(-0.02, 0.02, size=(num_envs, 6)).astype(np.float32)
    obs, rewards, dones = env.step(actions)
    print(f"Step completed. Output shape: {obs.shape}")
    print("MuJoCo Warp test completed successfully.")


if __name__ == "__main__":
    test_warp()
