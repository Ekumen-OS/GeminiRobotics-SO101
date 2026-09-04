"""Pytest fixtures and test environment configuration."""

import os
import sys
import tempfile
from pathlib import Path

import pytest

import robot.calibration
import simulation.sim_calibration
from simulation.sim_calibration import compute_sim_calibration

if sys.platform.startswith("linux") and "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"

# Isolate default calibration artifact paths so tests never write into config/
_TEST_CALIB_DIR = tempfile.TemporaryDirectory()
_ISOLATED_SIM_PATH = Path(_TEST_CALIB_DIR.name) / "workspace_homography_sim.npz"
_ISOLATED_REAL_PATH = Path(_TEST_CALIB_DIR.name) / "workspace_homography.npz"

simulation.sim_calibration.DEFAULT_SIM_SAVE_PATH = _ISOLATED_SIM_PATH
robot.calibration.DEFAULT_SIM_SAVE_PATH = _ISOLATED_SIM_PATH
robot.calibration.DEFAULT_SAVE_PATH = _ISOLATED_REAL_PATH


@pytest.fixture
def real_homography_path(tmp_path: Path) -> Path:
    """Build synthetic real-mode homography in tmp_path from workspace corner geometry.

    Args:
        tmp_path: Temporary directory fixture provided by pytest.

    Returns:
        Path to the generated real-mode homography .npz file.
    """
    save_path = tmp_path / "workspace_homography.npz"
    compute_sim_calibration(save_path=save_path)
    return save_path


@pytest.fixture
def sim_homography_path(tmp_path: Path) -> Path:
    """Build synthetic sim-mode homography in tmp_path from workspace corner geometry.

    Args:
        tmp_path: Temporary directory fixture provided by pytest.

    Returns:
        Path to the generated sim-mode homography .npz file.
    """
    save_path = tmp_path / "workspace_homography_sim.npz"
    compute_sim_calibration(save_path=save_path)
    return save_path
