"""Shared ArUco workspace calibrator singleton registry and calibration summaries."""

import threading

from robot.calibration import ArucoWorkspaceCalibrator
from robot.settings import get_settings

_calibrator_lock = threading.Lock()
_global_calibrator: ArucoWorkspaceCalibrator | None = None


def get_calibrator(mode: str = "real") -> ArucoWorkspaceCalibrator:
    """Return the shared ArucoWorkspaceCalibrator singleton instance.

    Args:
        mode: Operational mode ('real' or 'sim').

    Returns:
        The active ArucoWorkspaceCalibrator instance.
    """
    global _global_calibrator
    with _calibrator_lock:
        if _global_calibrator is None:
            if mode == "sim":
                from simulation.sim_calibration import init_sim_workspace_calibration

                _global_calibrator = init_sim_workspace_calibration()
            else:
                _global_calibrator = ArucoWorkspaceCalibrator(mode=mode)
        elif mode == "sim" and not _global_calibrator.is_calibrated:
            from simulation.sim_calibration import init_sim_workspace_calibration

            _global_calibrator = init_sim_workspace_calibration(calibrator=_global_calibrator)
        return _global_calibrator


def get_ready_calibrator(mode: str = "real") -> ArucoWorkspaceCalibrator:
    """Return the calibrator for ``mode`` with its homography loaded.

    Args:
        mode: Operational mode ('real' or 'sim').

    Returns:
        The calibrated ArucoWorkspaceCalibrator instance for ``mode``.

    Raises:
        CalibrationUnavailable: If the workspace is not calibrated.
    """
    calib = get_calibrator(mode=mode)
    is_sim = mode == "sim"
    settings = get_settings()
    target_path = settings.calibration_sim_save_path if is_sim else settings.calibration_save_path
    if is_sim and (calib.mode != "sim" or not calib.is_calibrated):
        if target_path.exists():
            if calib.load_calibration(target_path):
                calib.mode = "sim"
        else:
            from simulation.sim_calibration import init_sim_workspace_calibration

            calib = init_sim_workspace_calibration(calibrator=calib)
    elif not is_sim and (calib.mode != "real" or not calib.is_calibrated):
        if target_path.exists():
            if calib.load_calibration(target_path):
                calib.mode = "real"
        elif calib.mode == "sim":
            calib.homography_matrix = None
            calib.mode = "real"
    calib.require_calibrated()
    return calib


def get_calibration_status_summary() -> str:
    """Return a formatted string describing current calibration status for the UI.

    Returns:
        Human-readable calibration status string.
    """
    calib = get_calibrator()
    status = calib.get_calibration_status()
    if status["calibrated"]:
        err = status.get("reprojection_error")
        err_str = (
            f" (err: {err:.1f}mm)"
            if isinstance(err, (int, float)) and not isinstance(err, bool)
            else ""
        )
        return f"✅ Calibrated{err_str} — {status['markers_detected']}/4 markers visible"
    elif status["markers_detected"] > 0:
        return f"⚠️ Incomplete: {status['markers_detected']}/4 markers visible (need 4)"
    else:
        return "❌ Not calibrated (no markers detected in view)"


def save_workspace_calibration_file() -> str:
    """Save current workspace homography to disk and return status message.

    Returns:
        Status message string from the calibrator.
    """
    calib = get_calibrator()
    return calib.save_calibration()
