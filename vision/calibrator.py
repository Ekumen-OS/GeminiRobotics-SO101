"""Shared ArUco workspace calibrator singleton registry and calibration summaries."""

import threading

from robot.calibration import ArucoWorkspaceCalibrator

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
