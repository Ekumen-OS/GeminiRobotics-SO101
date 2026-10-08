"""Physical grasp classification engine for SO-101 gripper."""

from robot.constants import GRIPPER_CLOSE_DEG
from robot.contracts import GraspClassification, GripperReading
from robot.settings import Settings, get_settings


def classify_grasp(
    reading: GripperReading,
    settings: Settings | None = None,
) -> GraspClassification:
    """Classify physical grasp outcome based on measured gripper state.

    Evaluates whether the gripper closed on empty air, stalled on a grasped object,
    or failed to produce a valid sensory reading. The closed-on-air threshold is the
    calibrated empty-jaw aperture (``settings.gripper_empty_aperture_deg``, falling back to
    ``GRIPPER_CLOSE_DEG``) plus ``settings.grasp_empty_margin_deg``.

    Args:
        reading: GripperReading containing aperture_deg, load, and source.
        settings: Optional Settings instance. If None, uses get_settings().

    Returns:
        GraspClassification containing grasped (bool | None), aperture_deg,
        closed_on_air_threshold_deg, load, and reason.
    """
    if settings is None:
        settings = get_settings()

    empty_margin = float(settings.grasp_empty_margin_deg)
    # Anchor to the calibrated empty-jaw reading when known; the servo's fully-closed angle
    # depends on the LeRobot calibration and is not necessarily GRIPPER_CLOSE_DEG.
    anchor = settings.gripper_empty_aperture_deg
    empty_aperture = float(GRIPPER_CLOSE_DEG if anchor is None else anchor)
    threshold = float(empty_aperture + empty_margin)
    aperture = float(reading["aperture_deg"])
    load = reading["load"]
    source = reading["source"]

    if source == "unavailable":
        return {
            "grasped": None,
            "aperture_deg": aperture,
            "closed_on_air_threshold_deg": threshold,
            "load": load,
            "reason": "sensor_unavailable",
        }

    if aperture <= threshold:
        return {
            "grasped": False,
            "aperture_deg": aperture,
            "closed_on_air_threshold_deg": threshold,
            "load": load,
            "reason": "closed_on_air",
        }

    # Aperture is above the empty margin (jaws stalled before fully closing)
    if settings.grasp_load_min is not None:
        if load is None or float(load) < float(settings.grasp_load_min):
            return {
                "grasped": False,
                "aperture_deg": aperture,
                "closed_on_air_threshold_deg": threshold,
                "load": load,
                "reason": "no_load",
            }

    return {
        "grasped": True,
        "aperture_deg": aperture,
        "closed_on_air_threshold_deg": threshold,
        "load": load,
        "reason": "aperture",
    }
