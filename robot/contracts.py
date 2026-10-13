"""Typed contract definitions for robot perception outputs."""

from typing import Literal, NotRequired, TypedDict


class GripperReading(TypedDict):
    """Measured or commanded gripper joint state.

    Attributes:
        aperture_deg: Gripper joint angle in degrees.
        load: Measured actuator load or force, or None if unavailable/unsupported.
        source: Provenance of the reading ('measured', 'commanded', or 'unavailable').
    """

    aperture_deg: float
    load: float | None
    source: Literal["measured", "commanded", "unavailable"]


class GraspClassification(TypedDict):
    """Classification of physical grasp outcome based on gripper sensor reading.

    Attributes:
        grasped: True if object held between jaws, False if closed on air / no load,
            None if sensor is unavailable.
        aperture_deg: Gripper aperture angle in degrees at measurement time.
        closed_on_air_threshold_deg: Threshold angle below which gripper is deemed empty.
        load: Measured load/force or None.
        reason: Classification rationale ('closed_on_air', 'aperture', 'no_load',
            or 'sensor_unavailable').
    """

    grasped: bool | None
    aperture_deg: float
    closed_on_air_threshold_deg: float
    load: float | None
    reason: str


class MeasuredOffsetMm(TypedDict):
    """Table-plane displacement from the jaw to the target centre, in robot millimetres."""

    dx_mm: float
    dy_mm: float
    distance_mm: float


class AlignmentMeasurement(TypedDict):
    """Alignment payload of the closed pick loop (ARCHITECTURE.md §3.2), all in robot mm.

    Attributes:
        gripper_xy_mm: Jaw (grasp site) XY in robot mm, or None if the jaw was not located.
        target_xy_mm: Locked target centre XY in robot mm.
        measured_offset: Target centre minus jaw, or None when the jaw is unlocated.
        is_aligned: True/False when observable; None for the kinematic fallback (FK is blind to
            calibration and encoder bias) or when the jaw is unlocated.
        alignment_tolerance_mm: Tolerance used for ``is_aligned``.
        gripper_source: 'aruco', 'kinematic' or 'none'.
    """

    gripper_xy_mm: list[float] | None
    target_xy_mm: list[float]
    measured_offset: MeasuredOffsetMm | None
    is_aligned: bool | None
    alignment_tolerance_mm: float
    gripper_source: str
    er2_gripper_point_1000: NotRequired[list[float]]
