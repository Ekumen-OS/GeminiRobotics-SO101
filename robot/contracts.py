"""Typed contract definitions for robot perception outputs."""

from typing import Literal, TypedDict


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


class MeasuredOffset(TypedDict):
    """Measured 2D coordinate displacement between gripper fingertip and target contact."""

    dy: float
    dx: float
    distance: float


class VisualAlignmentResponse(TypedDict):
    """Contract schema for compute_visual_alignment payload defined in ARCHITECTURE.md §3.2."""

    gripper_point_1000: list[float] | None
    target_point_1000: list[float]
    measured_offset: MeasuredOffset | None
    is_aligned: bool | None
    alignment_tolerance: float
    gripper_source: str


class GraspElevationResponse(TypedDict):
    """Contract schema for verify_grasp_elevation payload defined in ARCHITECTURE.md §3.2."""

    object_elevated: bool | None
    object_detected_on_table: bool
    grasped: bool | None
    confidence: float
