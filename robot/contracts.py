"""Typed contract definitions for robot perception outputs."""

from typing import TypedDict


class MeasuredOffset(TypedDict):
    """Measured 2D coordinate displacement between gripper fingertip and target contact."""

    dy: float
    dx: float
    distance: float


class VisualAlignmentResponse(TypedDict):
    """Contract schema for compute_visual_alignment payload defined in ARCHITECTURE.md §3.2."""

    gripper_point_1000: list[float]
    target_point_1000: list[float]
    measured_offset: MeasuredOffset
    is_aligned: bool
    alignment_tolerance: float


class GraspElevationResponse(TypedDict):
    """Contract schema for verify_grasp_elevation payload defined in ARCHITECTURE.md §3.2."""

    object_elevated: bool
    object_detected_on_table: bool
    grasped: bool
    confidence: float
