"""Typed contract definitions for Gemini Live API robot tool execution payloads."""

from typing import Any, Literal, NotRequired, TypedDict

from robot.contracts import AlignmentMeasurement, GraspClassification

ErrorType = Literal[
    "argument",
    "calibration",
    "perception",
    "hardware",
    "internal",
    "descent_gate",
    "transport_gate",
    "retry_exhausted",
]


class ToolErrorResponse(TypedDict):
    """Contract schema for tool execution error payloads defined in ARCHITECTURE.md §3.3."""

    error: str
    error_type: ErrorType
    robot_moved: bool


class MoveToSpatialPointResponse(TypedDict):
    """Contract schema for move_to_spatial_point payload defined in ARCHITECTURE.md §3.3."""

    status: str
    target_xyz_m: NotRequired[list[float]]
    executed_qpos_deg: NotRequired[list[float]]
    approach: NotRequired[str]
    perception: NotRequired[AlignmentMeasurement | dict[str, Any]]
    visual_hint: NotRequired[str]
    warnings: NotRequired[list[str]]
    error: NotRequired[str]
    error_type: NotRequired[ErrorType]
    robot_moved: NotRequired[bool]
    applied_correction_mm: NotRequired[list[float] | None]
    target_xy_mm: NotRequired[list[float]]
    gripper_source: NotRequired[str | None]
    grasp_verification: NotRequired[dict[str, Any] | None]
    guardrail_warning: NotRequired[str | None]
    lift_note: NotRequired[str]


class GripperToolResponse(TypedDict):
    """Contract schema for open_gripper / close_gripper payloads defined in ARCHITECTURE.md §3.3."""

    status: str
    gripper_state: NotRequired[str]
    gripper_deg: NotRequired[float]
    executed_qpos_deg: NotRequired[list[float]]
    visual_hint: NotRequired[str]
    grasp: NotRequired[GraspClassification | None]
    warnings: NotRequired[list[str]]
    error: NotRequired[str]
    error_type: NotRequired[ErrorType]
    robot_moved: NotRequired[bool]
