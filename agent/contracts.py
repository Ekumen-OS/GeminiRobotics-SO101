"""Typed contract definitions for Gemini Live API robot tool execution payloads."""

from typing import Any, NotRequired, TypedDict

from robot.contracts import VisualAlignmentResponse


class MoveToSpatialPointResponse(TypedDict):
    """Contract schema for move_to_spatial_point payload defined in ARCHITECTURE.md §3.3."""

    status: str
    target_xyz_m: NotRequired[list[float]]
    executed_qpos_deg: NotRequired[list[float]]
    approach: NotRequired[str]
    perception: NotRequired[VisualAlignmentResponse | dict[str, Any]]
    visual_hint: NotRequired[str]
    warnings: NotRequired[list[str]]
    error: NotRequired[str]


class GripperToolResponse(TypedDict):
    """Contract schema for open_gripper / close_gripper payloads defined in ARCHITECTURE.md §3.3."""

    status: str
    gripper_state: NotRequired[str]
    gripper_deg: NotRequired[float]
    executed_qpos_deg: NotRequired[list[float]]
    visual_hint: NotRequired[str]
    warnings: NotRequired[list[str]]
    error: NotRequired[str]
