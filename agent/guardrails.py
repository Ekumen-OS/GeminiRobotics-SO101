"""Guardrails for closed-loop grasping, visual servoing, and motion gating.

Provides bias correction (OffsetCorrector), multi-modal grasp verdict synthesis
(decide_grasp), and per-instruction safety tracking (GuardrailState).
"""

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GraspVerdict:
    """Consolidated verdict combining physical sensor and visual elevation votes.

    Attributes:
        verdict: 'confirmed', 'missed', or 'unverified'.
        grasped: True if confirmed, False if missed, None if unverified.
        reason: Explanation of the combined decision.
        transport_allowed: True only when verdict is 'confirmed'.
    """

    verdict: str
    grasped: bool | None
    reason: str
    transport_allowed: bool

    def to_dict(self) -> dict[str, Any]:
        """Convert verdict to dictionary representation for tool payload."""
        return {
            "verdict": self.verdict,
            "grasped": self.grasped,
            "reason": self.reason,
            "transport_allowed": self.transport_allowed,
        }


def decide_grasp(
    aperture_grasped: bool | None,
    elevation_grasped: bool | None,
) -> GraspVerdict:
    """Synthesize multi-modal grasp verdict from physical sensor and visual elevation.

    Rules:
        - aperture False -> missed (sensor wins, cube absent or empty jaws).
        - aperture True and elevation True -> confirmed (both sensor & vision agree).
        - aperture True and elevation False -> missed (object still resting on table).
        - any None -> unverified (sensor or perception unavailable).

    Args:
        aperture_grasped: Verdict from robot.read_gripper_state() and classify_grasp().
        elevation_grasped: Verdict from verify_grasp_elevation().

    Returns:
        GraspVerdict instance.
    """
    if aperture_grasped is False:
        return GraspVerdict(
            verdict="missed",
            grasped=False,
            reason="closed_on_air",
            transport_allowed=False,
        )

    if aperture_grasped is True and elevation_grasped is True:
        return GraspVerdict(
            verdict="confirmed",
            grasped=True,
            reason="verified_by_sensor_and_vision",
            transport_allowed=True,
        )

    if aperture_grasped is True and elevation_grasped is False:
        return GraspVerdict(
            verdict="missed",
            grasped=False,
            reason="object_still_on_table",
            transport_allowed=False,
        )

    # Any None vote (aperture is None or elevation is None)
    reason = "sensor_unavailable" if aperture_grasped is None else "perception_unavailable"
    return GraspVerdict(
        verdict="unverified",
        grasped=None,
        reason=reason,
        transport_allowed=False,
    )


class OffsetCorrector:
    """Bias corrector for visual servoing and spatial alignment.

    Stores measured [dy, dx] at hover (where offset = target - gripper) and applies
    it to subsequent moves towards that target before homography, clamping to ±60 units.
    Consumed once applied or when model self-corrects so corrections do not stack.
    """

    def __init__(self, max_clamp: float = 60.0, match_distance: float = 5.0) -> None:
        """Initialize offset corrector.

        Args:
            max_clamp: Maximum coordinate shift allowed in normalized units (default ±60).
            match_distance: Distance radius in normalized units to match target (default 5).
        """
        self.max_clamp = max_clamp
        self.match_distance = match_distance
        self.target_1000: list[float] | None = None
        self.offset: list[float] | None = None
        self.correction_note: str | None = None

    def record_hover(self, target_1000: list[float], offset: list[float]) -> None:
        """Record measured offset [dy, dx] for a hover target.

        Args:
            target_1000: Target coordinates [y, x] in [0, 1000].
            offset: Measured offset [dy, dx] where offset = target - gripper.
        """
        self.target_1000 = [float(target_1000[0]), float(target_1000[1])]
        self.offset = [float(offset[0]), float(offset[1])]
        self.correction_note = None

    def consume_correction(
        self, point_1000: list[float]
    ) -> tuple[list[float], list[float] | None, str | None]:
        """Apply stored offset if requested point is within match_distance of recorded target.

        Rules:
            - If requested point is within 5 units of recorded hover target:
              apply measured correction (clamped to ±max_clamp), consume offset,
              and return (corrected_point_1000, applied_correction_1000, None).
            - If requested point is within 5 units of (target + offset):
              model already corrected, consume offset without applying,
              and return (point_1000, None, "model-corrected").
            - Otherwise: no-op, do not consume, return (point_1000, None, None).

        Args:
            point_1000: Requested target point [y, x].

        Returns:
            Tuple of (point_1000, applied_correction_1000, correction_note).
            If no correction was applied, applied_correction_1000 is None.
        """
        if self.target_1000 is None or self.offset is None:
            return [float(point_1000[0]), float(point_1000[1])], None, None

        dist_target = math.hypot(
            point_1000[0] - self.target_1000[0],
            point_1000[1] - self.target_1000[1],
        )

        target_plus_offset = [
            self.target_1000[0] + self.offset[0],
            self.target_1000[1] + self.offset[1],
        ]
        dist_model_corrected = math.hypot(
            point_1000[0] - target_plus_offset[0],
            point_1000[1] - target_plus_offset[1],
        )

        if dist_target <= self.match_distance:
            # Model re-issued the same target -> apply measured correction once
            dy_corr = max(-self.max_clamp, min(self.max_clamp, self.offset[0]))
            dx_corr = max(-self.max_clamp, min(self.max_clamp, self.offset[1]))

            corrected_y = max(0.0, min(1000.0, float(point_1000[0]) + dy_corr))
            corrected_x = max(0.0, min(1000.0, float(point_1000[1]) + dx_corr))

            applied = [dy_corr, dx_corr]
            self.target_1000 = None
            self.offset = None
            self.correction_note = None
            return [corrected_y, corrected_x], applied, None

        if dist_model_corrected <= self.match_distance:
            # Model already applied the correction -> consume without applying
            self.target_1000 = None
            self.offset = None
            self.correction_note = "model-corrected"
            return [float(point_1000[0]), float(point_1000[1])], None, "model-corrected"

        # Otherwise no-op
        return [float(point_1000[0]), float(point_1000[1])], None, None

    def reset(self) -> None:
        """Clear recorded hover targets and offsets."""
        self.target_1000 = None
        self.offset = None
        self.correction_note = None


@dataclass
class GuardrailState:
    """Per-instruction state tracking for closed-loop grasping safety gates.

    Carried in ToolContext and reset by interrupt() or new instruction.
    """

    last_hover_target: list[float] | None = None
    last_hover_offset: list[float] | None = None
    last_hover_is_aligned: bool | None = None
    corrector: OffsetCorrector = field(default_factory=OffsetCorrector)
    gripper_state: str = "open"
    gripper_closed: bool = False
    last_grasp_target: list[float] | None = None
    last_grasp_reason: str | None = None
    pre_grasp_frame: bytes | None = None
    transport_allowed: bool = False
    retry_count: int = 0
    escalated: bool = False
    attempt_counted: bool = False

    def record_miss_or_unverified(self, reason: str) -> None:
        """Record a failed grasp verification, incrementing retry counter."""
        if not self.attempt_counted:
            self.retry_count += 1
            self.attempt_counted = True
        self.transport_allowed = False
        self.last_grasp_reason = reason
        if self.retry_count >= 3:
            self.escalated = True

    def open_gripper(self) -> None:
        """Clear closed-state variables when the gripper is opened."""
        self.gripper_closed = False
        self.gripper_state = "open"
        self.transport_allowed = False
        self.pre_grasp_frame = None
        self.attempt_counted = False

    def reset(self) -> None:
        """Reset state for a fresh instruction."""
        self.last_hover_target = None
        self.last_hover_offset = None
        self.last_hover_is_aligned = None
        self.corrector.reset()
        self.gripper_state = "open"
        self.gripper_closed = False
        self.last_grasp_target = None
        self.last_grasp_reason = None
        self.pre_grasp_frame = None
        self.transport_allowed = False
        self.retry_count = 0
        self.escalated = False
        self.attempt_counted = False
