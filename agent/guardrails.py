"""Guardrails for closed-loop grasping: phases, target lock, retry/escalation, grasp verdict.

The pick loop is an explicit state machine (``Phase``) instead of a handful of booleans, and the
object being picked is a ``TargetLock`` (a physical object identified once, from a frame taken
before any motion) instead of an image point that every call re-derives. All geometry here is in
robot millimetres in the table plane.
"""

import math
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# A request whose table point lies within this distance of the lock centre is the same object.
# It absorbs the model's own pointing drift (~25 mm at a 5 cm hover); two same-colour objects
# closer than this are ambiguous anyway.
LOCK_MATCH_RADIUS_MM = 40.0
# Hovers allowed at one lock without a successful descent, grasp misses allowed per lock, and
# consecutive refusals (no successful motion in between) before the loop escalates to the user.
MAX_HOVERS_PER_LOCK = 4
MAX_GRASP_ATTEMPTS = 3
MAX_CONSECUTIVE_REFUSALS = 3
# A hover must still be where it was measured: current FK XY within this of the commanded XY.
HOVER_STILL_VALID_MM = 10.0
# The lift is verified at no less than this height (cm): below ~4.5 cm a lifted cube still
# overlaps its table footprint in the image (IoU up to 0.6 at 3 cm vs <= 0.3 at 5 cm).
MIN_LIFT_HEIGHT_CM = 5.0
# A blob wider than this cannot be one graspable object (two same-colour objects merged).
MAX_GRASPABLE_WIDTH_MM = 55.0

ESCALATION_MESSAGES = {
    "grasp_attempts": "Grasp failed 3 times — waiting for the user.",
    "refusals": "Three consecutive refusals without progress — waiting for the user.",
    "hovers": "Could not align over the target after 4 hovers — waiting for the user.",
    "alignment_unobservable": (
        "Alignment cannot be verified (no jaw tag) and DESCENT_GATE=block — waiting for the user."
    ),
}


class Phase(str, Enum):
    """Where the pick cycle is."""

    APPROACH = "approach"
    HELD_UNVERIFIED = "held_unverified"
    CLOSED_EMPTY = "closed_empty"
    TRANSPORT = "transport"


@dataclass
class GraspVerdict:
    """Consolidated verdict combining the post-lift gripper sensor and the visual veto.

    Attributes:
        verdict: 'confirmed', 'missed', or 'unverified'.
        grasped: True if confirmed, False if missed, None if unverified.
        reason: Explanation of the combined decision.
        transport_allowed: True only when verdict is 'confirmed'.
        consumes_attempt: Whether this verdict counts against the retry budget.
        vision: The visual vote that went into the decision ('on_table', 'not_seen_on_table',
            'unavailable', or None when vision was not consulted).
    """

    verdict: str
    grasped: bool | None
    reason: str
    transport_allowed: bool
    consumes_attempt: bool = True
    vision: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert verdict to dictionary representation for tool payload."""
        out: dict[str, Any] = {
            "verdict": self.verdict,
            "grasped": self.grasped,
            "reason": self.reason,
            "transport_allowed": self.transport_allowed,
        }
        if self.vision is not None:
            out["vision"] = self.vision
        return out


def decide_grasp(aperture_after_lift: bool | None, vote: str | None) -> GraspVerdict:
    """Decide the grasp outcome after the lift: the sensor decides, vision can only veto.

    ======================  ==========================  ==========================================
    post-lift aperture      vision vote                 verdict
    ======================  ==========================  ==========================================
    False (empty/slipped)   any                         missed (closed_on_air)
    None (sensor failed)    any                         unverified (sensor_unavailable)
    True                    on_table                    missed (object_still_on_table)
    True                    unavailable / None          unverified (perception_unavailable)
    True                    not_seen_on_table           confirmed (sensor)
    ======================  ==========================  ==========================================

    An ``unavailable`` vision vote asks for a re-verify (a second lift) and does not consume a
    retry attempt: the grasp may be fine, the camera just could not say.

    Args:
        aperture_after_lift: ``classify_grasp(...)["grasped"]`` re-read after the lift.
        vote: 'on_table', 'not_seen_on_table', 'unavailable', or None.

    Returns:
        GraspVerdict instance.
    """
    if aperture_after_lift is False:
        return GraspVerdict("missed", False, "closed_on_air", False, True, vote)
    if aperture_after_lift is None:
        return GraspVerdict("unverified", None, "sensor_unavailable", False, True, vote)
    if vote == "on_table":
        return GraspVerdict("missed", False, "object_still_on_table", False, True, vote)
    if vote == "not_seen_on_table":
        return GraspVerdict("confirmed", True, "sensor", True, False, vote)
    return GraspVerdict("unverified", None, "perception_unavailable", False, False, "unavailable")


@dataclass
class HoverRecord:
    """A measured hover: where the arm was sent and what the jaw locator saw.

    Attributes:
        lock_id: Lock the hover belongs to.
        commanded_xy_mm: Table XY the hover was commanded to (lock centre plus correction).
        is_aligned: True/False when the jaw was observed, None when alignment is unobservable.
        residual_mm: Lock centre minus measured jaw XY ([dx, dy]) or None.
        source: Jaw locator source ('aruco', 'kinematic', 'none').
    """

    lock_id: int
    commanded_xy_mm: tuple[float, float]
    is_aligned: bool | None
    residual_mm: tuple[float, float] | None
    source: str


@dataclass
class TargetLock:
    """A physical object the loop has committed to.

    Attributes:
        lock_id: Sequential identifier (a new object gets a new id).
        color: Colour class of the object, if known.
        centre_xy_mm: Footprint centre in robot mm ([x, y]).
        centre_1000: The same point as a normalised [y, x] image point.
        bbox_px: Bounding box of the blob at lock time (x, y, w, h), or None for a raw lock.
        width_mm: Bottom-edge width in mm, or None.
        raw_point: The [y, x] point the model sent when the lock was created.
        stale: True after a miss followed by open_gripper; the next request re-acquires.
        c_mm: Accumulated bias correction ([dx, dy] mm) applied to hovers and the descent.
        hovers_since_descent: Hovers at this lock since the last successful descent.
        grasp_attempts: Missed or unverified grasps at this lock.
        refusals: Refusals issued while this lock was active.
    """

    lock_id: int
    color: str | None
    centre_xy_mm: tuple[float, float]
    centre_1000: list[float]
    bbox_px: list[int] | None = None
    width_mm: float | None = None
    raw_point: list[float] = field(default_factory=list)
    stale: bool = False
    c_mm: tuple[float, float] = (0.0, 0.0)
    hovers_since_descent: int = 0
    grasp_attempts: int = 0
    refusals: int = 0

    def distance_to_mm(self, xy_mm: tuple[float, float] | list[float]) -> float:
        """Return the table distance from the lock centre to ``xy_mm``."""
        return math.hypot(xy_mm[0] - self.centre_xy_mm[0], xy_mm[1] - self.centre_xy_mm[1])

    def commanded_xy_mm(self) -> tuple[float, float]:
        """Return the table XY the arm is commanded to: lock centre plus correction."""
        return (self.centre_xy_mm[0] + self.c_mm[0], self.centre_xy_mm[1] + self.c_mm[1])


class GuardrailState:
    """Per-session state of the closed pick loop, mutated in place and guarded by an epoch.

    ``begin_instruction()`` and ``on_interrupt()`` increment ``epoch``. A tool handler captures
    the epoch when it starts and commits its writes through ``commit(epoch, fn)``, which drops
    them if the epoch has moved on: a handler thread that outlives an interrupt cannot corrupt
    the state of the next instruction.
    """

    def __init__(self) -> None:
        """Create a fresh state: APPROACH phase, no lock, nothing escalated."""
        self._lock = threading.RLock()
        self.epoch = 0
        self.phase: Phase = Phase.APPROACH
        self.target: TargetLock | None = None
        self._next_lock_id = 1
        self.last_motion: str | None = None
        self.last_hover: HoverRecord | None = None
        self.last_commanded_gripper: str | None = None
        self.last_grasp_reason: str | None = None
        self.last_verdict: dict[str, Any] | None = None
        self.grasp_xy_mm: tuple[float, float] | None = None
        self.consecutive_refusals = 0
        self.escalated = False
        self.escalation_reason: str | None = None
        self.attempt_counted = False

    # -- derived views -------------------------------------------------------------------
    @property
    def gripper_closed(self) -> bool:
        """True while the gripper is (believed) closed: any phase other than APPROACH."""
        return self.phase != Phase.APPROACH

    @property
    def transport_allowed(self) -> bool:
        """True only after a confirmed lift."""
        return self.phase == Phase.TRANSPORT

    @property
    def gripper_state(self) -> str:
        """Gripper state label used in tool payloads."""
        return "open" if self.phase == Phase.APPROACH else "closed"

    @property
    def retry_count(self) -> int:
        """Missed or unverified grasps at the current lock."""
        return self.target.grasp_attempts if self.target is not None else 0

    @property
    def escalation_message(self) -> str:
        """User-facing text for the active escalation."""
        return ESCALATION_MESSAGES.get(
            self.escalation_reason or "", ESCALATION_MESSAGES["grasp_attempts"]
        )

    # -- epoch ---------------------------------------------------------------------------
    def commit(self, epoch: int, fn: Callable[["GuardrailState"], None]) -> bool:
        """Apply ``fn`` to the state if ``epoch`` is still current.

        Args:
            epoch: The epoch the calling handler captured when it started.
            fn: Mutation to run under the state lock.

        Returns:
            True if applied, False if the write was dropped as stale.
        """
        with self._lock:
            if epoch != self.epoch:
                return False
            fn(self)
            return True

    def begin_instruction(
        self,
        phase: Phase = Phase.APPROACH,
    ) -> None:
        """Start a new instruction: reset per-instruction state, keep the physical situation.

        The lock, corrector, retry counters and escalation reset; the phase (derived by the
        caller from the last commanded gripper target and the gripper sensor) and the last
        verdict survive, so a confirmed hold carries across an instruction boundary.

        Args:
            phase: Phase derived from the gripper's actual state.
        """
        with self._lock:
            self.epoch += 1
            self.phase = phase
            self.target = None
            self.last_motion = None
            self.last_hover = None
            self.consecutive_refusals = 0
            self.escalated = False
            self.escalation_reason = None
            self.attempt_counted = False
            if phase == Phase.APPROACH:
                self.last_verdict = None
                self.last_grasp_reason = None
                self.grasp_xy_mm = None

    def on_interrupt(self) -> None:
        """Handle an interrupt: invalidate in-flight handlers and drop the lock and hover.

        The phase is kept: an interrupt does not open the gripper.
        """
        with self._lock:
            self.epoch += 1
            self.target = None
            self.last_motion = None
            self.last_hover = None
            self.consecutive_refusals = 0

    def derive_phase(
        self,
        grasped: bool | None,
        last_commanded_gripper: str | None,
    ) -> Phase:
        """Derive the phase at an instruction boundary from the gripper sensor.

        HOME's partly closed gripper (-42 deg) classifies as "grasped", so the sensor alone is
        not enough: only a gripper this session commanded closed can be holding something.

        Args:
            grasped: ``classify_grasp(read_gripper_state())["grasped"]`` (None if unavailable).
            last_commanded_gripper: 'open', 'closed' or None (never commanded).

        Returns:
            The phase to start the instruction in.
        """
        if last_commanded_gripper != "closed":
            return Phase.APPROACH
        if grasped is True:
            confirmed = bool(self.last_verdict and self.last_verdict.get("verdict") == "confirmed")
            return Phase.TRANSPORT if confirmed else Phase.HELD_UNVERIFIED
        return Phase.CLOSED_EMPTY

    # -- target lock ---------------------------------------------------------------------
    def new_lock(
        self,
        color: str | None,
        centre_xy_mm: tuple[float, float],
        centre_1000: list[float],
        bbox_px: list[int] | None,
        width_mm: float | None,
        raw_point: list[float],
    ) -> TargetLock:
        """Create and activate a lock on a new object (counters start at zero)."""
        with self._lock:
            lock = TargetLock(
                lock_id=self._next_lock_id,
                color=color,
                centre_xy_mm=centre_xy_mm,
                centre_1000=centre_1000,
                bbox_px=bbox_px,
                width_mm=width_mm,
                raw_point=list(raw_point),
            )
            self._next_lock_id += 1
            self.target = lock
            self.last_hover = None
            return lock

    def reacquire(
        self,
        lock: TargetLock,
        centre_xy_mm: tuple[float, float],
        centre_1000: list[float],
        bbox_px: list[int] | None,
        width_mm: float | None,
    ) -> None:
        """Refresh a stale lock from a new detection, keeping its correction and counters."""
        with self._lock:
            lock.centre_xy_mm = centre_xy_mm
            lock.centre_1000 = centre_1000
            lock.bbox_px = bbox_px
            lock.width_mm = width_mm
            lock.stale = False
            lock.hovers_since_descent = 0
            self.last_hover = None

    # -- loop bookkeeping ----------------------------------------------------------------
    def escalate(self, reason: str) -> None:
        """Stop the loop and wait for the user."""
        with self._lock:
            self.escalated = True
            self.escalation_reason = reason

    def record_refusal(self) -> None:
        """Count a refusal; escalate after too many in a row without a successful motion."""
        with self._lock:
            self.consecutive_refusals += 1
            if self.target is not None:
                self.target.refusals += 1
            if self.consecutive_refusals >= MAX_CONSECUTIVE_REFUSALS and not self.escalated:
                self.escalate("refusals")

    def record_motion_success(self) -> None:
        """A motion tool succeeded: the refusal streak is over."""
        with self._lock:
            self.consecutive_refusals = 0

    def record_miss_or_unverified(self, reason: str, consumes_attempt: bool = True) -> None:
        """Record a failed grasp verification against the lock's retry budget.

        Args:
            reason: Why the grasp did not verify.
            consumes_attempt: False for a verdict that only asks for a re-verify.
        """
        with self._lock:
            self.last_grasp_reason = reason
            if not consumes_attempt:
                return
            if not self.attempt_counted:
                self.attempt_counted = True
                if self.target is not None:
                    self.target.grasp_attempts += 1
            if self.target is not None and self.target.grasp_attempts >= MAX_GRASP_ATTEMPTS:
                if not self.escalated:
                    self.escalate("grasp_attempts")

    def on_closed(self, phase: Phase, grasp_xy_mm: tuple[float, float] | None) -> None:
        """The gripper was commanded closed.

        Args:
            phase: HELD_UNVERIFIED, or CLOSED_EMPTY when the aperture shows empty jaws.
            grasp_xy_mm: Where the grasp happened (FK jaw XY).
        """
        with self._lock:
            self.phase = phase
            self.last_commanded_gripper = "closed"
            self.grasp_xy_mm = grasp_xy_mm
            self.last_verdict = None
            self.last_motion = "close"

    def on_opened(self) -> None:
        """The gripper was commanded open: the cycle ends; a missed lock goes stale."""
        with self._lock:
            was = self.phase
            self.phase = Phase.APPROACH
            self.last_commanded_gripper = "open"
            self.attempt_counted = False
            self.last_verdict = None
            self.last_motion = "open"
            self.last_hover = None
            self.consecutive_refusals = 0
            if was == Phase.TRANSPORT:
                self.target = None  # released: the cycle is complete
            elif self.target is not None:
                self.target.stale = True
