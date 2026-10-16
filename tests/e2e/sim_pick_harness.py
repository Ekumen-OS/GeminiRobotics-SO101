"""Scripted pick-and-place harness for the MuJoCo simulation (no LLM).

Drives the real tool boundary (``execute_tool_call_sync``) with a real ``SO101Robot``, the sim
workspace calibration, and a frame provider that renders ``render_rgb_frame(640, 480)`` to JPEG.
A scripted "model" follows the system prompt: open, hover at 3 cm, re-hover only when a response
says so, descend, close, lift, transport, place, open, retract (or open and retry, at most 3
attempts). Every claim is checked against MuJoCo ground truth, so a verdict that contradicts the
physics is counted instead of trusted.

Not a test module (no ``test_`` prefix): ``tests/test_sim_pick_e2e.py`` runs it.
"""

import contextlib
import dataclasses
import re
import statistics
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import cv2
import mujoco
import numpy as np

from agent.guardrails import GuardrailState
from agent.tools import execute_tool_call_sync
from robot.interface import RobotInterface
from robot.registry import get_active_robot, set_active_robot
from robot.settings import Settings, get_settings, set_settings
from simulation.sim_calibration import init_sim_workspace_calibration
from simulation.sim_robot import SO101Robot
from vision.calibrator import get_calibrator
from vision.detector import detect_workspace_objects

# Pixel noise of the emulated model's pointing, in normalised [0, 1000] units.
MODEL_POINT_SIGMA = 8.0
MAX_HOVERS_PER_ATTEMPT = 4
MAX_ATTEMPTS = 3
CUBE_LIFTED_Z_M = 0.03
CUBE_HALF_M = 0.015
# Default placement area (m): reachable, clear of the HOME jaw, inside the calibrated mat.
DEFAULT_X_RANGE = (0.20, 0.30)
DEFAULT_Y_RANGE = (-0.10, 0.10)
# Table point (m) the scripted model places cubes at.
DESTINATION_XY_M = (0.25, -0.14)

_NEXT_CALL_MOVE = re.compile(
    r"move_to_spatial_point\(point=\[\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\],\s*z_height_cm=([-\d.]+)\)"
)


class PanOffsetRobot(RobotInterface):
    """Test-only wrapper that adds a fixed shoulder-pan encoder offset.

    The arm physically moves to ``q + delta`` but reports ``q_true - delta``, so forward
    kinematics from the reported joints is blind to the offset while the camera sees it.

    Attributes:
        inner: The wrapped ``SO101Robot``.
        delta_rad: Pan offset in radians.
    """

    mode = "sim"

    def __init__(self, inner: SO101Robot, delta_deg: float) -> None:
        """Wrap a sim robot.

        Args:
            inner: The robot to wrap.
            delta_deg: Shoulder-pan offset in degrees.
        """
        self.inner = inner
        self.delta_rad = float(np.radians(delta_deg))

    def _shift(self, qpos: np.ndarray, sign: float) -> np.ndarray:
        out = np.asarray(qpos, dtype=np.float64).copy()
        out[0] += sign * self.delta_rad
        return out

    def get_joint_positions(self) -> np.ndarray:
        """Return the reported joints (true pan minus the offset)."""
        return self._shift(self.inner.get_joint_positions(), -1.0)

    def set_joint_positions(self, qpos: np.ndarray) -> None:
        """Teleport to the commanded joints plus the offset."""
        self.inner.set_joint_positions(self._shift(qpos, 1.0))

    def move_to_pose(
        self,
        target_qpos: np.ndarray,
        duration_sec: float = 1.0,
        dt: float = 0.002,
        cancel_epoch: int | None = None,
    ) -> None:
        """Move to the commanded joints plus the offset."""
        self.inner.move_to_pose(
            self._shift(target_qpos, 1.0),
            duration_sec=duration_sec,
            dt=dt,
            cancel_epoch=cancel_epoch,
        )

    def interrupt(self) -> None:
        """Interrupt the wrapped robot."""
        self.inner.interrupt()

    @property
    def interrupt_epoch(self) -> int:  # type: ignore[override]
        """The wrapped robot's interrupt epoch."""
        return self.inner.interrupt_epoch

    def read_gripper_state(self) -> Any:
        """Return the wrapped robot's gripper reading."""
        return self.inner.read_gripper_state()

    def __getattr__(self, name: str) -> Any:
        """Delegate everything else (rendering, end-effector pose) to the wrapped robot."""
        return getattr(self.inner, name)


class ScriptedGripperRobot(RobotInterface):
    """Test-only wrapper whose gripper sensor can be scripted (e.g. stalled on nothing visible).

    Attributes:
        inner: The wrapped ``SO101Robot``.
        reading: When set, returned by ``read_gripper_state`` instead of the physics reading.
    """

    mode = "sim"

    def __init__(self, inner: SO101Robot) -> None:
        """Wrap a sim robot.

        Args:
            inner: The robot to wrap.
        """
        self.inner = inner
        self.reading: dict[str, Any] | None = None

    def get_joint_positions(self) -> np.ndarray:
        """Return the wrapped robot's joints."""
        return self.inner.get_joint_positions()

    def set_joint_positions(self, qpos: np.ndarray) -> None:
        """Teleport the wrapped robot."""
        self.inner.set_joint_positions(qpos)

    def move_to_pose(
        self,
        target_qpos: np.ndarray,
        duration_sec: float = 1.0,
        dt: float = 0.002,
        cancel_epoch: int | None = None,
    ) -> None:
        """Move the wrapped robot (the base-class no-op would otherwise shadow __getattr__)."""
        self.inner.move_to_pose(
            target_qpos, duration_sec=duration_sec, dt=dt, cancel_epoch=cancel_epoch
        )

    def interrupt(self) -> None:
        """Interrupt the wrapped robot."""
        self.inner.interrupt()

    @property
    def interrupt_epoch(self) -> int:  # type: ignore[override]
        """The wrapped robot's interrupt epoch."""
        return self.inner.interrupt_epoch

    def read_gripper_state(self) -> Any:
        """Return the scripted reading if set, else the physics reading."""
        if self.reading is not None:
            return dict(self.reading)
        return self.inner.read_gripper_state()

    def __getattr__(self, name: str) -> Any:
        """Delegate everything else to the wrapped robot."""
        return getattr(self.inner, name)


@dataclass
class PlacementMetrics:
    """Outcome of one scripted pick-and-place, measured against MuJoCo truth."""

    cube_start_xy_m: tuple[float, float]
    lifted: bool = False
    placed: bool = False
    placement_error_mm: float | None = None
    jaw_to_cube_at_close_mm: float | None = None
    contradictions: int = 0
    lift_refusals: int = 0
    refusals: dict[str, int] = field(default_factory=dict)
    refusal_loops: int = 0
    attempts: int = 0
    max_hovers_in_attempt: int = 0
    locks_created: int = 0
    transcript: list[str] = field(default_factory=list)


@dataclass
class ScenarioResult:
    """Metrics of one scenario (a set of placements)."""

    name: str
    placements: list[PlacementMetrics]
    wall_s: float = 0.0

    def summary(self) -> dict[str, Any]:
        """Aggregate the placements into the report metrics.

        Returns:
            Dictionary of counts and medians.
        """
        n = len(self.placements)
        jaw = [p.jaw_to_cube_at_close_mm for p in self.placements if p.jaw_to_cube_at_close_mm]
        place = [p.placement_error_mm for p in self.placements if p.placement_error_mm is not None]
        refusals: dict[str, int] = {}
        for p in self.placements:
            for k, v in p.refusals.items():
                refusals[k] = refusals.get(k, 0) + v
        return {
            "n": n,
            "lifted": sum(p.lifted for p in self.placements),
            "placed_within_15mm": sum(
                p.placed and p.placement_error_mm is not None and p.placement_error_mm <= 15.0
                for p in self.placements
            ),
            "jaw_to_cube_at_close_median_mm": statistics.median(jaw) if jaw else None,
            "jaw_to_cube_at_close_le_3mm": sum(
                p.jaw_to_cube_at_close_mm is not None and p.jaw_to_cube_at_close_mm <= 3.0
                for p in self.placements
            ),
            "jaw_to_cube_at_close_le_4mm": sum(
                p.jaw_to_cube_at_close_mm is not None and p.jaw_to_cube_at_close_mm <= 4.0
                for p in self.placements
            ),
            "placement_error_median_mm": statistics.median(place) if place else None,
            "contradictions": sum(p.contradictions for p in self.placements),
            "lift_refusals": sum(p.lift_refusals for p in self.placements),
            "refusal_loops": sum(p.refusal_loops for p in self.placements),
            "refusals": refusals,
            "max_hovers_in_attempt": max(
                (p.max_hovers_in_attempt for p in self.placements), default=0
            ),
            "attempts_total": sum(p.attempts for p in self.placements),
            "wall_s": round(self.wall_s, 1),
        }


def format_metrics_table(results: list[ScenarioResult]) -> str:
    """Render scenario summaries as a fixed-width table.

    Args:
        results: Scenario results, one column each.

    Returns:
        Multi-line string.
    """
    summaries = [r.summary() for r in results]
    keys = list(summaries[0].keys()) if summaries else []
    width = max([len(r.name) for r in results] + [14]) + 2
    lines = ["metric".ljust(34) + "".join(r.name.ljust(width) for r in results)]
    for key in keys:
        cells = []
        for s in summaries:
            v = s[key]
            cells.append((f"{v:.1f}" if isinstance(v, float) else str(v)).ljust(width))
        lines.append(key.ljust(34) + "".join(cells))
    return "\n".join(lines)


@contextlib.contextmanager
def settings_override(**overrides: Any) -> Iterator[Settings]:
    """Temporarily replace fields of the active settings.

    Args:
        **overrides: Settings field values.

    Yields:
        The temporary Settings instance.
    """
    original = get_settings()
    set_settings(dataclasses.replace(original, **overrides))
    try:
        yield get_settings()
    finally:
        set_settings(original)


class SimPickHarness:
    """Scripted picker over a real ``SO101Robot`` with MuJoCo ground truth."""

    def __init__(
        self,
        pan_offset_deg: float = 0.0,
        mode: str = "sim",
        jpeg_quality: int = 98,
        scripted_gripper: bool = False,
        descend_after_hover_cap: bool = False,
        point_offset_1000: tuple[float, float] = (0.0, 0.0),
    ) -> None:
        """Create the sim robot, calibration and frame provider.

        Args:
            pan_offset_deg: Hidden shoulder-pan encoder offset (0 = unbiased).
            mode: Tool routing mode.
            jpeg_quality: JPEG quality of the frames given to the tools.
            scripted_gripper: Wrap the robot so ``harness.robot.reading`` can script the
                gripper sensor.
            descend_after_hover_cap: After the hover cap, descend anyway instead of giving up.
            point_offset_1000: Constant [dy, dx] added to the model pointing point.
        """
        self.descend_after_hover_cap = descend_after_hover_cap
        self.point_offset_1000 = point_offset_1000
        self.sim = SO101Robot()
        self.robot: RobotInterface = (
            PanOffsetRobot(self.sim, pan_offset_deg) if pan_offset_deg else self.sim
        )
        if scripted_gripper:
            self.robot = ScriptedGripperRobot(self.sim)
        self.mode = mode
        self.jpeg_quality = jpeg_quality
        self.calib = get_calibrator(mode="sim")
        if not self.calib.is_calibrated or getattr(self.calib, "mode", None) != "sim":
            init_sim_workspace_calibration(calibrator=self.calib)
        self._cube_adr = self.sim.model.jnt_qposadr[self.sim.model.joint("target_joint").id]
        self.state = GuardrailState()
        self.log: list[str] = []
        # An EGL context is bound to the thread that created it, while tool handlers run on
        # changing worker threads: every render goes through one dedicated thread.
        self._render_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="harness-render")
        self._previous_active = get_active_robot()
        set_active_robot(self.robot)

    # -- truth -----------------------------------------------------------------------------
    def cube_xyz(self) -> np.ndarray:
        """Return the true cube centre (m)."""
        return self.sim.data.qpos[self._cube_adr : self._cube_adr + 3].copy()

    def site_xyz(self) -> np.ndarray:
        """Return the true grasp-site position (m)."""
        return self.sim.get_end_effector_pose()[0].copy()

    def place_cube(self, x: float, y: float, yaw: float = 0.0) -> None:
        """Teleport the cube to a table position (m) and let it settle.

        Args:
            x: Forward position (m).
            y: Lateral position (m).
            yaw: Rotation about the vertical axis (rad).
        """
        d = self.sim.data
        a = self._cube_adr
        d.qpos[a : a + 3] = [x, y, CUBE_HALF_M]
        d.qpos[a + 3 : a + 7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
        d.qvel[:] = 0.0
        mujoco.mj_forward(self.sim.model, d)
        self.sim.step(100)

    def reset(self, cube_xy: tuple[float, float], yaw: float = 0.0) -> None:
        """Reset the arm to HOME, the cube to ``cube_xy`` and the guardrail state."""
        self.sim.reset("HOME")
        self.sim.interrupt_epoch = 0
        self.sim._stop_requested = False
        if isinstance(self.robot, ScriptedGripperRobot):
            self.robot.reading = None
        self.place_cube(cube_xy[0], cube_xy[1], yaw)
        self.state = GuardrailState()

    # -- tool boundary ---------------------------------------------------------------------
    def frame_rgb(self) -> np.ndarray:
        """Render the workspace camera (RGB, 640x480) on the dedicated render thread."""
        return self._render_pool.submit(self.sim.render_rgb_frame, 640, 480).result()

    def jpeg(self) -> bytes:
        """Return a freshly rendered frame as JPEG bytes."""
        bgr = cv2.cvtColor(self.frame_rgb(), cv2.COLOR_RGB2BGR)
        ok, enc = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        assert ok
        return enc.tobytes()

    def fresh_jpeg(self, t_after: float, timeout_s: float = 0.5) -> bytes | None:
        """Return a frame stamped after ``t_after`` (the sim is converged when a tool asks).

        Every render is stamped with ``time.monotonic()`` at render time, exactly like the live
        buffers; a frame older than ``t_after`` is never returned.

        Args:
            t_after: Frame must be stamped after this ``time.monotonic()`` value.
            timeout_s: Longest time to keep rendering.

        Returns:
            JPEG bytes, or None if no frame was stamped late enough in time.
        """
        deadline = time.monotonic() + timeout_s
        while True:
            jpeg = self.jpeg()
            if time.monotonic() > t_after:
                return jpeg
            if time.monotonic() >= deadline:
                return None

    def call(self, name: str, **args: Any) -> dict[str, Any]:
        """Execute one tool call through the real boundary.

        Args:
            name: Tool name.
            **args: Tool arguments.

        Returns:
            The tool response dictionary.
        """
        result = execute_tool_call_sync(
            SimpleNamespace(name=name, args=args, id=f"harness_{len(self.log)}"),
            self.robot,
            get_latest_jpeg_fn=self.jpeg,
            get_fresh_jpeg_fn=self.fresh_jpeg,
            mode=self.mode,
            guardrail_state=self.state,
        )
        self.log.append(f"{name}({args}) -> {_brief(result)}")
        return dict(result)

    def image_point_for(self, x_m: float, y_m: float) -> list[float]:
        """Return the normalised [y, x] image point of a table position (m)."""
        px, py = self.calib.robot_mm_to_pixel(x_m * 1000.0, y_m * 1000.0)  # type: ignore[misc]
        return [py / 480.0 * 1000.0, px / 640.0 * 1000.0]

    def model_point(self, rng: np.random.Generator) -> list[float]:
        """Emulate the model pointing at the cube: detected blob centre plus Gaussian noise."""
        objs = [o for o in detect_workspace_objects(self.frame_rgb()) if o["color"] == "red"]
        assert objs, "harness could not see the cube on the initial frame"
        y, x = objs[0]["center_1000"]
        noise = rng.normal(0.0, MODEL_POINT_SIGMA, size=2)
        return [
            float(np.clip(y + noise[0] + self.point_offset_1000[0], 0, 1000)),
            float(np.clip(x + noise[1] + self.point_offset_1000[1], 0, 1000)),
        ]

    def close(self) -> None:
        """Release the simulator and restore the active robot."""
        set_active_robot(self._previous_active)
        self._render_pool.submit(self.sim.close).result()
        self._render_pool.shutdown()

    # -- scripted model --------------------------------------------------------------------
    def run_placement(
        self, cube_xy: tuple[float, float], seed: int, yaw: float = 0.0
    ) -> PlacementMetrics:
        """Run one scripted pick-and-place from a fresh scene.

        Args:
            cube_xy: Cube start position (m).
            seed: RNG seed for the model's pointing noise.
            yaw: Cube orientation in radians (default 0.0).

        Returns:
            Metrics measured against MuJoCo truth.
        """
        rng = np.random.default_rng(seed)
        self.reset(cube_xy, yaw=yaw)
        m = PlacementMetrics(cube_start_xy_m=cube_xy)
        self.log = []
        last_refusal: tuple[str, str] | None = None

        def note(name: str, resp: dict[str, Any]) -> None:
            nonlocal last_refusal
            et = resp.get("error_type")
            if et:
                m.refusals[et] = m.refusals.get(et, 0) + 1
                key = (name, str(et))
                if key == last_refusal:
                    m.refusal_loops += 1
                last_refusal = key
            else:
                last_refusal = None

        def do(name: str, **args: Any) -> dict[str, Any]:
            resp = self.call(name, **args)
            note(name, resp)
            return resp

        point = self.model_point(rng)
        do("open_gripper")
        dest = self.image_point_for(*DESTINATION_XY_M)

        done = False
        for attempt in range(1, MAX_ATTEMPTS + 1):
            m.attempts = attempt
            hovers = 0
            resp: dict[str, Any] = {}
            descended = False
            hover_args: dict[str, Any] = {"point": point, "z_height_cm": 3.0}
            # Hover until aligned (or told otherwise), then descend.
            while hovers < MAX_HOVERS_PER_ATTEMPT and not descended:
                hovers += 1
                resp = do("move_to_spatial_point", approach="top_down", **hover_args)
                m.max_hovers_in_attempt = max(m.max_hovers_in_attempt, hovers)
                if resp.get("error_type") == "retry_exhausted":
                    break
                nxt = _next_move_call(resp)
                if "error" in resp and nxt is None:
                    break
                if nxt is not None and nxt[2] >= 3.0:
                    hover_args = {"point": nxt[:2], "z_height_cm": nxt[2]}
                    continue
                perception = resp.get("perception") or {}
                if perception.get("is_aligned") is False:
                    continue
                descent = do(
                    "move_to_spatial_point", point=point, z_height_cm=0.0, approach="top_down"
                )
                if "error" in descent:
                    nxt = _next_move_call(descent)
                    if descent.get("error_type") == "retry_exhausted":
                        break
                    if nxt is not None and nxt[2] >= 3.0:
                        hover_args = {"point": nxt[:2], "z_height_cm": nxt[2]}
                        continue
                    break
                descended = True
            if not descended and self.descend_after_hover_cap and hovers >= MAX_HOVERS_PER_ATTEMPT:
                # A model that ignores the "re-hover" hint after the cap and tries the descent
                # anyway (what DESCENT_GATE=warn permits); used for the corrector-off A/B run.
                descent = do(
                    "move_to_spatial_point", point=point, z_height_cm=0.0, approach="top_down"
                )
                descended = "error" not in descent
            if not descended:
                break

            m.jaw_to_cube_at_close_mm = float(
                np.linalg.norm(self.cube_xyz()[:2] - self.site_xyz()[:2]) * 1000.0
            )
            close = do("close_gripper")
            if close.get("gripper_state") == "closed_empty":
                if self._cube_between_jaws():
                    m.contradictions += 1
                do("open_gripper")
                continue

            lift = do("move_to_spatial_point", point=point, z_height_cm=5.0, approach="top_down")
            held = self._cube_held()
            verdict = (lift.get("grasp_verification") or {}).get("verdict")
            if lift.get("error_type") in ("transport_gate", "descent_gate", "perception"):
                m.lift_refusals += 1
            if verdict == "confirmed" and not held:
                m.contradictions += 1
            if verdict == "missed" and held:
                m.contradictions += 1
            m.lifted = m.lifted or held

            if verdict != "confirmed":
                do("open_gripper")
                if lift.get("error_type") == "retry_exhausted":
                    break
                continue

            # Transport, place, release, retract.
            do("move_to_spatial_point", point=dest, z_height_cm=5.0, approach="top_down")
            do("move_to_spatial_point", point=dest, z_height_cm=1.0, approach="top_down")
            do("open_gripper")
            m.locks_created = self.state._next_lock_id - 1
            do("move_to_spatial_point", point=dest, z_height_cm=5.0, approach="top_down")
            done = True
            break

        cube = self.cube_xyz()
        if done:
            err = np.linalg.norm(cube[:2] - np.array(DESTINATION_XY_M)) * 1000.0
            m.placement_error_mm = float(err)
            m.placed = bool(cube[2] < 0.03)
        if not m.locks_created:
            m.locks_created = self.state._next_lock_id - 1
        m.transcript = list(self.log)
        return m

    def _cube_held(self) -> bool:
        """True if the cube is genuinely lifted and between the jaws (truth)."""
        cube, site = self.cube_xyz(), self.site_xyz()
        return bool(cube[2] > CUBE_LIFTED_Z_M and np.linalg.norm(cube - site) < 0.030)

    def _cube_between_jaws(self) -> bool:
        """True if the cube sits between the pads at grasp height (truth)."""
        cube, site = self.cube_xyz(), self.site_xyz()
        return bool(np.linalg.norm(cube[:2] - site[:2]) < 0.008 and abs(cube[2] - 0.015) < 0.004)


def _next_move_call(resp: dict[str, Any]) -> tuple[float, float, float] | None:
    """Parse the ``move_to_spatial_point(point=[y, x], z_height_cm=z)`` a refusal names."""
    for text in (resp.get("visual_hint"), resp.get("error")):
        if isinstance(text, str):
            match = _NEXT_CALL_MOVE.search(text)
            if match:
                return float(match.group(1)), float(match.group(2)), float(match.group(3))
    return None


def _brief(resp: dict[str, Any]) -> str:
    keys = ("status", "error_type", "gripper_state")
    parts = [f"{k}={resp[k]}" for k in keys if k in resp]
    if "error" in resp:
        parts.append(f"error={str(resp['error'])[:90]!r}")
    verification = resp.get("grasp_verification")
    if verification:
        parts.append(f"verdict={verification.get('verdict')}/{verification.get('reason')}")
    perception = resp.get("perception") or {}
    if perception:
        parts.append(
            f"aligned={perception.get('is_aligned')} src={perception.get('gripper_source')}"
        )
    return " ".join(parts)


def run_scenario(
    name: str,
    n_placements: int,
    seed: int = 0,
    pan_offset_deg: float = 0.0,
    x_range: tuple[float, float] = DEFAULT_X_RANGE,
    y_range: tuple[float, float] = DEFAULT_Y_RANGE,
    settings_overrides: dict[str, Any] | None = None,
    progress: Callable[[str], None] | None = None,
    descend_after_hover_cap: bool = False,
    random_yaw: bool = False,
    point_offset_1000: tuple[float, float] = (0.0, 0.0),
) -> ScenarioResult:
    """Run ``n_placements`` seeded placements and collect truth-based metrics.

    Args:
        name: Scenario label for the report.
        n_placements: Number of pick-and-place runs.
        seed: Master seed (cube positions and model pointing noise).
        pan_offset_deg: Hidden shoulder-pan encoder offset in degrees.
        x_range: Cube x range (m).
        y_range: Cube y range (m).
        settings_overrides: Settings fields to override for the run (e.g. bias_correction).
        progress: Optional callback receiving one line per placement.
        descend_after_hover_cap: Descend anyway after the hover cap (corrector-off A/B runs).
        random_yaw: Whether to sample random yaw in [0, pi/2) for each placement.
        point_offset_1000: Constant [dy, dx] added to the model pointing point.

    Returns:
        ScenarioResult with one PlacementMetrics per run.
    """
    rng = np.random.default_rng(seed)
    t0 = time.monotonic()
    placements: list[PlacementMetrics] = []
    with settings_override(**(settings_overrides or {})):
        harness = SimPickHarness(
            pan_offset_deg=pan_offset_deg,
            descend_after_hover_cap=descend_after_hover_cap,
            point_offset_1000=point_offset_1000,
        )
        try:
            for i in range(n_placements):
                xy = (float(rng.uniform(*x_range)), float(rng.uniform(*y_range)))
                yaw = float(rng.uniform(0.0, np.pi / 2.0)) if random_yaw else 0.0
                result = harness.run_placement(xy, seed=int(rng.integers(1 << 31)), yaw=yaw)
                placements.append(result)
                if progress is not None:
                    progress(
                        f"[{name}] #{i} cube=({xy[0]:.3f},{xy[1]:.3f}) lifted={result.lifted} "
                        f"placed={result.placed} err={result.placement_error_mm} "
                        f"jaw@close={result.jaw_to_cube_at_close_mm} refusals={result.refusals}"
                    )
        finally:
            harness.close()
    return ScenarioResult(name=name, placements=placements, wall_s=time.monotonic() - t0)
