"""Simulation runtime worker owning physics stepping and offscreen RGB rendering."""

import atexit
import logging
import threading
import time
from collections.abc import Callable
from typing import Any

import cv2
import numpy as np

from robot.interface import RobotInterface
from robot.settings import get_settings
from vision.patterns import (
    generate_raw_sim_placeholder_pattern,
    generate_sim_placeholder_pattern,
)

logger = logging.getLogger(__name__)

try:
    import mujoco

    _SIM_ERRORS = (RuntimeError, ValueError, mujoco.FatalError)
except (ImportError, AttributeError):
    _SIM_ERRORS = (RuntimeError, ValueError)

_last_sim_error_time: float = 0.0


def _log_sim_error_rate_limited(err: Exception) -> None:
    """Log simulation stepping/rendering error rate-limited to once per 5 seconds."""
    global _last_sim_error_time
    now = time.time()
    if now - _last_sim_error_time >= 5.0:
        _last_sim_error_time = now
        logger.error(f"Sim stepping/rendering error: {err}")


def _robot_can_render(robot: Any | None) -> bool:
    """Return True only when ``robot`` really implements ``render_rgb_frame``.

    ``RobotInterface`` defines ``render_rgb_frame`` as a method that raises
    ``NotImplementedError`` (physical arms inherit it), so ``hasattr`` alone is true for every
    robot. Only a subclass that overrides it (simulation) can render a frame.

    Args:
        robot: Active robot instance, or None.

    Returns:
        True when the robot overrides ``render_rgb_frame``.
    """
    if robot is None:
        return False
    render = getattr(type(robot), "render_rgb_frame", None)
    return render is not None and render is not RobotInterface.render_rgb_frame


def _estimate_sim_steps_per_frame(robot: Any | None, frame_interval: float) -> int:
    """Estimate MuJoCo physics steps needed per rendered frame.

    Args:
        robot: Active simulation robot object.
        frame_interval: Frame period in seconds.

    Returns:
        Number of simulation steps to execute for near-real-time playback.
    """
    if robot is None:
        return 1

    try:
        model = getattr(robot, "model", None)
        opt = getattr(model, "opt", None)
        timestep = getattr(opt, "timestep", None)
        if timestep is None:
            return 1
        ts_float = float(timestep)
        if ts_float <= 0.0:
            return 1
        return max(1, int(round(frame_interval / ts_float)))
    except (AttributeError, TypeError, ValueError):
        return 1


def _sim_frame_with_fallback(
    robot: Any | None,
    width: int,
    height: int,
) -> np.ndarray:
    """Render simulation frame if available, else return a placeholder.

    Args:
        robot: Active robot object for current mode.
        width: Frame width.
        height: Frame height.

    Returns:
        RGB frame as uint8 numpy array.
    """
    if _robot_can_render(robot):
        try:
            return robot.render_rgb_frame(width=width, height=height)
        except _SIM_ERRORS as err:
            _log_sim_error_rate_limited(err)
        except Exception as err:
            logger.warning(f"Sim render failed. Falling back to placeholder. Cause: {err}")

    fallback = generate_sim_placeholder_pattern()
    return cv2.resize(fallback, (width, height), interpolation=cv2.INTER_LINEAR)


class _SingletonSimRuntime:
    """Singleton worker that owns sim stepping + rendering for UI streaming.

    This avoids concurrent MuJoCo access across multiple Gradio callback threads.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._robot: Any | None = None
        self._latest_raw_frame: np.ndarray | None = None
        self._latest_frame: np.ndarray | None = None
        self._frame_sink: Callable[[np.ndarray | None], None] | None = None

    def set_frame_sink(self, sink: Callable[[np.ndarray | None], None] | None) -> None:
        """Register a callback to receive frames published by the simulation worker loop."""
        self._frame_sink = sink

    def _loop(self) -> None:
        settings = get_settings()
        frame_interval = 1.0 / settings.stream_fps
        stream_width = settings.stream_width
        stream_height = settings.stream_height

        while not self._stop_event.is_set():
            start = time.time()

            try:
                with self._lock:
                    robot = self._robot

                if robot is None:
                    raw_frame = cv2.resize(
                        generate_raw_sim_placeholder_pattern(),
                        (stream_width, stream_height),
                        interpolation=cv2.INTER_LINEAR,
                    )
                else:
                    sim_steps = _estimate_sim_steps_per_frame(
                        robot=robot,
                        frame_interval=frame_interval,
                    )
                    if hasattr(robot, "step"):
                        try:
                            robot.step(sim_steps)
                        except _SIM_ERRORS as e:
                            _log_sim_error_rate_limited(e)

                    raw_frame = _sim_frame_with_fallback(
                        robot=robot,
                        width=stream_width,
                        height=stream_height,
                    )

                with self._frame_lock:
                    self._latest_raw_frame = raw_frame
                    self._latest_frame = raw_frame

                if self._frame_sink is not None:
                    self._frame_sink(raw_frame)
            except _SIM_ERRORS as e:
                _log_sim_error_rate_limited(e)
                with self._frame_lock:
                    self._latest_raw_frame = None
                    self._latest_frame = None
                if self._frame_sink is not None:
                    self._frame_sink(None)
            except Exception as exc:
                logger.exception(f"Unexpected sim runtime loop error: {exc}")
                with self._frame_lock:
                    self._latest_raw_frame = None
                    self._latest_frame = None
                if self._frame_sink is not None:
                    self._frame_sink(None)

            elapsed = time.time() - start
            if elapsed < frame_interval:
                self._stop_event.wait(timeout=max(0.0, frame_interval - elapsed))

    def ensure_started(self, robot: Any | None) -> None:
        """Start worker once and update active robot reference."""
        with self._lock:
            if self._robot != robot:
                self._robot = robot
                with self._frame_lock:
                    self._latest_raw_frame = None
                    self._latest_frame = None
            if self._thread is not None and self._thread.is_alive():
                return

            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._loop,
                name="sim-runtime",
                daemon=True,
            )
            self._thread.start()

    def get_latest_raw_frame(self, robot: Any | None) -> np.ndarray:
        """Return most recent un-annotated raw simulation frame, bootstrapping runtime if needed.

        Args:
            robot: Active simulation robot object.

        Returns:
            RGB frame as uint8 numpy array.
        """
        self.ensure_started(robot)

        with self._frame_lock:
            if self._latest_raw_frame is not None:
                return self._latest_raw_frame.copy()
            if self._latest_frame is not None:
                return self._latest_frame.copy()

        settings = get_settings()
        stream_width = settings.stream_width
        stream_height = settings.stream_height

        if _robot_can_render(robot):
            frame = _sim_frame_with_fallback(
                robot=robot,
                width=stream_width,
                height=stream_height,
            )
            if frame is not None:
                with self._frame_lock:
                    self._latest_raw_frame = frame
                    self._latest_frame = frame
                return frame.copy()

        # First frame race: return deterministic raw fallback until worker publishes one.
        return cv2.resize(
            generate_raw_sim_placeholder_pattern(),
            (stream_width, stream_height),
            interpolation=cv2.INTER_LINEAR,
        )

    def get_latest_frame(self, robot: Any | None) -> np.ndarray:
        """Return most recent frame, bootstrapping runtime if needed.

        Args:
            robot: Active simulation robot object.

        Returns:
            RGB frame as uint8 numpy array.
        """
        self.ensure_started(robot)

        with self._frame_lock:
            if self._latest_frame is not None:
                return self._latest_frame.copy()

        settings = get_settings()
        stream_width = settings.stream_width
        stream_height = settings.stream_height

        if _robot_can_render(robot):
            frame = _sim_frame_with_fallback(
                robot=robot,
                width=stream_width,
                height=stream_height,
            )
            if frame is not None:
                with self._frame_lock:
                    self._latest_raw_frame = frame
                    self._latest_frame = frame
                return frame.copy()

        # First frame race: return deterministic fallback until worker publishes one.
        return cv2.resize(
            generate_sim_placeholder_pattern(),
            (stream_width, stream_height),
            interpolation=cv2.INTER_LINEAR,
        )

    def get_latest_raw_jpeg(self, robot: Any | None = None) -> bytes | None:
        """Return the most recent un-annotated raw frame as JPEG encoded bytes.

        Args:
            robot: Active simulation robot object.

        Returns:
            Pristine JPEG bytes, or None if encoding fails.
        """
        frame = self.get_latest_raw_frame(robot)
        if frame is None:
            return None
        # Convert RGB to BGR for cv2.imencode to produce correct JPEG colors.
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        success, encoded = cv2.imencode(".jpg", bgr_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 98])
        if success:
            return encoded.tobytes()
        return None

    def get_latest_jpeg(self, robot: Any | None = None) -> bytes | None:
        """Return the most recent frame as JPEG encoded bytes."""
        frame = self.get_latest_frame(robot)
        if frame is None:
            return None
        # Convert RGB to BGR for cv2.imencode to produce correct JPEG colors.
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        success, encoded = cv2.imencode(".jpg", bgr_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if success:
            return encoded.tobytes()
        return None

    def stop(self) -> None:
        """Stop worker thread cleanly."""
        self._stop_event.set()
        with self._lock:
            thread = self._thread
            self._thread = None
            self._robot = None

        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)

        with self._frame_lock:
            self._latest_raw_frame = None
            self._latest_frame = None


_SIM_RUNTIME = _SingletonSimRuntime()
atexit.register(_SIM_RUNTIME.stop)


def ensure_sim_runtime_started(robot: Any | None) -> None:
    """Start simulation runtime singleton if not running and bind active robot."""
    _SIM_RUNTIME.ensure_started(robot)


def stop_sim_runtime() -> None:
    """Stop simulation runtime singleton thread cleanly."""
    _SIM_RUNTIME.stop()


def get_sim_video_frame(robot: Any | None = None) -> np.ndarray:
    """Return a single live simulation frame for timer-driven UI updates.

    Args:
        robot: Active simulation robot object.

    Returns:
        RGB frame as uint8 numpy array.
    """
    return _SIM_RUNTIME.get_latest_frame(robot=robot)
