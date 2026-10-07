"""Tests for the shared camera capture and the browser MJPEG stream (/api/stream)."""

import asyncio
import threading
from collections.abc import Generator
from typing import Any

import cv2
import numpy as np
import pytest

import vision as stream_mod
import vision.capture as capture_mod
import vision.frames as frames_mod
import vision.sim_runtime as sim_runtime_mod
from robot.settings import Settings
from ui.app import mjpeg_frames

JPEG_SOI = b"\xff\xd8"


def _noise_frame(seed: int = 0) -> np.ndarray:
    """Return a deterministic noisy RGB frame (compresses poorly, so quality changes size)."""
    return np.random.default_rng(seed).integers(0, 255, (480, 640, 3), dtype=np.uint8)


def _decode_jpeg(data: bytes) -> np.ndarray:
    """Decode JPEG bytes to an RGB array."""
    bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


@pytest.fixture
def frames(clean_stream_state: None) -> Generator[None, None, None]:
    """Publish a raw frame to the global buffers and restore them afterwards."""
    raw = _noise_frame()
    with frames_mod._global_frame_lock:
        saved = (
            frames_mod._global_latest_raw_frame,
            frames_mod._global_latest_annotated_frame,
            frames_mod._global_latest_frame,
        )
        frames_mod._global_latest_raw_frame = raw.copy()
        frames_mod._global_latest_annotated_frame = raw.copy()
        frames_mod._global_latest_frame = raw.copy()
    yield
    with frames_mod._global_frame_lock:
        (
            frames_mod._global_latest_raw_frame,
            frames_mod._global_latest_annotated_frame,
            frames_mod._global_latest_frame,
        ) = saved


def test_web_jpeg_quality_changes_size_but_model_path_stays_full_quality(frames: None) -> None:
    """Lower web quality shrinks the browser JPEG; the raw (model) JPEG is unaffected by it."""
    low = stream_mod.get_latest_jpeg(quality=30)
    high = stream_mod.get_latest_jpeg(quality=95)
    assert low is not None and high is not None
    assert low.startswith(JPEG_SOI) and len(low) < len(high)

    raw_a = stream_mod.get_latest_raw_jpeg()
    stream_mod.get_latest_jpeg(quality=10)
    assert stream_mod.get_latest_raw_jpeg() == raw_a


def test_raw_jpeg_is_free_of_overlays_even_when_annotated_differs(frames: None) -> None:
    """The model path must stay pristine: annotated-only changes never reach the raw buffer."""
    raw = np.full((480, 640, 3), 128, dtype=np.uint8)
    annotated = raw.copy()
    annotated[100:300, 100:300] = (255, 0, 0)  # simulated overlay box
    with frames_mod._global_frame_lock:
        frames_mod._global_latest_raw_frame = raw
        frames_mod._global_latest_annotated_frame = annotated
        frames_mod._global_latest_frame = annotated

    raw_patch = _decode_jpeg(stream_mod.get_latest_raw_jpeg() or b"")[150:250, 150:250]
    web_patch = _decode_jpeg(stream_mod.get_latest_jpeg(quality=95) or b"")[150:250, 150:250]
    assert np.abs(raw_patch.astype(int) - 128).max() < 8  # still flat grey: no overlay leaked
    assert web_patch[..., 0].mean() > 230 and web_patch[..., 2].mean() < 30  # overlay present


def test_mjpeg_frames_emit_multipart_jpeg_chunks(frames: None) -> None:
    """The browser stream is a multipart body with one valid JPEG per part."""

    async def first_two() -> list[bytes]:
        gen = mjpeg_frames(quality=70, fps=60)
        try:
            return [await gen.__anext__(), await gen.__anext__()]
        finally:
            await gen.aclose()

    chunks = asyncio.run(first_two())
    for chunk in chunks:
        assert chunk.startswith(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n")
        body = chunk[len(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n") :]
        assert body.startswith(JPEG_SOI) and body.endswith(b"\r\n")


def test_web_stream_settings_defaults_and_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """WEB_STREAM_FPS / WEB_JPEG_QUALITY are configurable and default to bandwidth-friendly."""
    monkeypatch.delenv("WEB_STREAM_FPS", raising=False)
    monkeypatch.delenv("WEB_JPEG_QUALITY", raising=False)
    defaults = Settings.from_env()
    assert (defaults.web_stream_fps, defaults.web_jpeg_quality) == (15, 75)

    monkeypatch.setenv("WEB_STREAM_FPS", "8")
    monkeypatch.setenv("WEB_JPEG_QUALITY", "50")
    tuned = Settings.from_env()
    assert (tuned.web_stream_fps, tuned.web_jpeg_quality) == (8, 50)


@pytest.fixture
def fake_capture(monkeypatch: pytest.MonkeyPatch) -> Generator[list[tuple[int, str]], None, None]:
    """Replace ``stream_video`` with a generator that idles until the stream token changes."""
    started: list[tuple[int, str]] = []
    ready = threading.Event()

    def fake_stream_video(camera_index: int, mode: str, robot: Any = None):
        my_id = capture_mod._current_stream_id
        started.append((camera_index, mode))
        ready.set()
        while capture_mod._current_stream_id == my_id:
            yield None
            ready.wait(0.005)

    monkeypatch.setattr(capture_mod, "stream_video", fake_stream_video)
    monkeypatch.setattr(capture_mod, "_capture_thread", None)
    monkeypatch.setattr(capture_mod, "_capture_key", None)
    yield started
    capture_mod.prepare_stream_switch()  # let the fake generator exit
    thread = capture_mod._capture_thread
    if thread is not None:
        thread.join(timeout=2)


def _wait_until(predicate, timeout: float = 2.0) -> bool:
    """Poll ``predicate`` until true or timeout."""
    event = threading.Event()
    for _ in range(int(timeout / 0.01)):
        if predicate():
            return True
        event.wait(0.01)
    return False


def test_second_page_load_joins_running_capture(fake_capture: list[tuple[int, str]]) -> None:
    """A second tab must share the running capture, not restart or replace it."""
    assert stream_mod.ensure_camera_capture(2, "real") == 2
    assert _wait_until(lambda: fake_capture == [(2, "real")])

    # Second tab loads with a different dropdown default; it must join camera 2.
    assert stream_mod.ensure_camera_capture(0, "real") == 2
    assert fake_capture == [(2, "real")]


def test_switch_restarts_only_when_camera_changes_or_forced(
    fake_capture: list[tuple[int, str]],
) -> None:
    """Switching to the same camera is a no-op; a new camera or force=True restarts capture."""
    stream_mod.ensure_camera_capture(2, "real")
    assert _wait_until(lambda: len(fake_capture) == 1)

    stream_mod.switch_camera_capture(2, "real")
    assert len(fake_capture) == 1

    stream_mod.switch_camera_capture(0, "real")
    assert _wait_until(lambda: fake_capture[-1] == (0, "real"))

    stream_mod.switch_camera_capture(0, "real", force=True)
    assert _wait_until(lambda: len(fake_capture) == 3)


@pytest.fixture
def empty_frames(clean_stream_state: None) -> Generator[None, None, None]:
    """Clear the shared frame buffers (as at startup, before the first capture frame)."""
    with frames_mod._global_frame_lock:
        saved = (
            frames_mod._global_latest_raw_frame,
            frames_mod._global_latest_annotated_frame,
            frames_mod._global_latest_frame,
        )
        frames_mod._global_latest_raw_frame = None
        frames_mod._global_latest_annotated_frame = None
        frames_mod._global_latest_frame = None
    yield
    with frames_mod._global_frame_lock:
        (
            frames_mod._global_latest_raw_frame,
            frames_mod._global_latest_annotated_frame,
            frames_mod._global_latest_frame,
        ) = saved


def test_robot_can_render_only_when_render_is_overridden(mock_robot: Any) -> None:
    """Physical arms inherit a raising ``render_rgb_frame``; ``hasattr`` alone cannot tell."""
    from robot.interface import RobotInterface

    class Renderer(RobotInterface):
        def set_joint_positions(self, qpos: np.ndarray) -> None:
            pass

        def get_joint_positions(self) -> np.ndarray:
            return np.zeros(6)

        def render_rgb_frame(self, width: int = 640, height: int = 480) -> np.ndarray:
            return np.zeros((height, width, 3), dtype=np.uint8)

    assert hasattr(mock_robot, "render_rgb_frame")  # the trap: true for every RobotInterface
    assert sim_runtime_mod._robot_can_render(mock_robot) is False
    assert sim_runtime_mod._robot_can_render(Renderer()) is True
    assert sim_runtime_mod._robot_can_render(None) is False


def test_empty_buffers_with_physical_robot_do_not_attempt_sim_render(
    empty_frames: None, mock_robot: Any, capsys: pytest.CaptureFixture
) -> None:
    """Regression: polling before the first camera frame logged 'Sim render failed' per poll."""
    from ui.state import set_active_robot

    set_active_robot(mock_robot)  # a robot that cannot render, like the real arm
    assert stream_mod.get_latest_jpeg(quality=70) is not None
    assert stream_mod.get_latest_raw_jpeg() is not None
    captured = capsys.readouterr()
    assert "Sim render failed" not in captured.out + captured.err


def test_mjpeg_waits_for_first_frame_instead_of_serving_sim_placeholder(
    empty_frames: None,
) -> None:
    """Real mode must not flash the 'MODE: SIMULATION' placeholder while the camera warms up."""
    assert stream_mod.has_published_frame() is False

    async def run() -> bytes:
        gen = mjpeg_frames(quality=70, fps=100)
        # asyncio.wait (unlike wait_for) does not cancel the pending step, which would end the
        # generator; we only want to observe that it has produced nothing yet.
        pending = asyncio.ensure_future(gen.__anext__())
        try:
            done, _ = await asyncio.wait({pending}, timeout=0.3)
            assert not done, "stream emitted a frame before any camera frame was published"
            with frames_mod._global_frame_lock:
                frames_mod._global_latest_frame = _noise_frame()
                frames_mod._global_latest_annotated_frame = _noise_frame()
            return await asyncio.wait_for(pending, timeout=2)
        finally:
            pending.cancel()
            await gen.aclose()

    assert asyncio.run(run()).startswith(b"--frame")


def test_stop_camera_capture_joins_thread_and_clears_key(fake_capture) -> None:
    """Shutdown stops the shared capture thread so no V4L2 read is pending at interpreter exit."""
    stream_mod.ensure_camera_capture(4, "real")
    assert _wait_until(lambda: len(fake_capture) == 1)
    thread = capture_mod._capture_thread
    assert thread is not None and thread.is_alive()

    stream_mod.stop_camera_capture(timeout_s=2.0)

    assert not thread.is_alive()
    assert capture_mod._capture_thread is None
    assert capture_mod._capture_key is None
    stream_mod.stop_camera_capture()  # idempotent when nothing is running
