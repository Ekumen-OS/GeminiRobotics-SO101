"""Central configuration settings for GeminiRobotics-SO101."""

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _get_platform_serial_port() -> str:
    """Return platform-appropriate default serial port path without reading env."""
    if sys.platform == "darwin":
        return "/dev/tty.usbmodem5B415332861"
    return "/dev/ttyACM0"


def _lerobot_calibration_root() -> Path:
    """Return the LeRobot calibration cache root, mirroring `lerobot.constants`.

    Resolution order matches LeRobot so that files written by `lerobot-calibrate`
    are found without importing the package (which may not be installed).

    Returns:
        Path to the calibration directory (may not exist).
    """
    explicit = os.environ.get("HF_LEROBOT_CALIBRATION")
    if explicit:
        return Path(explicit).expanduser()
    lerobot_home = os.environ.get("HF_LEROBOT_HOME")
    if lerobot_home:
        return Path(lerobot_home).expanduser() / "calibration"
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home).expanduser() / "lerobot" / "calibration"
    return Path.home() / ".cache" / "huggingface" / "lerobot" / "calibration"


@dataclass(frozen=True)
class Settings:
    """Immutable runtime settings for hardware, Gemini agent, and UI services."""

    serial_port: str = field(default_factory=_get_platform_serial_port)
    gemini_model_id: str = "gemini-robotics-er-2-streaming-preview"
    google_api_key: str | None = None
    google_cloud_project: str | None = None
    google_cloud_location: str = "us-central1"
    server_host: str = "0.0.0.0"
    server_port: int = 7860
    follower_robot_id: str = "follower_droid"
    motor_calibration_path: Path | None = None
    calibration_config_path: Path = field(
        default_factory=lambda: _PROJECT_ROOT / "config" / "workspace_calibration.json"
    )
    calibration_save_path: Path = field(
        default_factory=lambda: _PROJECT_ROOT / "config" / "workspace_homography.npz"
    )
    calibration_sim_save_path: Path = field(
        default_factory=lambda: _PROJECT_ROOT / "config" / "workspace_homography_sim.npz"
    )
    alignment_tolerance_px: int = 25
    stream_fps: int = 30
    stream_width: int = 640
    stream_height: int = 480
    # Browser MJPEG stream (/api/stream) only; the model path always gets full-quality frames.
    web_stream_fps: int = 15
    web_jpeg_quality: int = 75
    # Preferred hardware camera: explicit /dev/video index, or case-insensitive device-name match
    camera_index: int | None = None
    camera_name: str | None = None

    @property
    def credentials_mode(self) -> str:
        """Return credentials mode identifier ('api_key', 'vertex', or 'none')."""
        if self.google_api_key:
            return "api_key"
        if self.google_cloud_project:
            return "vertex"
        return "none"

    @property
    def motor_calibration_candidates(self) -> tuple[Path, ...]:
        """Return candidate paths for the follower's LeRobot motor calibration file.

        An explicit `motor_calibration_path` wins outright. Otherwise the LeRobot
        cache location written by `lerobot-calibrate` is preferred, with the
        in-repo `config/` copy kept as a user-maintained backup.

        Returns:
            Candidate paths in priority order.
        """
        if self.motor_calibration_path is not None:
            return (self.motor_calibration_path,)
        filename = f"{self.follower_robot_id}.json"
        return (
            _lerobot_calibration_root() / "robots" / "so101_follower" / filename,
            _PROJECT_ROOT / "config" / filename,
        )

    @classmethod
    def from_env(cls, **overrides: Any) -> "Settings":
        """Resolve settings from environment variables and explicit overrides.

        Args:
            **overrides: Explicit overrides taking precedence over environment variables.

        Returns:
            Resolved Settings instance.
        """

        def _resolve_int(override_val: Any, env_names: tuple[str, ...], default: int) -> int:
            if override_val is not None:
                try:
                    return int(override_val)
                except (ValueError, TypeError):
                    return default
            for name in env_names:
                env_val = os.environ.get(name)
                if env_val is not None and env_val.strip() != "":
                    try:
                        return int(env_val)
                    except (ValueError, TypeError):
                        pass
            return default

        def _resolve_str(override_val: Any, env_names: tuple[str, ...], default: str) -> str:
            if override_val is not None:
                return str(override_val)
            for name in env_names:
                env_val = os.environ.get(name)
                if env_val is not None and env_val.strip() != "":
                    return env_val
            return default

        def _resolve_opt_str(override_val: Any, env_names: tuple[str, ...]) -> str | None:
            if override_val is not None:
                return str(override_val) if override_val != "" else None
            for name in env_names:
                env_val = os.environ.get(name)
                if env_val is not None and env_val.strip() != "":
                    return env_val
            return None

        def _resolve_path(override_val: Any, env_names: tuple[str, ...], default: Path) -> Path:
            if override_val is not None:
                return Path(override_val)
            for name in env_names:
                env_val = os.environ.get(name)
                if env_val is not None and env_val.strip() != "":
                    return Path(env_val)
            return default

        # Serial port resolution: explicit override > ROBOT_PORT > SO101_PORT > platform default
        serial_port = _resolve_str(
            overrides.get("serial_port"),
            ("ROBOT_PORT", "SO101_PORT"),
            _get_platform_serial_port(),
        )

        gemini_model_id = _resolve_str(
            overrides.get("gemini_model_id"),
            ("GEMINI_MODEL", "GEMINI_ROBOTICS_MODEL"),
            cls.gemini_model_id,
        )

        google_api_key = _resolve_opt_str(
            overrides.get("google_api_key"),
            ("GOOGLE_API_KEY",),
        )

        google_cloud_project = _resolve_opt_str(
            overrides.get("google_cloud_project"),
            ("GOOGLE_CLOUD_PROJECT",),
        )

        google_cloud_location = _resolve_str(
            overrides.get("google_cloud_location"),
            ("GOOGLE_CLOUD_LOCATION",),
            cls.google_cloud_location,
        )

        server_host = _resolve_str(
            overrides.get("server_host"),
            ("SERVER_HOST", "HOST"),
            cls.server_host,
        )

        server_port = _resolve_int(
            overrides.get("server_port"),
            ("SERVER_PORT", "PORT"),
            cls.server_port,
        )

        follower_robot_id = _resolve_str(
            overrides.get("follower_robot_id"),
            ("FOLLOWER_ROBOT_ID", "ROBOT_ID"),
            cls.follower_robot_id,
        )

        motor_calib_override = _resolve_opt_str(
            overrides.get("motor_calibration_path"),
            ("MOTOR_CALIBRATION_PATH",),
        )
        motor_calibration_path = (
            Path(motor_calib_override) if motor_calib_override is not None else None
        )

        default_calib_cfg = _PROJECT_ROOT / "config" / "workspace_calibration.json"
        calibration_config_path = _resolve_path(
            overrides.get("calibration_config_path"),
            ("CALIBRATION_CONFIG_PATH",),
            default_calib_cfg,
        )

        default_calib_save = _PROJECT_ROOT / "config" / "workspace_homography.npz"
        calibration_save_path = _resolve_path(
            overrides.get("calibration_save_path"),
            ("CALIBRATION_SAVE_PATH",),
            default_calib_save,
        )

        default_sim_save = _PROJECT_ROOT / "config" / "workspace_homography_sim.npz"
        calibration_sim_save_path = _resolve_path(
            overrides.get("calibration_sim_save_path"),
            ("CALIBRATION_SIM_SAVE_PATH",),
            default_sim_save,
        )

        # Single source of truth for tolerance: dataclass field default
        default_tolerance = cls.alignment_tolerance_px
        alignment_tolerance_px = _resolve_int(
            overrides.get("alignment_tolerance_px"),
            ("ALIGNMENT_TOLERANCE_PX", "ALIGNMENT_TOLERANCE"),
            default_tolerance,
        )

        stream_fps = _resolve_int(
            overrides.get("stream_fps"),
            ("STREAM_FPS",),
            cls.stream_fps,
        )

        stream_width = _resolve_int(
            overrides.get("stream_width"),
            ("STREAM_WIDTH",),
            cls.stream_width,
        )

        stream_height = _resolve_int(
            overrides.get("stream_height"),
            ("STREAM_HEIGHT",),
            cls.stream_height,
        )

        web_stream_fps = _resolve_int(
            overrides.get("web_stream_fps"),
            ("WEB_STREAM_FPS",),
            cls.web_stream_fps,
        )
        web_jpeg_quality = _resolve_int(
            overrides.get("web_jpeg_quality"),
            ("WEB_JPEG_QUALITY",),
            cls.web_jpeg_quality,
        )

        camera_index_raw = _resolve_opt_str(overrides.get("camera_index"), ("CAMERA_INDEX",))
        camera_index: int | None = None
        if camera_index_raw is not None:
            try:
                camera_index = int(camera_index_raw)
            except (ValueError, TypeError):
                camera_index = None

        camera_name = _resolve_opt_str(overrides.get("camera_name"), ("CAMERA_NAME",))

        return cls(
            serial_port=serial_port,
            gemini_model_id=gemini_model_id,
            google_api_key=google_api_key,
            google_cloud_project=google_cloud_project,
            google_cloud_location=google_cloud_location,
            server_host=server_host,
            server_port=server_port,
            follower_robot_id=follower_robot_id,
            motor_calibration_path=motor_calibration_path,
            calibration_config_path=calibration_config_path,
            calibration_save_path=calibration_save_path,
            calibration_sim_save_path=calibration_sim_save_path,
            alignment_tolerance_px=alignment_tolerance_px,
            stream_fps=stream_fps,
            stream_width=stream_width,
            stream_height=stream_height,
            web_stream_fps=web_stream_fps,
            web_jpeg_quality=web_jpeg_quality,
            camera_index=camera_index,
            camera_name=camera_name,
        )


_current_settings: Settings | None = None


def get_settings() -> Settings:
    """Retrieve active settings instance, resolving from environment if uninitialized.

    Returns:
        The active Settings singleton instance.
    """
    global _current_settings
    if _current_settings is None:
        _current_settings = Settings.from_env()
    return _current_settings


def set_settings(settings: Settings) -> None:
    """Explicitly assign global settings instance (e.g. at startup or in test harness).

    Args:
        settings: New Settings instance to activate globally.
    """
    global _current_settings
    _current_settings = settings
