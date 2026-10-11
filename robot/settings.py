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
    # Physical grasp sensor settings (Phase 3a)
    # Calibrated empty-jaw aperture on this arm; None anchors the threshold to GRIPPER_CLOSE_DEG
    gripper_empty_aperture_deg: float | None = None
    grasp_empty_margin_deg: float = 6.0
    gripper_settle_s: float = 0.6
    grasp_load_min: float | None = None
    # Closed-loop grasping settings (Phase 3d)
    descent_gate: str | None = None
    gripper_tag_offset_px: tuple[float, float] = (0.0, 0.0)
    gripper_locator_er2: bool = False
    vision_settle_s: float = 0.3

    def get_descent_gate(self, mode: str = "real") -> str:
        """Resolve descent gate policy ('block', 'warn', or 'off').

        Defaults to 'block' in sim and 'warn' in real unless explicitly overridden.

        Args:
            mode: Operational mode ('real' or 'sim').

        Returns:
            Active descent gate policy string.
        """
        env_gate = os.environ.get("DESCENT_GATE")
        if env_gate:
            cleaned = env_gate.strip().lower()
            if cleaned in ("block", "warn", "off"):
                return cleaned
        if self.descent_gate in ("block", "warn", "off"):
            return self.descent_gate
        return "block" if mode == "sim" else "warn"

    def get_vision_settle_s(self) -> float:
        """Resolve vision settling duration in seconds.

        Returns:
            Settling duration in seconds.
        """
        env_settle = os.environ.get("VISION_SETTLE_S")
        if env_settle is not None and env_settle.strip() != "":
            try:
                return float(env_settle)
            except (ValueError, TypeError):
                pass
        return self.vision_settle_s

    def get_gripper_locator_er2(self) -> bool:
        """Resolve whether ER2 visual pointing gripper localization is enabled.

        Returns:
            True if enabled in environment or settings, else False.
        """
        env_er2 = os.environ.get("GRIPPER_LOCATOR_ER2")
        if env_er2 is not None:
            return env_er2.strip().lower() in ("1", "true", "yes")
        return self.gripper_locator_er2

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

        def _resolve_float(override_val: Any, env_names: tuple[str, ...], default: float) -> float:
            if override_val is not None:
                try:
                    return float(override_val)
                except (ValueError, TypeError):
                    return default
            for name in env_names:
                env_val = os.environ.get(name)
                if env_val is not None and env_val.strip() != "":
                    try:
                        return float(env_val)
                    except (ValueError, TypeError):
                        pass
            return default

        def _resolve_opt_float(override_val: Any, env_names: tuple[str, ...]) -> float | None:
            if override_val is not None:
                try:
                    return float(override_val)
                except (ValueError, TypeError):
                    return None
            for name in env_names:
                env_val = os.environ.get(name)
                if env_val is not None and env_val.strip() != "":
                    try:
                        return float(env_val)
                    except (ValueError, TypeError):
                        pass
            return None

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

        gripper_empty_aperture_deg = _resolve_opt_float(
            overrides.get("gripper_empty_aperture_deg"),
            ("GRIPPER_EMPTY_APERTURE_DEG",),
        )

        grasp_empty_margin_deg = _resolve_float(
            overrides.get("grasp_empty_margin_deg"),
            ("GRASP_EMPTY_MARGIN_DEG",),
            cls.grasp_empty_margin_deg,
        )

        gripper_settle_s = _resolve_float(
            overrides.get("gripper_settle_s"),
            ("GRIPPER_SETTLE_S",),
            cls.gripper_settle_s,
        )

        grasp_load_min = _resolve_opt_float(
            overrides.get("grasp_load_min"),
            ("GRASP_LOAD_MIN",),
        )

        descent_gate_raw = _resolve_opt_str(overrides.get("descent_gate"), ("DESCENT_GATE",))
        descent_gate = descent_gate_raw.strip().lower() if descent_gate_raw else None
        if descent_gate not in ("block", "warn", "off"):
            descent_gate = None

        gripper_tag_offset_px = cls.gripper_tag_offset_px
        if overrides.get("gripper_tag_offset_px") is not None:
            ov = overrides["gripper_tag_offset_px"]
            if isinstance(ov, (tuple, list)) and len(ov) >= 2:
                gripper_tag_offset_px = (float(ov[0]), float(ov[1]))
        else:
            tag_offset_str = os.environ.get("GRIPPER_TAG_OFFSET_PX")
            if tag_offset_str and "," in tag_offset_str:
                try:
                    parts = [float(p.strip()) for p in tag_offset_str.split(",") if p.strip()]
                    if len(parts) >= 2:
                        gripper_tag_offset_px = (parts[0], parts[1])
                except (ValueError, TypeError):
                    pass

        gripper_locator_er2 = False
        if overrides.get("gripper_locator_er2") is not None:
            gripper_locator_er2 = bool(overrides["gripper_locator_er2"])
        else:
            er2_env = os.environ.get("GRIPPER_LOCATOR_ER2", "")
            gripper_locator_er2 = er2_env.strip().lower() in ("1", "true", "yes")

        vision_settle_s = _resolve_float(
            overrides.get("vision_settle_s"),
            ("VISION_SETTLE_S",),
            cls.vision_settle_s,
        )

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
            gripper_empty_aperture_deg=gripper_empty_aperture_deg,
            grasp_empty_margin_deg=grasp_empty_margin_deg,
            gripper_settle_s=gripper_settle_s,
            grasp_load_min=grasp_load_min,
            descent_gate=descent_gate,
            gripper_tag_offset_px=gripper_tag_offset_px,
            gripper_locator_er2=gripper_locator_er2,
            vision_settle_s=vision_settle_s,
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
