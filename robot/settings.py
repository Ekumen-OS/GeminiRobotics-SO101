"""Central configuration settings for GeminiRobotics-SO101."""

import dataclasses
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _get_platform_serial_port() -> str:
    """Return platform-appropriate default serial port path without reading env.

    Returns:
        Default serial device path for the host platform.
    """
    return "/dev/tty.usbmodem5B415332861" if sys.platform == "darwin" else "/dev/ttyACM0"


def _lerobot_calibration_root() -> Path:
    """Return the LeRobot calibration cache root, mirroring `lerobot.constants`.

    Resolution order matches LeRobot so that files written by `lerobot-calibrate`
    are found without importing the package (which may not be installed).

    Returns:
        Path to the calibration directory (may not exist).
    """
    for key, sub in (
        ("HF_LEROBOT_CALIBRATION", None),
        ("HF_LEROBOT_HOME", "calibration"),
        ("HF_HOME", "lerobot/calibration"),
    ):
        val = os.environ.get(key)
        if val:
            p = Path(val).expanduser()
            return p / sub if sub else p
    return Path.home() / ".cache" / "huggingface" / "lerobot" / "calibration"


def _parse_xyz(val: str) -> tuple[float, float, float]:
    """Parse comma-separated X,Y,Z coordinates from a string.

    Args:
        val: Comma-separated string of three floats (e.g. "x,y,z").

    Returns:
        Tuple of (x, y, z) float coordinates.

    Raises:
        ValueError: If val does not contain exactly three valid floats.
    """
    parts = [float(p.strip()) for p in val.split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError(f"Expected 3 comma-separated floats, got: {val!r}")
    return (parts[0], parts[1], parts[2])


def _parse_descent_gate(val: str) -> str:
    """Parse descent gate policy string.

    Args:
        val: Input policy string.

    Returns:
        Validated policy string ('block', 'warn', or 'off').

    Raises:
        ValueError: If val is not 'block', 'warn', or 'off'.
    """
    v = val.lower()
    if v in ("block", "warn", "off"):
        return v
    raise ValueError(f"Expected 'block', 'warn', or 'off', got: {val!r}")


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
    calibration_config_path: Path = _PROJECT_ROOT / "config" / "workspace_calibration.json"
    calibration_save_path: Path = _PROJECT_ROOT / "config" / "workspace_homography.npz"
    calibration_sim_save_path: Path = _PROJECT_ROOT / "config" / "workspace_homography_sim.npz"
    alignment_tolerance_mm: float = 4.0
    stream_fps: int = 30
    stream_width: int = 640
    stream_height: int = 480
    web_stream_fps: int = 15
    web_jpeg_quality: int = 75
    camera_index: int | None = None
    camera_name: str | None = None
    gripper_empty_aperture_deg: float | None = None
    grasp_empty_margin_deg: float = 6.0
    gripper_settle_s: float = 0.6
    grasp_load_min: float | None = None
    descent_gate: str | None = None
    gripper_tag_local_offset_mm: tuple[float, float, float] = (-25.4, 25.5, 78.6)
    # 62.2° horizontal FOV: Arducam IMX219 3.04 mm lens datasheet spec (Amazon listing states 60°).
    # Sensor-mode check confirmed 640x480 uses the full sensor width (vertical crop vs 1280x720,
    # marker horizontal spacing ratio 1.000). With the measured corners, 62.2° recovers camera
    # height z = 358 mm vs 376 mm at 60.0°, landing closer to the 345 mm tape cross-check.
    camera_hfov_deg: float = 62.2
    camera_focus: int = 144
    vision_settle_s: float = 0.3
    table_z_mm: float = 0.0

    def get_table_z_mm(self, mode: str = "real") -> float:
        """Resolve table surface height in robot frame (mm).

        Always 0.0 in sim; returns calibrated table_z_mm in real mode.

        Args:
            mode: Operational mode ('real' or 'sim').

        Returns:
            Table z coordinate in mm in robot frame.
        """
        return 0.0 if mode == "sim" else self.table_z_mm

    def get_descent_gate(self, mode: str = "real") -> str:
        """Resolve descent gate policy ('block', 'warn', or 'off').

        Defaults to 'block' in sim and 'warn' in real unless explicitly overridden.

        Args:
            mode: Operational mode ('real' or 'sim').

        Returns:
            Active descent gate policy string.
        """
        if self.descent_gate in ("block", "warn", "off"):
            return self.descent_gate
        return "block" if mode == "sim" else "warn"

    @property
    def credentials_mode(self) -> str:
        """Return credentials mode identifier ('api_key', 'vertex', or 'none').

        Returns:
            Credentials mode string ('api_key', 'vertex', or 'none').
        """
        if self.google_api_key:
            return "api_key"
        return "vertex" if self.google_cloud_project else "none"

    @property
    def motor_calibration_candidates(self) -> tuple[Path, ...]:
        """Return candidate paths for the follower's LeRobot motor calibration file.

        Returns:
            Tuple of candidate Path objects to search.
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
        defaults = {
            f.name: (
                f.default_factory() if f.default_factory is not dataclasses.MISSING else f.default
            )
            for f in dataclasses.fields(cls)
        }
        resolved: dict[str, Any] = {}
        for name, env_var, parser in _CONFIG_SPEC:
            default = defaults[name]
            if overrides.get(name) is not None:
                val = overrides[name]
                if isinstance(val, str) and parser is not str:
                    try:
                        resolved[name] = parser(val)
                    except (ValueError, TypeError):
                        logger.warning(
                            "Invalid override for %s=%r; using default %r", name, val, default
                        )
                        resolved[name] = default
                elif parser is Path and not isinstance(val, Path):
                    resolved[name] = Path(val)
                elif parser is _parse_xyz and not isinstance(val, tuple):
                    resolved[name] = tuple(float(x) for x in val)
                else:
                    resolved[name] = val
            else:
                env_val = os.environ.get(env_var)
                if env_val is not None and env_val.strip() != "":
                    raw = env_val.strip()
                    try:
                        resolved[name] = parser(raw)
                    except (ValueError, TypeError):
                        logger.warning(
                            "Invalid value for %s=%r; using default %r", env_var, raw, default
                        )
                        resolved[name] = default
                else:
                    resolved[name] = default

        return cls(**resolved)


_CONFIG_SPEC: tuple[tuple[str, str, Callable[[str], Any]], ...] = (
    ("serial_port", "ROBOT_PORT", str),
    ("gemini_model_id", "GEMINI_MODEL", str),
    ("google_api_key", "GOOGLE_API_KEY", str),
    ("google_cloud_project", "GOOGLE_CLOUD_PROJECT", str),
    ("google_cloud_location", "GOOGLE_CLOUD_LOCATION", str),
    ("server_host", "SERVER_HOST", str),
    ("server_port", "SERVER_PORT", int),
    ("follower_robot_id", "FOLLOWER_ROBOT_ID", str),
    ("motor_calibration_path", "MOTOR_CALIBRATION_PATH", Path),
    ("calibration_config_path", "CALIBRATION_CONFIG_PATH", Path),
    ("calibration_save_path", "CALIBRATION_SAVE_PATH", Path),
    ("calibration_sim_save_path", "CALIBRATION_SIM_SAVE_PATH", Path),
    ("alignment_tolerance_mm", "ALIGNMENT_TOLERANCE_MM", float),
    ("stream_fps", "STREAM_FPS", int),
    ("stream_width", "STREAM_WIDTH", int),
    ("stream_height", "STREAM_HEIGHT", int),
    ("web_stream_fps", "WEB_STREAM_FPS", int),
    ("web_jpeg_quality", "WEB_JPEG_QUALITY", int),
    ("camera_index", "CAMERA_INDEX", int),
    ("camera_name", "CAMERA_NAME", str),
    ("gripper_empty_aperture_deg", "GRIPPER_EMPTY_APERTURE_DEG", float),
    ("grasp_empty_margin_deg", "GRASP_EMPTY_MARGIN_DEG", float),
    ("gripper_settle_s", "GRIPPER_SETTLE_S", float),
    ("grasp_load_min", "GRASP_LOAD_MIN", float),
    ("descent_gate", "DESCENT_GATE", _parse_descent_gate),
    ("gripper_tag_local_offset_mm", "GRIPPER_TAG_LOCAL_OFFSET_MM", _parse_xyz),
    ("camera_hfov_deg", "CAMERA_HFOV_DEG", float),
    ("camera_focus", "CAMERA_FOCUS", int),
    ("vision_settle_s", "VISION_SETTLE_S", float),
    ("table_z_mm", "TABLE_Z_MM", float),
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


def table_height_to_robot_z_m(height_m: float, mode: str = "real") -> float:
    """Convert table-relative height in metres to robot-frame z in metres.

    Args:
        height_m: Height above the table plane in metres.
        mode: Operational mode ('real' or 'sim'). In sim, table_z is always 0.0.

    Returns:
        Robot-frame z coordinate in metres.
    """
    table_z_mm = get_settings().get_table_z_mm(mode)
    return height_m + (table_z_mm / 1000.0)


def robot_z_to_table_height_mm(z_mm: float, mode: str = "real") -> float:
    """Convert robot-frame z in millimetres to table-relative height in millimetres.

    Args:
        z_mm: Robot-frame z coordinate in millimetres.
        mode: Operational mode ('real' or 'sim'). In sim, table_z is always 0.0.

    Returns:
        Height above table plane in millimetres.
    """
    table_z_mm = get_settings().get_table_z_mm(mode)
    return z_mm - table_z_mm
