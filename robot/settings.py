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
    # Closed-loop alignment tolerance in robot millimetres (jaw vs target centre, table plane)
    alignment_tolerance_mm: float = 4.0
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
    # Jaw tag centre relative to the grasp site, in the gripper frame (mm): x = jaw closing
    # direction, y = hinge axis, z = towards the wrist. Default: the simulated tag mount.
    gripper_tag_local_offset_mm: tuple[float, float, float] = (-14.0, 27.0, 63.0)
    # Nominal horizontal field of view (deg) of the workspace camera, for the camera model.
    camera_hfov_deg: float = 70.0
    # Apply the per-lock mm bias correction learned from jaw-tag residuals (1) or not (0).
    bias_correction: bool = True
    gripper_locator_er2: bool = False
    # Model used for the optional ER2 pointing query. Deliberately not ``gemini_model_id``: that
    # default is the streaming (Live API) model, which does not serve generate_content.
    er2_locator_model: str = "gemini-robotics-er-2-preview"
    vision_settle_s: float = 0.3
    # Robot-frame x coordinate of the table mat origin (mm) on the real arm. Always 0.0 in sim.
    table_x_mm: float = 0.0
    # Robot-frame y coordinate of the table mat origin (mm) on the real arm. Always 0.0 in sim.
    table_y_mm: float = 0.0
    # Robot-frame z coordinate of the table surface (mm) on the real arm. Always 0.0 in sim.
    table_z_mm: float = 0.0

    def get_table_x_mm(self, mode: str = "real") -> float:
        """Resolve table x coordinate in robot frame (mm).

        Always 0.0 in sim; returns calibrated table_x_mm in real mode.

        Args:
            mode: Operational mode ('real' or 'sim').

        Returns:
            Table x coordinate in mm in robot frame.
        """
        if mode == "sim":
            return 0.0
        env_val = os.environ.get("TABLE_X_MM")
        if env_val is not None and env_val.strip() != "":
            try:
                return float(env_val)
            except (ValueError, TypeError):
                pass
        return self.table_x_mm

    def get_table_y_mm(self, mode: str = "real") -> float:
        """Resolve table y coordinate in robot frame (mm).

        Always 0.0 in sim; returns calibrated table_y_mm in real mode.

        Args:
            mode: Operational mode ('real' or 'sim').

        Returns:
            Table y coordinate in mm in robot frame.
        """
        if mode == "sim":
            return 0.0
        env_val = os.environ.get("TABLE_Y_MM")
        if env_val is not None and env_val.strip() != "":
            try:
                return float(env_val)
            except (ValueError, TypeError):
                pass
        return self.table_y_mm

    def get_table_z_mm(self, mode: str = "real") -> float:
        """Resolve table surface height in robot frame (mm).

        Always 0.0 in sim; returns calibrated table_z_mm in real mode.

        Args:
            mode: Operational mode ('real' or 'sim').

        Returns:
            Table z coordinate in mm in robot frame.
        """
        if mode == "sim":
            return 0.0
        env_val = os.environ.get("TABLE_Z_MM")
        if env_val is not None and env_val.strip() != "":
            try:
                return float(env_val)
            except (ValueError, TypeError):
                pass
        return self.table_z_mm

    def get_table_offset_mm(self, mode: str = "real") -> tuple[float, float, float]:
        """Resolve table (x, y, z) offsets in robot frame (mm).

        Always (0.0, 0.0, 0.0) in sim; returns (table_x, table_y, table_z) in real.

        Args:
            mode: Operational mode ('real' or 'sim').

        Returns:
            Tuple of (table_x_mm, table_y_mm, table_z_mm).
        """
        return (self.get_table_x_mm(mode), self.get_table_y_mm(mode), self.get_table_z_mm(mode))

    def get_table_offsets_mm(self, mode: str = "real") -> tuple[float, float, float]:
        """Alias for get_table_offset_mm."""
        return self.get_table_offset_mm(mode)

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

        alignment_tolerance_mm = _resolve_float(
            overrides.get("alignment_tolerance_mm"),
            ("ALIGNMENT_TOLERANCE_MM",),
            cls.alignment_tolerance_mm,
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

        gripper_tag_local_offset_mm = cls.gripper_tag_local_offset_mm
        tag_override = overrides.get("gripper_tag_local_offset_mm")
        tag_text = os.environ.get("GRIPPER_TAG_LOCAL_OFFSET_MM")
        if tag_override is not None:
            if isinstance(tag_override, (tuple, list)) and len(tag_override) == 3:
                gripper_tag_local_offset_mm = (
                    float(tag_override[0]),
                    float(tag_override[1]),
                    float(tag_override[2]),
                )
        elif tag_text and tag_text.strip():
            try:
                parts = [float(p.strip()) for p in tag_text.split(",") if p.strip()]
                if len(parts) == 3:
                    gripper_tag_local_offset_mm = (parts[0], parts[1], parts[2])
            except (ValueError, TypeError):
                pass

        camera_hfov_deg = _resolve_float(
            overrides.get("camera_hfov_deg"), ("CAMERA_HFOV_DEG",), cls.camera_hfov_deg
        )

        bias_override = overrides.get("bias_correction")
        if bias_override is not None:
            bias_correction = bool(bias_override)
        else:
            bias_env = os.environ.get("BIAS_CORRECTION")
            bias_correction = (
                cls.bias_correction
                if bias_env is None or bias_env.strip() == ""
                else bias_env.strip().lower() in ("1", "true", "yes")
            )

        gripper_locator_er2 = False
        if overrides.get("gripper_locator_er2") is not None:
            gripper_locator_er2 = bool(overrides["gripper_locator_er2"])
        else:
            er2_env = os.environ.get("GRIPPER_LOCATOR_ER2", "")
            gripper_locator_er2 = er2_env.strip().lower() in ("1", "true", "yes")

        er2_locator_model = _resolve_str(
            overrides.get("er2_locator_model"),
            ("ER2_LOCATOR_MODEL",),
            cls.er2_locator_model,
        )

        vision_settle_s = _resolve_float(
            overrides.get("vision_settle_s"),
            ("VISION_SETTLE_S",),
            cls.vision_settle_s,
        )

        table_x_mm = _resolve_float(
            overrides.get("table_x_mm"),
            ("TABLE_X_MM",),
            cls.table_x_mm,
        )

        table_y_mm = _resolve_float(
            overrides.get("table_y_mm"),
            ("TABLE_Y_MM",),
            cls.table_y_mm,
        )

        table_z_mm = _resolve_float(
            overrides.get("table_z_mm"),
            ("TABLE_Z_MM",),
            cls.table_z_mm,
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
            alignment_tolerance_mm=alignment_tolerance_mm,
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
            gripper_tag_local_offset_mm=gripper_tag_local_offset_mm,
            camera_hfov_deg=camera_hfov_deg,
            bias_correction=bias_correction,
            gripper_locator_er2=gripper_locator_er2,
            er2_locator_model=er2_locator_model,
            vision_settle_s=vision_settle_s,
            table_x_mm=table_x_mm,
            table_y_mm=table_y_mm,
            table_z_mm=table_z_mm,
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


def table_to_robot_xy_mm(
    x_table_mm: float, y_table_mm: float, mode: str = "real"
) -> tuple[float, float]:
    """Convert table-mat-relative (X, Y) in millimetres to robot-frame (X, Y) in millimetres.

    Args:
        x_table_mm: Forward coordinate on the table mat in millimetres.
        y_table_mm: Lateral coordinate on the table mat in millimetres.
        mode: Operational mode ('real' or 'sim'). In sim, offsets are always 0.0.

    Returns:
        Tuple of (x_robot_mm, y_robot_mm) in robot base frame.
    """
    settings = get_settings()
    return (
        x_table_mm + settings.get_table_x_mm(mode),
        y_table_mm + settings.get_table_y_mm(mode),
    )


def robot_to_table_xy_mm(
    x_robot_mm: float, y_robot_mm: float, mode: str = "real"
) -> tuple[float, float]:
    """Convert robot-frame (X, Y) in millimetres to table-mat-relative (X, Y) in millimetres.

    Args:
        x_robot_mm: Forward coordinate in robot base frame in millimetres.
        y_robot_mm: Lateral coordinate in robot base frame in millimetres.
        mode: Operational mode ('real' or 'sim'). In sim, offsets are always 0.0.

    Returns:
        Tuple of (x_table_mm, y_table_mm) in table mat coordinates.
    """
    settings = get_settings()
    return (
        x_robot_mm - settings.get_table_x_mm(mode),
        y_robot_mm - settings.get_table_y_mm(mode),
    )


def table_to_robot_xyz_mm(
    x_table_mm: float, y_table_mm: float, z_table_mm: float, mode: str = "real"
) -> tuple[float, float, float]:
    """Convert table-mat-relative (X, Y, Z) in mm to robot-frame (X, Y, Z) in mm.

    Args:
        x_table_mm: Forward coordinate on table mat in millimetres.
        y_table_mm: Lateral coordinate on table mat in millimetres.
        z_table_mm: Height above table surface in millimetres.
        mode: Operational mode ('real' or 'sim'). In sim, offsets are always 0.0.

    Returns:
        Tuple of (x_robot_mm, y_robot_mm, z_robot_mm) in robot base frame.
    """
    settings = get_settings()
    return (
        x_table_mm + settings.get_table_x_mm(mode),
        y_table_mm + settings.get_table_y_mm(mode),
        z_table_mm + settings.get_table_z_mm(mode),
    )


def robot_to_table_xyz_mm(
    x_robot_mm: float, y_robot_mm: float, z_robot_mm: float, mode: str = "real"
) -> tuple[float, float, float]:
    """Convert robot-frame (X, Y, Z) in mm to table-mat-relative (X, Y, Z) in mm.

    Args:
        x_robot_mm: Forward coordinate in robot base frame in millimetres.
        y_robot_mm: Lateral coordinate in robot base frame in millimetres.
        z_robot_mm: Robot-frame z coordinate in millimetres.
        mode: Operational mode ('real' or 'sim'). In sim, offsets are always 0.0.

    Returns:
        Tuple of (x_table_mm, y_table_mm, z_table_mm) in table mat coordinates.
    """
    settings = get_settings()
    return (
        x_robot_mm - settings.get_table_x_mm(mode),
        y_robot_mm - settings.get_table_y_mm(mode),
        z_robot_mm - settings.get_table_z_mm(mode),
    )


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
