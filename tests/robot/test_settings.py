"""Tests for centralized table-driven Settings and environment variable resolution."""

import logging
from pathlib import Path
from typing import Any

import pytest

from robot.settings import Settings, _get_platform_serial_port

_FIELDS_SPEC = [
    ("serial_port", "ROBOT_PORT", "/dev/ttyCUSTOM", "/dev/ttyCUSTOM", None),
    ("gemini_model_id", "GEMINI_MODEL", "custom-model", "custom-model", None),
    ("google_api_key", "GOOGLE_API_KEY", "test-key-123", "test-key-123", None),
    ("google_cloud_project", "GOOGLE_CLOUD_PROJECT", "proj-456", "proj-456", None),
    ("google_cloud_location", "GOOGLE_CLOUD_LOCATION", "europe-west1", "europe-west1", None),
    ("server_host", "SERVER_HOST", "127.0.0.1", "127.0.0.1", None),
    ("server_port", "SERVER_PORT", "9090", 9090, "not_an_int"),
    ("follower_robot_id", "FOLLOWER_ROBOT_ID", "custom_arm", "custom_arm", None),
    (
        "motor_calibration_path",
        "MOTOR_CALIBRATION_PATH",
        "/tmp/cal.json",
        Path("/tmp/cal.json"),
        None,
    ),
    (
        "calibration_config_path",
        "CALIBRATION_CONFIG_PATH",
        "/tmp/custom_ws.json",
        Path("/tmp/custom_ws.json"),
        None,
    ),
    (
        "calibration_save_path",
        "CALIBRATION_SAVE_PATH",
        "/tmp/custom_h.npz",
        Path("/tmp/custom_h.npz"),
        None,
    ),
    (
        "calibration_sim_save_path",
        "CALIBRATION_SIM_SAVE_PATH",
        "/tmp/custom_h_sim.npz",
        Path("/tmp/custom_h_sim.npz"),
        None,
    ),
    ("alignment_tolerance_mm", "ALIGNMENT_TOLERANCE_MM", "2.5", 2.5, "not_a_float"),
    ("stream_fps", "STREAM_FPS", "60", 60, "not_an_int"),
    ("stream_width", "STREAM_WIDTH", "1280", 1280, "not_an_int"),
    ("stream_height", "STREAM_HEIGHT", "720", 720, "not_an_int"),
    ("web_stream_fps", "WEB_STREAM_FPS", "20", 20, "not_an_int"),
    ("web_jpeg_quality", "WEB_JPEG_QUALITY", "90", 90, "not_an_int"),
    ("camera_index", "CAMERA_INDEX", "4", 4, "not_an_int"),
    ("camera_name", "CAMERA_NAME", "Brio4K", "Brio4K", None),
    ("gripper_empty_aperture_deg", "GRIPPER_EMPTY_APERTURE_DEG", "-72.0", -72.0, "not_a_float"),
    ("grasp_empty_margin_deg", "GRASP_EMPTY_MARGIN_DEG", "8.0", 8.0, "not_a_float"),
    ("gripper_settle_s", "GRIPPER_SETTLE_S", "1.5", 1.5, "not_a_float"),
    ("grasp_load_min", "GRASP_LOAD_MIN", "120.0", 120.0, "not_a_float"),
    ("descent_gate", "DESCENT_GATE", "off", "off", "invalid_gate"),
    (
        "gripper_tag_local_offset_mm",
        "GRIPPER_TAG_LOCAL_OFFSET_MM",
        "1.0, 2.0, 3.0",
        (1.0, 2.0, 3.0),
        "invalid,tuple",
    ),
    ("camera_hfov_deg", "CAMERA_HFOV_DEG", "65.0", 65.0, "not_a_float"),
    ("vision_settle_s", "VISION_SETTLE_S", "0.05", 0.05, "not_a_float"),
    ("table_x_mm", "TABLE_X_MM", "15.0", 15.0, "not_a_float"),
    ("table_y_mm", "TABLE_Y_MM", "-25.0", -25.0, "not_a_float"),
    ("table_z_mm", "TABLE_Z_MM", "-30.0", -30.0, "not_a_float"),
]


@pytest.mark.parametrize(
    ("field_name", "env_var", "valid_env_val", "expected_val", "invalid_env_val"),
    _FIELDS_SPEC,
)
def test_settings_env_var_resolution(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    field_name: str,
    env_var: str,
    valid_env_val: str,
    expected_val: Any,
    invalid_env_val: str | None,
) -> None:
    """Verify that every setting resolves correctly when unset, set, or invalid."""
    # 1. Unset environment variable -> returns default value
    monkeypatch.delenv(env_var, raising=False)
    default_settings = Settings.from_env()
    actual_default = getattr(default_settings, field_name)
    if field_name == "serial_port":
        assert actual_default == _get_platform_serial_port()
    elif field_name == "descent_gate":
        assert actual_default is None

    # 2. Valid environment variable -> returns parsed value
    monkeypatch.setenv(env_var, valid_env_val)
    parsed_settings = Settings.from_env()
    assert getattr(parsed_settings, field_name) == expected_val

    # 3. Invalid environment variable -> returns default value and logs a WARN naming env var
    if invalid_env_val is not None:
        monkeypatch.setenv(env_var, invalid_env_val)
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            fallback_settings = Settings.from_env()
        assert getattr(fallback_settings, field_name) == actual_default
        assert any(
            env_var in record.message and invalid_env_val in record.message
            for record in caplog.records
        )


def test_settings_override_beats_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify explicit overrides win over environment variables."""
    monkeypatch.setenv("SERVER_PORT", "9999")
    monkeypatch.setenv("TABLE_X_MM", "50.0")

    s = Settings.from_env(server_port=8888, table_x_mm=10.0)
    assert s.server_port == 8888
    assert s.table_x_mm == 10.0


def test_settings_sim_mode_zeros_table_offsets() -> None:
    """Verify getters return 0.0 in sim mode and the configured values in real mode."""
    s = Settings(table_x_mm=12.0, table_y_mm=-34.0, table_z_mm=-24.0)
    assert s.get_table_x_mm("sim") == 0.0
    assert s.get_table_y_mm("sim") == 0.0
    assert s.get_table_z_mm("sim") == 0.0

    assert s.get_table_x_mm("real") == 12.0
    assert s.get_table_y_mm("real") == -34.0
    assert s.get_table_z_mm("real") == -24.0


def test_settings_get_descent_gate_mode_defaults() -> None:
    """Verify descent gate defaults to block in sim and warn in real when unset or invalid."""
    s_unset = Settings(descent_gate=None)
    assert s_unset.get_descent_gate("sim") == "block"
    assert s_unset.get_descent_gate("real") == "warn"

    s_configured = Settings(descent_gate="off")
    assert s_configured.get_descent_gate("sim") == "off"
    assert s_configured.get_descent_gate("real") == "off"

    s_invalid = Settings(descent_gate="unknown")
    assert s_invalid.get_descent_gate("sim") == "block"
    assert s_invalid.get_descent_gate("real") == "warn"


def test_dropped_env_aliases_are_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that dropped aliases (PORT, HOST, SO101_PORT, etc.) do not affect settings."""
    monkeypatch.delenv("SERVER_PORT", raising=False)
    monkeypatch.delenv("SERVER_HOST", raising=False)
    monkeypatch.delenv("ROBOT_PORT", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.delenv("FOLLOWER_ROBOT_ID", raising=False)

    monkeypatch.setenv("PORT", "5555")
    monkeypatch.setenv("HOST", "192.168.1.100")
    monkeypatch.setenv("SO101_PORT", "/dev/ttyDROPPED")
    monkeypatch.setenv("GEMINI_ROBOTICS_MODEL", "dropped-model")
    monkeypatch.setenv("ROBOT_ID", "dropped_robot")
    monkeypatch.setenv("ALIGNMENT_TOLERANCE", "99")

    s = Settings.from_env()
    assert s.server_port == 7860
    assert s.server_host == "0.0.0.0"
    assert s.serial_port == _get_platform_serial_port()
    assert s.gemini_model_id == "gemini-robotics-er-2-streaming-preview"
    assert s.follower_robot_id == "follower_droid"
    assert s.alignment_tolerance_mm == 4.0
