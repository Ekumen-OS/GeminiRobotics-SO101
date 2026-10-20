"""Unit tests for main.py CLI arguments, tasks, and degraded real mode error handling."""

from unittest.mock import MagicMock

import pytest

import main
from robot.errors import HardwareError


def test_main_real_mode_missing_lerobot_exits_cleanly(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """ImportError from missing lerobot should exit with code 1 and install instructions."""

    def _mock_init(*args, **kwargs):
        raise ImportError("The 'lerobot' package is not installed or available")

    monkeypatch.setattr(main, "RealSO101Robot", _mock_init)
    monkeypatch.setattr("sys.argv", ["main.py", "--mode", "real"])

    with pytest.raises(SystemExit) as exc_info:
        main.main()

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert "uv sync --extra hw --extra ui" in combined


def test_main_real_mode_missing_calibration_exits_cleanly(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """FileNotFoundError for motor calibration should exit with code 1 and calibration hint."""
    calib_msg = (
        "No LeRobot motor calibration file found. Searched:\n"
        "  /path/to/calib.json\n"
        "Run `lerobot-calibrate --robot.type=so101_follower`"
    )

    def _mock_init(*args, **kwargs):
        raise FileNotFoundError(calib_msg)

    monkeypatch.setattr(main, "RealSO101Robot", _mock_init)
    monkeypatch.setattr("sys.argv", ["main.py", "--mode", "real"])

    with pytest.raises(SystemExit) as exc_info:
        main.main()

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert "No LeRobot motor calibration file found" in combined
    assert "lerobot-calibrate" in combined


def test_main_real_mode_hardware_error_enters_degraded_mode(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """HardwareError on bus connect should enter degraded mode without exiting."""
    mock_robot = MagicMock()

    def _mock_real_robot(*args, **kwargs):
        if kwargs.get("connect_on_init", True):
            raise HardwareError("Could not connect to Feetech serial bus")
        return mock_robot

    mock_launch_ui = MagicMock()
    monkeypatch.setattr(main, "RealSO101Robot", _mock_real_robot)
    monkeypatch.setattr("ui.app.launch_web_ui", mock_launch_ui)
    monkeypatch.setattr("sys.argv", ["main.py", "--mode", "real", "--port", "/dev/fake_port"])

    main.main()

    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert "Starting UI in degraded/disconnected real mode." in combined
    mock_launch_ui.assert_called_once()
    mock_robot.move_to_pose.assert_called_once()
    mock_robot.disconnect.assert_called_once()


def test_main_task_generate_markers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Running --task generate_markers should dispatch to setup_workspace_interactive."""
    mock_wizard = MagicMock()
    monkeypatch.setattr("tools.setup_wizard.setup_workspace_interactive", mock_wizard)
    monkeypatch.setattr(
        "sys.argv",
        ["main.py", "--task", "generate_markers", "--non-interactive", "--reuse"],
    )

    main.main()

    mock_wizard.assert_called_once_with(
        output_dir="assets/aruco_markers",
        interactive=False,
        reuse=True,
    )


def test_main_task_check_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Running --task check_environment should dispatch to tools check_environment."""
    mock_check = MagicMock(return_value=0)
    monkeypatch.setattr("tools.check_env.check_environment", mock_check)
    monkeypatch.setattr("sys.argv", ["main.py", "--task", "check_environment", "--mode", "sim"])

    with pytest.raises(SystemExit) as exc_info:
        main.main()

    assert exc_info.value.code == 0
    mock_check.assert_called_once_with(mode="sim")
