"""Unit and simulation tests for the operator reach_check tool."""

import json
from pathlib import Path

import numpy as np
import pytest

from robot.errors import HardwareError
from robot.settings import get_settings
from tests.conftest import MockRobot
from tools.reach_check import build_reach_check_points, replay_reach_point, run_reach_check


def test_build_reach_check_points(tmp_path: Path) -> None:
    """Verify the 5-point list has the 4 marker centres and the mat centre in order."""
    fixture_cfg = {
        "marker_ids": {"top_left": 1, "top_right": 2, "bottom_right": 3, "bottom_left": 0},
        "workspace_corners_mm": {
            "top_left": {"x": 500.0, "y": -200.0},
            "top_right": {"x": 100.0, "y": -200.0},
            "bottom_right": {"x": 100.0, "y": 150.0},
            "bottom_left": {"x": 500.0, "y": 150.0},
        },
    }
    cfg_file = tmp_path / "custom_calibration.json"
    cfg_file.write_text(json.dumps(fixture_cfg), encoding="utf-8")

    points = build_reach_check_points(config_path=cfg_file)
    assert len(points) == 5
    assert points[0] == ("top_left (ID 1)", 500.0, -200.0)
    assert points[1] == ("top_right (ID 2)", 100.0, -200.0)
    assert points[2] == ("bottom_right (ID 3)", 100.0, 150.0)
    assert points[3] == ("bottom_left (ID 0)", 500.0, 150.0)
    assert points[4] == ("mat_centre", 300.0, -25.0)

    # Calling without arguments reads the active settings config dynamically without literals
    active_points = build_reach_check_points()
    assert len(active_points) == 5
    active_cfg = json.loads(get_settings().calibration_config_path.read_text(encoding="utf-8"))
    active_corners = active_cfg["workspace_corners_mm"]
    assert active_points[0][1] == active_corners["top_left"]["x"]
    assert active_points[0][2] == active_corners["top_left"]["y"]
    assert active_points[1][1] == active_corners["top_right"]["x"]
    assert active_points[1][2] == active_corners["top_right"]["y"]
    assert active_points[2][1] == active_corners["bottom_right"]["x"]
    assert active_points[2][2] == active_corners["bottom_right"]["y"]
    assert active_points[3][1] == active_corners["bottom_left"]["x"]
    assert active_points[3][2] == active_corners["bottom_left"]["y"]


def test_run_reach_check_sim_end_to_end(capsys: pytest.CaptureFixture[str]) -> None:
    """Run reach_check end-to-end in sim and verify FK vs truth and locator vs truth."""
    results = run_reach_check(mode="sim", interactive=False)
    out = capsys.readouterr().out
    assert "measure the miss from the fingertip midpoint to the marker centre" in out
    assert len(results) == 5

    for res in results:
        assert res.fk_vs_truth_mm is not None
        assert res.fk_vs_truth_mm <= 2.0

    loc_errors = [res.loc_vs_truth_mm for res in results if res.loc_vs_truth_mm is not None]
    assert loc_errors
    assert float(np.mean(loc_errors)) <= 3.0


def test_run_reach_check_fails_closed_on_hardware_error() -> None:
    """If the robot raises HardwareError during reach_check, it interrupts and re-raises."""
    robot = MockRobot()

    def broken_move(*args: object, **kwargs: object) -> None:
        raise HardwareError("Servo bus disconnected", robot_moved=True)

    robot.move_to_pose = broken_move  # type: ignore[method-assign]
    with pytest.raises(HardwareError, match="Servo bus disconnected"):
        run_reach_check(
            mode="real",
            interactive=False,
            robot=robot,
            frame_fn=lambda: None,
        )
    assert robot._stop_requested is True


def test_run_reach_check_marks_unreachable_and_saves_telemetry(tmp_path: Path) -> None:
    """Reach check marks points differing by >5 mm as UNREACHABLE and saves telemetry."""
    # Config with an extreme out-of-reach point
    out_of_reach_cfg = {
        "marker_ids": {"top_left": 1, "top_right": 2, "bottom_right": 3, "bottom_left": 0},
        "workspace_corners_mm": {
            "top_left": {"x": 1000.0, "y": -200.0},
            "top_right": {"x": 200.0, "y": -50.0},
            "bottom_right": {"x": 200.0, "y": 50.0},
            "bottom_left": {"x": 250.0, "y": 0.0},
        },
    }
    cfg_file = tmp_path / "extreme_cfg.json"
    cfg_file.write_text(json.dumps(out_of_reach_cfg), encoding="utf-8")
    telemetry_dir = tmp_path / "telemetry"

    results = run_reach_check(
        mode="sim",
        interactive=False,
        config_path=cfg_file,
        telemetry_dir=telemetry_dir,
    )

    # First point was at x=1000.0 mm (clearly unreachable for 350 mm arm)
    assert results[0].unreachable is True
    assert results[0].ik_err_mm > 5.0

    # Verify telemetry directory has saved point JSONs and PNGs
    saved_jsons = list(telemetry_dir.glob("*.json"))
    assert len(saved_jsons) == len(results)

    # Test offline replay
    for json_file in saved_jsons:
        replay_res = replay_reach_point(json_file)
        assert replay_res["name"] is not None
        assert len(replay_res["fk_xy_mm"]) == 2
        if replay_res["saved_loc_xy_mm"] is not None:
            assert replay_res["matches"] is True


def test_run_reach_check_hardware_error_logs_interrupt_failure_to_stderr(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """If robot.interrupt() raises HardwareError, reach_check logs to stderr and re-raises."""
    robot = MockRobot()

    def broken_move(*args: object, **kwargs: object) -> None:
        raise HardwareError("Servo failure", robot_moved=True)

    def broken_interrupt() -> None:
        raise HardwareError("Interrupt bus failed")

    robot.move_to_pose = broken_move  # type: ignore[method-assign]
    robot.interrupt = broken_interrupt  # type: ignore[method-assign]

    with pytest.raises(HardwareError, match="Servo failure"):
        run_reach_check(
            mode="real",
            interactive=False,
            robot=robot,
            frame_fn=lambda: None,
        )

    err = capsys.readouterr().err
    assert "Failed to interrupt robot: Interrupt bus failed" in err
