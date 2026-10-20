"""Unit and simulation tests for the operator reach_check tool."""

import numpy as np
import pytest

from robot.errors import HardwareError
from tests.conftest import MockRobot
from tools.reach_check import build_reach_check_points, run_reach_check


def test_build_reach_check_points() -> None:
    """Verify the 5-point list has the 4 marker centres and the mat centre in order."""
    points = build_reach_check_points()
    assert len(points) == 5
    assert points[0] == ("top_left (ID 1)", 414.0, -240.0)
    assert points[1] == ("top_right (ID 2)", 97.0, -240.0)
    assert points[2] == ("bottom_right (ID 3)", 97.0, 102.0)
    assert points[3] == ("bottom_left (ID 0)", 414.0, 102.0)
    assert points[4] == ("mat_centre", 255.5, -69.0)


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
