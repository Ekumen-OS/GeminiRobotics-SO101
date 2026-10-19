"""End-to-end scripted pick-and-place in the MuJoCo sim, judged against MuJoCo truth.

The default suite runs a 2-placement smoke. ``uv run pytest -m slow -s`` runs the 10-placement
scenarios and prints the metrics table.
"""

import numpy as np
import pytest

from tests.harness.sim_pick import (
    ScenarioResult,
    SimPickHarness,
    format_metrics_table,
    run_scenario,
)
from vision.detector import detect_workspace_objects


def _assert_clean(result: ScenarioResult) -> None:
    """Zero verdict/physics contradictions, zero lifts refused, zero refusal loops, <= 4 hovers."""
    summary = result.summary()
    assert summary["contradictions"] == 0, result.placements
    assert summary["lift_refusals"] == 0, summary["refusals"]
    assert summary["refusal_loops"] == 0, summary["refusals"]
    assert summary["max_hovers_in_attempt"] <= 4


def test_harness_smoke_runs_and_reports_metrics(capsys: pytest.CaptureFixture[str]) -> None:
    """Two unbiased placements: both lifted and placed, with every verdict matching the physics."""
    result = run_scenario("smoke", n_placements=2, seed=0)
    table = format_metrics_table([result])
    print(table)
    assert len(result.placements) == 2
    assert all(p.transcript for p in result.placements)
    summary = result.summary()
    assert summary["n"] == 2
    assert summary["lifted"] == 2
    assert summary["placed_within_15mm"] == 2
    _assert_clean(result)
    assert "metric" in capsys.readouterr().out


def test_drifted_model_point_regression() -> None:
    """One placement with offset (25, 68) between 40 mm table and 80 unit image radii."""
    cube_xy = (0.24, -0.02)
    offset = (25.0, 68.0)
    harness = SimPickHarness(point_offset_1000=offset)
    try:
        harness.reset(cube_xy)
        objs = [o for o in detect_workspace_objects(harness.frame_rgb()) if o["color"] == "red"]
        assert objs, "harness could not see the cube on the initial frame"
        blob_center = objs[0]["center_1000"]

        rng = np.random.default_rng(0)
        point = harness.model_point(rng)

        # Precondition asserts: test must sit in the gap between the two radii.
        img_dist = float(np.hypot(point[0] - blob_center[0], point[1] - blob_center[1]))
        assert img_dist < 80.0, f"Image distance {img_dist:.1f} must be < 80 units"

        px, py = harness.calib.gemini_to_pixel(point)
        point_table_xy_mm = harness.calib.pixel_to_robot_mm(px, py)
        cube_truth_mm = harness.cube_xyz()[:2] * 1000.0
        table_dist_mm = float(
            np.hypot(
                point_table_xy_mm[0] - cube_truth_mm[0], point_table_xy_mm[1] - cube_truth_mm[1]
            )
        )
        assert 40.0 <= table_dist_mm <= 75.0, (
            f"Table distance {table_dist_mm:.1f} mm must be between 40 and 75 mm"
        )

        result = harness.run_placement(cube_xy, seed=0)
        assert result.lifted
        assert (
            result.placed
            and result.placement_error_mm is not None
            and result.placement_error_mm <= 15.0
        )
        assert result.locks_created == 1
        assert result.refusal_loops == 0
        assert result.max_hovers_in_attempt <= 4
    finally:
        harness.close()


@pytest.mark.slow
def test_harness_unbiased_10_placements() -> None:
    """Unbiased: at least 9/10 cubes lifted and placed within 15 mm; no contradictions."""
    result = run_scenario("unbiased", n_placements=10, seed=1, progress=print)
    print(format_metrics_table([result]))
    summary = result.summary()
    assert summary["lifted"] >= 9
    assert summary["placed_within_15mm"] >= 9
    _assert_clean(result)


@pytest.mark.slow
def test_harness_biased_with_correction_10_placements() -> None:
    """1.5 deg pan offset, corrector on: jaw within 3 mm of the cube at close in >= 9/10."""
    result = run_scenario(
        "biased_corr_on", n_placements=10, seed=1, pan_offset_deg=1.5, progress=print
    )
    print(format_metrics_table([result]))
    summary = result.summary()
    assert summary["jaw_to_cube_at_close_le_3mm"] >= 9
    assert summary["placed_within_15mm"] >= 9
    _assert_clean(result)


@pytest.mark.slow
def test_harness_unbiased_random_yaw_10_placements() -> None:
    """Random yaw, unbiased: at least 8/10 placed within 15 mm; jaw <= 4 mm in >= 8/10."""
    result = run_scenario(
        "unbiased_rand_yaw", n_placements=10, seed=1, random_yaw=True, progress=print
    )
    print(format_metrics_table([result]))
    summary = result.summary()
    assert summary["placed_within_15mm"] >= 8
    assert summary["jaw_to_cube_at_close_le_4mm"] >= 8
    _assert_clean(result)


@pytest.mark.slow
def test_harness_biased_with_correction_random_yaw_10_placements() -> None:
    """Random yaw, 1.5 deg pan offset: at least 8/10 placed within 15 mm; jaw <= 4 mm in >= 8/10."""
    result = run_scenario(
        "biased_corr_on_rand_yaw",
        n_placements=10,
        seed=1,
        pan_offset_deg=1.5,
        random_yaw=True,
        progress=print,
    )
    print(format_metrics_table([result]))
    summary = result.summary()
    assert summary["placed_within_15mm"] >= 8
    assert summary["jaw_to_cube_at_close_le_4mm"] >= 8
    _assert_clean(result)


@pytest.mark.slow
def test_harness_biased_without_correction_shows_the_corrector_matters() -> None:
    """Same bias, corrector disabled: the median jaw-to-cube error stays at the bias (>= 5 mm)."""
    result = run_scenario(
        "biased_corr_off",
        n_placements=10,
        seed=1,
        pan_offset_deg=1.5,
        settings_overrides={"descent_gate": "warn"},
        descend_after_hover_cap=True,
        disable_corrector=True,
        progress=print,
    )
    print(format_metrics_table([result]))
    summary = result.summary()
    assert summary["jaw_to_cube_at_close_median_mm"] >= 5.0
    assert summary["contradictions"] == 0


@pytest.mark.slow
def test_harness_full_metrics_table() -> None:
    """Print the three-scenario metrics table of the report (seed 1)."""
    results = [
        run_scenario("unbiased", 10, seed=1),
        run_scenario("biased_corr_on", 10, seed=1, pan_offset_deg=1.5),
        run_scenario(
            "biased_corr_off",
            10,
            seed=1,
            pan_offset_deg=1.5,
            settings_overrides={"descent_gate": "warn"},
            descend_after_hover_cap=True,
            disable_corrector=True,
        ),
    ]
    print("\n" + format_metrics_table(results))
    assert all(r.summary()["n"] == 10 for r in results)


def test_pan_offset_wrapper_bias_is_invisible_to_fk() -> None:
    """A 1.5 deg pan encoder offset shifts the true site 6-7 mm while reported joints agree."""
    import numpy as np

    from robot.cartesian_ik import CartesianIK

    harness = SimPickHarness(pan_offset_deg=1.5)
    try:
        ik = CartesianIK()
        target = np.array([0.25, 0.0, 0.05])
        q = ik.solve_ik(target, initial_qpos=harness.robot.get_joint_positions())
        harness.robot.move_to_pose(q, duration_sec=1.0)
        fk_site = ik.forward_kinematics(harness.robot.get_joint_positions())
        true_site = harness.site_xyz()
        assert np.linalg.norm(fk_site[:2] - target[:2]) < 0.001  # FK believes it is on target
        lateral_mm = float(np.linalg.norm(true_site[:2] - target[:2]) * 1000.0)
        assert 5.0 <= lateral_mm <= 8.0, lateral_mm
    finally:
        harness.close()
