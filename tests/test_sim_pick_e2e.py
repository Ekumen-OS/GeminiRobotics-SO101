"""End-to-end scripted pick-and-place in the MuJoCo sim, judged against MuJoCo truth.

The default suite runs a 2-placement smoke. ``uv run pytest -m slow -s`` runs the 10-placement
scenarios and prints the metrics table.
"""

import pytest

from tests.sim_pick_harness import ScenarioResult, format_metrics_table, run_scenario


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
def test_harness_biased_without_correction_shows_the_corrector_matters() -> None:
    """Same bias, BIAS_CORRECTION=0: the median jaw-to-cube error stays at the bias (>= 5 mm)."""
    result = run_scenario(
        "biased_corr_off",
        n_placements=10,
        seed=1,
        pan_offset_deg=1.5,
        settings_overrides={"bias_correction": False, "descent_gate": "warn"},
        descend_after_hover_cap=True,
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
            settings_overrides={"bias_correction": False, "descent_gate": "warn"},
            descend_after_hover_cap=True,
        ),
    ]
    print("\n" + format_metrics_table(results))
    assert all(r.summary()["n"] == 10 for r in results)


def test_pan_offset_wrapper_bias_is_invisible_to_fk() -> None:
    """A 1.5 deg pan encoder offset shifts the true site 6-7 mm while reported joints agree."""
    import numpy as np

    from robot.cartesian_ik import CartesianIK
    from tests.sim_pick_harness import SimPickHarness

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
