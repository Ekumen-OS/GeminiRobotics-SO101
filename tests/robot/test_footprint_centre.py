"""Footprint-centre target estimation against MuJoCo truth (Phase 3e, Step 3)."""

import numpy as np
import pytest

from simulation.sim_calibration import init_sim_workspace_calibration
from tests.simulation.test_sim_physics import _Scene
from vision.calibrator import get_calibrator
from vision.detector import (
    color_from_label,
    detect_workspace_objects,
    locate_target_footprint,
    refine_point_to_nearest_object,
)


@pytest.fixture(scope="module")
def _shared_scene() -> _Scene:
    """One sim scene for the module with the sim calibration loaded on the shared calibrator."""
    init_sim_workspace_calibration(calibrator=get_calibrator(mode="sim"))
    scene = _Scene()
    yield scene
    scene.close()


@pytest.fixture
def sim_scene(_shared_scene: _Scene) -> _Scene:
    """The shared scene reset to HOME."""
    init_sim_workspace_calibration(calibrator=get_calibrator(mode="sim"))
    _shared_scene.robot.reset("HOME")
    return _shared_scene


def _footprint_errors_mm(scene: _Scene, yaw_range: float, seed: int) -> list[float]:
    homography = get_calibrator(mode="sim").homography_matrix
    rng = np.random.default_rng(seed)
    errors = []
    for _ in range(10):
        x, y = rng.uniform(0.17, 0.32), rng.uniform(-0.12, 0.12)
        scene.place(x, y, yaw=float(rng.uniform(-yaw_range, yaw_range)))
        objs = [
            o
            for o in detect_workspace_objects(scene.robot.render_rgb_frame(640, 480))
            if o["color"] == "red"
        ]
        assert objs
        fp = locate_target_footprint(objs[0], homography)
        truth = scene.cube()[:2] * 1000.0
        errors.append(float(np.linalg.norm(np.array(fp["centre_xy_mm"]) - truth)))
    return errors


def test_footprint_centre_within_5mm_sim(sim_scene: _Scene) -> None:
    """At yaw 0 the footprint centre is within 5 mm of the cube's true centre (10 placements)."""
    errors = _footprint_errors_mm(sim_scene, yaw_range=0.0, seed=0)
    print(f"footprint centre error yaw 0: mean {np.mean(errors):.2f} mm max {max(errors):.2f} mm")
    assert max(errors) <= 5.0


def test_footprint_centre_error_at_random_yaw_is_reported(sim_scene: _Scene) -> None:
    """Report (no threshold) the footprint error for randomly rotated cubes."""
    errors = _footprint_errors_mm(sim_scene, yaw_range=np.pi / 4, seed=1)
    print(
        f"footprint centre error random yaw: mean {np.mean(errors):.2f} mm max {max(errors):.2f} mm"
    )
    assert len(errors) == 10


def test_footprint_centre_random_yaw_within_4mm_sim(sim_scene: _Scene) -> None:
    """20 seeded placements, yaw uniform in [0, 90 deg); max error <= 4 mm."""
    homography = get_calibrator(mode="sim").homography_matrix
    rng = np.random.default_rng(42)
    errors: list[float] = []
    bucket_errors: dict[str, list[float]] = {
        "0-15": [],
        "15-30": [],
        "30-45": [],
        "45-60": [],
        "60-75": [],
        "75-90": [],
    }
    for _ in range(20):
        x, y = float(rng.uniform(0.17, 0.32)), float(rng.uniform(-0.12, 0.12))
        yaw = float(rng.uniform(0.0, np.pi / 2.0))
        sim_scene.place(x, y, yaw=yaw)
        objs = [
            o
            for o in detect_workspace_objects(sim_scene.robot.render_rgb_frame(640, 480))
            if o["color"] == "red"
        ]
        assert objs
        fp = locate_target_footprint(objs[0], homography)
        truth = sim_scene.cube()[:2] * 1000.0
        err = float(np.linalg.norm(np.array(fp["centre_xy_mm"]) - truth))
        errors.append(err)
        yaw_deg = float(np.degrees(yaw))
        if yaw_deg < 15.0:
            bucket_errors["0-15"].append(err)
        elif yaw_deg < 30.0:
            bucket_errors["15-30"].append(err)
        elif yaw_deg < 45.0:
            bucket_errors["30-45"].append(err)
        elif yaw_deg < 60.0:
            bucket_errors["45-60"].append(err)
        elif yaw_deg < 75.0:
            bucket_errors["60-75"].append(err)
        else:
            bucket_errors["75-90"].append(err)

    print(f"20 random yaw: mean {np.mean(errors):.2f} mm, max {max(errors):.2f} mm")
    for b_name, b_errs in bucket_errors.items():
        if b_errs:
            print(
                f"  Bucket {b_name} deg (n={len(b_errs)}): "
                f"mean {np.mean(b_errs):.2f} mm, max {max(b_errs):.2f} mm"
            )
    assert max(errors) <= 4.0


def test_footprint_beats_bottom_contact_by_a_wide_margin(sim_scene: _Scene) -> None:
    """The near-edge target is ~16 mm off; the footprint centre removes that bias."""
    calib = get_calibrator(mode="sim")
    sim_scene.place(0.24, 0.0)
    objs = [
        o
        for o in detect_workspace_objects(sim_scene.robot.render_rgb_frame(640, 480))
        if o["color"] == "red"
    ]
    by, bx = objs[0]["bottom_contact_1000"]
    edge_mm = calib.pixel_to_robot_mm(bx / 1000.0 * 640.0, by / 1000.0 * 480.0)
    truth = sim_scene.cube()[:2] * 1000.0
    edge_err = float(np.linalg.norm(edge_mm - truth))
    fp = locate_target_footprint(objs[0], calib.homography_matrix)
    fp_err = float(np.linalg.norm(np.array(fp["centre_xy_mm"]) - truth))
    assert edge_err > 10.0
    assert fp_err < 5.0


def test_refine_returns_footprint_centre_and_keeps_bottom_contact(sim_scene: _Scene) -> None:
    """refine_point_to_nearest_object returns the footprint centre; bottom_contact_1000 remains."""
    calib = get_calibrator(mode="sim")
    sim_scene.place(0.24, 0.02)
    frame = sim_scene.robot.render_rgb_frame(640, 480)
    objs = [o for o in detect_workspace_objects(frame) if o["color"] == "red"]
    point = objs[0]["center_1000"]
    refined, obj = refine_point_to_nearest_object(point, frame, calibrator=calib)
    assert obj is not None
    assert "bottom_contact_1000" in obj
    assert refined == obj["centre_1000"]
    truth = sim_scene.cube()[:2] * 1000.0
    assert float(np.linalg.norm(np.array(obj["centre_xy_mm"]) - truth)) < 5.0


def test_refine_does_not_snap_beyond_default_radius(sim_scene: _Scene) -> None:
    """A point far from every object stays where it is (no 450-unit snap, no single-object snap)."""
    calib = get_calibrator(mode="sim")
    sim_scene.place(0.24, 0.0)
    frame = sim_scene.robot.render_rgb_frame(640, 480)
    refined, obj = refine_point_to_nearest_object([100.0, 100.0], frame, calibrator=calib)
    assert obj is None
    assert refined == [100.0, 100.0]


def test_refine_colour_filter_only_matches_that_colour(sim_scene: _Scene) -> None:
    """With target_color='blue' the red cube is never matched, even when it is the nearest."""
    calib = get_calibrator(mode="sim")
    sim_scene.place(0.24, 0.0)
    frame = sim_scene.robot.render_rgb_frame(640, 480)
    objs = [o for o in detect_workspace_objects(frame) if o["color"] == "red"]
    point = objs[0]["center_1000"]
    _, obj_blue = refine_point_to_nearest_object(
        point, frame, target_color="blue", calibrator=calib
    )
    _, obj_red = refine_point_to_nearest_object(point, frame, target_color="red", calibrator=calib)
    assert obj_blue is None
    assert obj_red is not None


def test_color_from_label_words() -> None:
    """Colour words are extracted from free-text labels."""
    assert color_from_label("red cube") == "red"
    assert color_from_label("The BLUE block") == "blue"
    assert color_from_label("cube") is None
    assert color_from_label(None) is None
    assert color_from_label("") is None


def test_locate_target_footprint_rejects_degenerate_geometry() -> None:
    """A homography that collapses the bbox raises ValueError instead of returning garbage."""
    obj = {"bbox_px": [100, 100, 40, 40]}
    with pytest.raises(ValueError):
        locate_target_footprint(obj, np.zeros((3, 3)))
