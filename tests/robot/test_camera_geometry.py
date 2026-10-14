"""Camera model, geometry snapshot and parallax-free jaw locator (Phase 3e, Step 6)."""

import threading
from pathlib import Path

import mujoco
import numpy as np
import pytest

from robot.calibration import ArucoWorkspaceCalibrator, GeometrySnapshot
from robot.camera_model import PinholeCamera
from robot.errors import CalibrationUnavailable
from robot.perception import compute_alignment_mm
from robot.settings import Settings
from simulation.sim_calibration import compute_sim_calibration
from simulation.sim_robot import SO101Robot
from tests.conftest import GRIPPER_OPEN_RAD
from vision.gripper_locator import detect_jaw_tag, locate_jaw_xy_mm

SIM_HFOV_DEG = 57.822  # 2 * atan(tan(45 deg / 2) * 640 / 480)


def _mujoco_camera_centre_mm() -> np.ndarray:
    model = mujoco.MjModel.from_xml_path("simulation/assets/scene.xml")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return (
        data.cam_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "workspace_cam")]
        * 1000.0
    )


def _sim_snapshot(tmp_path: Path) -> GeometrySnapshot:
    path = tmp_path / "sim.npz"
    compute_sim_calibration(save_path=path)
    calib = ArucoWorkspaceCalibrator(save_path=path, mode="sim")
    snapshot = calib.geometry_snapshot()
    assert snapshot is not None
    return snapshot


def test_camera_model_sim_matches_mujoco(tmp_path: Path) -> None:
    """The camera derived from the homography sits within 5 mm of the MuJoCo camera (oracle)."""
    snapshot = _sim_snapshot(tmp_path)
    assert snapshot.camera is not None
    error_mm = float(np.linalg.norm(snapshot.camera.centre_mm - _mujoco_camera_centre_mm()))
    print(f"camera centre error vs MuJoCo: {error_mm:.3f} mm; centre {snapshot.camera.centre_mm}")
    assert error_mm <= 5.0
    assert snapshot.hfov_deg == pytest.approx(SIM_HFOV_DEG, abs=0.01)


def test_camera_model_project_and_ray_round_trip(tmp_path: Path) -> None:
    """project() and ray_to_plane() are inverse for points at any height."""
    camera = _sim_snapshot(tmp_path).camera
    assert camera is not None
    for xyz in ([250.0, 0.0, 0.0], [200.0, 80.0, 60.0], [300.0, -90.0, 120.0]):
        u, v = camera.project(xyz)
        x, y = camera.ray_to_plane(u, v, xyz[2])
        assert (x, y) == pytest.approx((xyz[0], xyz[1]), abs=1e-6)


def test_camera_model_robust_to_noise_and_fov_error() -> None:
    """Parallax-corrected XY at 58 mm: p95 <= 5 mm with 1 px noise and +-10 % field of view."""
    import cv2

    rng = np.random.default_rng(0)
    true_hfov = 60.0
    width, height = 640, 480
    focal = (width / 2.0) / np.tan(np.radians(true_hfov) / 2.0)
    K = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1.0]])
    centre = np.array([185.0, 458.0, 591.0])
    forward = np.array([0.186, 0.042, 0.096]) * 1000.0 - centre
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    R = np.vstack([right, down, forward])
    truth = PinholeCamera(K=K, R=R, t=-R @ centre)

    errors: list[float] = []
    for fov_scale in (0.9, 1.0, 1.1):
        # Fit the homography from noisy marker-like observations of table points.
        table = np.array(
            [[x, y] for x in (100.0, 240.0, 380.0) for y in (-190.0, 0.0, 190.0)], dtype=np.float64
        )
        pixels = np.array([truth.project([x, y, 0.0]) for x, y in table])
        noisy = pixels + rng.normal(0.0, 1.0, pixels.shape)
        H, _ = cv2.findHomography(noisy.astype(np.float32), table.astype(np.float32))
        G = np.linalg.inv(H)
        calib = ArucoWorkspaceCalibrator(config_path="config/workspace_calibration.json")
        snapshot = calib.set_geometry(
            H, G, 0.0, hfov_deg=true_hfov * fov_scale, image_size=(width, height)
        )
        for _ in range(200):
            x, y = rng.uniform(120.0, 360.0), rng.uniform(-170.0, 170.0)
            u, v = truth.project([x, y, 58.0])
            ex, ey = snapshot.pixel_to_plane_mm(
                u + rng.normal(0, 0.5), v + rng.normal(0, 0.5), 58.0
            )
            errors.append(float(np.hypot(ex - x, ey - y)))
    p95 = float(np.percentile(errors, 95))
    print(f"parallax-corrected XY error at 58 mm: mean {np.mean(errors):.2f} p95 {p95:.2f} mm")
    assert p95 <= 5.0


def test_snapshot_height_zero_is_exactly_the_homography(tmp_path: Path) -> None:
    """At height 0 the parallax correction is the identity: target and jaw share one geometry."""
    snapshot = _sim_snapshot(tmp_path)
    for u, v in ((100.0, 120.0), (320.0, 240.0), (500.0, 400.0)):
        assert snapshot.pixel_to_plane_mm(u, v, 0.0) == pytest.approx(
            snapshot.pixel_to_table_mm(u, v), abs=1e-9
        )
    assert snapshot.table_mm_to_pixel(*snapshot.pixel_to_table_mm(300.0, 200.0)) == pytest.approx(
        (300.0, 200.0), abs=1e-6
    )


def test_snapshot_without_camera_cannot_correct_parallax(tmp_path: Path) -> None:
    """A raised point with no camera model is refused rather than silently uncorrected."""
    snapshot = _sim_snapshot(tmp_path)
    no_camera = GeometrySnapshot(
        snapshot.homography, snapshot.inverse_homography, 0.0, snapshot.image_size, 70.0, None
    )
    assert no_camera.pixel_to_plane_mm(300.0, 200.0, 0.0) is not None
    with pytest.raises(CalibrationUnavailable):
        no_camera.pixel_to_plane_mm(300.0, 200.0, 60.0)


def test_old_npz_loads_and_derives_camera(old_format_real_npz: Path) -> None:
    """A calibration file without hfov/image keys loads unchanged and still gets a camera."""
    calib = ArucoWorkspaceCalibrator(mode="real")
    assert calib.load_calibration(old_format_real_npz) is True
    snapshot = calib.geometry_snapshot()
    assert snapshot is not None and snapshot.camera is not None
    assert snapshot.hfov_deg == pytest.approx(Settings().camera_hfov_deg)
    assert snapshot.image_size == (640, 480)
    assert snapshot.camera.centre_mm[2] > 0.0
    assert calib.homography_matrix is not None and calib.is_calibrated


def test_camera_centre_is_logged_on_load(
    old_format_real_npz: Path, capsys: pytest.CaptureFixture
) -> None:
    """Loading a calibration logs the derived camera centre for a tape-measure check."""
    calib = ArucoWorkspaceCalibrator(mode="real")
    calib.load_calibration(old_format_real_npz)
    out = capsys.readouterr().out
    assert "[Calibration] [CAMERA] centre_mm=[" in out
    assert "hfov=70.0" in out


def test_snapshot_swap_is_atomic(tmp_path: Path) -> None:
    """Readers never see an H from one update paired with the inverse from another."""
    base = _sim_snapshot(tmp_path)
    H1, G1 = np.array(base.homography), np.array(base.inverse_homography)
    shift = np.array([[1.0, 0, 7.0], [0, 1.0, -3.0], [0, 0, 1.0]])
    H2 = H1 @ shift
    G2 = np.linalg.inv(H2)
    calib = ArucoWorkspaceCalibrator(config_path="config/workspace_calibration.json")
    calib.set_geometry(H1, G1, 0.0, hfov_deg=SIM_HFOV_DEG, image_size=(640, 480))
    stop = threading.Event()
    torn: list[float] = []

    def writer() -> None:
        flip = False
        while not stop.is_set():
            pair = (H2, G2) if flip else (H1, G1)
            calib.set_geometry(pair[0], pair[1], 0.0)
            flip = not flip

    def reader() -> None:
        for _ in range(1500):
            snap = calib.geometry_snapshot()
            assert snap is not None
            residual = float(np.abs(snap.homography @ snap.inverse_homography - np.eye(3)).max())
            if residual > 1e-6:
                torn.append(residual)

    thread = threading.Thread(target=writer)
    thread.start()
    readers = [threading.Thread(target=reader) for _ in range(3)]
    for t in readers:
        t.start()
    for t in readers:
        t.join()
    stop.set()
    thread.join()
    assert torn == []


# -- jaw locator ------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def _shared_sim(tmp_path_factory: pytest.TempPathFactory):
    """One SO101Robot and sim geometry snapshot for the module."""
    robot = SO101Robot(render_viewer=False)
    snapshot = _sim_snapshot(tmp_path_factory.mktemp("sim_geometry"))
    yield robot, snapshot
    robot.close()


@pytest.fixture
def sim_scene(_shared_sim):
    """The shared robot reset to HOME and the sim geometry snapshot."""
    robot, snapshot = _shared_sim
    robot.reset("HOME")
    robot.interrupt_epoch = 0
    return robot, snapshot


def test_jaw_locator_parallax_free_sim(sim_scene) -> None:
    """Jaw XY from the tag is within 3 mm of the MuJoCo site XY at 12 seeded poses (1-8 cm)."""
    from robot.cartesian_ik import CartesianIK

    robot, snapshot = sim_scene
    ik = CartesianIK()
    rng = np.random.default_rng(3)
    errors: list[float] = []
    naive: list[float] = []
    for _ in range(12):
        robot.reset("HOME")
        x, y, z = rng.uniform(0.17, 0.30), rng.uniform(-0.12, 0.12), rng.uniform(0.01, 0.08)
        q = ik.solve_ik(np.array([x, y, z]), initial_qpos=robot.get_joint_positions())
        q[5] = GRIPPER_OPEN_RAD
        robot.move_to_pose(q, duration_sec=1.5)
        frame = robot.render_rgb_frame(640, 480)
        loc = locate_jaw_xy_mm(frame, snapshot, robot.get_joint_positions(), mode="sim")
        assert loc is not None, f"tag not detected at {(x, y, z)}"
        truth = robot.get_end_effector_pose()[0][:2] * 1000.0
        errors.append(float(np.linalg.norm(np.array(loc.xy_mm) - truth)))
        # What the 3d locator effectively did: take the tag pixel as if it sat on the table.
        raw = np.array(snapshot.pixel_to_table_mm(*loc.tag_pixel))
        naive.append(float(np.linalg.norm(raw - truth)))
    print(
        f"jaw XY error: mean {np.mean(errors):.2f} max {max(errors):.2f} mm "
        f"(table-plane shortcut: mean {np.mean(naive):.1f} mm)"
    )
    assert max(errors) <= 3.0
    assert np.mean(naive) > 20.0  # the parallax the old image-space comparison suffered


def test_jaw_locator_sees_a_pan_encoder_bias_that_fk_cannot(sim_scene) -> None:
    """With a 1.5 deg pan offset the tag-based XY follows the real arm; FK follows the belief."""
    from tests.e2e.sim_pick_harness import PanOffsetRobot

    robot, snapshot = sim_scene
    biased = PanOffsetRobot(robot, 1.5)
    from robot.cartesian_ik import CartesianIK

    ik = CartesianIK()
    target = np.array([0.25, 0.0, 0.03])
    q = ik.solve_ik(target, initial_qpos=biased.get_joint_positions())
    q[5] = GRIPPER_OPEN_RAD
    biased.move_to_pose(q, duration_sec=1.5)
    frame = robot.render_rgb_frame(640, 480)
    loc = locate_jaw_xy_mm(frame, snapshot, biased.get_joint_positions(), mode="sim")
    assert loc is not None
    truth = robot.get_end_effector_pose()[0][:2] * 1000.0
    fk = ik.forward_kinematics(biased.get_joint_positions())[:2] * 1000.0
    assert float(np.linalg.norm(np.array(loc.xy_mm) - truth)) < 3.0
    assert float(np.linalg.norm(fk - truth)) > 5.0


def test_jaw_locator_returns_none_without_tag_snapshot_or_camera(sim_scene) -> None:
    """No tag, no calibration or no camera model: None (the caller falls back to FK)."""
    robot, snapshot = sim_scene
    blank = np.full((480, 640, 3), 120, dtype=np.uint8)
    q = robot.get_joint_positions()
    assert locate_jaw_xy_mm(blank, snapshot, q) is None
    assert locate_jaw_xy_mm(robot.render_rgb_frame(640, 480), None, q) is None
    no_camera = GeometrySnapshot(
        snapshot.homography, snapshot.inverse_homography, 0.0, snapshot.image_size, 70.0, None
    )
    robot.move_to_pose(np.r_[q[:5], GRIPPER_OPEN_RAD], duration_sec=0.2)
    frame = robot.render_rgb_frame(640, 480)
    if detect_jaw_tag(frame) is not None:
        assert locate_jaw_xy_mm(frame, no_camera, robot.get_joint_positions()) is None


# -- alignment payload ------------------------------------------------------------------------
def test_alignment_payload_is_in_mm_with_euclidean_tolerance() -> None:
    """is_aligned compares the Euclidean mm distance with ALIGNMENT_TOLERANCE_MM."""
    within = compute_alignment_mm([240.0, 0.0], [243.0, 0.0], "aruco", tolerance_mm=4.0)
    assert within["is_aligned"] is True
    assert within["measured_offset"] == {"dx_mm": -3.0, "dy_mm": 0.0, "distance_mm": 3.0}
    outside = compute_alignment_mm([240.0, 0.0], [243.0, 3.5], "aruco", tolerance_mm=4.0)
    assert outside["is_aligned"] is False
    assert outside["alignment_tolerance_mm"] == 4.0


def test_alignment_is_unobservable_for_fk_and_missing_jaw() -> None:
    """FK cannot see calibration or encoder bias; an unlocated jaw has no alignment either."""
    fk = compute_alignment_mm([240.0, 0.0], [240.0, 0.0], "kinematic")
    assert fk["is_aligned"] is None
    assert fk["measured_offset"] is not None
    none = compute_alignment_mm([240.0, 0.0], None, "none")
    assert none["is_aligned"] is None and none["measured_offset"] is None
    assert none["gripper_source"] == "none"


def test_new_settings_defaults_and_env_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    """CAMERA_HFOV_DEG, ALIGNMENT_TOLERANCE_MM, GRIPPER_TAG_LOCAL_OFFSET_MM, BIAS_CORRECTION."""
    for name in (
        "CAMERA_HFOV_DEG",
        "ALIGNMENT_TOLERANCE_MM",
        "GRIPPER_TAG_LOCAL_OFFSET_MM",
        "BIAS_CORRECTION",
        "ER2_LOCATOR_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)
    defaults = Settings.from_env()
    assert defaults.camera_hfov_deg == 70.0
    assert defaults.alignment_tolerance_mm == 4.0
    assert defaults.gripper_tag_local_offset_mm == (-14.0, 27.0, 63.0)
    assert defaults.bias_correction is True
    assert defaults.er2_locator_model == "gemini-robotics-er-2-preview"

    monkeypatch.setenv("CAMERA_HFOV_DEG", "62.5")
    monkeypatch.setenv("ALIGNMENT_TOLERANCE_MM", "2.5")
    monkeypatch.setenv("GRIPPER_TAG_LOCAL_OFFSET_MM", "1, 2, 3")
    monkeypatch.setenv("BIAS_CORRECTION", "0")
    monkeypatch.setenv("ER2_LOCATOR_MODEL", "my-er2")
    custom = Settings.from_env()
    assert custom.camera_hfov_deg == 62.5
    assert custom.alignment_tolerance_mm == 2.5
    assert custom.gripper_tag_local_offset_mm == (1.0, 2.0, 3.0)
    assert custom.bias_correction is False
    assert custom.er2_locator_model == "my-er2"
    assert not hasattr(custom, "gripper_tag_offset_px")


def _fixed_finger_mesh_face_x_mm(model: mujoco.MjModel) -> float:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper")
    mesh_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MESH, "wrist_roll_follower_so101_v1")
    v_adr = model.mesh_vertadr[mesh_id]
    v_num = model.mesh_vertnum[mesh_id]
    raw_verts = model.mesh_vert[v_adr : v_adr + v_num].copy()

    geom_id = -1
    for g in range(model.ngeom):
        if model.geom_bodyid[g] == body_id and model.geom_dataid[g] == mesh_id:
            geom_id = g
            break
    assert geom_id >= 0, "wrist_roll_follower_so101_v1 geom not found on gripper"

    g_pos = model.geom_pos[geom_id]
    g_quat = model.geom_quat[geom_id]
    rot = np.zeros((3, 3))
    mujoco.mju_quat2Mat(rot.ravel(), g_quat)
    body_verts = (rot @ raw_verts.T).T + g_pos

    fingertip_mask = (body_verts[:, 2] < -0.075) & (body_verts[:, 2] > -0.103)
    ft_verts = body_verts[fingertip_mask]
    return float(np.max(ft_verts[:, 0]) * 1000.0)


def test_tcp_clears_real_fixed_finger() -> None:
    """A 30 mm cube centred on TCP has >= 4.5 mm clearance to the fixed finger mesh face."""
    model = mujoco.MjModel.from_xml_path("simulation/assets/scene.xml")
    mesh_face_x_mm = _fixed_finger_mesh_face_x_mm(model)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
    tcp_x_mm = float(model.site_pos[site_id][0] * 1000.0)
    cube_near_face_x_mm = tcp_x_mm - 15.0
    clearance_mm = cube_near_face_x_mm - mesh_face_x_mm
    print(f"TCP clearance to fixed finger: {clearance_mm:.2f} mm")
    assert clearance_mm >= 4.5


def test_fixed_pad_matches_finger_mesh() -> None:
    """The inner face of pad_fixed is within 0.5 mm of the finger mesh face."""
    model = mujoco.MjModel.from_xml_path("simulation/assets/scene.xml")
    mesh_face_x_mm = _fixed_finger_mesh_face_x_mm(model)
    pad_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "pad_fixed")
    pad_inner_face_x_mm = float((model.geom_pos[pad_id][0] + model.geom_size[pad_id][0]) * 1000.0)
    diff_mm = abs(pad_inner_face_x_mm - mesh_face_x_mm)
    print(
        f"pad_fixed inner face vs mesh: pad={pad_inner_face_x_mm:.2f} mm, "
        f"mesh={mesh_face_x_mm:.2f} mm, diff={diff_mm:.2f} mm"
    )
    assert diff_mm <= 0.5


def test_tag_offset_default_matches_model() -> None:
    """Settings default equals tag position minus grasp site position, within 0.5 mm."""
    model = mujoco.MjModel.from_xml_path("simulation/assets/scene.xml")
    tag_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "aruco_jaw_tag")
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
    tag_pos_mm = model.geom_pos[tag_id] * 1000.0
    site_pos_mm = model.site_pos[site_id] * 1000.0
    derived_offset = tag_pos_mm - site_pos_mm
    default_offset = np.array(Settings().gripper_tag_local_offset_mm)
    diff = np.abs(default_offset - derived_offset)
    print(f"Derived tag offset: {derived_offset}, default: {default_offset}, diff: {diff}")
    assert np.all(diff <= 0.5)
