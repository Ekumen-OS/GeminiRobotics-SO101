"""MuJoCo ground-truth tests of the simulated gripper, scene and motion (Phase 3e, Step 2)."""

import mujoco
import numpy as np
import pytest

from agent.motion import plan_approach_segments, run_segments
from robot.cartesian_ik import CartesianIK
from robot.errors import HardwareError
from robot.grasp import classify_grasp
from simulation.sim_robot import HOME_CLEARANCE_M, HOME_JAW_XY_M, SO101Robot
from vision.detector import detect_workspace_objects
from vision.gripper_locator import detect_jaw_tag

CUBE_HALF = 0.015
GRIPPER_OPEN_RAD = np.radians(75.0)
GRIPPER_CLOSED_RAD = np.radians(-75.0)


class _Scene:
    """A SO101Robot plus cube access and IK helpers for the physics tests."""

    def __init__(self) -> None:
        self.robot = SO101Robot()
        self.ik = CartesianIK()
        self._adr = self.robot.model.jnt_qposadr[self.robot.model.joint("target_joint").id]

    def cube(self) -> np.ndarray:
        return self.robot.data.qpos[self._adr : self._adr + 3].copy()

    def place(self, x: float, y: float, yaw: float = 0.0) -> None:
        d = self.robot.data
        d.qpos[self._adr : self._adr + 3] = [x, y, CUBE_HALF]
        d.qpos[self._adr + 3 : self._adr + 7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
        d.qvel[:] = 0.0
        mujoco.mj_forward(self.robot.model, d)
        self.robot.step(100)

    def goto(self, xyz: tuple[float, float, float], gripper: float, duration: float = 1.0) -> None:
        q = self.ik.solve_ik(
            np.array(xyz), initial_qpos=self.robot.get_joint_positions(), approach="top_down"
        )
        q[5] = gripper
        self.robot.move_to_pose(q, duration_sec=duration)

    def site(self) -> np.ndarray:
        return self.robot.get_end_effector_pose()[0]

    def close(self) -> None:
        self.robot.close()


@pytest.fixture(scope="module")
def _shared_scene() -> _Scene:
    """One sim scene for the module (model load and renderer setup dominate the test time)."""
    s = _Scene()
    yield s
    s.close()


@pytest.fixture
def scene(_shared_scene: _Scene) -> _Scene:
    """The shared scene reset to HOME with the default cube position."""
    _shared_scene.robot.reset("HOME")
    _shared_scene.robot.interrupt_epoch = 0
    _shared_scene.place(0.24, 0.0)
    return _shared_scene


def _pick(scene: _Scene, x: float, y: float, yaw: float = 0.0) -> tuple[float, float]:
    """Descend onto a cube at (x, y), close, lift to 5 cm; return (cube z, cube-site distance)."""
    scene.robot.reset("HOME")
    scene.place(x, y, yaw)
    scene.robot.move_to_pose(
        np.r_[scene.robot.get_joint_positions()[:5], GRIPPER_OPEN_RAD], duration_sec=0.5
    )
    scene.goto((x, y, 0.08), GRIPPER_OPEN_RAD)
    scene.goto((x, y, 0.05), GRIPPER_OPEN_RAD, 0.6)
    scene.goto((x, y, 0.0), GRIPPER_OPEN_RAD)
    scene.goto((x, y, 0.0), GRIPPER_CLOSED_RAD, 0.5)
    scene.goto((x, y, 0.05), GRIPPER_CLOSED_RAD)
    cube = scene.cube()
    return float(cube[2]), float(np.linalg.norm(cube - scene.site()))


@pytest.mark.slow
def test_sim_true_centre_grasp_lifts_cube(scene: _Scene) -> None:
    """A grasp commanded at the cube's true centre lifts it in at least 9 of 10 placements."""
    rng = np.random.default_rng(0)
    lifted = 0
    for _ in range(10):
        x, y = rng.uniform(0.20, 0.30), rng.uniform(-0.10, 0.10)
        cube_z, dist = _pick(scene, x, y)
        # Held through the 5 cm lift: well above the table and still between the pads.
        lifted += bool(cube_z > 0.03 and dist < 0.03)
    print(f"true-centre grasp lifted {lifted}/10")
    assert lifted >= 9


def test_sim_true_centre_grasp_lifts_cube_smoke(scene: _Scene) -> None:
    """Default-suite guard: one true-centre grasp lifts the default cube and holds it."""
    cube_z, dist = _pick(scene, 0.24, 0.0)
    assert cube_z > 0.03
    assert dist < 0.03


def test_sim_reset_and_first_hover_do_not_disturb_cube(scene: _Scene) -> None:
    """Resetting to HOME and the first hover from HOME move the default cube under 2 mm."""
    robot = scene.robot
    default = scene.cube().copy()
    robot.reset("HOME")
    robot.step(200)
    assert np.linalg.norm(scene.cube() - default) * 1000.0 < 2.0

    target = np.array([default[0], default[1], 0.03])
    segments = plan_approach_segments(scene.ik, robot.get_joint_positions(), target)
    run_segments(robot, segments)
    moved_mm = float(np.linalg.norm(scene.cube() - default) * 1000.0)
    print(f"cube moved {moved_mm:.2f} mm")
    assert moved_mm < 2.0


def test_sim_randomize_target_keeps_clear_of_home_jaw(scene: _Scene) -> None:
    """Randomised cube positions never fall inside the HOME clearance disc."""
    np.random.seed(0)
    for _ in range(200):
        pos = scene.robot.randomize_target()
        assert np.hypot(pos[0] - HOME_JAW_XY_M[0], pos[1] - HOME_JAW_XY_M[1]) >= HOME_CLEARANCE_M


def test_home_jaw_constant_matches_forward_kinematics(scene: _Scene) -> None:
    """HOME_JAW_XY_M is the grasp-site XY at HOME (so the clearance disc is centred right)."""
    scene.robot.reset("HOME")
    site = scene.site()
    assert abs(site[0] - HOME_JAW_XY_M[0]) < 0.005
    assert abs(site[1] - HOME_JAW_XY_M[1]) < 0.005


@pytest.mark.slow
def test_sim_jaw_tag_detected_and_cube_visible(scene: _Scene) -> None:
    """Jaw tag seen in >= 11/12 poses at 640x480; the cube stays >= 70 % visible at a 3 cm hover."""
    robot = scene.robot
    rng = np.random.default_rng(3)
    detected = 0
    for _ in range(12):
        robot.reset("HOME")
        scene.place(0.28, -0.12)
        x, y, z = rng.uniform(0.15, 0.30), rng.uniform(-0.12, 0.12), rng.uniform(0.01, 0.08)
        robot.move_to_pose(
            np.r_[robot.get_joint_positions()[:5], GRIPPER_OPEN_RAD], duration_sec=0.4
        )
        scene.goto((x, y, z), GRIPPER_OPEN_RAD, 1.5)
        detected += detect_jaw_tag(robot.render_rgb_frame(640, 480)) is not None
    print(f"jaw tag detected in {detected}/12 poses")
    assert detected >= 11

    ratios = []
    for x, y in ((0.24, 0.0), (0.20, 0.08), (0.28, -0.08), (0.22, -0.06), (0.26, 0.10)):
        robot.reset("HOME")
        scene.place(x, y)
        alone = _red_area(robot)
        robot.move_to_pose(
            np.r_[robot.get_joint_positions()[:5], GRIPPER_OPEN_RAD], duration_sec=0.4
        )
        scene.goto((x, y, 0.03), GRIPPER_OPEN_RAD, 1.5)
        ratios.append(_red_area(robot) / alone)
    print("cube visible area at 3 cm hover / unoccluded:", [round(r, 2) for r in ratios])
    assert min(ratios) >= 0.70


def _red_area(robot: SO101Robot) -> float:
    objs = [
        o for o in detect_workspace_objects(robot.render_rgb_frame(640, 480)) if o["color"] == "red"
    ]
    return max((o["area"] for o in objs), default=0.0) or 1e-9


def test_sim_floor_has_no_hsv_object_class(scene: _Scene) -> None:
    """The neutral floor is not detected as blue (or any colour) when the arm is at HOME."""
    scene.place(5.0, 5.0)  # cube far off the mat
    objs = detect_workspace_objects(scene.robot.render_rgb_frame(640, 480))
    assert [o for o in objs if o["color"] == "blue"] == []


def test_sim_close_on_air_reads_empty(scene: _Scene) -> None:
    """Closing with nothing between the pads reads the empty-jaw aperture."""
    robot = scene.robot
    scene.place(0.28, -0.12)
    scene.goto((0.22, 0.0, 0.03), GRIPPER_OPEN_RAD)
    scene.goto((0.22, 0.0, 0.03), GRIPPER_CLOSED_RAD, 0.5)
    reading = robot.read_gripper_state()
    grasp = classify_grasp(reading)
    assert grasp["grasped"] is False
    assert reading["aperture_deg"] <= grasp["closed_on_air_threshold_deg"]
    assert reading["aperture_deg"] < -70.0


def test_sim_close_on_cube_reads_stalled(scene: _Scene) -> None:
    """Closing on the cube stalls the jaws well above the empty value (object vs air)."""
    scene.place(0.24, 0.0)
    scene.goto((0.24, 0.0, 0.08), GRIPPER_OPEN_RAD)
    scene.goto((0.24, 0.0, 0.0), GRIPPER_OPEN_RAD)
    scene.goto((0.24, 0.0, 0.0), GRIPPER_CLOSED_RAD, 0.5)
    reading = scene.robot.read_gripper_state()
    grasp = classify_grasp(reading)
    assert grasp["grasped"] is True
    assert reading["aperture_deg"] > -40.0


def test_multi_segment_move_honours_interrupt(scene: _Scene) -> None:
    """An interrupt between two segments cancels the remaining segments."""
    robot = scene.robot
    robot.reset("HOME")
    target = np.array([0.26, 0.10, 0.03])
    segments = plan_approach_segments(scene.ik, robot.get_joint_positions(), target)
    assert len(segments) == 3  # lift, travel, descend (HOME is a low pose, target is far)

    real_move = robot.move_to_pose
    calls = {"n": 0}

    def interrupt_after_first(*args: object, **kwargs: object) -> None:
        real_move(*args, **kwargs)
        calls["n"] += 1
        if calls["n"] == 1:
            robot.interrupt()  # lands after the first segment, before the next call

    robot.move_to_pose = interrupt_after_first  # type: ignore[method-assign]
    with pytest.raises(HardwareError) as exc:
        run_segments(robot, segments)
    assert exc.value.robot_moved is True
    assert calls["n"] == 1  # the second move_to_pose never ran
    # The arm stopped at the end of the vertical lift: raised, not at the target.
    site = scene.site()
    assert site[2] > 0.05
    assert np.hypot(site[0] - target[0], site[1] - target[1]) > 0.05


def test_first_move_from_low_pose_does_not_sweep_below_6cm(scene: _Scene) -> None:
    """HOME -> far hover: vertical lift first, travel at >= 5.5 cm, descend only over the target."""
    robot = scene.robot
    robot.reset("HOME")
    start = scene.site().copy()
    target = np.array([0.26, 0.10, 0.03])
    segments = plan_approach_segments(scene.ik, robot.get_joint_positions(), target)

    segment_no = {"n": 0}
    lift_xy_drift: list[float] = []
    travel_heights: list[float] = []
    real_step, real_move = robot.step, robot.move_to_pose

    def counting_move(*args: object, **kwargs: object) -> None:
        segment_no["n"] += 1
        real_move(*args, **kwargs)

    def spy_step(n: int = 1) -> None:
        real_step(n)
        site = scene.site()
        if segment_no["n"] == 1:
            lift_xy_drift.append(float(np.hypot(site[0] - start[0], site[1] - start[1])))
        elif segment_no["n"] == 2:
            travel_heights.append(float(site[2]))

    robot.step = spy_step  # type: ignore[method-assign]
    robot.move_to_pose = counting_move  # type: ignore[method-assign]
    run_segments(robot, segments)
    robot.step, robot.move_to_pose = real_step, real_move  # type: ignore[method-assign]

    assert max(lift_xy_drift) < 0.02, max(lift_xy_drift)
    assert travel_heights
    # The travel segment may only start below 6 cm by the settling lag of the lift.
    assert min(travel_heights) > 0.052, min(travel_heights)
    final = scene.site()
    assert np.linalg.norm(final - target) < 0.003


def test_move_to_pose_converges_before_returning(scene: _Scene) -> None:
    """After move_to_pose every arm joint is within 0.6 deg of the goal and nearly at rest."""
    robot = scene.robot
    goal = np.radians([20.0, -30.0, 40.0, 30.0, 10.0, 0.0])
    robot.move_to_pose(goal, duration_sec=0.6)
    err = np.degrees(np.abs(robot.get_joint_positions()[:5] - goal[:5]))
    assert float(err.max()) < 0.6
    assert float(np.abs(robot.get_joint_velocities()[:5]).max()) < 0.05


def test_compute_sim_calibration() -> None:
    """Test synthetic homography computation from MuJoCo camera projection."""
    import tempfile
    from pathlib import Path

    from simulation.sim_calibration import compute_sim_calibration, init_sim_workspace_calibration

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = Path(tmp_dir) / "sim_homography.npz"
        h_mat, inv_h, err = compute_sim_calibration(save_path=save_path)

        assert h_mat.shape == (3, 3)
        assert inv_h.shape == (3, 3)
        assert err < 0.01  # Sub-millimeter error for exact geometric projection
        assert save_path.exists()

        calibrator = init_sim_workspace_calibration(save_path=save_path)
        assert calibrator.is_calibrated is True

        # Test center point projection roundtrip
        px, py = calibrator.gemini_to_pixel([500.0, 500.0])
        xyz = calibrator.pixel_to_robot_xyz(px, py, z_m=0.03)
        assert 0.10 <= xyz[0] <= 0.45  # In front of robot base in workspace
        assert -0.30 <= xyz[1] <= 0.30  # Within workspace lateral span
        assert np.isclose(xyz[2], 0.03)


def test_sim_robot_camera_rendering(scene: _Scene) -> None:
    """Test SO101Robot rendering from workspace_cam in MuJoCo simulation."""
    try:
        frame = scene.robot.render_rgb_frame(width=640, height=480, camera="workspace_cam")
    except (mujoco.FatalError, RuntimeError) as err:
        pytest.skip(f"MuJoCo offscreen rendering requires hardware OpenGL/EGL context: {err}")
    assert frame.shape == (480, 640, 3)
    assert frame.dtype == np.uint8
    assert np.mean(frame) > 10.0


def test_sim_robot_randomize_target(scene: _Scene) -> None:
    """Verify target manipulation object randomization bounds in simulation."""
    new_pos = scene.robot.randomize_target(
        x_range=(0.15, 0.22), y_range=(-0.10, 0.10), z_height=0.015
    )
    assert 0.15 <= new_pos[0] <= 0.22
    assert -0.10 <= new_pos[1] <= 0.10
    assert new_pos[2] == 0.015
