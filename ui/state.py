"""Global robot state synchronization manager for REST API and 3D visualizers."""

import struct
import threading
import time
from pathlib import Path

import mujoco
import numpy as np

from robot.constants import JOINT_NAMES, PRESET_POSES
from robot.interface import RobotInterface

_state_lock = threading.Lock()
_active_robot: RobotInterface | None = None
_latest_target_deg: list[float] = [round(float(np.degrees(v)), 1) for v in PRESET_POSES["HOME"]]
_latest_target_rad: list[float] = [float(v) for v in PRESET_POSES["HOME"]]

# Headless MuJoCo kinematics model for computing accurate 3D geom world transforms
_scene_xml_path = Path(__file__).resolve().parent.parent / "simulation" / "assets" / "scene.xml"
_mj_model = mujoco.MjModel.from_xml_path(str(_scene_xml_path))
_mj_data = mujoco.MjData(_mj_model)
_joint_ids = [_mj_model.joint(name).id for name in JOINT_NAMES]

# Pre-identify visual geom indices and mesh filenames
_visual_geom_indices: list[int] = []
_visual_geom_specs: list[dict] = []

for _idx in range(_mj_model.ngeom):
    if _mj_model.geom_group[_idx] == 2:  # Visual group in MuJoCo
        _mesh_id = _mj_model.geom_dataid[_idx]
        _mesh_name = mujoco.mj_id2name(_mj_model, mujoco.mjtObj.mjOBJ_MESH, _mesh_id)
        _visual_geom_indices.append(_idx)
        _visual_geom_specs.append(
            {
                "id": _idx,
                "mesh": f"{_mesh_name}.stl",
                "is_servo": "sts3215" in _mesh_name,
            }
        )


def _ensure_centered_meshes() -> None:
    """Ensure exact MuJoCo-centered STL meshes exist for WebGL rendering."""
    out_dir = Path(__file__).resolve().parent.parent / "simulation" / "assets" / "centered_meshes"
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(_mj_model.nmesh):
        name = mujoco.mj_id2name(_mj_model, mujoco.mjtObj.mjOBJ_MESH, i)
        stl_file = out_dir / f"{name}.stl"
        if stl_file.exists() and stl_file.stat().st_size > 84:
            continue

        vert_adr = _mj_model.mesh_vertadr[i]
        vert_num = _mj_model.mesh_vertnum[i]
        face_adr = _mj_model.mesh_faceadr[i]
        face_num = _mj_model.mesh_facenum[i]

        verts = _mj_model.mesh_vert[vert_adr : vert_adr + vert_num]
        faces = _mj_model.mesh_face[face_adr : face_adr + face_num]

        with open(stl_file, "wb") as f:
            header = f"MuJoCo centered mesh: {name}".encode("ascii")[:80].ljust(80, b"\0")
            f.write(header)
            f.write(struct.pack("<I", face_num))
            for face in faces:
                v1, v2, v3 = verts[face[0]], verts[face[1]], verts[face[2]]
                f.write(struct.pack("<3f", 0.0, 0.0, 0.0))
                f.write(struct.pack("<3f", float(v1[0]), float(v1[1]), float(v1[2])))
                f.write(struct.pack("<3f", float(v2[0]), float(v2[1]), float(v2[2])))
                f.write(struct.pack("<3f", float(v3[0]), float(v3[1]), float(v3[2])))
                f.write(struct.pack("<H", 0))


_ensure_centered_meshes()


def set_active_robot(robot: RobotInterface | None) -> None:
    """Set the globally active robot controller reference.

    Args:
        robot: Initialized robot hardware or simulation controller instance.
    """
    global _active_robot
    with _state_lock:
        _active_robot = robot


def get_active_robot() -> RobotInterface | None:
    """Retrieve the globally active robot controller reference.

    Returns:
        The active RobotInterface instance (hardware or simulation) if set,
        or None if no robot is currently active.
    """
    global _active_robot
    with _state_lock:
        return _active_robot


def set_target_joints_deg(deg: list[float]) -> None:
    """Update target joint positions in degrees and radians.

    Args:
        deg: Target joint positions in degrees (list of 6 floats).
    """
    global _latest_target_deg, _latest_target_rad
    with _state_lock:
        _latest_target_deg = [round(float(v), 1) for v in deg]
        _latest_target_rad = [float(v) for v in np.radians(deg, dtype=np.float32)]


def get_robot_state_dict() -> dict:
    """Retrieve the latest robot state dictionary for REST/JSON clients.

    Returns:
        Dictionary containing target/measured joint angles and exact 3D world transforms
        for every visual mesh geom.
    """
    global _active_robot, _latest_target_deg, _latest_target_rad
    with _state_lock:
        target_deg = list(_latest_target_deg)
        target_rad = list(_latest_target_rad)
        robot = _active_robot

    measured_rad = list(target_rad)
    if robot is not None and hasattr(robot, "get_joint_positions"):
        try:
            curr = robot.get_joint_positions()
            if len(curr) == len(JOINT_NAMES):
                measured_rad = [float(v) for v in curr]
        except Exception:
            pass

    measured_deg = [round(float(np.degrees(v)), 1) for v in measured_rad]

    # Compute exact MuJoCo world transforms for all visual geoms and target object
    geoms: list[dict] = []
    quat_buf = np.zeros(4, dtype=np.float64)

    # Use active robot simulation model & data if available
    # (contains real physical contacts and dynamic target physics)
    if robot is not None and hasattr(robot, "model") and hasattr(robot, "data"):
        m_inst = robot.model
        d_inst = robot.data
        lock_ctx = getattr(robot, "_sim_lock", None)
    else:
        m_inst = _mj_model
        d_inst = _mj_data
        lock_ctx = None

    def _extract_transforms():
        nonlocal geoms, quat_buf
        if lock_ctx is None:
            # Update kinematics on fallback model
            for i, j_id in enumerate(_joint_ids):
                qpos_adr = m_inst.jnt_qposadr[j_id]
                d_inst.qpos[qpos_adr] = measured_rad[i]
            mujoco.mj_forward(m_inst, d_inst)

        for spec, geom_idx in zip(_visual_geom_specs, _visual_geom_indices, strict=False):
            pos = [round(float(v), 6) for v in d_inst.geom_xpos[geom_idx]]
            mujoco.mju_mat2Quat(quat_buf, d_inst.geom_xmat[geom_idx])
            three_quat = [
                round(float(v), 6) for v in [quat_buf[1], quat_buf[2], quat_buf[3], quat_buf[0]]
            ]
            geoms.append(
                {
                    "mesh": spec["mesh"],
                    "pos": pos,
                    "quat": three_quat,
                    "is_servo": spec["is_servo"],
                }
            )

        # Target object body transforms (body 'target')
        try:
            target_body_id = m_inst.body("target").id
            t_pos = [round(float(v), 6) for v in d_inst.xpos[target_body_id]]
            w, x, y, z = d_inst.xquat[target_body_id]
            t_quat = [round(float(v), 6) for v in [x, y, z, w]]
        except Exception:
            t_pos = [round(float(v), 6) for v in d_inst.geom_xpos[-1]]
            t_quat = [0.0, 0.0, 0.0, 1.0]

        return t_pos, t_quat

    with _state_lock:
        if lock_ctx is not None:
            with lock_ctx:
                target_pos, target_quat = _extract_transforms()
        else:
            target_pos, target_quat = _extract_transforms()

    return {
        "target_deg": target_deg,
        "target_rad": target_rad,
        "measured_deg": measured_deg,
        "measured_rad": measured_rad,
        "geoms": geoms,
        "target_pos": target_pos,
        "target_quat": target_quat,
        "joint_names": JOINT_NAMES,
        "timestamp": time.time(),
    }
