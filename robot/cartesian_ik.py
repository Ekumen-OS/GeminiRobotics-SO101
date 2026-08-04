"""Cartesian Inverse Kinematics (IK) solver for the SO-101 robot arm."""

from pathlib import Path

import mujoco
import numpy as np

from robot.constants import JOINT_NAMES


class CartesianIK:
    """Numerical Inverse Kinematics solver using MuJoCo Jacobian damped least-squares.

    Attributes:
        model: MuJoCo MjModel instance.
        data: MuJoCo MjData instance.
        site_id: End effector site ID.
        joint_ids: List of joint IDs.
    """

    def __init__(self, xml_path: str | Path | None = None):
        """Initialize Cartesian IK solver for SO-101 model.

        Args:
            xml_path: Path to the MJCF scene XML file. Defaults to simulation assets/scene.xml.
        """
        if xml_path is None:
            xml_path = Path(__file__).parent.parent / "simulation" / "assets" / "scene.xml"

        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)

        self.all_joint_ids = [self.model.joint(name).id for name in JOINT_NAMES]

        # SO-101 has 5 active spatial DOF joints (gripper jaw hinge does not move the site frame)
        arm_joint_names = [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
        ]
        self.joint_ids = [self.model.joint(name).id for name in arm_joint_names]

        try:
            self.site_id = self.model.site("gripperframe").id
        except KeyError:
            self.site_id = self.model.site("end_effector").id

    def solve_ik(
        self,
        target_pos: np.ndarray,
        target_mat: np.ndarray | None = None,
        initial_qpos: np.ndarray | None = None,
        nominal_qpos: np.ndarray | None = None,
        preserve_orientation: bool = False,
        max_iters: int = 200,
        tol: float = 1e-4,
        damping: float = 1e-3,
        null_weight: float = 5e-2,
    ) -> np.ndarray:
        """Solve numerical inverse kinematics for target 3D end-effector pose.

        Supports position and optional orientation tracking with Levenberg-Marquardt
        damped least-squares and null-space joint posture stabilization for 5-DOF arms.

        Args:
            target_pos: Target (x, y, z) 3D coordinate array in meters.
            target_mat: Optional target 3x3 orientation rotation matrix.
            initial_qpos: Starting joint angles configuration array (length 5 or 6).
            nominal_qpos: Preferred rest posture for null-space regularization.
            preserve_orientation: If True and target_mat is None, retain starting orientation.
            max_iters: Maximum IK optimization iterations.
            tol: Target distance and rotational error threshold in meters/radians.
            damping: Damped least-squares regularization factor (lambda).
            null_weight: Gain coefficient for null-space posture projection.

        Returns:
            A numpy array of 6 joint angles in radians (including gripper angle).
        """
        if initial_qpos is not None:
            target_ids = self.all_joint_ids if len(initial_qpos) >= 6 else self.joint_ids
            for i, j_id in enumerate(target_ids):
                qpos_adr = self.model.jnt_qposadr[j_id]
                self.data.qpos[qpos_adr] = initial_qpos[i]

        mujoco.mj_forward(self.model, self.data)

        if preserve_orientation and target_mat is None:
            target_mat = self.data.site_xmat[self.site_id].reshape(3, 3).copy()

        if nominal_qpos is None and initial_qpos is not None:
            nominal_qpos = initial_qpos[: len(self.joint_ids)].copy()
        elif nominal_qpos is not None:
            nominal_qpos = nominal_qpos[: len(self.joint_ids)].copy()
        else:
            nominal_qpos = np.array(
                [self.data.qpos[self.model.jnt_qposadr[j]] for j in self.joint_ids]
            )

        # Allocate Jacobian buffers (3 x nv for site position and rotation)
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))

        dof_indices = [self.model.jnt_dofadr[j_id] for j_id in self.joint_ids]
        n_dofs = len(dof_indices)
        I_dof = np.eye(n_dofs)

        for _ in range(max_iters):
            current_pos = self.data.site_xpos[self.site_id].copy()
            err_pos = target_pos - current_pos

            if target_mat is not None:
                current_mat = self.data.site_xmat[self.site_id].reshape(3, 3)
                err_rot = 0.5 * (
                    np.cross(current_mat[:, 0], target_mat[:, 0])
                    + np.cross(current_mat[:, 1], target_mat[:, 1])
                    + np.cross(current_mat[:, 2], target_mat[:, 2])
                )
                err = np.concatenate([err_pos, err_rot])
                mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.site_id)
                J_pos = jacp[:, dof_indices]
                J_rot = jacr[:, dof_indices]
                J = np.vstack([J_pos, J_rot])
            else:
                err = err_pos
                mujoco.mj_jacSite(self.model, self.data, jacp, None, self.site_id)
                J = jacp[:, dof_indices]

            if np.linalg.norm(err) < tol:
                break

            m_rows = J.shape[0]
            JJt_dmp = J @ J.T + (damping**2) * np.eye(m_rows)
            J_pinv = J.T @ np.linalg.solve(JJt_dmp, np.eye(m_rows))

            # Primary task: task-space error tracking
            dq_primary = J_pinv @ err

            # Secondary task: null-space posture centering toward nominal_qpos
            curr_qpos = np.array(
                [self.data.qpos[self.model.jnt_qposadr[j_id]] for j_id in self.joint_ids]
            )
            null_projector = I_dof - J_pinv @ J
            dq_null = null_projector @ (null_weight * (nominal_qpos - curr_qpos))

            dq = dq_primary + dq_null

            # Update qpos and enforce joint limits
            for i, j_id in enumerate(self.joint_ids):
                qpos_adr = self.model.jnt_qposadr[j_id]
                self.data.qpos[qpos_adr] += dq[i]

                jnt_range = self.model.jnt_range[j_id]
                self.data.qpos[qpos_adr] = np.clip(
                    self.data.qpos[qpos_adr], jnt_range[0], jnt_range[1]
                )

            mujoco.mj_forward(self.model, self.data)

        qpos_res = []
        for j_id in self.all_joint_ids:
            qpos_adr = self.model.jnt_qposadr[j_id]
            qpos_res.append(self.data.qpos[qpos_adr])

        return np.array(qpos_res, dtype=np.float32)
