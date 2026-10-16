"""Synthetic workspace calibration module for MuJoCo simulation mode.

Computes exact geometric homography mapping camera pixels to robot base coordinates
by projecting 3D workspace corner coordinates through the virtual MuJoCo camera.
"""

import json
import logging
from pathlib import Path

import cv2
import mujoco
import numpy as np

from robot.calibration import ArucoWorkspaceCalibrator
from robot.settings import get_settings

logger = logging.getLogger(__name__)

DEFAULT_SCENE_PATH = Path(__file__).resolve().parent / "assets" / "scene.xml"

# Half a pixel: continuous projection coordinates -> ArUco detector pixel-index coordinates.
PIXEL_CENTRE_OFFSET = 0.5


def project_world_to_pixel(
    point_world: np.ndarray | list[float],
    cam_pos: np.ndarray,
    cam_mat: np.ndarray,
    fovy_deg: float,
    img_shape: tuple[int, int] = (480, 640),
) -> tuple[float, float]:
    """Project a 3D world coordinate to 2D image pixel coordinates through a pinhole camera.

    Args:
        point_world: 3D point in world coordinates [X, Y, Z] (meters).
        cam_pos: Camera position in world coordinates (3,) (meters).
        cam_mat: Camera rotation matrix in world coordinates (3, 3).
        fovy_deg: Vertical field of view in degrees.
        img_shape: Output frame dimensions as (height, width).

    Returns:
        Tuple of (px, py) float pixel coordinates.
    """
    p_w = np.asarray(point_world, dtype=np.float64)
    # Transform point from world to camera coordinate frame
    p_cam = cam_mat.T @ (p_w - cam_pos)
    x_c = p_cam[0]
    y_c = p_cam[1]
    z_c = -p_cam[2]  # MuJoCo camera looks along -Z

    height, width = img_shape
    fovy_rad = np.radians(fovy_deg)
    f_y = (height / 2.0) / np.tan(fovy_rad / 2.0)
    f_x = f_y  # MuJoCo uses square pixels

    c_x = width / 2.0
    c_y = height / 2.0

    px = c_x + (x_c / z_c) * f_x
    py = c_y - (y_c / z_c) * f_y
    return float(px), float(py)


def compute_sim_calibration(
    xml_path: str | Path | None = None,
    model: mujoco.MjModel | None = None,
    data: mujoco.MjData | None = None,
    config_path: str | Path | None = None,
    save_path: str | Path | None = None,
    img_shape: tuple[int, int] = (480, 640),
    camera_name: str = "workspace_cam",
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute synthetic perspective homography matrix for the simulated workspace.

    Args:
        xml_path: Optional path to MuJoCo MJCF scene XML.
        model: Optional preloaded MuJoCo model instance.
        data: Optional preloaded MuJoCo data instance.
        config_path: Optional path to workspace calibration JSON.
        save_path: Optional path to save computed .npz calibration. Defaults to
            config/workspace_homography_sim.npz.
        img_shape: Image resolution as (height, width).
        camera_name: Name of the MuJoCo workspace camera.

    Returns:
        Tuple of (homography_matrix, inverse_homography_matrix, reprojection_error).

    Raises:
        ValueError: If camera is not found or configuration is invalid.
    """
    if model is None or data is None:
        scene_file = Path(xml_path) if xml_path else DEFAULT_SCENE_PATH
        model = mujoco.MjModel.from_xml_path(str(scene_file))
        data = mujoco.MjData(model)

    mujoco.mj_forward(model, data)

    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if cam_id < 0:
        raise ValueError(
            f"Camera '{camera_name}' not found in MuJoCo model. "
            "Ensure scene.xml contains the named camera."
        )

    cam_pos = data.cam_xpos[cam_id].copy()
    cam_mat = data.cam_xmat[cam_id].reshape(3, 3).copy()
    fovy_deg = float(model.cam_fovy[cam_id])

    cfg_file = Path(config_path) if config_path else get_settings().calibration_config_path
    if not cfg_file.exists():
        raise FileNotFoundError(f"Workspace calibration config not found at '{cfg_file}'.")

    config = json.loads(cfg_file.read_text())
    corners_cfg = config.get("workspace_corners_mm", {})

    expected_corners = ["top_left", "top_right", "bottom_right", "bottom_left"]
    src_pixels: list[list[float]] = []
    dst_robot_mm: list[list[float]] = []

    for corner_name in expected_corners:
        if corner_name not in corners_cfg:
            raise ValueError(f"Missing corner '{corner_name}' in workspace calibration config.")
        pt = corners_cfg[corner_name]
        x_mm = float(pt["x"])
        y_mm = float(pt["y"])

        px, py = project_world_to_pixel(
            point_world=[x_mm / 1000.0, y_mm / 1000.0, 0.0],
            cam_pos=cam_pos,
            cam_mat=cam_mat,
            fovy_deg=fovy_deg,
            img_shape=img_shape,
        )
        # OpenCV's ArUco detector reports corners in pixel-index coordinates (a pixel's centre is
        # at an integer), while the projection above is continuous (a pixel spans [i, i+1)). The
        # calibration this replaces is built from detected marker centres, and the jaw tag is
        # detected by the same detector, so use the detector's convention: shift by half a pixel.
        src_pixels.append([px - PIXEL_CENTRE_OFFSET, py - PIXEL_CENTRE_OFFSET])
        dst_robot_mm.append([x_mm, y_mm])

    src_arr = np.array(src_pixels, dtype=np.float32)
    dst_arr = np.array(dst_robot_mm, dtype=np.float32)

    homography_matrix, _ = cv2.findHomography(src_arr, dst_arr)
    if homography_matrix is None:
        homography_matrix = cv2.getPerspectiveTransform(src_arr[:4], dst_arr[:4])

    inverse_homography = np.linalg.inv(homography_matrix)

    # Calculate exact reprojection error
    src_homo = np.hstack([src_arr, np.ones((len(src_arr), 1), dtype=np.float32)])
    proj = (homography_matrix @ src_homo.T).T
    proj_xy = proj[:, :2] / proj[:, 2:3]
    reprojection_error = float(np.mean(np.linalg.norm(proj_xy - dst_arr, axis=1)))

    # Persist to disk
    target_save_path = Path(save_path) if save_path else get_settings().calibration_sim_save_path
    target_save_path.parent.mkdir(parents=True, exist_ok=True)
    hfov_deg = float(
        np.degrees(
            2.0 * np.arctan(np.tan(np.radians(fovy_deg) / 2.0) * img_shape[1] / img_shape[0])
        )
    )
    np.savez(
        target_save_path,
        homography=homography_matrix,
        inverse_homography=inverse_homography,
        reprojection_error=reprojection_error,
        hfov_deg=hfov_deg,
        image_size=np.array([img_shape[1], img_shape[0]]),
    )
    logger.info(
        f"[SimCalibration] Computed synthetic homography (err: {reprojection_error:.6f}mm) "
        f"and saved to {target_save_path}"
    )

    return homography_matrix, inverse_homography, reprojection_error


def init_sim_workspace_calibration(
    calibrator: ArucoWorkspaceCalibrator | None = None,
    config_path: str | Path | None = None,
    save_path: str | Path | None = None,
    xml_path: str | Path | None = None,
    camera_name: str = "workspace_cam",
) -> ArucoWorkspaceCalibrator:
    """Initialize and populate an ArucoWorkspaceCalibrator with synthetic sim calibration.

    Args:
        calibrator: Optional existing ArucoWorkspaceCalibrator instance.
        config_path: Optional calibration JSON config path.
        save_path: Optional target .npz path.
        xml_path: Optional scene XML path.
        camera_name: Name of workspace camera.

    Returns:
        The calibrated ArucoWorkspaceCalibrator instance.
    """
    target_save_path = Path(save_path) if save_path else get_settings().calibration_sim_save_path

    # Compute and save synthetic homography
    h_mat, inv_h, err = compute_sim_calibration(
        xml_path=xml_path,
        config_path=config_path,
        save_path=target_save_path,
        camera_name=camera_name,
    )

    if calibrator is None:
        calibrator = ArucoWorkspaceCalibrator(config_path=config_path)

    loaded = np.load(target_save_path)
    calibrator.set_geometry(
        h_mat,
        inv_h,
        err,
        hfov_deg=float(loaded["hfov_deg"]),
        image_size=(int(loaded["image_size"][0]), int(loaded["image_size"][1])),
        log_camera=True,
    )
    calibrator.mode = "sim"

    return calibrator
