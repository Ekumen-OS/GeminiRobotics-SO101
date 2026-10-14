"""Visual jaw localization: the jaw ArUco tag, parallax removed, in robot millimetres.

The tag sits above the table, so its image position is displaced from the grasp site's table
footprint by parallax that varies with height and position (a constant pixel offset cannot work).
The locator therefore does not compare image points: it takes the tag's pixel, intersects the
viewing ray with the horizontal plane at the tag's height (known from forward kinematics), and
subtracts the tag-to-grasp-site offset rotated by the forward-kinematics orientation. The result
is the grasp-site XY in the same table-plane millimetres as the target.
"""

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from robot.calibration import GeometrySnapshot
from robot.cartesian_ik import CartesianIK
from robot.errors import CalibrationUnavailable
from robot.logger import log_with_tag
from robot.registry import get_active_robot
from robot.settings import (
    get_settings,
    robot_z_to_table_height_mm,
    table_to_robot_xy_mm,
)

logger = logging.getLogger(__name__)

JAW_TAG_ID = 10

_ik_singleton: CartesianIK | None = None


def _get_ik() -> CartesianIK:
    """Return shared CartesianIK instance for FK computation."""
    global _ik_singleton
    if _ik_singleton is None:
        _ik_singleton = CartesianIK()
    return _ik_singleton


@dataclass(frozen=True)
class JawLocation:
    """A jaw position measured from the jaw tag.

    Attributes:
        xy_mm: Grasp-site XY in robot mm (parallax removed).
        tag_pixel: Pixel (u, v) of the tag centre.
        tag_height_mm: Tag-centre height above the table used for the parallax correction.
        site_pixel: Pixel of the site's table-plane footprint (for overlays), or None.
        source: Always 'aruco'.
    """

    xy_mm: tuple[float, float]
    tag_pixel: tuple[float, float]
    tag_height_mm: float
    site_pixel: tuple[float, float] | None
    source: str = "aruco"


def detect_jaw_tag(frame_rgb: np.ndarray | None, tag_id: int = JAW_TAG_ID) -> np.ndarray | None:
    """Detect the jaw ArUco tag and return its four corner pixels.

    Args:
        frame_rgb: RGB image array of shape (H, W, 3), or None.
        tag_id: Marker ID to search for (default 10).

    Returns:
        (4, 2) float array of corner pixels (u, v), or None if the tag is not visible.
    """
    if frame_rgb is None or getattr(frame_rgb, "size", 0) == 0:
        return None

    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    params = cv2.aruco.DetectorParameters()
    params.minMarkerPerimeterRate = 0.01
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        detect_fn = getattr(cv2.aruco, "detectMarkers")
        corners, ids, _ = detect_fn(gray, aruco_dict, parameters=params)

    if ids is not None and len(ids) > 0:
        for i, mid in enumerate(ids.flatten()):
            if int(mid) == tag_id:
                return np.asarray(corners[i], dtype=np.float64).reshape(4, 2)
    return None


def locate_jaw_xy_mm(
    frame_rgb: np.ndarray | None,
    snapshot: GeometrySnapshot | None,
    qpos_rad: np.ndarray,
    tag_offset_mm: tuple[float, float, float] | None = None,
    ik: CartesianIK | None = None,
    mode: str | None = None,
) -> JawLocation | None:
    """Locate the grasp site in robot millimetres from the jaw tag.

    Args:
        frame_rgb: Measurement frame.
        snapshot: Geometry snapshot (homography and camera model) taken for this call.
        qpos_rad: Measured joint angles (radians); they give the tag height and the gripper
            orientation. The shoulder-pan encoder bias the corrector fixes is deliberately *not*
            observable here: the tag pixel is what the camera really saw.
        tag_offset_mm: Tag centre relative to the grasp site in the gripper frame (mm);
            defaults to ``settings.gripper_tag_local_offset_mm``.
        ik: Kinematics solver (shared singleton by default).
        mode: Operational mode ('real' or 'sim') for table height resolution.

    Returns:
        JawLocation, or None if the tag is not visible, the calibration is missing, or there is
        no camera model to remove the parallax with.
    """
    if snapshot is None:
        return None
    corners = detect_jaw_tag(frame_rgb)
    if corners is None:
        return None

    if mode is None:
        active = get_active_robot()
        mode = getattr(active, "mode", "real") if active is not None else "real"

    offset = np.asarray(
        tag_offset_mm if tag_offset_mm is not None else get_settings().gripper_tag_local_offset_mm,
        dtype=np.float64,
    )
    solver = ik if ik is not None else _get_ik()
    site_m, rotation = solver.gripper_frame_pose(np.asarray(qpos_rad, dtype=np.float64))
    site_mm = site_m * 1000.0
    tag_offset_world = rotation @ offset
    tag_z_robot_mm = float(site_mm[2] + tag_offset_world[2])
    tag_height_mm = robot_z_to_table_height_mm(tag_z_robot_mm, mode=mode)

    centre_px = corners.mean(axis=0)
    try:
        tag_xy = snapshot.pixel_to_plane_mm(float(centre_px[0]), float(centre_px[1]), tag_height_mm)
    except CalibrationUnavailable as e:
        log_with_tag(logger, "WARN", f"Jaw locator unavailable: {e}")
        return None
    xy = (float(tag_xy[0] - tag_offset_world[0]), float(tag_xy[1] - tag_offset_world[1]))
    try:
        site_px = snapshot.table_mm_to_pixel(xy[0], xy[1])
    except (ZeroDivisionError, FloatingPointError):
        site_px = None
    robot_xy = table_to_robot_xy_mm(xy[0], xy[1], mode=mode)
    return JawLocation(
        xy_mm=robot_xy,
        tag_pixel=(float(centre_px[0]), float(centre_px[1])),
        tag_height_mm=tag_height_mm,
        site_pixel=site_px,
    )
