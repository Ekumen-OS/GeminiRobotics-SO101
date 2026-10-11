"""Visual gripper localization module.

Locates the robot gripper in image coordinates using fiducial markers (jaw ArUco tag)
with forward kinematics (kinematic projection) fallback.
"""

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from robot.cartesian_ik import CartesianIK
from robot.settings import get_settings

_ik_singleton: CartesianIK | None = None


def _get_ik() -> CartesianIK:
    """Return shared CartesianIK instance for FK computation."""
    global _ik_singleton
    if _ik_singleton is None:
        _ik_singleton = CartesianIK()
    return _ik_singleton


@dataclass
class GripperLocation:
    """Gripper location result with source provenance and confidence.

    Attributes:
        point_1000: [y, x] normalized coordinates in [0, 1000] range.
        source: Provenance source ('aruco', 'kinematic', 'er2').
        confidence: Confidence score in [0.0, 1.0].
    """

    point_1000: list[float]
    source: str
    confidence: float


def locate_gripper_aruco(
    frame_rgb: np.ndarray,
    tag_id: int = 10,
    offset_px: tuple[float, float] = (0.0, 0.0),
) -> GripperLocation | None:
    """Locate the gripper in an RGB frame using jaw ArUco tag.

    Args:
        frame_rgb: RGB image array of shape (H, W, 3).
        tag_id: Marker ID to search for (default 10).
        offset_px: (offset_x, offset_y) pixel shift from marker center to fingertips.

    Returns:
        GripperLocation if marker is detected, else None.
    """
    if frame_rgb is None or getattr(frame_rgb, "size", 0) == 0:
        return None

    h, w = frame_rgb.shape[:2]
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
                marker_corners = corners[i].reshape(4, 2)
                center = np.mean(marker_corners, axis=0)
                px = float(center[0] + offset_px[0])
                py = float(center[1] + offset_px[1])
                y_norm = max(0.0, min(1000.0, (py / h) * 1000.0))
                x_norm = max(0.0, min(1000.0, (px / w) * 1000.0))
                return GripperLocation(
                    point_1000=[float(y_norm), float(x_norm)],
                    source="aruco",
                    confidence=0.95,
                )

    return None


def locate_gripper_kinematic(
    calibrator: Any,
    robot: Any,
    frame_shape: tuple[int, int] = (480, 640),
) -> GripperLocation | None:
    """Locate the gripper using forward kinematics and workspace homography projection.

    Args:
        calibrator: ArucoWorkspaceCalibrator instance with valid inverse homography.
        robot: Robot controller providing get_joint_positions().
        frame_shape: Camera image dimensions as (height, width).

    Returns:
        GripperLocation if FK projection succeeds, else None.
    """
    if robot is None or not hasattr(robot, "get_joint_positions"):
        return None
    if calibrator is None or not getattr(calibrator, "is_calibrated", False):
        return None

    try:
        qpos_rad = robot.get_joint_positions()
        if len(qpos_rad) < 5:
            return None

        ik = _get_ik()
        ee_xyz = ik.forward_kinematics(qpos_rad)
        x_mm = float(ee_xyz[0] * 1000.0)
        y_mm = float(ee_xyz[1] * 1000.0)

        proj = calibrator.robot_mm_to_pixel(x_mm, y_mm)
        if proj is None:
            return None

        px, py = proj
        h, w = frame_shape
        y_norm = max(0.0, min(1000.0, (py / h) * 1000.0))
        x_norm = max(0.0, min(1000.0, (px / w) * 1000.0))

        return GripperLocation(
            point_1000=[float(y_norm), float(x_norm)],
            source="kinematic",
            confidence=0.5,
        )
    except (RuntimeError, ValueError, TypeError, IndexError):
        return None


def locate_gripper_1000(
    frame_rgb: np.ndarray | None,
    calibrator: Any | None = None,
    robot: Any | None = None,
    fallback_to_fk: bool = True,
) -> GripperLocation | None:
    """Locate gripper position [y, x] in normalized 0-1000 image coordinates.

    Tries jaw ArUco tag detection first. If not found and fallback_to_fk is True,
    falls back to forward kinematics projected through workspace homography.

    Args:
        frame_rgb: Optional RGB frame from camera.
        calibrator: Optional ArucoWorkspaceCalibrator instance.
        robot: Optional robot controller.
        fallback_to_fk: Whether to attempt FK fallback if tag is not detected.

    Returns:
        GripperLocation if found, else None.
    """
    settings = get_settings()

    if frame_rgb is not None:
        loc = locate_gripper_aruco(
            frame_rgb,
            tag_id=10,
            offset_px=settings.gripper_tag_offset_px,
        )
        if loc is not None:
            return loc

    if fallback_to_fk and robot is not None and calibrator is not None:
        frame_shape = frame_rgb.shape[:2] if frame_rgb is not None else (480, 640)
        return locate_gripper_kinematic(calibrator, robot, frame_shape=frame_shape)

    return None
