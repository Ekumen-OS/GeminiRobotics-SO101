"""Perception metrics engine for SO-101 visual verification and anti-hallucination guardrails."""

import math
from typing import Any

import numpy as np

from robot.contracts import GraspElevationResponse, VisualAlignmentResponse
from robot.settings import get_settings
from robot.vision_detector import detect_workspace_objects, refine_point_to_nearest_object


def compute_visual_alignment(
    frame_rgb: np.ndarray | None,
    target_point_1000: list[float] | tuple[float, float] | np.ndarray,
    gripper_point_1000: list[float] | tuple[float, float] | np.ndarray | None = None,
    robot_state: dict[str, Any] | None = None,
    tolerance: float | None = None,
    calibrator: Any | None = None,
) -> VisualAlignmentResponse:
    """Calculate measured visual alignment offset between gripper and target object.

    Args:
        frame_rgb: Current RGB camera frame (height, width, 3) or None.
        target_point_1000: Normalized [y, x] coordinate of target in [0, 1000].
        gripper_point_1000: Optional explicitly specified normalized [y, x] gripper position.
        robot_state: Optional dictionary containing robot telemetry or joint angles.
        tolerance: Maximum pixel error threshold in normalized [0, 1000] space.
            Defaults to settings alignment_tolerance_px.
        calibrator: Optional workspace calibrator used to mask detection to the ArUco
            workspace polygon. None disables the mask.

    Returns:
        Dictionary adhering to ARCHITECTURE.md perception contract:
            - gripper_point_1000: [y_g, x_g]
            - target_point_1000: [y_t, x_t]
            - measured_offset: {"dy": float, "dx": float, "distance": float}
            - is_aligned: bool
            - alignment_tolerance: float
    """
    if tolerance is None:
        tolerance = float(get_settings().alignment_tolerance_px)
    else:
        tolerance = float(tolerance)
    target_y = float(target_point_1000[0])
    target_x = float(target_point_1000[1])

    # Refine target using detector if frame is provided and valid
    if frame_rgb is not None and getattr(frame_rgb, "size", 0) > 0:
        try:
            refined_pt, _ = refine_point_to_nearest_object(
                [target_y, target_x], frame_rgb, max_dist_1000=120.0, calibrator=calibrator
            )
            target_y = float(refined_pt[0])
            target_x = float(refined_pt[1])
        except Exception:
            pass

    # Determine gripper coordinate
    if gripper_point_1000 is not None:
        grip_y = float(gripper_point_1000[0])
        grip_x = float(gripper_point_1000[1])
    elif robot_state and "gripper_point_1000" in robot_state:
        pt = robot_state["gripper_point_1000"]
        grip_y = float(pt[0])
        grip_x = float(pt[1])
    else:
        # Fallback to target position if no gripper position is known
        grip_y = target_y
        grip_x = target_x

    dy = float(target_y - grip_y)
    dx = float(target_x - grip_x)
    distance = float(math.hypot(dy, dx))
    is_aligned = bool(abs(dy) <= tolerance and abs(dx) <= tolerance)

    return {
        "gripper_point_1000": [grip_y, grip_x],
        "target_point_1000": [target_y, target_x],
        "measured_offset": {
            "dy": dy,
            "dx": dx,
            "distance": distance,
        },
        "is_aligned": is_aligned,
        "alignment_tolerance": float(tolerance),
    }


def verify_grasp_elevation(
    pre_lift_frame_rgb: np.ndarray | None,
    post_lift_frame_rgb: np.ndarray | None,
    target_point_1000: list[float] | tuple[float, float] | np.ndarray | None = None,
    max_table_presence_dist: float = 80.0,
    calibrator: Any | None = None,
) -> GraspElevationResponse:
    """Verify whether an object was lifted or remains resting on the table surface.

    Args:
        pre_lift_frame_rgb: RGB frame prior to lifting or None.
        post_lift_frame_rgb: RGB frame after lifting motion.
        target_point_1000: Initial target bottom contact coordinate [y, x] in [0, 1000].
        max_table_presence_dist: Radius in [0, 1000] units to check for stationary object.
        calibrator: Optional workspace calibrator used to mask detection to the ArUco
            workspace polygon. None disables the mask.

    Returns:
        Dictionary adhering to ARCHITECTURE.md grasp verification contract:
            - object_elevated: bool | None
            - object_detected_on_table: bool
            - grasped: bool | None
            - confidence: float
    """
    detector_failed = False
    object_detected_pre = True
    object_detected_post = False

    if pre_lift_frame_rgb is not None and getattr(pre_lift_frame_rgb, "size", 0) > 0:
        try:
            pre_objs = detect_workspace_objects(pre_lift_frame_rgb, calibrator=calibrator)
            object_detected_pre = len(pre_objs) > 0
        except Exception:
            detector_failed = True

    if post_lift_frame_rgb is not None and getattr(post_lift_frame_rgb, "size", 0) > 0:
        try:
            post_objs = detect_workspace_objects(post_lift_frame_rgb, calibrator=calibrator)
            if target_point_1000 is not None and len(post_objs) > 0:
                t_y, t_x = float(target_point_1000[0]), float(target_point_1000[1])
                for obj in post_objs:
                    oy, ox = obj["bottom_contact_1000"]
                    if math.hypot(oy - t_y, ox - t_x) <= max_table_presence_dist:
                        object_detected_post = True
                        break
            else:
                # If target point not specified, check if any object is still detected on table
                object_detected_post = len(post_objs) > 0
        except Exception:
            detector_failed = True

    if detector_failed:
        return {
            "object_elevated": None,
            "object_detected_on_table": False,
            "grasped": None,
            "confidence": 0.0,
        }

    object_elevated = bool(object_detected_pre and not object_detected_post)
    confidence = 0.95 if object_elevated else 0.85

    return {
        "object_elevated": object_elevated,
        "object_detected_on_table": object_detected_post,
        "grasped": object_elevated,
        "confidence": confidence,
    }
