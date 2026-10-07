"""OpenCV-assisted multi-color salient object detector for table workspace."""

from typing import Any

import cv2
import numpy as np

# HSV color range definitions (OpenCV H: 0..180, S: 0..255, V: 0..255)
COLOR_HSV_RANGES: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
    "red": [
        (np.array([0, 70, 70], dtype=np.uint8), np.array([12, 255, 255], dtype=np.uint8)),
        (np.array([168, 70, 70], dtype=np.uint8), np.array([180, 255, 255], dtype=np.uint8)),
    ],
    "blue": [
        (np.array([95, 80, 60], dtype=np.uint8), np.array([135, 255, 255], dtype=np.uint8)),
    ],
    "green": [
        (np.array([35, 70, 60], dtype=np.uint8), np.array([85, 255, 255], dtype=np.uint8)),
    ],
    "yellow": [
        (np.array([18, 90, 80], dtype=np.uint8), np.array([34, 255, 255], dtype=np.uint8)),
    ],
}

DEFAULT_MIN_OBJECT_AREA_PX = 100
MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))


def get_workspace_polygon_from_calibrator(
    frame_rgb: np.ndarray,
    calibrator: Any | None = None,
) -> np.ndarray | None:
    """Extract workspace polygon from detected ArUco markers or saved calibration.

    Args:
        frame_rgb: Input image in RGB format (H, W, 3).
        calibrator: Optional ArucoWorkspaceCalibrator instance.

    Returns:
        NumPy array of shape (4, 2) with corner coordinates, or None.
    """
    if calibrator is None:
        return None
    try:
        detected = calibrator.detect_markers(frame_rgb)
        marker_ids = calibrator.config.get("marker_ids", {})
        expected_ids = [
            int(marker_ids.get("top_left", 1)),
            int(marker_ids.get("top_right", 2)),
            int(marker_ids.get("bottom_right", 3)),
            int(marker_ids.get("bottom_left", 0)),
        ]
        pts = []
        for mid in expected_ids:
            if mid in detected:
                pts.append(np.mean(detected[mid], axis=0))
        if len(pts) == 4:
            return np.array(pts, dtype=np.float32)
    except (KeyError, ValueError, TypeError, AttributeError, cv2.error):
        pass
    return None


def detect_workspace_objects(
    frame_rgb: np.ndarray,
    workspace_polygon_px: np.ndarray | None = None,
    min_area: int = DEFAULT_MIN_OBJECT_AREA_PX,
    calibrator: Any | None = None,
) -> list[dict[str, Any]]:
    """Detect salient colored objects on the tabletop workspace.

    Args:
        frame_rgb: Input image in RGB format (H, W, 3).
        workspace_polygon_px: Optional (N, 2) array of polygon vertex pixels
            defining the workspace boundary. If None, tries to detect ArUco boundaries.
        min_area: Minimum contour pixel area to consider as an object.
        calibrator: Optional workspace calibrator instance to derive the boundary polygon.

    Returns:
        List of detected object dictionaries containing color, bounding box,
        bottom contact pixel coordinate, and normalized [y, x] in [0, 1000].
    """
    if frame_rgb is None or frame_rgb.size == 0:
        return []

    h, w = frame_rgb.shape[:2]
    hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)

    if workspace_polygon_px is None and calibrator is not None:
        workspace_polygon_px = get_workspace_polygon_from_calibrator(
            frame_rgb, calibrator=calibrator
        )

    # Workspace polygon mask (optional)
    mask_roi: np.ndarray | None = None
    if workspace_polygon_px is not None and len(workspace_polygon_px) >= 3:
        mask_roi = np.zeros((h, w), dtype=np.uint8)
        pts = np.int32([workspace_polygon_px])
        cv2.fillPoly(mask_roi, pts, 255)

    detected: list[dict[str, Any]] = []

    for color_name, ranges in COLOR_HSV_RANGES.items():
        color_mask = np.zeros((h, w), dtype=np.uint8)
        for lower, upper in ranges:
            part = cv2.inRange(hsv, lower, upper)
            color_mask = cv2.bitwise_or(color_mask, part)

        if mask_roi is not None:
            color_mask = cv2.bitwise_and(color_mask, mask_roi)

        # Clean noise
        opened = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, MORPH_KERNEL)
        cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, MORPH_KERNEL)

        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area < min_area:
                continue

            bx, by, bw, bh = cv2.boundingRect(cnt)
            # Bottom contact center (where the object rests on the table surface)
            bot_x = float(bx + (bw / 2.0))
            bot_y = float(by + bh)

            # Centroid
            m = cv2.moments(cnt)
            cx = float(m["m10"] / m["m00"]) if m["m00"] != 0 else float(bx + bw / 2.0)
            cy = float(m["m01"] / m["m00"]) if m["m00"] != 0 else float(by + bh / 2.0)

            norm_bot_y = float(np.clip((bot_y / h) * 1000.0, 0.0, 1000.0))
            norm_bot_x = float(np.clip((bot_x / w) * 1000.0, 0.0, 1000.0))
            norm_cx = float(np.clip((cx / w) * 1000.0, 0.0, 1000.0))
            norm_cy = float(np.clip((cy / h) * 1000.0, 0.0, 1000.0))

            detected.append(
                {
                    "color": color_name,
                    "area": area,
                    "bbox_px": [int(bx), int(by), int(bw), int(bh)],
                    "bottom_contact_px": [bot_x, bot_y],
                    "bottom_contact_1000": [norm_bot_y, norm_bot_x],
                    "center_1000": [norm_cy, norm_cx],
                }
            )

    # Sort objects by descending area
    detected.sort(key=lambda item: item["area"], reverse=True)
    return detected


def refine_point_to_nearest_object(
    point_1000: list[float] | tuple[float, float] | np.ndarray,
    frame_rgb: np.ndarray,
    max_dist_1000: float = 450.0,
    target_color: str | None = None,
    workspace_polygon_px: np.ndarray | None = None,
    min_area: int = DEFAULT_MIN_OBJECT_AREA_PX,
    calibrator: Any | None = None,
) -> tuple[list[float], dict[str, Any] | None]:
    """Refine a candidate VLA target point to the nearest detected tabletop object.

    Args:
        point_1000: Normalized [y, x] coordinate in [0, 1000].
        frame_rgb: RGB frame corresponding to the scene.
        max_dist_1000: Maximum allowed Euclidean distance in normalized coordinate
            space to accept a snap refinement.
        target_color: Optional preferred color name (red, blue, green, yellow).
        workspace_polygon_px: Optional polygon vertices in pixels.
        min_area: Minimum contour pixel area to consider.
        calibrator: Optional ArucoWorkspaceCalibrator instance.

    Returns:
        Tuple of (refined_point_1000, detected_object_dict_or_None).
        If no candidate is found within max_dist_1000, returns original point and None.
    """
    if len(point_1000) != 2 or frame_rgb is None or frame_rgb.size == 0:
        return [float(point_1000[0]), float(point_1000[1])], None

    objects = detect_workspace_objects(
        frame_rgb,
        workspace_polygon_px=workspace_polygon_px,
        min_area=min_area,
        calibrator=calibrator,
    )
    if not objects:
        return [float(point_1000[0]), float(point_1000[1])], None

    py, px = float(point_1000[0]), float(point_1000[1])

    best_obj: dict[str, Any] | None = None
    best_dist = float("inf")

    # If target_color specified, prioritize matching color
    candidate_objects = objects
    if target_color:
        matching = [obj for obj in objects if obj["color"].lower() == target_color.lower()]
        if matching:
            candidate_objects = matching

    for obj in candidate_objects:
        obj_y, obj_x = obj["bottom_contact_1000"]
        dist = float(np.hypot(obj_y - py, obj_x - px))
        if dist < best_dist and dist <= max_dist_1000:
            best_dist = dist
            best_obj = obj

    if best_obj is not None:
        return list(best_obj["bottom_contact_1000"]), best_obj

    # If no object within max_dist_1000, fallback to single object if exactly 1 exists
    if len(candidate_objects) == 1:
        single_obj = candidate_objects[0]
        obj_y, obj_x = single_obj["bottom_contact_1000"]
        dist = float(np.hypot(obj_y - py, obj_x - px))
        if dist <= 600.0:
            return list(single_obj["bottom_contact_1000"]), single_obj

    return [py, px], None
