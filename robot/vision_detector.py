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
# Continuous blob coordinates -> pixel-index coordinates of the ArUco-based homography.
PIXEL_INDEX_SHIFT = 0.5
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

            # Extent of the blob's lowest pixel rows: the object's near bottom edge. The bbox
            # centre is not the bottom-edge centre (vertical edges lean in perspective).
            pts = cnt.reshape(-1, 2)
            low = pts[pts[:, 1] >= by + bh - 3]
            bottom_edge = [float(low[:, 0].min()), float(low[:, 0].max() + 1), float(by + bh)]

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
                    "bottom_edge_px": bottom_edge,
                    "bottom_contact_px": [bot_x, bot_y],
                    "bottom_contact_1000": [norm_bot_y, norm_bot_x],
                    "center_1000": [norm_cy, norm_cx],
                }
            )

    # Sort objects by descending area
    detected.sort(key=lambda item: item["area"], reverse=True)
    return detected


def _apply_homography(homography: np.ndarray, px: float, py: float) -> np.ndarray:
    """Map an image pixel through a 3x3 homography to a 2D point.

    Args:
        homography: 3x3 matrix.
        px: Horizontal pixel coordinate.
        py: Vertical pixel coordinate.

    Returns:
        (2,) array.

    Raises:
        ValueError: If the point maps to infinity.
    """
    vec = homography @ np.array([px, py, 1.0], dtype=np.float64)
    if abs(vec[2]) < 1e-9:
        raise ValueError("Point maps to infinity under the homography")
    return np.array([vec[0] / vec[2], vec[1] / vec[2]], dtype=np.float64)


def locate_target_footprint(
    obj: dict[str, Any],
    homography: np.ndarray,
    image_shape: tuple[int, int] = (480, 640),
) -> dict[str, Any]:
    """Estimate where a detected object stands on the table (its footprint centre).

    The lowest blob pixels are the object's near bottom edge, not its centre (15.7 mm off on a
    30 mm cube). The footprint centre is the midpoint of that bottom edge (``bottom_edge_px``,
    falling back to the bbox bottom row) mapped through the table homography, shifted by half
    the edge width (in mm) along the table-plane normal of that edge, on the side away from the
    camera.

    Args:
        obj: Detection from ``detect_workspace_objects`` (needs ``bbox_px``).
        homography: 3x3 pixel -> robot-mm homography of the table plane.
        image_shape: Frame dimensions as (height, width), for the normalised result.

    Returns:
        Dictionary with ``centre_xy_mm`` ([x, y] robot mm), ``centre_1000`` ([y, x] normalised),
        ``width_mm`` (bottom-edge width) and ``bbox_px``.

    Raises:
        ValueError: If the homography is singular or the geometry degenerates.
    """
    bx, by, bw, bh = obj["bbox_px"]
    left_px, right_px, bottom_py = obj.get("bottom_edge_px") or [bx, bx + bw, by + bh]
    # Blob coordinates are continuous (a pixel spans [i, i+1)); the homography is built from
    # ArUco detections, which use pixel-index coordinates (a pixel's centre is at i).
    left_px, right_px, bottom_py = (
        left_px - PIXEL_INDEX_SHIFT,
        right_px - PIXEL_INDEX_SHIFT,
        bottom_py - PIXEL_INDEX_SHIFT,
    )
    mid_px = (left_px + right_px) / 2.0
    bottom_mid = _apply_homography(homography, mid_px, bottom_py)
    bottom_left = _apply_homography(homography, float(left_px), bottom_py)
    bottom_right = _apply_homography(homography, float(right_px), bottom_py)
    top_mid = _apply_homography(homography, mid_px, float(by) - PIXEL_INDEX_SHIFT)
    edge = bottom_right - bottom_left
    width_mm = float(np.linalg.norm(edge))
    if width_mm < 1e-9:
        raise ValueError("Degenerate bbox: the bottom edge has no width")
    # Away from the camera, perpendicular to the object's own bottom edge in the table plane.
    # (The bbox top maps to a point displaced by the object's height, so the direction towards
    # it leans sideways for off-axis objects; the edge normal does not.)
    normal = np.array([-edge[1], edge[0]]) / width_mm
    if float(normal @ (top_mid - bottom_mid)) < 0.0:
        normal = -normal
    centre = bottom_mid + normal * (width_mm / 2.0)

    inverse = np.linalg.inv(homography)
    cpx = _apply_homography(inverse, float(centre[0]), float(centre[1])) + PIXEL_INDEX_SHIFT
    h, w = image_shape
    centre_1000 = [
        float(np.clip(cpx[1] / h * 1000.0, 0.0, 1000.0)),
        float(np.clip(cpx[0] / w * 1000.0, 0.0, 1000.0)),
    ]
    return {
        "centre_xy_mm": [float(centre[0]), float(centre[1])],
        "centre_1000": centre_1000,
        "width_mm": width_mm,
        "bbox_px": [int(bx), int(by), int(bw), int(bh)],
    }


def color_from_label(label: Any) -> str | None:
    """Return the colour word (red, blue, green, yellow) contained in a tool label, if any.

    Args:
        label: Free-text label from the model (e.g. "red cube"), or None.

    Returns:
        Lower-case colour name, or None if the label names no known colour.
    """
    if not label:
        return None
    words = str(label).lower().replace("-", " ").replace("_", " ").split()
    for color in COLOR_HSV_RANGES:
        if color in words:
            return color
    return None


DEFAULT_REFINE_RADIUS_1000 = 80.0


def refine_point_to_nearest_object(
    point_1000: list[float] | tuple[float, float] | np.ndarray,
    frame_rgb: np.ndarray,
    max_dist_1000: float = DEFAULT_REFINE_RADIUS_1000,
    target_color: str | None = None,
    workspace_polygon_px: np.ndarray | None = None,
    min_area: int = DEFAULT_MIN_OBJECT_AREA_PX,
    calibrator: Any | None = None,
    homography: np.ndarray | None = None,
) -> tuple[list[float], dict[str, Any] | None]:
    """Snap a candidate VLA target point to the nearest detected tabletop object.

    The match distance is measured to the nearest of the object's blob centre, bottom contact
    and (when a homography is available) footprint centre, so a point anywhere on the object's
    image matches it. No object farther than ``max_dist_1000`` is ever snapped to.

    Args:
        point_1000: Normalized [y, x] coordinate in [0, 1000].
        frame_rgb: RGB frame corresponding to the scene.
        max_dist_1000: Maximum allowed distance in normalized coordinate space.
        target_color: Optional colour name (red, blue, green, yellow). When given, only objects
            of that colour can match (no match if none is present).
        workspace_polygon_px: Optional polygon vertices in pixels.
        min_area: Minimum contour pixel area to consider.
        calibrator: Optional ArucoWorkspaceCalibrator instance (workspace mask; also the
            source of the homography when ``homography`` is not given).
        homography: Optional 3x3 pixel -> robot-mm homography for the footprint centre.

    Returns:
        Tuple of (refined_point_1000, detected_object_dict_or_None). The refined point is the
        object's footprint centre when a homography is available, else its ``bottom_contact_1000``
        (kept for backward compatibility). The object dict additionally carries ``centre_1000``,
        ``centre_xy_mm`` and ``width_mm`` when the footprint could be computed.
        If no candidate is found within ``max_dist_1000``, returns the original point and None.
    """
    if len(point_1000) != 2 or frame_rgb is None or frame_rgb.size == 0:
        return [float(point_1000[0]), float(point_1000[1])], None

    objects = detect_workspace_objects(
        frame_rgb,
        workspace_polygon_px=workspace_polygon_px,
        min_area=min_area,
        calibrator=calibrator,
    )
    original = [float(point_1000[0]), float(point_1000[1])]
    if target_color:
        objects = [obj for obj in objects if obj["color"].lower() == target_color.lower()]
    if not objects:
        return original, None

    if homography is None and calibrator is not None:
        homography = getattr(calibrator, "homography_matrix", None)

    py, px = original
    best_obj: dict[str, Any] | None = None
    best_dist = float("inf")
    for obj in objects:
        if homography is not None:
            try:
                obj.update(locate_target_footprint(obj, homography, frame_rgb.shape[:2]))
            except (ValueError, np.linalg.LinAlgError):
                pass
        anchors = [obj["center_1000"], obj["bottom_contact_1000"]]
        if "centre_1000" in obj:
            anchors.append(obj["centre_1000"])
        dist = min(float(np.hypot(a[0] - py, a[1] - px)) for a in anchors)
        if dist < best_dist and dist <= max_dist_1000:
            best_dist = dist
            best_obj = obj

    if best_obj is None:
        return original, None
    refined = best_obj.get("centre_1000", best_obj["bottom_contact_1000"])
    return [float(refined[0]), float(refined[1])], best_obj
