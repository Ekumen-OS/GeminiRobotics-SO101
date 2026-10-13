"""OpenCV-assisted multi-color salient object detector for table workspace."""

from typing import Any

import cv2
import numpy as np
from scipy.optimize import minimize

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

            # Lowest boundary of the blob (bottom-most pixel in each column)
            blob_mask = np.zeros((bh, bw), dtype=np.uint8)
            cv2.drawContours(blob_mask, [cnt], -1, 255, -1, offset=(-bx, -by))
            bottom_boundary: list[list[int]] = []
            for col_x in range(bw):
                nz = np.where(blob_mask[:, col_x])[0]
                if len(nz) > 0:
                    bottom_boundary.append([int(bx + col_x), int(by + int(nz.max()))])

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
                    "bottom_boundary_px": bottom_boundary,
                    "contour": cnt,
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
    camera: Any | None = None,
) -> dict[str, Any]:
    """Estimate where a detected object stands on the table (its footprint centre).

    For rotated cubes, the lowest boundary of the blob (bottom-most pixel in each column)
    forms visible bottom edges on the table. Maps these points through the table homography,
    fits the near corner and edge directions (yaw mod 90 deg), and optionally refines via
    silhouette IoU matching.

    Args:
        obj: Detection from ``detect_workspace_objects`` (needs ``bbox_px``).
        homography: 3x3 pixel -> robot-mm homography of the table plane.
        image_shape: Frame dimensions as (height, width), for the normalised result.
        camera: Optional PinholeCamera model (recovered from homography if omitted).

    Returns:
        Dictionary with ``centre_xy_mm`` ([x, y] robot mm), ``centre_1000`` ([y, x] normalised),
        ``width_mm``, ``yaw_deg`` (estimated yaw mod 90 deg) and ``bbox_px``.

    Raises:
        ValueError: If the homography is singular or the geometry degenerates.
    """
    bx, by, bw, bh = obj["bbox_px"]
    inv_h = np.linalg.inv(homography)

    if camera is None:
        try:
            from robot.camera_model import PinholeCamera

            camera = PinholeCamera.from_homography(homography)
        except Exception:
            camera = None

    cam_xy = (
        np.asarray(camera.centre_mm[:2], dtype=np.float64)
        if camera is not None
        else np.array([185.0, 458.0], dtype=np.float64)
    )

    cnt = obj.get("contour")
    col_pts: np.ndarray
    if "bottom_boundary_px" in obj and len(obj["bottom_boundary_px"]) >= 4:
        col_pts = np.asarray(obj["bottom_boundary_px"], dtype=np.float64)
    elif cnt is not None and len(cnt) >= 4:
        blob_mask = np.zeros((bh, bw), dtype=np.uint8)
        cv2.drawContours(blob_mask, [cnt], -1, 255, -1, offset=(-bx, -by))
        extracted: list[list[float]] = []
        for col_x in range(bw):
            nz = np.where(blob_mask[:, col_x])[0]
            if len(nz) > 0:
                extracted.append([float(bx + col_x), float(by + nz.max())])
        col_pts = np.asarray(extracted, dtype=np.float64)
    else:
        left_px, right_px, bottom_py = obj.get("bottom_edge_px") or [bx, bx + bw, by + bh]
        col_pts = np.array([[left_px, bottom_py], [right_px, bottom_py]], dtype=np.float64)

    # Filter out pixels that are far higher than the lowest row
    if len(col_pts) >= 4:
        max_py = float(col_pts[:, 1].max())
        col_pts = col_pts[col_pts[:, 1] >= max_py - float(bh) * 0.4]

    t_pts = np.array(
        [
            _apply_homography(homography, p[0] - PIXEL_INDEX_SHIFT, p[1] - PIXEL_INDEX_SHIFT)
            for p in col_pts
        ],
        dtype=np.float64,
    )

    # Check physical width across the bottom edge / boundary
    left_px, right_px, bottom_py = obj.get("bottom_edge_px") or [bx, bx + bw, by + bh]
    b_left = _apply_homography(
        homography, float(left_px) - PIXEL_INDEX_SHIFT, float(bottom_py) - PIXEL_INDEX_SHIFT
    )
    b_right = _apply_homography(
        homography, float(right_px) - PIXEL_INDEX_SHIFT, float(bottom_py) - PIXEL_INDEX_SHIFT
    )
    edge_span_mm = float(np.linalg.norm(b_right - b_left))
    pts_span_mm = float(np.linalg.norm(t_pts[-1] - t_pts[0])) if len(t_pts) > 1 else 0.0
    measured_width = max(edge_span_mm, pts_span_mm)

    if len(t_pts) < 4:
        end_a = t_pts[0]
        end_b = t_pts[-1]
        edge = end_b - end_a
        w_mm = float(np.linalg.norm(edge))
        if w_mm < 1e-9:
            raise ValueError("Degenerate bbox: the bottom edge has no width")
        mid = (end_a + end_b) / 2.0
        normal = np.array([-edge[1], edge[0]]) / w_mm
        if float(normal @ (mid - cam_xy)) < 0.0:
            normal = -normal
        s = float(np.clip(w_mm, 15.0, 50.0))
        centre = mid + normal * (s / 2.0)
        yaw_rad = float(np.arctan2(edge[1], edge[0]))
        yaw_deg = float(np.degrees(yaw_rad) % 90.0)
        cpx = _apply_homography(inv_h, float(centre[0]), float(centre[1])) + PIXEL_INDEX_SHIFT
        h, w_img = image_shape
        centre_1000 = [
            float(np.clip(cpx[1] / h * 1000.0, 0.0, 1000.0)),
            float(np.clip(cpx[0] / w_img * 1000.0, 0.0, 1000.0)),
        ]
        return {
            "centre_xy_mm": [float(centre[0]), float(centre[1])],
            "centre_1000": centre_1000,
            "width_mm": measured_width if measured_width > 50.0 else s,
            "yaw_deg": yaw_deg,
            "bbox_px": [int(bx), int(by), int(bw), int(bh)],
        }

    # L-shape fit
    to_cam = cam_xy - t_pts.mean(axis=0)
    norm_to_cam = float(np.linalg.norm(to_cam))
    if norm_to_cam > 1e-9:
        to_cam /= norm_to_cam
    init_corner = t_pts[np.argmax(t_pts @ to_cam)]

    def l_loss(params: list[float] | np.ndarray) -> float:
        cx, cy, th = float(params[0]), float(params[1]), float(params[2])
        u1 = np.array([np.cos(th), np.sin(th)])
        u2 = np.array([-np.sin(th), np.cos(th)])
        c_pt = np.array([cx, cy])
        if float(u1 @ (c_pt - cam_xy)) < 0.0:
            u1 = -u1
        if float(u2 @ (c_pt - cam_xy)) < 0.0:
            u2 = -u2
        diff = t_pts - c_pt
        p1 = diff @ u1
        p2 = diff @ u2
        d1 = p2**2 + np.maximum(0.0, -p1) ** 2 + np.maximum(0.0, p1 - 30.0) ** 2
        d2 = p1**2 + np.maximum(0.0, -p2) ** 2 + np.maximum(0.0, p2 - 30.0) ** 2
        return float(np.sum(np.minimum(d1, d2)))

    best_loss = float("inf")
    best_th = 0.0
    for th_test in np.linspace(0.0, np.pi / 2.0, 19):
        loss_val = l_loss([init_corner[0], init_corner[1], th_test])
        if loss_val < best_loss:
            best_loss = loss_val
            best_th = th_test

    res = minimize(
        l_loss,
        [init_corner[0], init_corner[1], best_th],
        method="Nelder-Mead",
        options={"maxiter": 100},
    )
    cx, cy, th = float(res.x[0]), float(res.x[1]), float(res.x[2])
    u1 = np.array([np.cos(th), np.sin(th)])
    u2 = np.array([-np.sin(th), np.cos(th)])
    c_pt = np.array([cx, cy])
    if float(u1 @ (c_pt - cam_xy)) < 0.0:
        u1 = -u1
    if float(u2 @ (c_pt - cam_xy)) < 0.0:
        u2 = -u2
    final_centre = c_pt + 15.0 * (u1 + u2)
    final_th = th

    # Silhouette IoU refinement when camera and contour are available
    if camera is not None and cnt is not None:
        blob_mask = np.zeros((bh, bw), dtype=np.uint8)
        cv2.drawContours(blob_mask, [cnt], -1, 255, -1, offset=(-bx, -by))

        def iou_cost(params: list[float] | np.ndarray) -> float:
            px, py, pyaw = float(params[0]), float(params[1]), float(params[2])
            c, s = np.cos(pyaw), np.sin(pyaw)
            rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
            corners_3d = []
            for dx in (-15.0, 15.0):
                for dy in (-15.0, 15.0):
                    for dz in (-15.0, 15.0):
                        corners_3d.append([px, py, 15.0] + rot @ np.array([dx, dy, dz]))
            uvs = np.array([camera.project(pt) for pt in corners_3d], dtype=np.float32)
            hull = cv2.convexHull(uvs)
            hull_crop = hull - np.array([bx, by], dtype=np.float32)
            sim_mask = np.zeros((bh, bw), dtype=np.uint8)
            cv2.fillPoly(sim_mask, [np.int32(hull_crop)], 255)
            inter = cv2.countNonZero(cv2.bitwise_and(blob_mask, sim_mask))
            union = cv2.countNonZero(cv2.bitwise_or(blob_mask, sim_mask))
            if union == 0:
                return 1.0
            return 1.0 - (float(inter) / float(union))

        res_iou = minimize(
            iou_cost,
            [final_centre[0], final_centre[1], th],
            method="Nelder-Mead",
            options={"maxiter": 40, "xatol": 0.5, "fatol": 0.01},
        )
        final_centre = res_iou.x[:2]
        final_th = float(res_iou.x[2])

    yaw_deg = float(np.degrees(final_th) % 90.0)
    cpx = (
        _apply_homography(inv_h, float(final_centre[0]), float(final_centre[1])) + PIXEL_INDEX_SHIFT
    )
    h, w_img = image_shape
    centre_1000 = [
        float(np.clip(cpx[1] / h * 1000.0, 0.0, 1000.0)),
        float(np.clip(cpx[0] / w_img * 1000.0, 0.0, 1000.0)),
    ]
    return {
        "centre_xy_mm": [float(final_centre[0]), float(final_centre[1])],
        "centre_1000": centre_1000,
        "width_mm": measured_width if measured_width > 50.0 else 30.0,
        "yaw_deg": yaw_deg,
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

    camera = None
    if calibrator is not None:
        if homography is None:
            homography = getattr(calibrator, "homography_matrix", None)
        geom_fn = getattr(calibrator, "geometry_snapshot", None)
        if callable(geom_fn):
            snapshot = geom_fn()
            if snapshot is not None:
                camera = getattr(snapshot, "camera", None)

    py, px = original
    best_obj: dict[str, Any] | None = None
    best_dist = float("inf")
    for obj in objects:
        if homography is not None:
            try:
                obj.update(
                    locate_target_footprint(obj, homography, frame_rgb.shape[:2], camera=camera)
                )
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
