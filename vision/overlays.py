"""Visual overlay toggle states and workspace calibration overlay renderer."""

import cv2
import numpy as np

from robot.calibration import ArucoWorkspaceCalibrator

_calibration_overlay_enabled: bool = False
_target_overlay_enabled: bool = True


def set_calibration_overlay(enabled: bool) -> None:
    """Enable or disable the ArUco calibration overlay on live camera feeds.

    Args:
        enabled: True to enable overlay drawing, False to disable.
    """
    global _calibration_overlay_enabled
    _calibration_overlay_enabled = bool(enabled)


def is_calibration_overlay_enabled() -> bool:
    """Check if the calibration overlay is currently enabled.

    Returns:
        True if overlay is active, False otherwise.
    """
    return _calibration_overlay_enabled


def set_target_overlay_enabled(enabled: bool) -> None:
    """Enable or disable the model target overlay on live camera feeds.

    Args:
        enabled: True to enable overlay drawing, False to disable.
    """
    global _target_overlay_enabled
    _target_overlay_enabled = bool(enabled)


def is_target_overlay_enabled() -> bool:
    """Check if the model target overlay is currently enabled.

    Returns:
        True if overlay is active, False otherwise.
    """
    return _target_overlay_enabled


def draw_workspace_overlay(
    calibrator: ArucoWorkspaceCalibrator,
    frame_rgb: np.ndarray,
    detected_markers: dict[int, np.ndarray] | None = None,
    show_markers: bool = True,
) -> np.ndarray:
    """Draw detected ArUco corners, calibrated workspace boundary, and coordinate axes.

    Args:
        calibrator: Active ArucoWorkspaceCalibrator instance.
        frame_rgb: Input RGB frame image.
        detected_markers: Optional pre-detected markers dict.
        show_markers: Whether to outline detected marker polygons.

    Returns:
        Annotated RGB frame array.
    """
    annotated = frame_rgb.copy()
    if detected_markers is None:
        detected_markers = calibrator.last_detected_markers

    # 1. Draw detected marker boundaries and IDs
    if show_markers and detected_markers:
        for marker_id, corners in detected_markers.items():
            pts = corners.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

            center = np.mean(corners, axis=0).astype(np.int32)
            cv2.putText(
                annotated,
                f"ID:{marker_id}",
                (int(center[0]) - 15, int(center[1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

    # 2. Draw workspace bounding polygon
    marker_ids = calibrator.config.get("marker_ids", {})
    corners_cfg = calibrator.config.get("workspace_corners_mm", {})
    id_order = [
        int(marker_ids.get("top_left", 1)),
        int(marker_ids.get("top_right", 2)),
        int(marker_ids.get("bottom_right", 3)),
        int(marker_ids.get("bottom_left", 0)),
    ]

    # Draw polygon: use detected corners if all 4 visible, or projected corners if calibrated
    poly_pts = None
    if all(mid in detected_markers for mid in id_order):
        poly_pts = np.array(
            [np.mean(detected_markers[mid], axis=0) for mid in id_order],
            dtype=np.int32,
        ).reshape((-1, 1, 2))
    elif calibrator.is_calibrated:
        pt_tl = calibrator.robot_mm_to_pixel(
            float(corners_cfg.get("top_left", {}).get("x", 380)),
            float(corners_cfg.get("top_left", {}).get("y", -190)),
        )
        pt_tr = calibrator.robot_mm_to_pixel(
            float(corners_cfg.get("top_right", {}).get("x", 100)),
            float(corners_cfg.get("top_right", {}).get("y", -190)),
        )
        pt_br = calibrator.robot_mm_to_pixel(
            float(corners_cfg.get("bottom_right", {}).get("x", 100)),
            float(corners_cfg.get("bottom_right", {}).get("y", 190)),
        )
        pt_bl = calibrator.robot_mm_to_pixel(
            float(corners_cfg.get("bottom_left", {}).get("x", 380)),
            float(corners_cfg.get("bottom_left", {}).get("y", 190)),
        )
        if all(p is not None for p in [pt_tl, pt_tr, pt_br, pt_bl]):
            poly_pts = np.array([pt_tl, pt_tr, pt_br, pt_bl], dtype=np.int32).reshape((-1, 1, 2))

    if poly_pts is not None:
        cv2.polylines(annotated, [poly_pts], isClosed=True, color=(0, 230, 255), thickness=2)

        bl_pt = poly_pts[3][0]
        cv2.putText(
            annotated,
            "CALIBRATED WORKSPACE",
            (int(bl_pt[0]) + 5, int(bl_pt[1]) + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 230, 255),
            1,
            cv2.LINE_AA,
        )

    # 3. Draw Robot coordinate axes using true homography projection
    # Prefer origin near robot base (top_right, near ruler and robot base)
    # so +X steps into the workspace (Left) and +Y steps towards front of table (Down)
    axis_corner_key = "top_right" if "top_right" in corners_cfg else "bottom_left"
    axis_id = int(marker_ids.get(axis_corner_key, 2))
    axis_phys = corners_cfg.get(axis_corner_key, {"x": 100.0, "y": -190.0})
    ox_mm = float(axis_phys["x"])
    oy_mm = float(axis_phys["y"])

    origin: tuple[float, float] | None = None
    if axis_id in detected_markers:
        mean_pt = np.mean(detected_markers[axis_id], axis=0)
        origin = (float(mean_pt[0]), float(mean_pt[1]))
    elif calibrator.is_calibrated:
        origin = calibrator.robot_mm_to_pixel(ox_mm, oy_mm)

    if origin is not None:
        axis_len_px = 65.0
        tip_x: tuple[int, int] | None = None
        tip_y: tuple[int, int] | None = None

        if calibrator.is_calibrated and calibrator.inverse_homography is not None:
            # Step +80mm in physical robot space
            px_x_step = calibrator.robot_mm_to_pixel(ox_mm + 80.0, oy_mm)
            px_y_step = calibrator.robot_mm_to_pixel(ox_mm, oy_mm + 80.0)

            if px_x_step is not None:
                dx = px_x_step[0] - origin[0]
                dy = px_x_step[1] - origin[1]
                norm_x = (dx * dx + dy * dy) ** 0.5
                if norm_x > 1e-3:
                    tip_x = (
                        int(origin[0] + axis_len_px * dx / norm_x),
                        int(origin[1] + axis_len_px * dy / norm_x),
                    )

            if px_y_step is not None:
                dx = px_y_step[0] - origin[0]
                dy = px_y_step[1] - origin[1]
                norm_y = (dx * dx + dy * dy) ** 0.5
                if norm_y > 1e-3:
                    tip_y = (
                        int(origin[0] + axis_len_px * dx / norm_y),
                        int(origin[1] + axis_len_px * dy / norm_y),
                    )

        # Fallback if homography unavailable
        if tip_x is None:
            tip_x = (int(origin[0]), int(origin[1] - axis_len_px))
        if tip_y is None:
            tip_y = (int(origin[0] - axis_len_px), int(origin[1]))

        origin_pt = (int(origin[0]), int(origin[1]))

        # +X axis points Forward in robot space - Red
        cv2.arrowedLine(annotated, origin_pt, tip_x, (255, 50, 50), 2, tipLength=0.25)
        cv2.putText(
            annotated,
            "+X (Robot Fwd)",
            (tip_x[0] + 5, tip_x[1] + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 50, 50),
            1,
            cv2.LINE_AA,
        )

        # +Y axis points Left in robot space - Green
        cv2.arrowedLine(annotated, origin_pt, tip_y, (50, 255, 50), 2, tipLength=0.25)
        cv2.putText(
            annotated,
            "+Y (Robot Left)",
            (tip_y[0] + 5, tip_y[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (50, 255, 50),
            1,
            cv2.LINE_AA,
        )

    return annotated
