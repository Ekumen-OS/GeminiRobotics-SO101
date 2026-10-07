"""ArUco marker generation and true-to-scale printable calibration sheet creation."""

import json
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from robot.calibration import _get_default_config_path

logger = logging.getLogger(__name__)


def generate_aruco_markers(
    output_dir: str | Path = "assets/aruco_markers",
    marker_size_px: int = 400,
    dictionary_name: str = "DICT_4X4_50",
    marker_ids: list[int] | None = None,
) -> list[Path]:
    """Generate printable ArUco marker PNG images for workspace calibration.

    Args:
        output_dir: Output directory path where marker PNGs are saved.
        marker_size_px: Marker image resolution in pixels.
        dictionary_name: ArUco dictionary identifier string.
        marker_ids: List of marker IDs to generate. Defaults to [0, 1, 2, 3].

    Returns:
        List of Path objects for the generated PNG files.
    """
    if marker_ids is None:
        marker_ids = [0, 1, 2, 3]

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    dict_attr = getattr(cv2.aruco, dictionary_name, cv2.aruco.DICT_4X4_50)
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_attr)

    created_files: list[Path] = []
    for mid in marker_ids:
        marker_img = cv2.aruco.generateImageMarker(aruco_dict, mid, marker_size_px)
        # Add white border for printing
        bordered = cv2.copyMakeBorder(marker_img, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=255)
        # Annotate marker ID text
        cv2.putText(
            bordered,
            f"ID {mid} (DICT_4X4_50)",
            (35, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            0,
            1,
            cv2.LINE_AA,
        )

        file_path = out_path / f"marker_{mid}.png"
        cv2.imwrite(str(file_path), bordered)
        created_files.append(file_path)
        logger.info(f"[ArUco] Generated {file_path}")

    return created_files


def generate_calibration_sheet(
    config_data: dict[str, Any] | None = None,
    output_dir: str | Path = "assets/aruco_markers",
    sheet_width_px: int = 2400,
    sheet_height_px: int = 1900,
) -> Path:
    """Generate a single printable high-resolution workspace calibration mat sheet.

    Creates an all-in-one true-to-scale poster/sheet containing all 4 corner markers,
    dimension guides, robot base alignment indicator, and coordinate axes strictly matching
    the physical aspect ratio and base offset from configuration.

    Args:
        config_data: Optional dictionary containing calibration settings.
        output_dir: Directory where the calibration sheet PNG will be saved.
        sheet_width_px: Canvas width in pixels.
        sheet_height_px: Canvas height in pixels.

    Returns:
        Path to the generated calibration sheet PNG file.
    """
    if config_data is None:
        cfg_file = _get_default_config_path()
        if cfg_file.exists():
            try:
                config_data = json.loads(cfg_file.read_text())
            except Exception:
                config_data = {}
        else:
            config_data = {}

    corners = config_data.get("workspace_corners_mm", {})
    tl_cfg = corners.get("top_left", {"x": 380, "y": 280})
    tr_cfg = corners.get("top_right", {"x": 380, "y": -200})
    br_cfg = corners.get("bottom_right", {"x": 120, "y": -200})
    bl_cfg = corners.get("bottom_left", {"x": 120, "y": 280})

    all_x = [float(c["x"]) for c in [tl_cfg, tr_cfg, br_cfg, bl_cfg]]
    all_y = [float(c["y"]) for c in [tl_cfg, tr_cfg, br_cfg, bl_cfg]]
    x_max, x_min = max(all_x), min(all_x)
    y_max, y_min = max(all_y), min(all_y)

    delta_x = x_max - x_min
    delta_y = y_max - y_min
    y_mid = (y_max + y_min) / 2.0

    dict_name = config_data.get("aruco_dictionary", "DICT_4X4_50")
    dict_attr = getattr(cv2.aruco, dict_name, cv2.aruco.DICT_4X4_50)
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_attr)

    marker_ids = config_data.get("marker_ids", {})
    id_tl = int(marker_ids.get("top_left", 1))
    id_tr = int(marker_ids.get("top_right", 2))
    id_br = int(marker_ids.get("bottom_right", 3))
    id_bl = int(marker_ids.get("bottom_left", 0))

    # Create crisp white canvas
    canvas = np.full((sheet_height_px, sheet_width_px, 3), 255, dtype=np.uint8)

    # Compute true-to-scale uniform scaling (pixels per mm)
    mat_top_v = 170
    margin_bottom = 200
    avail_w = sheet_width_px - 380
    avail_h = sheet_height_px - mat_top_v - margin_bottom
    scale = min(avail_w / max(1.0, delta_y), avail_h / max(1.0, (x_max + 10)))

    def pt(x: float, y: float) -> tuple[int, int]:
        """Map physical (X_mm, Y_mm) to canvas pixel (u, v)."""
        u = int(sheet_width_px / 2.0 - (y - y_mid) * scale)
        v = int(mat_top_v + (x_max - x) * scale)
        return (u, v)

    # Physical corner coordinates on canvas
    tl = pt(x_max, y_max)
    tr = pt(x_max, y_min)
    br = pt(x_min, y_min)
    bl = pt(x_min, y_max)
    base = pt(0.0, 0.0)

    # Draw grid background inside mat area every 50mm
    step_mm = 50.0
    # X grid lines (horizontal on canvas)
    cur_x = x_min
    while cur_x <= x_max + 1e-4:
        v_line = pt(cur_x, 0.0)[1]
        cv2.line(canvas, (tl[0], v_line), (tr[0], v_line), (242, 245, 248), 1)
        cur_x += step_mm

    # Y grid lines (vertical on canvas)
    cur_y = y_min
    while cur_y <= y_max + 1e-4:
        u_line = pt(x_min, cur_y)[0]
        cv2.line(canvas, (u_line, tl[1]), (u_line, bl[1]), (242, 245, 248), 1)
        cur_y += step_mm

    # Outer mat border with true physical aspect ratio
    cv2.rectangle(canvas, tl, br, (51, 65, 85), 3)

    # Generate and place the 4 ArUco markers
    marker_px = max(160, min(240, int(scale * 45)))
    m_tl = cv2.aruco.generateImageMarker(aruco_dict, id_tl, marker_px)
    m_tr = cv2.aruco.generateImageMarker(aruco_dict, id_tr, marker_px)
    m_br = cv2.aruco.generateImageMarker(aruco_dict, id_br, marker_px)
    m_bl = cv2.aruco.generateImageMarker(aruco_dict, id_bl, marker_px)

    def _paste_marker(marker_img: np.ndarray, x: int, y: int, label: str, id_num: int) -> None:
        rgb_m = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)
        canvas[y : y + marker_px, x : x + marker_px] = rgb_m
        cv2.rectangle(canvas, (x - 2, y - 2), (x + marker_px + 2, y + marker_px + 2), (0, 0, 0), 2)
        cv2.putText(
            canvas,
            f"ARUCO ID {id_num}",
            (x, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            label,
            (x, y + marker_px + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (70, 80, 95),
            1,
            cv2.LINE_AA,
        )

    # Paste markers strictly inside each corner
    p_pad = 20
    _paste_marker(
        m_tl,
        tl[0] + p_pad,
        tl[1] + p_pad,
        f"Top-Left (X:{tl_cfg['x']}mm, Y:+{tl_cfg['y']}mm)",
        id_tl,
    )
    _paste_marker(
        m_tr,
        tr[0] - p_pad - marker_px,
        tr[1] + p_pad,
        f"Top-Right (X:{tr_cfg['x']}mm, Y:{tr_cfg['y']}mm)",
        id_tr,
    )
    _paste_marker(
        m_br,
        br[0] - p_pad - marker_px,
        br[1] - p_pad - marker_px,
        f"Bottom-Right (X:{br_cfg['x']}mm, Y:{br_cfg['y']}mm)",
        id_br,
    )
    _paste_marker(
        m_bl,
        bl[0] + p_pad,
        bl[1] - p_pad - marker_px,
        f"Bottom-Left (X:{bl_cfg['x']}mm, Y:+{bl_cfg['y']}mm)",
        id_bl,
    )

    # Header title banner
    cv2.putText(
        canvas,
        "SO-101 ROBOTIC ARM - WORKSPACE CALIBRATION SHEET",
        (tl[0], 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (15, 23, 42),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"True-to-Scale Reference | Width (Y): {delta_y:.0f} mm | Depth (X): {delta_x:.0f} mm",
        (tl[0], 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (100, 116, 139),
        1,
        cv2.LINE_AA,
    )

    # Draw Center Crosshair
    cx = (tl[0] + tr[0]) // 2
    cy = (tl[1] + bl[1]) // 2
    cv2.drawMarker(canvas, (cx, cy), (160, 175, 190), cv2.MARKER_CROSS, 40, 2)
    cv2.putText(
        canvas,
        f"WORKSPACE CENTER (X:{(x_max + x_min) / 2:.0f}, Y:{y_mid:.0f})",
        (cx - 120, cy - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (100, 116, 139),
        1,
        cv2.LINE_AA,
    )

    # Draw Coordinate Axes indicator in center
    axis_ox, axis_oy = cx - 180, cy + 80
    cv2.arrowedLine(
        canvas,
        (axis_ox, axis_oy),
        (axis_ox, axis_oy - 100),
        (220, 38, 38),
        3,
        tipLength=0.25,
    )
    cv2.putText(
        canvas,
        "+X (Forward)",
        (axis_ox + 10, axis_oy - 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 38, 38),
        2,
        cv2.LINE_AA,
    )

    cv2.arrowedLine(
        canvas,
        (axis_ox, axis_oy),
        (axis_ox - 100, axis_oy),
        (22, 163, 74),
        3,
        tipLength=0.25,
    )
    cv2.putText(
        canvas,
        "+Y (Left)",
        (axis_ox - 110, axis_oy - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (22, 163, 74),
        2,
        cv2.LINE_AA,
    )

    # Robot Base Alignment Guide precisely positioned at true physical (X=0, Y=0)
    base_box_w, base_box_h = 340, 80
    base_bx1 = base[0] - (base_box_w // 2)
    base_by1 = base[1] - 30

    # Draw dashed/dotted connection line from base to near edge of mat
    cv2.line(canvas, (base[0], bl[1]), (base[0], base_by1), (220, 38, 38), 2)

    cv2.rectangle(
        canvas,
        (base_bx1, base_by1),
        (base_bx1 + base_box_w, base_by1 + base_box_h),
        (30, 41, 59),
        2,
    )
    cv2.drawMarker(
        canvas,
        (base[0], base_by1 + (base_box_h // 2)),
        (220, 38, 38),
        cv2.MARKER_TILTED_CROSS,
        25,
        2,
    )
    cv2.putText(
        canvas,
        "ROBOT BASE ALIGNMENT (X=0, Y=0)",
        (base_bx1 + 15, base_by1 + 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (15, 23, 42),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"Align arm base center here ({x_min:.0f}mm from mat edge)",
        (base_bx1 + 10, base_by1 + 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (100, 116, 139),
        1,
        cv2.LINE_AA,
    )

    # Save output sheet
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    sheet_file = out_p / "workspace_calibration_sheet.png"
    cv2.imwrite(str(sheet_file), canvas)
    logger.info(f"[ArUco] Generated true-to-scale calibration sheet: {sheet_file}")
    return sheet_file
