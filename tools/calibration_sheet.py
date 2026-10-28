"""ArUco marker generation and workspace placement guide creation."""

import datetime
import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_MARKER_IDS: dict[str, int] = {
    "top_left": 1,
    "top_right": 2,
    "bottom_right": 3,
    "bottom_left": 0,
}


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
        if mid == 10:
            # Jaw tag (ID 10): 25 mm outer tile with 25:32 (78.125%) active black pattern
            border_px = int(round(marker_size_px * (3.5 / 25.0)))
            bordered = cv2.copyMakeBorder(
                marker_img,
                border_px,
                border_px,
                border_px,
                border_px,
                cv2.BORDER_CONSTANT,
                value=255,
            )
        else:
            bordered = cv2.copyMakeBorder(
                marker_img, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=255
            )
            cv2.putText(
                bordered,
                f"ID {mid} ({dictionary_name})",
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


def _draw_text_box(
    canvas: np.ndarray,
    lines: list[str],
    anchor_xy: tuple[int, int],
    font_scale: float = 0.65,
    thickness: int = 2,
    bg_color: tuple[int, int, int] = (255, 255, 255),
    border_color: tuple[int, int, int] = (160, 174, 192),
    text_color: tuple[int, int, int] = (15, 23, 42),
    padding: int = 10,
    align_h: str = "center",
    align_v: str = "center",
) -> None:
    """Draw multi-line text inside a bordered opaque rectangle."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    sizes = [cv2.getTextSize(line, font, font_scale, thickness)[0] for line in lines]
    max_w = max(w for w, _ in sizes)
    line_h = max(h for _, h in sizes)
    gap = int(round(line_h * 0.45))
    box_w = max_w + 2 * padding
    box_h = len(lines) * line_h + (len(lines) - 1) * gap + 2 * padding

    ax, ay = anchor_xy
    x1 = ax - box_w // 2 if align_h == "center" else (ax if align_h == "left" else ax - box_w)
    y1 = ay - box_h // 2 if align_v == "center" else (ay if align_v == "top" else ay - box_h)
    x2, y2 = x1 + box_w, y1 + box_h

    cv2.rectangle(canvas, (x1, y1), (x2, y2), bg_color, -1)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), border_color, 2, cv2.LINE_AA)

    curr_y = y1 + padding + line_h
    for idx, line in enumerate(lines):
        w = sizes[idx][0]
        tx = (
            x1 + (box_w - w) // 2
            if align_h == "center"
            else (x1 + padding if align_h == "left" else x2 - padding - w)
        )
        cv2.putText(
            canvas, line, (tx, curr_y), font, font_scale, text_color, thickness, cv2.LINE_AA
        )
        curr_y += line_h + gap


def _draw_dim_arrow(
    canvas: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    color: tuple[int, int, int] = (51, 65, 85),
) -> None:
    """Draw a double-headed dimension line with perpendicular end ticks."""
    x1, y1 = pt1
    x2, y2 = pt2
    cv2.line(canvas, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    dist = float(np.hypot(x2 - x1, y2 - y1))
    if dist < 1e-3:
        return
    ux, uy = (x2 - x1) / dist, (y2 - y1) / dist
    nx, ny = -uy, ux
    for px, py in ((x1, y1), (x2, y2)):
        cv2.line(
            canvas,
            (int(round(px - nx * 12)), int(round(py - ny * 12))),
            (int(round(px + nx * 12)), int(round(py + ny * 12))),
            color,
            2,
            cv2.LINE_AA,
        )
    if dist > 28:
        for px, py, s in ((x1, y1, 1.0), (x2, y2, -1.0)):
            for side in (-0.5, 0.5):
                cv2.line(
                    canvas,
                    (px, py),
                    (
                        int(round(px + (s * ux + side * nx) * 14)),
                        int(round(py + (s * uy + side * ny) * 14)),
                    ),
                    color,
                    2,
                    cv2.LINE_AA,
                )


def generate_placement_guide(
    corners_pan_mm: dict[str, tuple[float, float]],
    marker_size_mm: float,
    dictionary_name: str,
    pan_offset_x_mm: float,
    output_dir: str | Path = "assets/aruco_markers",
    marker_ids: dict[str, int] | None = None,
    date_str: str | None = None,
) -> Path:
    """Generate an A4-printable workspace placement guide diagram.

    Args:
        corners_pan_mm: Four workspace corner centres in pan-axis coordinates (mm), keyed by
            'top_left', 'top_right', 'bottom_right', and 'bottom_left'.
        marker_size_mm: Physical side length of each ArUco marker (mm).
        dictionary_name: OpenCV ArUco dictionary name (e.g. 'DICT_4X4_50').
        pan_offset_x_mm: Forward X offset (mm) from robot base origin to shoulder-pan axis.
        output_dir: Output directory where workspace_placement_guide.png is saved.
        marker_ids: Optional mapping from corner name to marker ID.
        date_str: Optional ISO date string for the footer.

    Returns:
        Path to the saved placement guide PNG image.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    tl_pan = corners_pan_mm["top_left"]
    tr_pan = corners_pan_mm["top_right"]
    br_pan = corners_pan_mm["bottom_right"]
    bl_pan = corners_pan_mm["bottom_left"]

    near_x = min(tr_pan[0], br_pan[0])
    far_x = max(tl_pan[0], bl_pan[0])
    depth = far_x - near_x
    right_y = min(tl_pan[1], tr_pan[1])
    left_y = max(bl_pan[1], br_pan[1])
    width = left_y - right_y
    lat_offset = (left_y + right_y) / 2.0

    dict_attr = getattr(cv2.aruco, dictionary_name, cv2.aruco.DICT_4X4_50)
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_attr)
    ids = marker_ids if marker_ids is not None else _DEFAULT_MARKER_IDS
    if date_str is None:
        date_str = datetime.date.today().isoformat()

    w_px, h_px = 3508, 2480
    canvas = np.full((h_px, w_px, 3), 255, dtype=np.uint8)

    x_min_w = min(-115.0, near_x - 30.0)
    x_max_w = far_x + 65.0
    y_min_w = min(right_y - 110.0, -110.0)
    y_max_w = max(left_y + 110.0, 110.0)

    scale = min((w_px - 600) / (y_max_w - y_min_w), (h_px - 500) / (x_max_w - x_min_w))
    u_center = w_px // 2 - 40
    v_center = (220 + h_px - 120) // 2
    x_w_center = (x_max_w + x_min_w) / 2.0
    y_w_center = (y_max_w + y_min_w) / 2.0

    def pt(x_pan: float, y_pan: float) -> tuple[int, int]:
        return (
            int(round(u_center - (y_pan - y_w_center) * scale)),
            int(round(v_center - (x_pan - x_w_center) * scale)),
        )

    # 1. Header Banner (BGR colors)
    cv2.rectangle(canvas, (0, 0), (w_px, 200), (252, 250, 248), -1)
    cv2.line(canvas, (0, 200), (w_px, 200), (225, 213, 203), 2)
    cv2.putText(
        canvas,
        "Rectangle-only layout — place marker centres at these distances "
        "from the shoulder-pan axis",
        (60, 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (42, 23, 15),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "NOTE: Rectangle only, axis-aligned with the robot (+X forward along HOME, "
        "+Y to robot's left). Free-form or rotated layouts are not supported.",
        (60, 125),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (28, 28, 185),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"Tape measurement origin (0, 0) is at the shoulder-pan rotation axis. "
        f"Robot base offset (+{pan_offset_x_mm:.1f} mm in X) is added automatically to config.",
        (60, 168),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (105, 85, 71),
        2,
        cv2.LINE_AA,
    )

    # 3. Robot centerline, base footprint, and pan axis origin
    pan_u, pan_v = pt(0.0, 0.0)
    cv2.line(canvas, pt(x_min_w + 15.0, 0.0), pt(far_x + 25.0, 0.0), (235, 225, 215), 2)

    b_tl = pt(10.0, -45.0)
    b_br = pt(-pan_offset_x_mm - 50.0, 45.0)
    cv2.rectangle(canvas, b_tl, b_br, (245, 243, 240), -1)
    cv2.rectangle(canvas, b_tl, b_br, (160, 150, 140), 2, cv2.LINE_AA)
    _draw_text_box(
        canvas,
        [f"Robot Base Footprint (Origin: -{pan_offset_x_mm:.1f} mm X)"],
        (pan_u, b_br[1] + 14),
        font_scale=0.55,
        thickness=1,
        border_color=(160, 150, 140),
        text_color=(90, 80, 70),
        align_h="center",
        align_v="top",
    )

    # Red crosshair at Pan Axis Origin (0, 0) — BGR red is (38, 38, 220)
    red_bgr = (38, 38, 220)
    cv2.circle(canvas, (pan_u, pan_v), 15, red_bgr, 3, cv2.LINE_AA)
    cv2.circle(canvas, (pan_u, pan_v), 4, red_bgr, -1, cv2.LINE_AA)
    cv2.line(canvas, (pan_u - 22, pan_v), (pan_u + 22, pan_v), red_bgr, 2, cv2.LINE_AA)
    cv2.line(canvas, (pan_u, pan_v - 22), (pan_u, pan_v + 22), red_bgr, 2, cv2.LINE_AA)
    _draw_text_box(
        canvas,
        ["Pan Axis Origin (0, 0)", "Measurement Datum"],
        (pan_u, pan_v + 32),
        font_scale=0.58,
        thickness=2,
        bg_color=(242, 242, 254),
        border_color=red_bgr,
        text_color=(28, 28, 185),
        align_h="center",
        align_v="top",
    )

    # Axis arrows (+X forward, +Y robot left, -Y robot right) clear of footprint and labels
    x_arrow_px = min(130, max(85, int(round(near_x * scale * 0.42))))
    y_arrow_px = max(180, (b_tl[0] - b_br[0]) // 2 + 55)
    cv2.arrowedLine(
        canvas, (pan_u, pan_v), (pan_u, pan_v - x_arrow_px), red_bgr, 3, cv2.LINE_AA, tipLength=0.2
    )
    cv2.putText(
        canvas,
        "+X (Forward)",
        (pan_u - 165, pan_v - x_arrow_px + 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        red_bgr,
        2,
        cv2.LINE_AA,
    )

    teal_bgr = (120, 140, 15)
    cv2.arrowedLine(
        canvas,
        (pan_u, pan_v),
        (pan_u - y_arrow_px, pan_v),
        teal_bgr,
        3,
        cv2.LINE_AA,
        tipLength=0.15,
    )
    cv2.putText(
        canvas,
        "+Y (Robot Left)",
        (pan_u - y_arrow_px - 195, pan_v + 7),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        teal_bgr,
        2,
        cv2.LINE_AA,
    )
    cv2.arrowedLine(
        canvas,
        (pan_u, pan_v),
        (pan_u + y_arrow_px, pan_v),
        teal_bgr,
        3,
        cv2.LINE_AA,
        tipLength=0.15,
    )
    cv2.putText(
        canvas,
        "-Y (Robot Right)",
        (pan_u + y_arrow_px + 14, pan_v + 7),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        teal_bgr,
        2,
        cv2.LINE_AA,
    )

    # 4. Workspace rectangle boundary
    p_tl, p_tr = pt(far_x, right_y), pt(near_x, right_y)
    p_br, p_bl = pt(near_x, left_y), pt(far_x, left_y)
    rect_pts = np.array([p_tl, p_tr, p_br, p_bl], dtype=np.int32)
    cv2.polylines(canvas, [rect_pts], True, (50, 40, 30), 3, cv2.LINE_AA)

    # 5. Markers and corner callouts
    marker_px = max(100, min(170, int(round(marker_size_mm * scale))))
    half_m = marker_px // 2
    marker_defs = [
        ("top_left", ids["top_left"], tl_pan, p_tl, "right", "bottom"),
        ("top_right", ids["top_right"], tr_pan, p_tr, "right", "top"),
        ("bottom_right", ids["bottom_right"], br_pan, p_br, "left", "top"),
        ("bottom_left", ids["bottom_left"], bl_pan, p_bl, "left", "bottom"),
    ]
    for corner_name, mid, coords_pan, (mu, mv), side_h, side_v in marker_defs:
        m_img = cv2.aruco.generateImageMarker(aruco_dict, int(mid), marker_px)
        m_bgr = cv2.cvtColor(m_img, cv2.COLOR_GRAY2BGR)
        x1, y1 = mu - half_m, mv - half_m
        canvas[y1 : y1 + marker_px, x1 : x1 + marker_px] = m_bgr
        cv2.rectangle(
            canvas, (x1 - 2, y1 - 2), (x1 + marker_px + 2, y1 + marker_px + 2), (30, 20, 15), 2
        )
        cv2.circle(canvas, (mu, mv), 5, red_bgr, -1, cv2.LINE_AA)
        cv2.line(canvas, (mu - 12, mv), (mu + 12, mv), red_bgr, 2, cv2.LINE_AA)
        cv2.line(canvas, (mu, mv - 12), (mu, mv + 12), red_bgr, 2, cv2.LINE_AA)

        off_u = (half_m + 16) if side_h == "right" else -(half_m + 16)
        off_v = (half_m + 16) if side_v == "top" else -(half_m + 16)
        _draw_text_box(
            canvas,
            [
                f"{corner_name} (ID {mid})",
                f"({coords_pan[0]:+.1f}, {coords_pan[1]:+.1f}) mm",
                "from pan axis",
            ],
            (mu + off_u, mv + off_v),
            font_scale=0.6,
            thickness=2,
            border_color=(42, 23, 15),
            align_h="left" if side_h == "right" else "right",
            align_v="top" if side_v == "top" else "bottom",
        )

    # 6. Dimension lines (placed clear of marker boxes and axis labels)
    # Right-side X dimensions (near X and depth) placed well past right-hand marker callouts
    dim_x_u = p_tr[0] + half_m + 350
    cv2.line(canvas, (pan_u, pan_v), (dim_x_u + 15, pan_v), (180, 165, 150), 1, cv2.LINE_AA)
    cv2.line(canvas, (p_tr[0], p_tr[1]), (dim_x_u + 15, p_tr[1]), (180, 165, 150), 1, cv2.LINE_AA)
    cv2.line(canvas, (p_tl[0], p_tl[1]), (dim_x_u + 15, p_tl[1]), (180, 165, 150), 1, cv2.LINE_AA)
    _draw_dim_arrow(canvas, (dim_x_u, pan_v), (dim_x_u, p_tr[1]))
    _draw_text_box(
        canvas,
        [f"near X: {near_x:.1f} mm"],
        (dim_x_u + 18, (pan_v + p_tr[1]) // 2),
        font_scale=0.68,
        align_h="left",
    )
    _draw_dim_arrow(canvas, (dim_x_u, p_tr[1]), (dim_x_u, p_tl[1]))
    _draw_text_box(
        canvas,
        [f"depth: {depth:.1f} mm"],
        (dim_x_u + 18, (p_tr[1] + p_tl[1]) // 2),
        font_scale=0.68,
        align_h="left",
    )

    # Top width dimension
    dim_w_v = p_tl[1] - half_m - 45
    cv2.line(canvas, (p_bl[0], p_bl[1]), (p_bl[0], dim_w_v - 15), (180, 165, 150), 1, cv2.LINE_AA)
    cv2.line(canvas, (p_tl[0], p_tl[1]), (p_tl[0], dim_w_v - 15), (180, 165, 150), 1, cv2.LINE_AA)
    _draw_dim_arrow(canvas, (p_bl[0], dim_w_v), (p_tl[0], dim_w_v))
    _draw_text_box(
        canvas,
        [f"width: {width:.1f} mm"],
        ((p_bl[0] + p_tl[0]) // 2, dim_w_v - 15),
        font_scale=0.72,
        align_h="center",
        align_v="bottom",
    )

    # Lateral offset dimension just below the near edge, label to the right of the span
    rect_center_u = (p_tl[0] + p_bl[0]) // 2
    dim_lat_v = p_tr[1] + half_m + 38
    if abs(lat_offset) > 2.0:
        cv2.line(canvas, (pan_u, p_tr[1]), (pan_u, dim_lat_v + 12), (180, 165, 150), 1, cv2.LINE_AA)
        cv2.line(
            canvas,
            (rect_center_u, p_tr[1]),
            (rect_center_u, dim_lat_v + 12),
            (180, 165, 150),
            1,
            cv2.LINE_AA,
        )
        _draw_dim_arrow(canvas, (pan_u, dim_lat_v), (rect_center_u, dim_lat_v))
        _draw_text_box(
            canvas,
            [f"lateral offset: {lat_offset:+.1f} mm"],
            (max(pan_u, rect_center_u) + 20, dim_lat_v),
            font_scale=0.62,
            align_h="left",
            align_v="center",
        )
    else:
        _draw_text_box(
            canvas,
            ["lateral offset: 0.0 mm (centred)"],
            (pan_u, dim_lat_v),
            font_scale=0.62,
            align_h="center",
            align_v="center",
        )

    # 7. Legend Box and Footer
    _draw_text_box(
        canvas,
        [
            "DIAGRAM LEGEND",
            "  • Red Crosshair: Pan Rotation Axis Origin (0, 0)",
            "  • Dark Outer Border: Workspace Rectangle Boundary",
            "  • Marker Squares: ArUco Marker Tiles (place centres at given coords)",
            "  • Dimension Lines: Tape Measure Distance from Pan Axis Origin",
        ],
        (60, 225),
        font_scale=0.55,
        thickness=1,
        border_color=(215, 200, 190),
        text_color=(75, 65, 51),
        align_h="left",
        align_v="top",
        padding=12,
    )

    cv2.rectangle(canvas, (0, h_px - 85), (w_px, h_px), (252, 250, 248), -1)
    cv2.line(canvas, (0, h_px - 85), (w_px, h_px - 85), (225, 213, 203), 2)
    footer_text = (
        f"Marker Size: {marker_size_mm:.1f} mm | ArUco Dictionary: {dictionary_name} | "
        f"Base Offset: +{pan_offset_x_mm:.1f} mm X | Date: {date_str}"
    )
    cv2.putText(
        canvas,
        footer_text,
        (w_px // 2 - 460, h_px - 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (120, 110, 100),
        2,
        cv2.LINE_AA,
    )

    guide_path = out_path / "workspace_placement_guide.png"
    cv2.imwrite(str(guide_path), canvas)
    logger.info(f"[ArUco] Generated workspace placement guide: {guide_path}")
    return guide_path


def generate_jaw_tag_sheet(
    output_dir: str | Path = "assets/aruco_markers",
    marker_id: int = 10,
    tag_size_mm: float = 25.0,
    dpi: int = 300,
    dictionary_name: str = "DICT_4X4_50",
) -> Path:
    """Generate a printable 1:1 scale jaw ArUco marker sheet.

    Args:
        output_dir: Directory where the sheet PNG will be saved.
        marker_id: Marker ID to generate. Defaults to 10.
        tag_size_mm: Physical side length of marker in millimeters. Defaults to 25.0.
        dpi: Print resolution in dots per inch. Defaults to 300 DPI.
        dictionary_name: ArUco dictionary identifier string.

    Returns:
        Path to the generated jaw tag sheet PNG file.
    """
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    w_px = int(round((210.0 / 25.4) * dpi))
    h_px = int(round((297.0 / 25.4) * dpi))
    canvas = np.full((h_px, w_px, 3), 255, dtype=np.uint8)

    dict_attr = getattr(cv2.aruco, dictionary_name, cv2.aruco.DICT_4X4_50)
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_attr)
    tag_size_px = int(round((tag_size_mm / 25.4) * dpi))
    marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, tag_size_px)
    marker_rgb = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2RGB)

    cx, cy = w_px // 2, int(h_px * 0.3)
    x1, y1 = cx - tag_size_px // 2, cy - tag_size_px // 2
    canvas[y1 : y1 + tag_size_px, x1 : x1 + tag_size_px] = marker_rgb

    cut_pad_px = int(round((5.0 / 25.4) * dpi))
    cv2.rectangle(
        canvas,
        (x1 - cut_pad_px, y1 - cut_pad_px),
        (x1 + tag_size_px + cut_pad_px, y1 + tag_size_px + cut_pad_px),
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )

    ruler_len_px = int(round((50.0 / 25.4) * dpi))
    rx = cx - ruler_len_px // 2
    ry = cy + tag_size_px // 2 + int(round((25.0 / 25.4) * dpi))
    cv2.line(canvas, (rx, ry), (rx + ruler_len_px, ry), (0, 0, 0), 2, cv2.LINE_AA)
    cv2.line(canvas, (rx, ry - 10), (rx, ry + 10), (0, 0, 0), 2, cv2.LINE_AA)
    cv2.line(
        canvas,
        (rx + ruler_len_px, ry - 10),
        (rx + ruler_len_px, ry + 10),
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        canvas,
        f"SO-101 Gripper Jaw ArUco Tag ({tag_size_mm:g} mm, ID {marker_id})",
        (cx - 280, int(h_px * 0.15)),
        font,
        0.8,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "50 mm Scale Line (must measure exactly 50 mm when printed at 100%)",
        (rx - 80, ry + 25),
        font,
        0.45,
        (60, 60, 60),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas, "Mounting Instructions:", (cx - 250, ry + 70), font, 0.6, (0, 0, 0), 2, cv2.LINE_AA
    )
    instructions = [
        "1. Print this sheet at 100% scale (no fit-to-page / borderless).",
        "2. Verify the 50 mm line with a physical ruler.",
        f"3. Cut out the {tag_size_mm:g} mm ArUco marker along the inner border.",
        "4. Mount it on the gripper housing, facing the camera, clear of the fingers.",
        "5. Measure its centre against the grasp site and set GRIPPER_TAG_LOCAL_OFFSET_MM.",
    ]
    for idx, inst in enumerate(instructions):
        cv2.putText(
            canvas, inst, (cx - 240, ry + 105 + idx * 30), font, 0.45, (50, 50, 50), 1, cv2.LINE_AA
        )

    sheet_file = out_p / "jaw_tag_sheet.png"
    cv2.imwrite(str(sheet_file), canvas)
    logger.info(f"[ArUco] Generated jaw tag sheet: {sheet_file}")
    return sheet_file
