"""ArUco marker generation and workspace placement guide creation."""

import datetime
import functools
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


_DPI = 300
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_INK = (42, 23, 15)
_MUTED = (110, 100, 90)
_RED = (38, 38, 200)
_TEAL = (120, 140, 15)
_GUIDE = (200, 190, 180)
_BOX_BORDER = (175, 165, 155)


def _px(mm: float) -> int:
    """Convert a printed length in millimetres to pixels at 300 dpi.

    Args:
        mm: Length on paper in millimetres.

    Returns:
        Length in pixels.
    """
    return int(round(mm / 25.4 * _DPI))


@functools.lru_cache(maxsize=32)
def _font(size_pt: float, bold: bool = False) -> tuple[float, int]:
    """Return the Hershey font scale and stroke thickness that print at ``size_pt``.

    The cap height is set to 0.7 em, the usual ratio for sans-serif type, so text sized here
    matches the same point size in a word processor when the page is printed at 300 dpi.

    Args:
        size_pt: Typographic point size on paper.
        bold: Use a heavier stroke.

    Returns:
        Tuple of (font_scale, thickness) for ``cv2.putText``.
    """
    cap_px = 0.7 * size_pt * _DPI / 72.0
    thickness = max(2, int(round(cap_px / (7.0 if bold else 11.0))))
    h1 = cv2.getTextSize("H", _FONT, 1.0, thickness)[0][1]
    h2 = cv2.getTextSize("H", _FONT, 2.0, thickness)[0][1]
    return (cap_px - (2 * h1 - h2)) / (h2 - h1), thickness


def _text_size(text: str, size_pt: float, bold: bool = False) -> tuple[int, int]:
    """Measure rendered text width and cap height in pixels.

    Args:
        text: Text to measure (ASCII only; Hershey fonts have no other glyphs).
        size_pt: Typographic point size on paper.
        bold: Use a heavier stroke.

    Returns:
        Tuple of (width_px, height_px).
    """
    scale, thickness = _font(size_pt, bold)
    (w, h), _ = cv2.getTextSize(text, _FONT, scale, thickness)
    return w, h


def _put_text(
    canvas: np.ndarray,
    text: str,
    origin: tuple[int, int],
    size_pt: float,
    color: tuple[int, int, int] = _INK,
    bold: bool = False,
) -> None:
    """Draw text with its bottom-left corner at ``origin``.

    Args:
        canvas: BGR image to draw on.
        text: Text to draw.
        origin: (x, baseline_y) in pixels.
        size_pt: Typographic point size on paper.
        color: BGR text colour.
        bold: Use a heavier stroke.
    """
    scale, thickness = _font(size_pt, bold)
    cv2.putText(canvas, text, origin, _FONT, scale, color, thickness, cv2.LINE_AA)


def _box_size(lines: list[str], size_pt: float, bold: bool = False) -> tuple[int, int]:
    """Measure the padded box that ``_draw_text_box`` draws for ``lines``.

    Args:
        lines: Text lines.
        size_pt: Typographic point size on paper.
        bold: Use a heavier stroke.

    Returns:
        Tuple of (box_width_px, box_height_px).
    """
    sizes = [_text_size(line, size_pt, bold) for line in lines]
    line_h = max(h for _, h in sizes)
    pad = int(round(line_h * 0.45))
    gap = int(round(line_h * 0.6))
    return (
        max(w for w, _ in sizes) + 2 * pad,
        len(lines) * line_h + (len(lines) - 1) * gap + 2 * pad,
    )


def _draw_text_box(
    canvas: np.ndarray,
    lines: list[str],
    anchor_xy: tuple[int, int],
    size_pt: float,
    align_h: str = "center",
    align_v: str = "center",
    bold: bool = False,
    text_color: tuple[int, int, int] = _INK,
    border_color: tuple[int, int, int] | None = _BOX_BORDER,
) -> tuple[int, int, int, int]:
    """Draw left-aligned multi-line text on an opaque white box.

    Args:
        canvas: BGR image to draw on.
        lines: Text lines.
        anchor_xy: Anchor point in pixels, interpreted through ``align_h`` and ``align_v``.
        size_pt: Typographic point size on paper.
        align_h: Which box edge sits on the anchor: 'left', 'center' or 'right'.
        align_v: Which box edge sits on the anchor: 'top', 'center' or 'bottom'.
        bold: Use a heavier stroke.
        text_color: BGR text colour.
        border_color: BGR border colour, or None for no border.

    Returns:
        The box as (x1, y1, x2, y2) in pixels.
    """
    box_w, box_h = _box_size(lines, size_pt, bold)
    line_h = max(_text_size(line, size_pt, bold)[1] for line in lines)
    pad = int(round(line_h * 0.45))
    gap = int(round(line_h * 0.6))

    ax, ay = anchor_xy
    x1 = ax - box_w // 2 if align_h == "center" else (ax if align_h == "left" else ax - box_w)
    y1 = ay - box_h // 2 if align_v == "center" else (ay if align_v == "top" else ay - box_h)
    x2, y2 = x1 + box_w, y1 + box_h

    cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 255, 255), -1)
    if border_color is not None:
        cv2.rectangle(canvas, (x1, y1), (x2, y2), border_color, 2, cv2.LINE_AA)
    baseline = y1 + pad + line_h
    for line in lines:
        _put_text(canvas, line, (x1 + pad, baseline), size_pt, text_color, bold)
        baseline += line_h + gap
    return x1, y1, x2, y2


def _draw_dim_arrow(
    canvas: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    thickness: int,
    color: tuple[int, int, int] = _INK,
) -> None:
    """Draw a double-headed dimension line with perpendicular end ticks.

    Args:
        canvas: BGR image to draw on.
        pt1: Start point in pixels.
        pt2: End point in pixels.
        thickness: Line thickness in pixels.
        color: BGR line colour.
    """
    x1, y1 = pt1
    x2, y2 = pt2
    dist = float(np.hypot(x2 - x1, y2 - y1))
    if dist < 1e-3:
        return
    cv2.line(canvas, pt1, pt2, color, thickness, cv2.LINE_AA)
    ux, uy = (x2 - x1) / dist, (y2 - y1) / dist
    nx, ny = -uy, ux
    tick, head = _px(2.0), min(_px(2.5), dist / 3.0)
    for (px, py), s in (((x1, y1), 1.0), ((x2, y2), -1.0)):
        cv2.line(
            canvas,
            (int(round(px - nx * tick)), int(round(py - ny * tick))),
            (int(round(px + nx * tick)), int(round(py + ny * tick))),
            color,
            thickness,
            cv2.LINE_AA,
        )
        for side in (-0.4, 0.4):
            tip = (
                int(round(px + (s * ux + side * nx) * head)),
                int(round(py + (s * uy + side * ny) * head)),
            )
            cv2.line(canvas, (px, py), tip, color, thickness, cv2.LINE_AA)


def _fmt_mm(value: float, signed: bool = False) -> str:
    """Format a millimetre value without a trailing '.0'.

    Args:
        value: Value in millimetres.
        signed: Prefix positive values with '+'.

    Returns:
        The formatted number, e.g. '414', '-69', '+12.5'.
    """
    r = round(value, 1)
    if r == 0:
        return "0"
    text = f"{r:.0f}" if abs(r - round(r)) < 1e-6 else f"{r:.1f}"
    return f"+{text}" if signed and r > 0 else text


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

    The page is A4 landscape at 300 dpi. The drawing is not to scale; every distance the
    operator needs is written on it. Text is sized in printed points so it stays readable on
    paper and when the PNG is shown scaled down.

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

    xs = [p[0] for p in corners_pan_mm.values()]
    ys = [p[1] for p in corners_pan_mm.values()]
    near_x, far_x = min(xs), max(xs)
    right_y, left_y = min(ys), max(ys)
    depth = far_x - near_x
    width = left_y - right_y
    lat_offset = (left_y + right_y) / 2.0
    has_lat = abs(lat_offset) > 2.0

    dict_attr = getattr(cv2.aruco, dictionary_name, cv2.aruco.DICT_4X4_50)
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_attr)
    ids = marker_ids if marker_ids is not None else _DEFAULT_MARKER_IDS
    if date_str is None:
        date_str = datetime.date.today().isoformat()

    w_px, h_px = _px(297.0), _px(210.0)
    canvas = np.full((h_px, w_px, 3), 255, dtype=np.uint8)
    margin, gap = _px(12.0), _px(2.5)
    thin, line_w, edge_w = 2, _px(0.35), _px(0.6)
    title_pt, body_pt, label_pt, dim_pt, axis_pt, small_pt = 24.0, 14.0, 15.0, 18.0, 14.0, 12.0

    # 1. Header and footer
    y = margin
    header = [
        ("Marker placement guide - rectangle layout", title_pt, True, _INK),
        (
            "Place each marker CENTRE at the values shown: millimetres, measured from the "
            "shoulder-pan axis (0, 0).",
            body_pt,
            False,
            _INK,
        ),
        (
            "Rectangle only, axis-aligned with the robot. Rotated or free-form layouts are "
            "not supported.",
            body_pt,
            True,
            _RED,
        ),
    ]
    for text, size, bold, color in header:
        y += _text_size(text, size, bold)[1]
        _put_text(canvas, text, (margin, y), size, color, bold)
        y += int(_text_size(text, size, bold)[1] * 0.7)
    header_bottom = y + gap
    cv2.line(canvas, (margin, header_bottom), (w_px - margin, header_bottom), _GUIDE, thin)

    footer = (
        f"Marker {_fmt_mm(marker_size_mm)} mm  |  {dictionary_name}  |  not to scale: use the "
        f"distances written on the drawing  |  {date_str}"
    )
    footer_w, footer_h = _text_size(footer, small_pt)
    _put_text(canvas, footer, ((w_px - footer_w) // 2, h_px - margin), small_pt, _MUTED)
    footer_top = h_px - margin - footer_h - gap
    cv2.line(canvas, (margin, footer_top), (w_px - margin, footer_top), _GUIDE, thin)

    # 2. Text that sits around the rectangle, measured up front to fit the drawing
    marker_lines = {
        name: [f"ID {ids[name]}  {name}", f"x {_fmt_mm(p[0])}  y {_fmt_mm(p[1])}"]
        for name, p in corners_pan_mm.items()
    }
    lab_w = max(_box_size(lines, label_pt)[0] for lines in marker_lines.values())
    near_lines = ["near X", f"{_fmt_mm(near_x)} mm"]
    depth_lines = ["depth", f"{_fmt_mm(depth)} mm"]
    dim_w = max(_box_size(near_lines, dim_pt, True)[0], _box_size(depth_lines, dim_pt, True)[0])
    width_lines = [f"width {_fmt_mm(width)} mm" + ("" if has_lat else " (centred on robot)")]
    lat_lines = ["lateral offset", f"{_fmt_mm(lat_offset, signed=True)} mm"]
    width_box_h = _box_size(width_lines, dim_pt, True)[1]
    lat_box_h = _box_size(lat_lines, dim_pt, True)[1] if has_lat else 0
    base_label = [f"robot base (origin {pan_offset_x_mm:.1f} mm behind pan axis)"]
    base_label_h = _box_size(base_label, small_pt)[1]
    pan_label = ["pan axis (0, 0)"]
    plus_y, minus_y, plus_x = "+Y (robot left)", "-Y (robot right)", "+X (forward)"

    marker_px_max = _px(13.0)
    half_max = marker_px_max // 2
    base_back_mm, base_half_mm = pan_offset_x_mm + 50.0, 45.0
    tick = gap // 2
    rows_above = 2 * gap + (lat_box_h + 2 * gap if has_lat else 0) + gap // 2 + width_box_h
    band_top, band_bottom = header_bottom + gap, footer_top - gap

    # 3. Scale: fit the rectangle plus its surrounding text inside the band
    fixed_w = 2 * (half_max + gap + lab_w) + 2 * gap + tick + dim_w
    scale = min(
        (w_px - 2 * margin - fixed_w) / width,
        (band_bottom - band_top - half_max - rows_above - gap - base_label_h)
        / (far_x + base_back_mm),
    )

    def layout(scale: float) -> dict[str, int]:
        """Horizontal extents (relative to the pan axis) of everything at this scale."""
        half_m = max(_px(3.5), min(half_max, int(round(marker_size_mm * scale / 2))))
        y_arrow = max(_px(16.0), int(round(base_half_mm * scale)) + _px(6.0))
        rect_l, rect_r = int(round(-left_y * scale)), int(round(-right_y * scale))
        base_r = int(round(base_half_mm * scale))
        minus_y_end = y_arrow + gap + _text_size(minus_y, axis_pt)[0]
        pan_label_end = base_r + gap + _box_size(pan_label, small_pt)[0]
        dim_u = max(rect_r + half_m + gap + lab_w, minus_y_end, pan_label_end) + 2 * gap
        return {
            "half_m": half_m,
            "y_arrow": y_arrow,
            "rect_l": rect_l,
            "rect_r": rect_r,
            "minus_y_end": minus_y_end,
            "dim_u": dim_u,
            "left": min(
                rect_l - half_m - gap - lab_w,
                -y_arrow - gap - _text_size(plus_y, axis_pt)[0],
            ),
            "right": dim_u + tick + gap + dim_w,
        }

    lay = layout(scale)
    while lay["right"] - lay["left"] > w_px - 2 * margin:
        scale *= 0.97
        lay = layout(scale)
    half_m = lay["half_m"]

    pan_u = (w_px - (lay["right"] + lay["left"])) // 2
    top_rel = -far_x * scale - half_m - rows_above
    bottom_rel = base_back_mm * scale + gap + base_label_h
    pan_v = int(round((band_top + band_bottom - (top_rel + bottom_rel)) / 2))

    def pt(x_pan: float, y_pan: float) -> tuple[int, int]:
        return int(round(pan_u - y_pan * scale)), int(round(pan_v - x_pan * scale))

    rect_l_u, rect_r_u = pan_u + lay["rect_l"], pan_u + lay["rect_r"]
    far_v, near_v = pt(far_x, 0.0)[1], pt(near_x, 0.0)[1]
    centre_u = pt(0.0, lat_offset)[0]
    dim_u = pan_u + lay["dim_u"]

    # 4. Robot base footprint, rectangle, centre lines
    b_tl, b_br = pt(10.0, base_half_mm), pt(-base_back_mm, -base_half_mm)
    cv2.rectangle(canvas, b_tl, b_br, (245, 243, 240), -1)
    cv2.rectangle(canvas, b_tl, b_br, _MUTED, line_w, cv2.LINE_AA)
    _draw_text_box(
        canvas, base_label, (pan_u, b_br[1] + gap), small_pt, "center", "top", text_color=_MUTED
    )

    lat_v = far_v - half_m - 2 * gap
    if has_lat:
        cv2.line(canvas, (pan_u, pan_v), (pan_u, lat_v - tick), _GUIDE, thin, cv2.LINE_AA)
        cv2.line(canvas, (centre_u, far_v), (centre_u, lat_v - tick), _GUIDE, thin, cv2.LINE_AA)

    order = ["top_left", "top_right", "bottom_right", "bottom_left"]
    rect_pts = np.array([pt(*corners_pan_mm[name]) for name in order], dtype=np.int32)
    cv2.polylines(canvas, [rect_pts], True, _INK, edge_w, cv2.LINE_AA)

    # 5. Markers and their labels (beside the marker, on the outer side)
    label_right_end: dict[int, int] = {}
    for name in order:
        mu, mv = pt(*corners_pan_mm[name])
        m_img = cv2.aruco.generateImageMarker(aruco_dict, int(ids[name]), 2 * half_m)
        canvas[mv - half_m : mv + half_m, mu - half_m : mu + half_m] = cv2.cvtColor(
            m_img, cv2.COLOR_GRAY2BGR
        )
        cross = _px(2.0)
        cv2.line(canvas, (mu - cross, mv), (mu + cross, mv), _RED, line_w, cv2.LINE_AA)
        cv2.line(canvas, (mu, mv - cross), (mu, mv + cross), _RED, line_w, cv2.LINE_AA)
        on_right = mu > centre_u
        box = _draw_text_box(
            canvas,
            marker_lines[name],
            (mu + half_m + gap if on_right else mu - half_m - gap, mv),
            label_pt,
            "left" if on_right else "right",
            "center",
        )
        if on_right:
            label_right_end[mv] = box[2]

    # 6. Axes at the pan origin
    cv2.circle(canvas, (pan_u, pan_v), _px(1.5), _RED, line_w, cv2.LINE_AA)
    y_arrow = lay["y_arrow"]
    for sign, text in ((-1, plus_y), (1, minus_y)):
        tip = (pan_u + sign * y_arrow, pan_v)
        cv2.arrowedLine(canvas, (pan_u, pan_v), tip, _TEAL, line_w, cv2.LINE_AA, tipLength=0.12)
        tw, th = _text_size(text, axis_pt)
        tx = tip[0] + gap if sign > 0 else tip[0] - gap - tw
        _put_text(canvas, text, (tx, pan_v + th // 2), axis_pt, _TEAL)
    x_arrow = int(min(_px(14.0), 0.55 * (pan_v - near_v - half_m)))
    cv2.arrowedLine(
        canvas, (pan_u, pan_v), (pan_u, pan_v - x_arrow), _RED, line_w, cv2.LINE_AA, tipLength=0.15
    )
    th = _text_size(plus_x, axis_pt)[1]
    _put_text(canvas, plus_x, (pan_u + gap, pan_v - x_arrow + th // 2), axis_pt, _RED)
    _draw_text_box(
        canvas,
        pan_label,
        (pt(0.0, -base_half_mm)[0] + gap, pan_v + gap),
        small_pt,
        "left",
        "top",
        text_color=_RED,
        border_color=None,
    )

    # 7. Dimensions: near X and depth on the right, lateral offset and width on top
    ext_lines = [(pan_u + lay["minus_y_end"] + gap, pan_v)]
    ext_lines += [(end + gap, v) for v, end in label_right_end.items()]
    for start_u, v in ext_lines:
        cv2.line(canvas, (start_u, v), (dim_u + tick, v), _GUIDE, thin, cv2.LINE_AA)
    _draw_dim_arrow(canvas, (dim_u, pan_v), (dim_u, near_v), line_w)
    _draw_dim_arrow(canvas, (dim_u, near_v), (dim_u, far_v), line_w)
    for lines, v_mid in ((near_lines, (pan_v + near_v) // 2), (depth_lines, (near_v + far_v) // 2)):
        _draw_text_box(
            canvas, lines, (dim_u + gap, v_mid), dim_pt, "left", "center", True, border_color=None
        )

    if has_lat:
        _draw_dim_arrow(canvas, (pan_u, lat_v), (centre_u, lat_v), line_w)
        lat_w = _box_size(lat_lines, dim_pt, True)[0]
        lat_x = (pan_u + centre_u) // 2 - lat_w // 2
        if lat_w <= rect_r_u - rect_l_u - 2 * gap:
            lat_x = min(max(lat_x, rect_l_u + gap), rect_r_u - gap - lat_w)
        _draw_text_box(
            canvas,
            lat_lines,
            (lat_x, lat_v - gap // 2),
            dim_pt,
            "left",
            "bottom",
            True,
            border_color=None,
        )
        width_v = lat_v - lat_box_h - 2 * gap
    else:
        width_v = lat_v
    for u in (rect_l_u, rect_r_u):
        cv2.line(canvas, (u, far_v - half_m - gap // 2), (u, width_v - tick), _GUIDE, thin)
    _draw_dim_arrow(canvas, (rect_l_u, width_v), (rect_r_u, width_v), line_w)
    _draw_text_box(
        canvas,
        width_lines,
        (centre_u, width_v - gap // 2),
        dim_pt,
        "center",
        "bottom",
        True,
        border_color=None,
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

    title = f"SO-101 gripper jaw ArUco tag ({tag_size_mm:g} mm, ID {marker_id})"
    title_w = _text_size(title, 18.0, bold=True)[0]
    _put_text(canvas, title, ((w_px - title_w) // 2, int(h_px * 0.15)), 18.0, (0, 0, 0), True)
    caption = "50 mm scale line: must measure exactly 50 mm when printed at 100%"
    caption_w, caption_h = _text_size(caption, 11.0)
    _put_text(canvas, caption, ((w_px - caption_w) // 2, ry + 2 * caption_h), 11.0, (60, 60, 60))

    left = _px(25.0)
    y = ry + _px(22.0)
    _put_text(canvas, "Mounting instructions:", (left, y), 14.0, (0, 0, 0), True)
    instructions = [
        "1. Print this sheet at 100% scale (no fit-to-page / borderless).",
        "2. Verify the 50 mm line with a physical ruler.",
        f"3. Cut out the {tag_size_mm:g} mm ArUco marker along the inner border.",
        "4. Mount it on the gripper housing, facing the camera, clear of the fingers.",
        "5. Measure its centre against the grasp site and set GRIPPER_TAG_LOCAL_OFFSET_MM.",
    ]
    step = int(_text_size("H", 12.0)[1] * 2.2)
    for idx, inst in enumerate(instructions, start=1):
        _put_text(canvas, inst, (left, y + idx * step + step // 3), 12.0, (50, 50, 50))

    sheet_file = out_p / "jaw_tag_sheet.png"
    cv2.imwrite(str(sheet_file), canvas)
    logger.info(f"[ArUco] Generated jaw tag sheet: {sheet_file}")
    return sheet_file
