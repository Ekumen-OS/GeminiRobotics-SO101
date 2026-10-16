"""Model-target blackboard and visual overlay renderer."""

import threading
from typing import Any

import cv2
import numpy as np

from vision.overlays import is_target_overlay_enabled

TARGET_PALETTE_RGB: list[tuple[int, int, int]] = [
    (0, 225, 255),  # Vibrant Cyan
    (16, 185, 129),  # Emerald Green
    (245, 158, 11),  # Amber
    (168, 85, 247),  # Violet
    (244, 63, 94),  # Rose / Coral
    (56, 189, 248),  # Sky Blue
]

_model_targets: list[dict[str, Any]] = []
_model_target_lock = threading.Lock()


def set_model_targets(
    targets: list[dict[str, Any] | tuple[float, float] | list[float]] | None,
) -> None:
    """Set the active list of 2D/3D model targets/objects from the VLA model.

    Args:
        targets: List of target dicts or points, or None to clear.
    """
    global _model_targets
    with _model_target_lock:
        if not targets:
            _model_targets = []
            return

        parsed: list[dict[str, Any]] = []
        for item in targets:
            if isinstance(item, dict):
                pt = item.get("point")
                box = item.get("box_2d")
                if pt is None and box is not None and len(box) == 4:
                    pt = ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)
                if pt is not None and len(pt) >= 2:
                    parsed.append(
                        {
                            "point": (float(pt[0]), float(pt[1])),
                            "label": item.get("label"),
                            "box_2d": [float(v) for v in box] if box and len(box) == 4 else None,
                            "color": item.get("color"),
                        }
                    )
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                parsed.append(
                    {
                        "point": (float(item[0]), float(item[1])),
                        "label": None,
                        "box_2d": None,
                        "color": None,
                    }
                )
        _model_targets = parsed


def get_model_targets() -> list[dict[str, Any]]:
    """Get the list of active model targets.

    Returns:
        List of target dictionaries.
    """
    with _model_target_lock:
        return [dict(t) for t in _model_targets]


def clear_model_targets() -> None:
    """Clear all active model targets."""
    set_model_targets(None)


def draw_model_target_overlay(frame: np.ndarray) -> np.ndarray:
    """Draw clean, minimalist bounding boxes and label badges for all detected model targets.

    Args:
        frame: Input image (RGB or BGR numpy array).

    Returns:
        Frame with minimalist target box overlay drawn for all active targets.
    """
    if not is_target_overlay_enabled():
        return frame

    targets = get_model_targets()
    if not targets:
        return frame

    h, w = frame.shape[:2]
    out = frame.copy()

    # Base design tokens
    color_dark_bg = (15, 23, 42)  # Dark slate pill background
    color_text = (248, 250, 252)  # Crisp white text
    font = cv2.FONT_HERSHEY_SIMPLEX

    for idx, tgt in enumerate(targets):
        norm_y, norm_x = tgt["point"]
        px = int(np.clip((norm_x / 1000.0) * w, 0, w - 1))
        py = int(np.clip((norm_y / 1000.0) * h, 0, h - 1))
        color = tgt.get("color") or TARGET_PALETTE_RGB[idx % len(TARGET_PALETTE_RGB)]

        # Determine bounding box bounds
        box_2d = tgt.get("box_2d")
        if box_2d is not None and len(box_2d) == 4:
            ymin, xmin, ymax, xmax = box_2d
            x1 = int(np.clip((xmin / 1000.0) * w, 0, w - 1))
            y1 = int(np.clip((ymin / 1000.0) * h, 0, h - 1))
            x2 = int(np.clip((xmax / 1000.0) * w, 0, w - 1))
            y2 = int(np.clip((ymax / 1000.0) * h, 0, h - 1))
            # Ensure proper min/max ordering
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1
        else:
            half_box = 28
            x1, y1 = max(2, px - half_box), max(2, py - half_box)
            x2, y2 = min(w - 3, px + half_box), min(h - 3, py + half_box)

        # 1. Subtle translucent fill (alpha 12%)
        overlay = out.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, 0.12, out, 0.88, 0, out)

        # 2. Crisp thin minimalist outline
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)

        # 3. Refined corner brackets
        bw, bh = x2 - x1, y2 - y1
        corner_len = min(7, max(3, bw // 4), max(3, bh // 4))
        # Top-Left
        cv2.line(out, (x1, y1), (x1 + corner_len, y1), (255, 255, 255), 2, cv2.LINE_AA)
        cv2.line(out, (x1, y1), (x1, y1 + corner_len), (255, 255, 255), 2, cv2.LINE_AA)
        # Top-Right
        cv2.line(out, (x2, y1), (x2 - corner_len, y1), (255, 255, 255), 2, cv2.LINE_AA)
        cv2.line(out, (x2, y1), (x2, y1 + corner_len), (255, 255, 255), 2, cv2.LINE_AA)
        # Bottom-Left
        cv2.line(out, (x1, y2), (x1 + corner_len, y2), (255, 255, 255), 2, cv2.LINE_AA)
        cv2.line(out, (x1, y2), (x1, y2 - corner_len), (255, 255, 255), 2, cv2.LINE_AA)
        # Bottom-Right
        cv2.line(out, (x2, y2), (x2 - corner_len, y2), (255, 255, 255), 2, cv2.LINE_AA)
        cv2.line(out, (x2, y2), (x2, y2 - corner_len), (255, 255, 255), 2, cv2.LINE_AA)

        # 4. Target centroid dot
        cv2.circle(out, (px, py), 3, color_dark_bg, -1, cv2.LINE_AA)
        cv2.circle(out, (px, py), 2, color, -1, cv2.LINE_AA)

        # 5. Floating Label Badge
        label = tgt.get("label")
        tag_text = label if label else f"TARGET #{idx + 1} [{int(norm_y)}, {int(norm_x)}]"
        font_scale = 0.38
        font_thickness = 1
        (t_w, t_h), _ = cv2.getTextSize(tag_text, font, font_scale, font_thickness)

        pad_x, pad_y = 6, 4
        dot_radius = 3
        dot_space = 8
        total_badge_w = t_w + dot_space + (pad_x * 2)
        total_badge_h = t_h + (pad_y * 2)

        bx1 = max(4, min(w - total_badge_w - 4, px - total_badge_w // 2))
        if y1 - total_badge_h - 6 > 4:
            by1 = y1 - total_badge_h - 6
        else:
            by1 = min(h - total_badge_h - 4, y2 + 6)
        bx2 = bx1 + total_badge_w
        by2 = by1 + total_badge_h

        # Semi-transparent dark pill background (alpha 85%)
        badge_layer = out.copy()
        cv2.rectangle(badge_layer, (bx1, by1), (bx2, by2), color_dark_bg, -1)
        cv2.addWeighted(badge_layer, 0.85, out, 0.15, 0, out)

        # Clean 1px border with target accent color
        cv2.rectangle(out, (bx1, by1), (bx2, by2), color, 1, cv2.LINE_AA)

        # Indicator dot & text
        dot_cx = bx1 + pad_x + dot_radius
        dot_cy = by1 + pad_y + (t_h // 2)
        cv2.circle(out, (dot_cx, dot_cy), dot_radius, color, -1, cv2.LINE_AA)

        text_x = bx1 + pad_x + dot_space
        text_y = by1 + pad_y + t_h
        cv2.putText(
            out,
            tag_text,
            (text_x, text_y),
            font,
            font_scale,
            color_text,
            font_thickness,
            cv2.LINE_AA,
        )

    return out
