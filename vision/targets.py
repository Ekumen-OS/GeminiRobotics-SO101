"""Model-target blackboard and visual overlay renderer."""

import threading
from typing import Any, Sequence

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
    targets: Sequence[dict[str, Any] | tuple[float, float] | list[float]] | None,
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
                            "active": bool(item.get("active", False)),
                        }
                    )
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                parsed.append(
                    {
                        "point": (float(item[0]), float(item[1])),
                        "label": None,
                        "box_2d": None,
                        "color": None,
                        "active": False,
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
    """Draw minimal target pointers for all detected model targets.

    Active targets are drawn at full opacity; stale targets are alpha-blended at ~50%
    opacity. Destination transport targets render as a small open square marker rather
    than a circular ring.

    Args:
        frame: Input image (RGB or BGR numpy array).

    Returns:
        Frame with minimal target pointers drawn for active and stale targets.
    """
    if not is_target_overlay_enabled():
        return frame

    targets = get_model_targets()
    if not targets:
        return frame

    h, w = frame.shape[:2]
    out = frame.copy()

    scale = w / 640.0
    ring_radius = max(4, int(round(7.0 * scale)))
    dot_radius = max(1, int(round(2.0 * scale)))
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.30, 0.35 * scale)
    font_thickness = 1

    # Draw stale targets first so active target renders crisply on top if overlapping
    indexed_targets = sorted(
        enumerate(targets),
        key=lambda pair: bool(pair[1].get("active", False)),
    )

    for idx, tgt in indexed_targets:
        norm_y, norm_x = tgt["point"]
        px = int(np.clip((norm_x / 1000.0) * w, 0, w - 1))
        py = int(np.clip((norm_y / 1000.0) * h, 0, h - 1))
        color = tgt.get("color") or TARGET_PALETTE_RGB[idx % len(TARGET_PALETTE_RGB)]
        is_active = bool(tgt.get("active", False))

        raw_label = tgt.get("label")
        label_text = str(raw_label).strip().lower() if raw_label else ""
        is_destination = "destination" in label_text

        # Inactive targets draw on a layer and get alpha-blended into out
        target_layer = out if is_active else out.copy()

        # 1. Marker shape
        if is_destination:
            sq_half = ring_radius
            top_left = (max(0, px - sq_half), max(0, py - sq_half))
            bottom_right = (min(w - 1, px + sq_half), min(h - 1, py + sq_half))
            cv2.rectangle(target_layer, top_left, bottom_right, color, 1, cv2.LINE_AA)
            cv2.circle(target_layer, (px, py), dot_radius, color, -1, cv2.LINE_AA)
        else:
            cv2.circle(target_layer, (px, py), ring_radius, color, 1, cv2.LINE_AA)
            cv2.circle(target_layer, (px, py), dot_radius, color, -1, cv2.LINE_AA)

        # 2. Minimal label: one line lowercase text, offset up and right of marker
        if label_text:
            (tw, th), _ = cv2.getTextSize(label_text, font, font_scale, font_thickness)
            offset_x = ring_radius + int(4 * scale)
            offset_y = -ring_radius + int(2 * scale)
            text_x = px + offset_x
            text_y = py + offset_y

            # Clamp text position within frame boundaries
            if text_x + tw > w - 2:
                text_x = max(2, w - 2 - tw)
            if text_y - th < 2:
                text_y = th + 2
            if text_y > h - 2:
                text_y = h - 2

            # Thin dark outline for legibility
            outline_color = (0, 0, 0)
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                cv2.putText(
                    target_layer,
                    label_text,
                    (text_x + dx, text_y + dy),
                    font,
                    font_scale,
                    outline_color,
                    font_thickness,
                    cv2.LINE_AA,
                )
            # Text in accent color
            cv2.putText(
                target_layer,
                label_text,
                (text_x, text_y),
                font,
                font_scale,
                color,
                font_thickness,
                cv2.LINE_AA,
            )

        # Alpha-blend stale target at ~50% opacity
        if not is_active:
            cv2.addWeighted(target_layer, 0.5, out, 0.5, 0, out)

    return out
