"""Visual overlay toggle states and accessors for live camera streams."""

_calibration_overlay_enabled: bool = False
_bounding_boxes_overlay_enabled: bool = False
_grid_overlay_enabled: bool = False
_crosshair_overlay_enabled: bool = False
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


def set_bounding_boxes_overlay(enabled: bool) -> None:
    """Enable or disable the bounding boxes overlay on live camera feeds.

    Args:
        enabled: True to enable overlay drawing, False to disable.
    """
    global _bounding_boxes_overlay_enabled
    _bounding_boxes_overlay_enabled = bool(enabled)


def is_bounding_boxes_overlay_enabled() -> bool:
    """Check if the bounding boxes overlay is currently enabled.

    Returns:
        True if overlay is active, False otherwise.
    """
    return _bounding_boxes_overlay_enabled


def set_grid_overlay(enabled: bool) -> None:
    """Enable or disable the workspace grid overlay on live camera feeds.

    Args:
        enabled: True to enable overlay drawing, False to disable.
    """
    global _grid_overlay_enabled
    _grid_overlay_enabled = bool(enabled)


def is_grid_overlay_enabled() -> bool:
    """Check if the workspace grid overlay is currently enabled.

    Returns:
        True if overlay is active, False otherwise.
    """
    return _grid_overlay_enabled


def set_crosshair_overlay(enabled: bool) -> None:
    """Enable or disable the crosshair / center marker overlay on live camera feeds.

    Args:
        enabled: True to enable overlay drawing, False to disable.
    """
    global _crosshair_overlay_enabled
    _crosshair_overlay_enabled = bool(enabled)


def is_crosshair_overlay_enabled() -> bool:
    """Check if the crosshair overlay is currently enabled.

    Returns:
        True if overlay is active, False otherwise.
    """
    return _crosshair_overlay_enabled


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
