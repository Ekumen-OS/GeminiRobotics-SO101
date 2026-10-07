"""Thread-safe active robot controller registry."""

import threading

from robot.interface import RobotInterface

_state_lock = threading.Lock()
_active_robot: RobotInterface | None = None


def set_active_robot(robot: RobotInterface | None) -> None:
    """Set the globally active robot controller reference.

    Args:
        robot: Initialized robot hardware or simulation controller instance.
    """
    global _active_robot
    with _state_lock:
        _active_robot = robot


def get_active_robot() -> RobotInterface | None:
    """Retrieve the globally active robot controller reference.

    Returns:
        The active RobotInterface instance (hardware or simulation) if set,
        or None if no robot is currently active.
    """
    global _active_robot
    with _state_lock:
        return _active_robot
