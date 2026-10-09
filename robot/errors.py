"""Core robotics error taxonomy for GeminiRobotics-SO101."""


class RoboticsError(Exception):
    """Base exception class for all robotics-layer errors."""

    def __init__(self, message: str = "", robot_moved: bool = False) -> None:
        """Initialize RoboticsError with message and robot motion state."""
        super().__init__(message)
        self.robot_moved = robot_moved


class HardwareError(RoboticsError):
    """Raised when physical robot hardware communication (bus write/read/connect) fails."""


class CalibrationUnavailable(RuntimeError, RoboticsError):
    """Raised when workspace calibration is missing, corrupt, or uninitialized.

    Subclasses RuntimeError so existing exception handlers and tests matching
    RuntimeError("Workspace is not calibrated...") continue to succeed.
    """

    def __init__(self, message: str = "", robot_moved: bool = False) -> None:
        """Initialize CalibrationUnavailable with message and robot motion state."""
        super().__init__(message)
        self.robot_moved = robot_moved


class PerceptionUnavailable(RoboticsError):
    """Raised when camera frames are missing/undecodable or visual perception fails."""


class ToolArgumentError(RoboticsError):
    """Raised when invalid tool parameters (e.g. joints, coordinates, presets) are supplied."""
