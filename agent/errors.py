"""Agent layer error taxonomy."""

from robot.errors import RoboticsError


class SessionError(ConnectionError, RoboticsError):
    """Raised when Gemini Live API session connection, send, or receive fails."""
