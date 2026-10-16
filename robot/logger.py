"""Unified ISO-8601 diagnostic logging engine for GeminiRobotics-SO101."""

import datetime
import logging
import sys
from typing import Any

# Map module names to standardized bracketed module labels
MODULE_NAME_MAP: dict[str, str] = {
    "agent.vla_controller": "VLAController",
    "agent.tools": "Tools",
    "agent.live_session": "LiveSession",
    "agent.response_assembler": "ResponseAssembler",
    "robot.perception": "Perception",
    "robot.calibration": "ArucoCalibrator",
    "robot.cartesian_ik": "CartesianIK",
    "robot.real_robot": "RealSO101Robot",
    "simulation.sim_robot": "SO101Robot",
    "simulation.sim_calibration": "SimCalibration",
    "tools.calibration_sheet": "CalibrationSheet",
    "tools.setup_wizard": "SetupWizard",
    "vision.capture": "Capture",
    "vision.camera": "Camera",
    "vision.calibrator": "Calibrator",
    "vision.frames": "Frames",
    "vision.sim_runtime": "SimRuntime",
    "vision.targets": "Targets",
    "vision.overlays": "Overlays",
    "vision.patterns": "Patterns",
    "ui.app": "WebUI",
    "ui.components.chat": "Chat",
}


class TerminalFormatter(logging.Formatter):
    """Format log records byte-for-byte matching the ISO-8601 millisecond terminal standard."""

    def format(self, record: logging.LogRecord) -> str:
        """Format the specified record as [YYYY-MM-DD HH:MM:SS.mmm] [MODULE] [TAG] message.

        Args:
            record: Log record to format.

        Returns:
            Formatted log message string.
        """
        dt = datetime.datetime.fromtimestamp(record.created)
        now_str = dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        module = getattr(record, "module_name", None) or getattr(record, "component", None)
        if not module:
            module = MODULE_NAME_MAP.get(record.name, record.name.rsplit(".", 1)[-1])

        tag = getattr(record, "tag", None)
        if not tag:
            tag = (
                "ERROR"
                if record.levelno >= logging.ERROR
                else ("WARN" if record.levelno >= logging.WARNING else record.levelname)
            )

        formatted = f"[{now_str}] [{module}] [{tag}] {record.getMessage()}"
        if record.exc_info and not record.exc_text:
            record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            formatted = f"{formatted.rstrip()}\n{record.exc_text}"
        if record.stack_info:
            formatted = f"{formatted.rstrip()}\n{self.formatStack(record.stack_info)}"

        return formatted


class TerminalStreamHandler(logging.Handler):
    """Stream handler routing ERROR/WARN to sys.stderr and others to sys.stdout."""

    def emit(self, record: logging.LogRecord) -> None:
        """Format and write log record to sys.stderr (WARN/ERROR) or sys.stdout (INFO/DEBUG).

        Args:
            record: Log record to format and write.
        """
        try:
            tag = getattr(record, "tag", None)
            is_err_or_warn = record.levelno >= logging.WARNING or tag in ("ERROR", "WARN")
            stream = sys.stderr if is_err_or_warn else sys.stdout
            stream.write(self.format(record) + "\n")
            if hasattr(stream, "flush"):
                stream.flush()
        except Exception:
            self.handleError(record)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure repository-wide diagnostic logging with ISO-8601 formatting and stream routing.

    Args:
        level: Root logging level (default: logging.INFO).
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    if not any(isinstance(h, TerminalStreamHandler) for h in root_logger.handlers):
        for h in [h for h in root_logger.handlers if isinstance(h, logging.StreamHandler)]:
            root_logger.removeHandler(h)
        handler = TerminalStreamHandler()
        handler.setFormatter(TerminalFormatter())
        root_logger.addHandler(handler)

    for noisy in ("google", "urllib3", "asyncio", "PIL", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def log_with_tag(
    logger: logging.Logger,
    tag: str,
    message: str,
    level: int = logging.INFO,
    module: str | None = None,
) -> None:
    """Emit a log message with an explicit semantic tag and optional module override.

    Args:
        logger: Destination logger instance.
        tag: Semantic tag (e.g. 'EXEC', 'DECISION', 'REASONING', 'TOOL_CALL', 'TOOL_RESULT').
        message: Log message string.
        level: Standard logging level integer.
        module: Optional module label override.
    """
    extra: dict[str, Any] = {"tag": tag}
    if module is not None:
        extra["module_name"] = module

    if tag in ("ERROR",) and level < logging.ERROR:
        level = logging.ERROR
    elif tag in ("WARN",) and level < logging.WARNING:
        level = logging.WARNING

    logger.log(level, message, extra=extra)
