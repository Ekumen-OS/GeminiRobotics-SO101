"""Unified ISO-8601 diagnostic logging engine for GeminiRobotics-SO101."""

import datetime
import logging
import sys
from typing import Any

# Map module names and logger identifiers to standardized bracketed module labels
MODULE_NAME_MAP: dict[str, str] = {
    "agent.vla_controller": "VLAController",
    "vla_controller": "VLAController",
    "VLAController": "VLAController",
    "agent.tools": "Tools",
    "tools": "Tools",
    "Tools": "Tools",
    "agent.live_session": "LiveSession",
    "live_session": "LiveSession",
    "LiveSession": "LiveSession",
    "agent.response_assembler": "ResponseAssembler",
    "response_assembler": "ResponseAssembler",
    "ResponseAssembler": "ResponseAssembler",
    "robot.perception": "Perception",
    "perception": "Perception",
    "Perception": "Perception",
    "robot.calibration": "ArucoCalibrator",
    "calibration": "ArucoCalibrator",
    "ArucoCalibrator": "ArucoCalibrator",
    "robot.cartesian_ik": "CartesianIK",
    "cartesian_ik": "CartesianIK",
    "CartesianIK": "CartesianIK",
    "robot.real_robot": "RealSO101Robot",
    "real_robot": "RealSO101Robot",
    "RealSO101Robot": "RealSO101Robot",
    "simulation.sim_robot": "SO101Robot",
    "sim_robot": "SO101Robot",
    "SO101Robot": "SO101Robot",
    "simulation.sim_calibration": "SimCalibration",
    "sim_calibration": "SimCalibration",
    "tools.calibration_sheet": "CalibrationSheet",
    "calibration_sheet": "CalibrationSheet",
    "CalibrationSheet": "CalibrationSheet",
    "tools.setup_wizard": "SetupWizard",
    "setup_wizard": "SetupWizard",
    "SetupWizard": "SetupWizard",
    "vision.capture": "Capture",
    "vision.camera": "Camera",
    "vision.calibrator": "Calibrator",
    "vision.frames": "Frames",
    "vision.sim_runtime": "SimRuntime",
    "vision.targets": "Targets",
    "vision.overlays": "Overlays",
    "vision.patterns": "Patterns",
    "ui.app": "WebUI",
    "app": "WebUI",
    "WebUI": "WebUI",
    "ui.components.chat": "Chat",
    "chat": "Chat",
    "Chat": "Chat",
}


class TerminalFormatter(logging.Formatter):
    """Format log records byte-for-byte matching the ISO-8601 millisecond terminal standard."""

    def format(self, record: logging.LogRecord) -> str:
        """Format the specified record as text.

        Format: [YYYY-MM-DD HH:MM:SS.mmm] [MODULE] [TAG] message
        """
        # Timestamp with millisecond precision
        dt = datetime.datetime.fromtimestamp(record.created)
        now_str = dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        # Resolve standardized module name
        module = getattr(record, "module_name", None) or getattr(record, "component", None)
        if not module:
            module = MODULE_NAME_MAP.get(record.name)
            if not module:
                if "." in record.name:
                    last_token = record.name.split(".")[-1]
                    module = MODULE_NAME_MAP.get(last_token, last_token)
                else:
                    module = record.name

        # Resolve semantic tag
        tag = getattr(record, "tag", None)
        if not tag:
            if record.levelno >= logging.ERROR:
                tag = "ERROR"
            elif record.levelno >= logging.WARNING:
                tag = "WARN"
            else:
                tag = record.levelname

        message = record.getMessage()
        formatted = f"[{now_str}] [{module}] [{tag}] {message}"

        if record.exc_info:
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            if not formatted.endswith("\n"):
                formatted += "\n"
            formatted += record.exc_text
        if record.stack_info:
            if not formatted.endswith("\n"):
                formatted += "\n"
            formatted += self.formatStack(record.stack_info)

        return formatted


class TerminalStreamHandler(logging.Handler):
    """Stream handler routing ERROR/WARN to sys.stderr and others to sys.stdout.

    Dynamically references sys.stdout and sys.stderr at emit time to ensure
    seamless interoperability with pytest's capsys fixture.
    """

    def emit(self, record: logging.LogRecord) -> None:
        """Format and write log record to the appropriate output stream."""
        try:
            custom_stream = getattr(record, "stream", None)
            tag = getattr(record, "tag", None)
            is_err_or_warn = record.levelno >= logging.WARNING or tag in ("ERROR", "WARN")

            if custom_stream is not None:
                stream = custom_stream
            elif is_err_or_warn:
                stream = sys.stderr
            else:
                stream = sys.stdout

            msg = self.format(record)
            stream.write(msg + "\n")
            if hasattr(stream, "flush"):
                stream.flush()
        except Exception:
            self.handleError(record)


_logging_initialized = False


def setup_logging(level: int = logging.INFO) -> None:
    """Configure repository-wide diagnostic logging with ISO-8601 formatting and stream routing.

    Args:
        level: Root logging level (default: logging.INFO).
    """
    global _logging_initialized

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    if not any(isinstance(h, TerminalStreamHandler) for h in root_logger.handlers):
        # Remove pre-existing standard stream handlers to avoid duplicate output
        handlers_to_remove = [
            h for h in root_logger.handlers if isinstance(h, logging.StreamHandler)
        ]
        for h in handlers_to_remove:
            root_logger.removeHandler(h)

        handler = TerminalStreamHandler()
        handler.setFormatter(TerminalFormatter())
        root_logger.addHandler(handler)

    # Silence verbose third-party loggers
    for noisy in ("google", "urllib3", "asyncio", "PIL", "matplotlib", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _logging_initialized = True


def log_with_tag(
    logger: logging.Logger,
    tag: str,
    message: str,
    level: int = logging.INFO,
    module: str | None = None,
    stream: Any = None,
) -> None:
    """Emit a log message with an explicit semantic tag and module override.

    Args:
        logger: Destination logger instance.
        tag: Semantic tag (e.g. 'EXEC', 'DECISION', 'REASONING', 'TOOL_CALL', 'TOOL_RESULT').
        message: Log message string.
        level: Standard logging level integer.
        module: Optional module label override.
        stream: Optional custom stream override (e.g. StringIO buffer).
    """
    extra: dict[str, Any] = {"tag": tag}
    if module is not None:
        extra["module_name"] = module
    if stream is not None:
        extra["stream"] = stream

    # Adjust default level if tag represents an error/warning
    if tag in ("ERROR",) and level < logging.ERROR:
        level = logging.ERROR
    elif tag in ("WARN",) and level < logging.WARNING:
        level = logging.WARNING

    logger.log(level, message, extra=extra)


def log_compat(module: str, tag: str, message: str, stream: Any = None) -> None:
    """Backward compatibility log function reproducing previous _log behavior.

    Args:
        module: Component name (e.g. 'Tools', 'Perception').
        tag: Semantic tag (e.g. 'INFO', 'WARN', 'ERROR').
        message: Log message content.
        stream: Optional target stream.
    """
    logger = logging.getLogger(module)
    level = (
        logging.WARNING if tag == "WARN" else (logging.ERROR if tag in ("ERROR",) else logging.INFO)
    )
    log_with_tag(logger, tag=tag, message=message, level=level, module=module, stream=stream)
