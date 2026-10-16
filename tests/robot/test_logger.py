"""Unit tests for robot/logger.py ISO-8601 logging engine and stream routing."""

import logging
import re

import pytest

from robot.logger import (
    MODULE_NAME_MAP,
    TerminalFormatter,
    TerminalStreamHandler,
    log_with_tag,
    setup_logging,
)

LOG_PATTERN = re.compile(
    r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}\] "
    r"\[(?P<module>[^\]]+)\] \[(?P<tag>[^\]]+)\] (?P<msg>.*)$"
)


@pytest.fixture(autouse=True)
def clean_logging_environment():
    """Ensure setup_logging is cleanly active for each test."""
    setup_logging(level=logging.INFO)
    yield
    setup_logging(level=logging.INFO)


def test_setup_logging_idempotent() -> None:
    """Calling setup_logging multiple times should not duplicate TerminalStreamHandler."""
    setup_logging(level=logging.INFO)
    setup_logging(level=logging.INFO)
    root = logging.getLogger()
    terminal_handlers = [h for h in root.handlers if isinstance(h, TerminalStreamHandler)]
    assert len(terminal_handlers) == 1
    assert isinstance(terminal_handlers[0].formatter, TerminalFormatter)


def test_stdout_routing_for_info(capsys: pytest.CaptureFixture[str]) -> None:
    """Informational logs should be emitted to sys.stdout and match standard format."""
    logger = logging.getLogger("agent.tools")
    logger.info("Executing pick sequence")

    captured = capsys.readouterr()
    assert "Executing pick sequence" in captured.out
    assert captured.err == ""

    match = LOG_PATTERN.match(captured.out.strip())
    assert match is not None
    assert match.group("module") == "Tools"
    assert match.group("tag") == "INFO"
    assert match.group("msg") == "Executing pick sequence"


def test_stderr_routing_for_warning(capsys: pytest.CaptureFixture[str]) -> None:
    """Warning logs should be routed to sys.stderr."""
    logger = logging.getLogger("robot.real_robot")
    logger.warning("Motor temperature elevated")

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Motor temperature elevated" in captured.err

    match = LOG_PATTERN.match(captured.err.strip())
    assert match is not None
    assert match.group("module") == "RealSO101Robot"
    assert match.group("tag") == "WARN"
    assert match.group("msg") == "Motor temperature elevated"


def test_stderr_routing_for_error(capsys: pytest.CaptureFixture[str]) -> None:
    """Error logs should be routed to sys.stderr."""
    logger = logging.getLogger("agent.vla_controller")
    logger.error("Controller loop failure")

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Controller loop failure" in captured.err

    match = LOG_PATTERN.match(captured.err.strip())
    assert match is not None
    assert match.group("module") == "VLAController"
    assert match.group("tag") == "ERROR"


def test_log_with_tag_custom_tags(capsys: pytest.CaptureFixture[str]) -> None:
    """Semantic tags should be preserved in brackets and routed according to tag severity."""
    logger = logging.getLogger("agent.vla_controller")

    log_with_tag(logger, "EXEC", "Dispatched trajectory waypoint")
    captured = capsys.readouterr()
    assert "Dispatched trajectory waypoint" in captured.out
    assert captured.err == ""
    match = LOG_PATTERN.match(captured.out.strip())
    assert match is not None
    assert match.group("tag") == "EXEC"
    assert match.group("module") == "VLAController"

    log_with_tag(logger, "WARN", "Divergence detected in solver")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Divergence detected in solver" in captured.err
    match = LOG_PATTERN.match(captured.err.strip())
    assert match is not None
    assert match.group("tag") == "WARN"

    log_with_tag(logger, "ERROR", "IK unreachable target")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "IK unreachable target" in captured.err
    match = LOG_PATTERN.match(captured.err.strip())
    assert match is not None
    assert match.group("tag") == "ERROR"


def test_module_name_mapping(capsys: pytest.CaptureFixture[str]) -> None:
    """Modules in MODULE_NAME_MAP should resolve to their mapped bracket labels."""
    for module_name, expected_label in MODULE_NAME_MAP.items():
        logger = logging.getLogger(module_name)
        logger.info("Probe module name")
        captured = capsys.readouterr()
        match = LOG_PATTERN.match(captured.out.strip())
        assert match is not None
        assert match.group("module") == expected_label, (
            f"Expected {expected_label} for {module_name}, got {match.group('module')}"
        )


def test_unmapped_module_fallback(capsys: pytest.CaptureFixture[str]) -> None:
    """Unmapped module names should fall back to their trailing token."""
    logger = logging.getLogger("unmapped.subsystem.worker")
    logger.info("Worker heartbeat")

    captured = capsys.readouterr()
    match = LOG_PATTERN.match(captured.out.strip())
    assert match is not None
    assert match.group("module") == "worker"

    single_token_logger = logging.getLogger("standalone")
    single_token_logger.info("Standalone event")
    captured = capsys.readouterr()
    match = LOG_PATTERN.match(captured.out.strip())
    assert match is not None
    assert match.group("module") == "standalone"


def test_log_with_tag_module_override(capsys: pytest.CaptureFixture[str]) -> None:
    """Passing module override should display that override as the bracketed module."""
    logger = logging.getLogger("robot.calibration")
    log_with_tag(logger, "CALIB", "Tag calibration step", module="CustomCalibrator")

    captured = capsys.readouterr()
    match = LOG_PATTERN.match(captured.out.strip())
    assert match is not None
    assert match.group("module") == "CustomCalibrator"
    assert match.group("tag") == "CALIB"


def test_exception_formatting(capsys: pytest.CaptureFixture[str]) -> None:
    """Exceptions should include formatted traceback in the log output."""
    logger = logging.getLogger("robot.real_robot")
    try:
        raise ValueError("Simulated sensor failure")
    except ValueError:
        logger.exception("Error during read")

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Error during read" in captured.err
    assert "Traceback (most recent call last):" in captured.err
    assert "ValueError: Simulated sensor failure" in captured.err
