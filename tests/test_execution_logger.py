"""Unit tests for the execution logger, system telemetry, and crash handler."""

import json
from pathlib import Path

import pytest

from parallax.diagnostics.execution_logger import (
    CrashReport,
    ExecutionLogger,
    capture_resource_snapshot,
    pipeline_stage,
)


def test_resource_snapshot() -> None:
    """Verify hardware and memory snapshot fields."""
    snap = capture_resource_snapshot()
    assert snap.system_ram_total_gb > 0
    assert snap.system_ram_percent >= 0.0
    assert snap.disk_free_gb >= 0.0


def test_execution_logger_file_output(tmp_path: Path) -> None:
    """Verify that ExecutionLogger appends structured events to log file."""
    log_file = tmp_path / "test_exec.log"
    crash_file = tmp_path / "test_crash.json"
    logger = ExecutionLogger(log_file=log_file, crash_file=crash_file)

    logger.log_info("Sample informational message")
    logger.log_warning("Sample warning message")
    logger.log_error("Sample error message")

    assert log_file.is_file()
    content = log_file.read_text(encoding="utf-8")
    assert "[INFO]" in content
    assert "Sample informational message" in content
    assert "[WARNING]" in content
    assert "Sample warning message" in content
    assert "[ERROR]" in content
    assert "Sample error message" in content


def test_pipeline_stage_context_manager(tmp_path: Path) -> None:
    """Verify pipeline_stage records duration and handles failures cleanly."""
    log_file = tmp_path / "test_exec.log"
    crash_file = tmp_path / "test_crash.json"
    logger = ExecutionLogger(log_file=log_file, crash_file=crash_file)

    with pipeline_stage("TestSuccessStage", logger=logger):
        x = 10 + 20
        assert x == 30

    content = log_file.read_text(encoding="utf-8")
    assert "STAGE START: [TestSuccessStage]" in content
    assert "STAGE FINISHED: [TestSuccessStage]" in content

    with (
        pytest.raises(ValueError, match="Synthetic Failure"),
        pipeline_stage("TestFailStage", logger=logger),
    ):
        raise ValueError("Synthetic Failure")

    content_after = log_file.read_text(encoding="utf-8")
    assert "STAGE START: [TestFailStage]" in content_after
    assert "STAGE FAILED: [TestFailStage] with ValueError: Synthetic Failure" in content_after


def test_crash_record_and_json_dump(tmp_path: Path) -> None:
    """Verify unhandled crash recording writes structured JSON."""
    log_file = tmp_path / "test_exec.log"
    crash_file = tmp_path / "test_crash.json"
    logger = ExecutionLogger(log_file=log_file, crash_file=crash_file)
    logger.active_stage = "UnitTestingStage"

    try:
        raise RuntimeError("Fatal unit test explosion")
    except RuntimeError as exc:
        crash = logger.record_crash(type(exc), exc, exc.__traceback__)

    assert isinstance(crash, CrashReport)
    assert crash.error_type == "RuntimeError"
    assert crash.error_message == "Fatal unit test explosion"
    assert crash.active_stage == "UnitTestingStage"
    assert "Fatal unit test explosion" in crash.traceback

    assert crash_file.is_file()
    data = json.loads(crash_file.read_text(encoding="utf-8"))
    assert data["error_type"] == "RuntimeError"
    assert data["active_stage"] == "UnitTestingStage"
    assert data["resources"]["system_ram_total_gb"] > 0
