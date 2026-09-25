"""Runtime execution logger, system telemetry, and global crash handler for Parallax.

Provides structured logging to reports/benchmark_execution.log, stage-level latency
and memory tracking, resource threshold warnings, and unhandled exception hooks.
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
import time
import traceback
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class SystemResourceSnapshot(BaseModel):
    """Snapshot of process and system memory and storage resources."""

    rss_mb: float = Field(description="Resident Set Size memory of process in MB")
    virtual_mb: float = Field(description="Virtual memory of process in MB")
    system_ram_used_gb: float = Field(description="System used RAM in GB")
    system_ram_total_gb: float = Field(description="System total RAM in GB")
    system_ram_percent: float = Field(description="System RAM utilization percentage")
    swap_used_gb: float = Field(description="Swap used in GB")
    swap_free_gb: float = Field(description="Swap free in GB")
    disk_free_gb: float = Field(description="Root disk free space in GB")


class CrashReport(BaseModel):
    """Structured representation of an unhandled exception or critical failure."""

    timestamp: str = Field(description="ISO-8601 timestamp of failure")
    error_type: str = Field(description="Exception class name")
    error_message: str = Field(description="String description of error")
    active_stage: str = Field(default="UNKNOWN", description="Pipeline stage active at failure")
    traceback: str = Field(description="Formatted Python traceback")
    resources: SystemResourceSnapshot = Field(description="System resources at failure")


def capture_resource_snapshot() -> SystemResourceSnapshot:
    """Capture process and system hardware resource metrics without third-party dependencies."""
    rss_mb = 0.0
    vsz_mb = 0.0
    pid = os.getpid()

    # Read from /proc/<pid>/status if available on Linux
    status_path = Path(f"/proc/{pid}/status")
    if status_path.is_file():
        try:
            for line in status_path.read_text().splitlines():
                if line.startswith("VmRSS:"):
                    rss_mb = float(line.split()[1]) / 1024.0
                elif line.startswith("VmSize:"):
                    vsz_mb = float(line.split()[1]) / 1024.0
        except OSError:
            pass

    # Read system memory from /proc/meminfo
    total_ram_gb = 16.0
    avail_ram_gb = 8.0
    swap_total_gb = 0.0
    swap_free_gb = 0.0
    meminfo_path = Path("/proc/meminfo")
    if meminfo_path.is_file():
        try:
            for line in meminfo_path.read_text().splitlines():
                if line.startswith("MemTotal:"):
                    total_ram_gb = float(line.split()[1]) / (1024.0 * 1024.0)
                elif line.startswith("MemAvailable:"):
                    avail_ram_gb = float(line.split()[1]) / (1024.0 * 1024.0)
                elif line.startswith("SwapTotal:"):
                    swap_total_gb = float(line.split()[1]) / (1024.0 * 1024.0)
                elif line.startswith("SwapFree:"):
                    swap_free_gb = float(line.split()[1]) / (1024.0 * 1024.0)
        except OSError:
            pass

    used_ram_gb = max(0.0, total_ram_gb - avail_ram_gb)
    ram_pct = (used_ram_gb / total_ram_gb * 100.0) if total_ram_gb > 0 else 0.0
    swap_used_gb = max(0.0, swap_total_gb - swap_free_gb)

    # Disk space
    disk_free_gb = 0.0
    try:
        usage = shutil.disk_usage("/")
        disk_free_gb = usage.free / (1024.0**3)
    except OSError:
        pass

    return SystemResourceSnapshot(
        rss_mb=round(rss_mb, 2),
        virtual_mb=round(vsz_mb, 2),
        system_ram_used_gb=round(used_ram_gb, 2),
        system_ram_total_gb=round(total_ram_gb, 2),
        system_ram_percent=round(ram_pct, 1),
        swap_used_gb=round(swap_used_gb, 2),
        swap_free_gb=round(swap_free_gb, 2),
        disk_free_gb=round(disk_free_gb, 2),
    )


class ExecutionLogger:
    """Manages file-based structured execution logging and telemetry alerts."""

    def __init__(
        self,
        log_file: Path | str = "reports/benchmark_execution.log",
        crash_file: Path | str = "reports/benchmark_crash.json",
    ) -> None:
        self.log_file = Path(log_file)
        self.crash_file = Path(crash_file)
        self.active_stage: str = "INITIALIZATION"

        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.crash_file.parent.mkdir(parents=True, exist_ok=True)

        self._logger = logging.getLogger("ParallaxExecution")
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False

        # Clear existing handlers
        self._logger.handlers.clear()

        # File Handler
        fh = logging.FileHandler(self.log_file, mode="a", encoding="utf-8")
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        fh.setFormatter(formatter)
        self._logger.addHandler(fh)

    def log_info(self, message: str) -> None:
        """Log informational event to file."""
        self._logger.info(message)

    def log_warning(self, message: str) -> None:
        """Log warning event to file."""
        self._logger.warning(message)

    def log_error(self, message: str) -> None:
        """Log error event to file."""
        self._logger.error(message)

    def check_memory_threshold(self, threshold_percent: float = 85.0) -> None:
        """Check system RAM and log a warning if usage exceeds threshold."""
        snap = capture_resource_snapshot()
        if snap.system_ram_percent >= threshold_percent:
            self.log_warning(
                f"RAM usage high: {snap.system_ram_percent:.1f}% "
                f"({snap.system_ram_used_gb:.1f} GB / {snap.system_ram_total_gb:.1f} GB). "
                f"Process RSS: {snap.rss_mb:.1f} MB, Swap used: {snap.swap_used_gb:.1f} GB."
            )

    def record_crash(
        self,
        exc_type: type[BaseException],
        exc_value: BaseException,
        tb: Any,
    ) -> CrashReport:
        """Record an unhandled exception to both log and structured crash JSON."""
        tb_str = "".join(traceback.format_exception(exc_type, exc_value, tb))
        resources = capture_resource_snapshot()

        crash = CrashReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            error_type=exc_type.__name__,
            error_message=str(exc_value),
            active_stage=self.active_stage,
            traceback=tb_str,
            resources=resources,
        )

        msg = (
            f"\n{'=' * 70}\n"
            f"CRITICAL FAILURE IN STAGE: [{self.active_stage}]\n"
            f"Error: {crash.error_type}: {crash.error_message}\n"
            f"Resources: RSS={resources.rss_mb:.1f}MB, "
            f"System RAM={resources.system_ram_percent:.1f}%, "
            f"Disk Free={resources.disk_free_gb:.1f}GB\n"
            f"Traceback:\n{crash.traceback}"
            f"{'=' * 70}\n"
        )
        self.log_error(msg)

        try:
            self.crash_file.write_text(crash.model_dump_json(indent=2), encoding="utf-8")
        except OSError as e:
            self.log_error(f"Failed to write crash JSON: {e}")

        return crash


_GLOBAL_LOGGER: ExecutionLogger | None = None


def get_global_logger() -> ExecutionLogger:
    """Retrieve or create the singleton execution logger."""
    global _GLOBAL_LOGGER
    if _GLOBAL_LOGGER is None:
        _GLOBAL_LOGGER = ExecutionLogger()
    return _GLOBAL_LOGGER


def install_global_exception_handler(
    log_file: Path | str = "reports/benchmark_execution.log",
    crash_file: Path | str = "reports/benchmark_crash.json",
) -> ExecutionLogger:
    """Install a global uncaught exception hook into sys.excepthook."""
    logger = ExecutionLogger(log_file=log_file, crash_file=crash_file)
    global _GLOBAL_LOGGER
    _GLOBAL_LOGGER = logger

    def _handler(exc_type: type[BaseException], exc_value: BaseException, tb: Any) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            logger.log_warning("Execution interrupted by user (KeyboardInterrupt / SIGINT).")
            sys.__excepthook__(exc_type, exc_value, tb)
            return

        logger.record_crash(exc_type, exc_value, tb)
        sys.__excepthook__(exc_type, exc_value, tb)

    sys.excepthook = _handler
    return logger


@contextmanager
def pipeline_stage(
    stage_name: str,
    logger: ExecutionLogger | None = None,
) -> Generator[None, None, None]:
    """Context manager tracking stage execution time, memory footprint, and failures."""
    active_logger = logger or get_global_logger()
    active_logger.active_stage = stage_name
    t0 = time.time()
    snap_before = capture_resource_snapshot()
    active_logger.log_info(
        f"STAGE START: [{stage_name}] | Process RSS: {snap_before.rss_mb:.1f} MB, "
        f"RAM: {snap_before.system_ram_percent:.1f}%"
    )

    try:
        yield
    except Exception as exc:
        active_logger.log_error(f"STAGE FAILED: [{stage_name}] with {type(exc).__name__}: {exc}")
        raise
    finally:
        elapsed = time.time() - t0
        snap_after = capture_resource_snapshot()
        delta_rss = snap_after.rss_mb - snap_before.rss_mb
        active_logger.log_info(
            f"STAGE FINISHED: [{stage_name}] in {elapsed:.2f}s | "
            f"Process RSS: {snap_after.rss_mb:.1f} MB (delta: {delta_rss:+.1f} MB)"
        )
