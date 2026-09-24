"""
Parallax Observability Module
=============================
Structured execution tracing, latency monitoring, and guardrail spans.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger("parallax.observability")


class SpanRecord(BaseModel):
    """Execution span within a trace pipeline."""

    span_name: str
    start_time: float
    duration_ms: float = 0.0
    provider: str | None = None
    guardrail_passed: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class TraceRecord(BaseModel):
    """Complete structured trace of an execution pipeline."""

    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    pipeline_name: str
    start_timestamp: float = Field(default_factory=time.time)
    total_latency_ms: float = 0.0
    success: bool = True
    spans: list[SpanRecord] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(self.model_dump(), default=str)


class PipelineTracer:
    """Manages active pipeline spans and captures structured trace metrics."""

    def __init__(self, pipeline_name: str, metadata: dict[str, Any] | None = None) -> None:
        self.record = TraceRecord(
            pipeline_name=pipeline_name,
            metadata=metadata or {},
        )
        self._start_time = time.perf_counter()

    @property
    def trace_id(self) -> str:
        return self.record.trace_id

    @contextmanager
    def span(
        self,
        name: str,
        provider: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Generator[SpanRecord, None, None]:
        """Record an atomic execution span within the trace."""
        span_obj = SpanRecord(
            span_name=name,
            start_time=time.time(),
            provider=provider,
            metadata=metadata or {},
        )
        start_perf = time.perf_counter()
        try:
            yield span_obj
        except Exception as exc:
            span_obj.guardrail_passed = False
            span_obj.error = str(exc)
            raise
        finally:
            span_obj.duration_ms = round((time.perf_counter() - start_perf) * 1000.0, 2)
            self.record.spans.append(span_obj)

    def finish(self, success: bool = True) -> TraceRecord:
        """Finalize the trace and calculate total latency."""
        self.record.total_latency_ms = round((time.perf_counter() - self._start_time) * 1000.0, 2)
        self.record.success = success
        logger.info(
            "TRACE [%s] pipeline=%s latency=%.2fms spans=%d success=%s",
            self.record.trace_id,
            self.record.pipeline_name,
            self.record.total_latency_ms,
            len(self.record.spans),
            self.record.success,
        )
        return self.record
