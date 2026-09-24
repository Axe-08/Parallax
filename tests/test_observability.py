"""Unit tests for Parallax observability and tracing."""

import time

from parallax.observability.tracer import PipelineTracer


def test_pipeline_tracer_records_spans():
    tracer = PipelineTracer("test_pipeline", metadata={"env": "test"})
    assert tracer.trace_id is not None

    with tracer.span("step_1", provider="unit_test"):
        time.sleep(0.01)

    with tracer.span("step_2", metadata={"items": 5}):
        pass

    record = tracer.finish(success=True)
    assert record.pipeline_name == "test_pipeline"
    assert record.success is True
    assert len(record.spans) == 2
    assert record.spans[0].span_name == "step_1"
    assert record.spans[0].duration_ms > 0
    assert record.spans[0].provider == "unit_test"
    assert record.spans[1].metadata["items"] == 5
