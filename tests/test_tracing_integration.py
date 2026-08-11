import json
from pathlib import Path

from src.backbones import MockBackbone
from src.config import load_config
from src.pipeline import C3Pipeline
from src.retrievers import (
    InMemoryMemoryStore,
    ProceduralJsonStore,
)
from src.schemas import C3Result, QueryState
from src.tracing import JsonlTraceSink, TelemetryRecord


ROOT = Path(__file__).resolve().parents[1]


class RecordingTraceSink:
    def __init__(self) -> None:
        self.results: list[C3Result] = []
        self.closed = False

    def emit(self, result: C3Result) -> None:
        self.results.append(result)

    def close(self) -> None:
        self.closed = True


def build_pipeline(
    *,
    trace_sink=None,
) -> C3Pipeline:
    config = load_config(
        ROOT / "configs/c3_lite_v2_2_final.yaml"
    )
    stopwords = set(
        config["query_analysis"]["stopwords"]
    )

    return C3Pipeline(
        config=config,
        memory_store=InMemoryMemoryStore.from_json_file(
            str(ROOT / "examples/demo_memories.json"),
            stopwords,
        ),
        procedure_store=ProceduralJsonStore.from_file(
            str(ROOT / "examples/demo_procedures.json"),
            stopwords,
        ),
        backbone=MockBackbone(),
        prompt_template=(
            ROOT / "prompts/answer_prompt.txt"
        ),
        trace_sink=trace_sink,
    )


def test_pipeline_emits_returned_result_once() -> None:
    sink = RecordingTraceSink()
    pipeline = build_pipeline(trace_sink=sink)

    result = pipeline.answer(
        QueryState(
            "What is my current MSc project scope?",
            "user01",
        )
    )

    assert len(sink.results) == 1
    assert sink.results[0] is result

    pipeline.close()
    assert sink.closed is True


def test_telemetry_record_excludes_sensitive_payload() -> None:
    pipeline = build_pipeline()

    result = pipeline.answer(
        QueryState(
            "What is my current MSc project scope?",
            "user01",
        )
    )

    record = TelemetryRecord.from_result(result)
    payload = record.to_dict()

    forbidden_keys = {
        "query",
        "user_id",
        "answer",
        "final_prompt",
        "selected_evidence",
        "debug",
    }

    assert forbidden_keys.isdisjoint(payload)
    assert payload["selected_count"] == len(
        result.selected_ids
    )
    assert payload["schema_version"] == "1.0"

    pipeline.close()


def test_jsonl_sink_writes_safe_single_line(
    tmp_path: Path,
) -> None:
    trace_path = (
        tmp_path
        / "telemetry"
        / "c3.jsonl"
    )

    sink = JsonlTraceSink(trace_path)
    pipeline = build_pipeline(trace_sink=sink)

    result = pipeline.answer(
        QueryState(
            "What is my current MSc project scope?",
            "user01",
        )
    )

    pipeline.close()

    lines = trace_path.read_text(
        encoding="utf-8"
    ).splitlines()

    assert len(lines) == 1

    payload = json.loads(lines[0])

    assert payload["schema_version"] == "1.0"
    assert payload["decision"] == result.decision.value
    assert payload["selected_ids"] == result.selected_ids
    assert payload["selected_count"] == len(
        result.selected_ids
    )

    forbidden_keys = {
        "query",
        "user_id",
        "answer",
        "final_prompt",
        "selected_evidence",
        "debug",
    }

    assert forbidden_keys.isdisjoint(payload)

    assert result.query not in lines[0]
    assert result.final_prompt not in lines[0]


class EmitFailingTraceSink:
    def emit(self, result: C3Result) -> None:
        del result
        raise RuntimeError("telemetry backend unavailable")

    def close(self) -> None:
        pass


class CloseFailingTraceSink:
    def emit(self, result: C3Result) -> None:
        del result

    def close(self) -> None:
        raise RuntimeError("telemetry close failed")


def test_trace_emit_failure_does_not_break_answer() -> None:
    from src.tracing import BestEffortTraceSink

    pipeline = build_pipeline(
        trace_sink=EmitFailingTraceSink()
    )

    result = pipeline.answer(
        QueryState(
            "What is my current MSc project scope?",
            "user01",
        )
    )

    assert isinstance(
        pipeline.trace_sink,
        BestEffortTraceSink,
    )
    assert pipeline.trace_sink.emit_failure_count == 1
    assert pipeline.trace_sink.last_error is not None
    assert "RuntimeError" in pipeline.trace_sink.last_error

    assert result.query
    assert result.decision is not None

    pipeline.close()


def test_trace_close_failure_is_isolated() -> None:
    from src.tracing import BestEffortTraceSink

    pipeline = build_pipeline(
        trace_sink=CloseFailingTraceSink()
    )

    assert isinstance(
        pipeline.trace_sink,
        BestEffortTraceSink,
    )

    pipeline.close()

    assert pipeline.trace_sink.close_failure_count == 1
    assert pipeline.trace_sink.last_error is not None
    assert "RuntimeError" in pipeline.trace_sink.last_error


def test_closed_jsonl_sink_raises_observability_error(
    tmp_path: Path,
) -> None:
    from src.errors import ObservabilityError

    sink = JsonlTraceSink(
        tmp_path / "closed.jsonl"
    )
    sink.close()

    pipeline = build_pipeline()

    result = pipeline.answer(
        QueryState(
            "What is my current MSc project scope?",
            "user01",
        )
    )

    try:
        sink.emit(result)
    except ObservabilityError as exc:
        assert isinstance(exc, RuntimeError)
    else:
        raise AssertionError(
            "Expected ObservabilityError."
        )

    pipeline.close()
