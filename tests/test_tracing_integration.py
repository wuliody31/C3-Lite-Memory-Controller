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
