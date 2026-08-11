from pathlib import Path
from typing import Any

import pytest

from src.backbones import MockBackbone
from src.config import load_config
from src.errors import ResourceCleanupError
from src.pipeline import C3Pipeline


ROOT = Path(__file__).resolve().parents[1]


class TrackingStore:
    def __init__(
        self,
        *,
        close_error: Exception | None = None,
    ) -> None:
        self.close_error = close_error
        self.close_count = 0

    def retrieve(self, **kwargs: Any) -> list[Any]:
        del kwargs
        return []

    def close(self) -> None:
        self.close_count += 1

        if self.close_error is not None:
            raise self.close_error


class TrackingTraceSink:
    def __init__(self) -> None:
        self.close_count = 0

    def emit(self, result: Any) -> None:
        del result

    def close(self) -> None:
        self.close_count += 1


def build_cleanup_pipeline(
    *,
    memory_store: TrackingStore,
    procedure_store: TrackingStore | None,
    trace_sink: TrackingTraceSink,
) -> C3Pipeline:
    config = load_config(
        ROOT / "configs/c3_lite_v2_2_final.yaml"
    )

    return C3Pipeline(
        config=config,
        memory_store=memory_store,
        procedure_store=procedure_store,
        backbone=MockBackbone(),
        prompt_template=(
            ROOT / "prompts/answer_prompt.txt"
        ),
        trace_sink=trace_sink,
    )


def test_cleanup_attempts_remaining_resources_after_failure() -> None:
    memory = TrackingStore(
        close_error=OSError(
            "memory close failed"
        )
    )
    procedure = TrackingStore()
    trace = TrackingTraceSink()

    pipeline = build_cleanup_pipeline(
        memory_store=memory,
        procedure_store=procedure,
        trace_sink=trace,
    )

    with pytest.raises(
        ResourceCleanupError,
        match="memory_store",
    ):
        pipeline.close()

    assert memory.close_count == 1
    assert procedure.close_count == 1
    assert trace.close_count == 1


def test_pipeline_close_is_idempotent_after_success() -> None:
    memory = TrackingStore()
    procedure = TrackingStore()
    trace = TrackingTraceSink()

    pipeline = build_cleanup_pipeline(
        memory_store=memory,
        procedure_store=procedure,
        trace_sink=trace,
    )

    pipeline.close()
    pipeline.close()

    assert memory.close_count == 1
    assert procedure.close_count == 1
    assert trace.close_count == 1


def test_pipeline_close_is_idempotent_after_failed_attempt() -> None:
    memory = TrackingStore(
        close_error=RuntimeError(
            "simulated cleanup failure"
        )
    )
    trace = TrackingTraceSink()

    pipeline = build_cleanup_pipeline(
        memory_store=memory,
        procedure_store=None,
        trace_sink=trace,
    )

    with pytest.raises(ResourceCleanupError):
        pipeline.close()

    pipeline.close()

    assert memory.close_count == 1
    assert trace.close_count == 1
