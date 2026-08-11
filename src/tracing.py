"""Observability contracts for C3 pipeline execution results."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, TextIO
from uuid import uuid4

from .errors import ObservabilityError
from .schemas import C3Result


class TraceSink(Protocol):
    """Receives completed C3 execution results for observability."""

    def emit(self, result: C3Result) -> None:
        ...

    def close(self) -> None:
        ...


class NullTraceSink:
    """Default sink preserving historical pipeline behaviour."""

    def emit(self, result: C3Result) -> None:
        del result

    def close(self) -> None:
        pass


class BestEffortTraceSink:
    """Isolate observability failures from the C3 answer path.

    The wrapped sink is allowed to fail without invalidating an otherwise
    successful pipeline result. Failure counters remain available for
    operational inspection.
    """

    def __init__(self, sink: TraceSink) -> None:
        self.sink = sink
        self.emit_failure_count = 0
        self.close_failure_count = 0
        self.last_error: str | None = None

    def emit(self, result: C3Result) -> None:
        try:
            self.sink.emit(result)
        except Exception as exc:
            self.emit_failure_count += 1
            self.last_error = (
                f"{type(exc).__name__}: {exc}"
            )

    def close(self) -> None:
        try:
            self.sink.close()
        except Exception as exc:
            self.close_failure_count += 1
            self.last_error = (
                f"{type(exc).__name__}: {exc}"
            )


@dataclass(frozen=True, slots=True)
class TelemetryRecord:
    """Privacy-conscious projection of one completed C3 execution."""

    schema_version: str
    event_id: str
    timestamp_utc: str
    decision: str
    query_mode: str
    selected_memory_types: tuple[str, ...]
    raw_retrieved_count: int
    ranked_candidate_count: int
    selected_count: int
    selected_ids: tuple[str, ...]
    coverage: float
    adequacy: float
    agreement: float
    latency_ms: float
    input_tokens: int | None
    output_tokens: int | None

    @classmethod
    def from_result(cls, result: C3Result) -> "TelemetryRecord":
        raw_ids = (
            result.raw_retrieved_ids
            if result.raw_retrieved_ids
            else result.retrieved_ids
        )
        ranked_ids = (
            result.ranked_candidate_ids
            if result.ranked_candidate_ids
            else result.retrieved_ids
        )

        return cls(
            schema_version="1.0",
            event_id=str(uuid4()),
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            decision=result.decision.value,
            query_mode=result.query_mode.value,
            selected_memory_types=tuple(
                result.selected_memory_types
            ),
            raw_retrieved_count=len(raw_ids),
            ranked_candidate_count=len(ranked_ids),
            selected_count=len(result.selected_ids),
            selected_ids=tuple(result.selected_ids),
            coverage=result.coverage,
            adequacy=result.adequacy,
            agreement=result.agreement,
            latency_ms=result.latency_ms,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "event_id": self.event_id,
            "timestamp_utc": self.timestamp_utc,
            "decision": self.decision,
            "query_mode": self.query_mode,
            "selected_memory_types": list(
                self.selected_memory_types
            ),
            "raw_retrieved_count": self.raw_retrieved_count,
            "ranked_candidate_count": self.ranked_candidate_count,
            "selected_count": self.selected_count,
            "selected_ids": list(self.selected_ids),
            "coverage": self.coverage,
            "adequacy": self.adequacy,
            "agreement": self.agreement,
            "latency_ms": self.latency_ms,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }


class JsonlTraceSink:
    """Append safe C3 telemetry records to a UTF-8 JSONL file.

    This sink intentionally persists a restricted telemetry projection
    rather than the full C3Result payload.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        self._handle: TextIO = self.path.open(
            "a",
            encoding="utf-8",
        )

    def emit(self, result: C3Result) -> None:
        if self._handle.closed:
            raise ObservabilityError(
                "Cannot emit telemetry to a closed JsonlTraceSink."
            )

        record = TelemetryRecord.from_result(result)

        json.dump(
            record.to_dict(),
            self._handle,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        self._handle.write("\n")
        self._handle.flush()

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()
