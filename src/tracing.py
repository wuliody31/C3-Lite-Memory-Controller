"""Observability contracts for C3 pipeline execution results."""

from __future__ import annotations

from typing import Protocol

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
