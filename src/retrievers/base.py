from __future__ import annotations

from typing import Protocol
from ..schemas import MemoryCandidate, MemoryType, QueryFeatures, QueryState


class MemoryStore(Protocol):
    def retrieve(self, *, memory_type: MemoryType, state: QueryState, features: QueryFeatures, top_k: int, include_archived: bool = False) -> list[MemoryCandidate]: ...
    def close(self) -> None: ...
