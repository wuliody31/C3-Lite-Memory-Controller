from __future__ import annotations

from datetime import datetime
from typing import Any

from ..schemas import MemoryCandidate, MemoryType, QueryFeatures, QueryState
from ..text_utils import overlap_ratio, tokenize


def _date(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


class InMemoryMemoryStore:
    """JSON-backed deterministic store used for development, tests and converted datasets."""
    def __init__(self, records: list[dict[str, Any]], stopwords: set[str] | None = None):
        self.stopwords = stopwords or set()
        self.records = [self._candidate(x) for x in records]

    @classmethod
    def from_json_file(cls, path: str, stopwords: set[str] | None = None) -> "InMemoryMemoryStore":
        import json
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict):
            data = data.get("memories", [])
        if not isinstance(data, list):
            raise ValueError("Memory JSON must be a list or {'memories': [...]}.")
        return cls(data, stopwords)

    def retrieve(self, *, memory_type: MemoryType, state: QueryState, features: QueryFeatures, top_k: int, include_archived: bool = False) -> list[MemoryCandidate]:
        q = tokenize(features.normalised_query, self.stopwords)
        outdated = {"outdated", "superseded", "archived", "invalid"}
        matches: list[tuple[float, MemoryCandidate]] = []
        for item in self.records:
            if item.user_id != state.user_id or item.memory_type != memory_type:
                continue
            if not include_archived and item.status.lower() in outdated:
                continue
            text = " ".join(x for x in [item.text, item.subject or "", item.predicate or "", item.object_value or ""] if x)
            lexical = overlap_ratio(q, tokenize(text, self.stopwords))
            entity_hits = sum(1 for entity in features.entities if entity.lower() in text.lower())
            matches.append((lexical + min(0.5, 0.2 * entity_hits), item))
        matches.sort(key=lambda x: (x[0], x[1].confidence, x[1].timestamp or datetime.min, x[1].memory_id), reverse=True)
        return [x[1] for x in matches[:top_k]]

    @staticmethod
    def _candidate(x: dict[str, Any]) -> MemoryCandidate:
        memory_id = str(x.get("memory_id") or x.get("id") or "")
        if not memory_id:
            raise ValueError("Each memory requires memory_id or id.")
        return MemoryCandidate(
            memory_id=memory_id,
            memory_type=MemoryType(str(x["memory_type"]).lower()),
            text=str(x.get("text", "")),
            user_id=str(x.get("user_id", "")),
            timestamp=_date(x.get("timestamp") or x.get("date")),
            subject=x.get("subject"), predicate=x.get("predicate"),
            object_value=x.get("object") or x.get("object_value"),
            status=str(x.get("status", "current")), confidence=float(x.get("confidence", 1.0)),
            importance=float(x.get("importance", 0.5)), authority=str(x.get("authority", "unknown")),
            source_ids=list(x.get("source_ids") or x.get("source_episode_ids") or []),
            relations=list(x.get("relations") or []), metadata=dict(x.get("metadata") or {}),
        )

    def close(self) -> None:
        return None
