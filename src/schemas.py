from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Iterable


class MemoryType(str, Enum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


class QueryMode(str, Enum):
    CURRENT = "current_state"
    HISTORICAL = "historical_state"
    TIMELINE = "timeline"
    PROCEDURAL = "procedural"
    ATEMPORAL = "atemporal"


class AnswerDecision(str, Enum):
    DIRECT = "direct"
    CAVEAT = "caveat"
    ABSTAIN = "abstain"


@dataclass(slots=True)
class QueryState:
    query: str
    user_id: str
    current_time: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    session_id: str | None = None
    recent_context: list[dict[str, Any]] = field(
        default_factory=list
    )


@dataclass(slots=True)
class QueryFeatures:
    normalised_query: str
    tokens: list[str]
    entities: list[str]
    temporal_expressions: list[str]
    query_mode: QueryMode
    task_type: str | None
    asks_current_state: bool
    asks_historical_state: bool
    asks_timeline: bool
    asks_procedure: bool
    asks_explanation: bool
    asks_conflict: bool
    information_needs: list[str]


@dataclass(slots=True)
class RouteDecision:
    selected_types: list[MemoryType]
    scores: dict[str, float]
    reasons: dict[str, list[str]]

    @property
    def confidence(self) -> float:
        if not self.selected_types:
            return 0.0
        return max(
            self.scores.get(memory_type.value, 0.0)
            for memory_type in self.selected_types
        )


@dataclass(slots=True)
class MemoryCandidate:
    memory_id: str
    memory_type: MemoryType
    text: str
    user_id: str
    timestamp: datetime | None = None
    subject: str | None = None
    predicate: str | None = None
    object_value: str | None = None
    status: str = "current"
    confidence: float = 1.0
    importance: float = 0.5
    authority: str = "unknown"
    source_ids: list[str] = field(default_factory=list)
    relations: list[dict[str, str]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    lexical_score: float = 0.0
    graph_entity_score: float = 0.0
    temporal_task_score: float = 0.0
    validity_score: float = 0.0
    source_confidence_score: float = 0.0
    route_compatibility_score: float = 0.0
    final_score: float = 0.0
    conflict_key: str | None = None
    conflict_penalty: float = 0.0
    resolution_action: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["memory_type"] = self.memory_type.value
        data["timestamp"] = (
            self.timestamp.isoformat()
            if self.timestamp
            else None
        )
        return data


@dataclass(slots=True)
class ConflictGroup:
    conflict_id: str
    key: str
    candidate_ids: list[str]
    conflict_type: str
    explicit: bool
    preferred_ids: list[str] = field(default_factory=list)
    historical_ids: list[str] = field(default_factory=list)
    excluded_ids: list[str] = field(default_factory=list)
    unresolved: bool = False
    explanation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ConfidenceResult:
    adequacy: float
    coverage: float
    agreement: float
    decision: AnswerDecision
    components: dict[str, float]


@dataclass(slots=True)
class C3Result:
    query: str
    user_id: str
    answer: str
    decision: AnswerDecision
    query_mode: QueryMode
    selected_memory_types: list[str]
    route_scores: dict[str, float]

    # Backward-compatible field. In C3 it represents the
    # post-ranking candidate list that reaches conflict handling.
    retrieved_ids: list[str]

    selected_ids: list[str]
    conflict_groups: list[ConflictGroup]
    coverage: float
    adequacy: float
    agreement: float
    final_prompt: str
    latency_ms: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    selected_evidence: list[MemoryCandidate] = field(
        default_factory=list
    )
    debug: dict[str, Any] = field(default_factory=dict)

    # New explicit retrieval-stage fields. Defaults keep older
    # baselines and external callers compatible.
    raw_retrieved_ids: list[str] = field(default_factory=list)
    ranked_candidate_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        raw_ids = (
            self.raw_retrieved_ids
            if self.raw_retrieved_ids
            else self.retrieved_ids
        )
        ranked_ids = (
            self.ranked_candidate_ids
            if self.ranked_candidate_ids
            else self.retrieved_ids
        )

        return {
            "query": self.query,
            "user_id": self.user_id,
            "answer": self.answer,
            "decision": self.decision.value,
            "query_mode": self.query_mode.value,
            "selected_memory_types": self.selected_memory_types,
            "route_scores": self.route_scores,
            "raw_retrieved_ids": raw_ids,
            "ranked_candidate_ids": ranked_ids,
            "retrieved_ids": self.retrieved_ids,
            "selected_ids": self.selected_ids,
            "retrieval_stages": {
                "raw_retrieved_ids": raw_ids,
                "ranked_candidate_ids": ranked_ids,
                "selected_ids": self.selected_ids,
            },
            "conflict_groups": [
                group.to_dict()
                for group in self.conflict_groups
            ],
            "coverage": self.coverage,
            "adequacy": self.adequacy,
            "agreement": self.agreement,
            "final_prompt": self.final_prompt,
            "latency_ms": self.latency_ms,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "selected_evidence": [
                candidate.to_dict()
                for candidate in self.selected_evidence
            ],
            "debug": self.debug,
        }


def unique_candidates(
    items: Iterable[MemoryCandidate],
) -> list[MemoryCandidate]:
    seen: set[str] = set()
    output: list[MemoryCandidate] = []

    for item in items:
        if item.memory_id not in seen:
            seen.add(item.memory_id)
            output.append(item)

    return output
