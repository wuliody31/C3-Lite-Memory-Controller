from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .evidence_requirements import EvidenceRequirement
from .schemas import MemoryType, QueryMode


@dataclass(slots=True)
class RequirementSpec:
    """Unified query-conditioned requirement vector for C3-v3.

    Paper notation:

        z_q = (
            T_q,
            tau_q,
            E_q,
            R_q,
            B_q
        )

    The structure is intentionally descriptive at this stage.
    It consolidates existing RC8.3 decisions without changing
    retrieval, ranking, conflict resolution, or evidence selection.
    """

    query: str

    # T_q: memory banks relevant to the query.
    memory_types: list[MemoryType]

    # tau_q: temporal interpretation of the query.
    temporal_mode: QueryMode

    # E_q: entities / subjects relevant to retrieval.
    entities: list[str] = field(default_factory=list)

    # Explicit temporal expressions such as dates or "before X".
    temporal_expressions: list[str] = field(default_factory=list)

    # Query-derived information needs.
    information_needs: list[str] = field(default_factory=list)

    # R_q: structured answer/evidence requirements.
    requirements: list[EvidenceRequirement] = field(
        default_factory=list
    )
    # S_q: query-conditioned semantic obligations.
    slots: list[RequirementSlot] = field(
    default_factory=list
    )

    # B_q and related evidence-set limits.
    token_budget: int = 0
    max_evidence: int = 0
    max_per_memory_type: int = 0

    explicit_cardinality: int | None = None

    # Query semantics retained for later arbitration/repair.
    asks_current_state: bool = False
    asks_historical_state: bool = False
    asks_timeline: bool = False
    asks_procedure: bool = False
    asks_explanation: bool = False
    asks_conflict: bool = False

    task_type: str | None = None

    # Kept for traceability; not part of the paper's core vector.
    route_scores: dict[str, float] = field(
        default_factory=dict
    )
    route_reasons: dict[str, list[str]] = field(
        default_factory=dict
    )
    compilation_reasons: list[str] = field(
        default_factory=list
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "memory_types": [
                memory_type.value
                for memory_type in self.memory_types
            ],
            "temporal_mode": self.temporal_mode.value,
            "entities": list(self.entities),
            "temporal_expressions": list(
                self.temporal_expressions
            ),
            "information_needs": list(
                self.information_needs
            ),
            "requirements": [
                requirement.to_dict()
                for requirement in self.requirements
            ],
            "slots": [
                slot.to_dict()
                for slot in self.slots
        ],
            "token_budget": self.token_budget,
            "max_evidence": self.max_evidence,
            "max_per_memory_type": (
                self.max_per_memory_type
            ),
            "explicit_cardinality": (
                self.explicit_cardinality
            ),
            "asks_current_state": (
                self.asks_current_state
            ),
            "asks_historical_state": (
                self.asks_historical_state
            ),
            "asks_timeline": self.asks_timeline,
            "asks_procedure": self.asks_procedure,
            "asks_explanation": (
                self.asks_explanation
            ),
            "asks_conflict": self.asks_conflict,
            "task_type": self.task_type,
            "route_scores": dict(
                self.route_scores
            ),
            "route_reasons": {
                key: list(values)
                for key, values
                in self.route_reasons.items()
            },
            "compilation_reasons": list(
                self.compilation_reasons
            ),
        }


from .schemas import QueryFeatures, RouteDecision

@dataclass(frozen=True, slots=True)
class RequirementSlot:
    """One query-conditioned semantic obligation."""

    slot_id: str
    kind: str
    target: str

    hard: bool = True
    min_count: int = 1
    distinct: bool = False

    temporal_role: str | None = None

    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "slot_id": self.slot_id,
            "kind": self.kind,
            "target": self.target,
            "hard": self.hard,
            "min_count": self.min_count,
            "distinct": self.distinct,
            "temporal_role": self.temporal_role,
            "description": self.description,
        }

@dataclass(slots=True)
class RequirementCompilation:
    """Compatibility bridge between RC8.3 and C3-v3."""

    spec: RequirementSpec
    features: QueryFeatures
    route: RouteDecision

