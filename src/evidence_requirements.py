from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from .schemas import (
    ConflictGroup,
    MemoryType,
    QueryFeatures,
    QueryMode,
    RouteDecision,
)


@dataclass(slots=True)
class EvidenceRequirement:
    """One evidence role that the selector should satisfy."""

    role: str
    min_count: int = 1
    hard: bool = True
    distinct: bool = False
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class EvidencePlan:
    """Query-conditioned evidence requirements and dynamic limits."""

    requirements: list[EvidenceRequirement]
    max_evidence: int
    max_per_memory_type: int
    token_budget: int
    explicit_cardinality: int | None = None
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "requirements": [
                requirement.to_dict()
                for requirement in self.requirements
            ],
            "max_evidence": self.max_evidence,
            "max_per_memory_type": self.max_per_memory_type,
            "token_budget": self.token_budget,
            "explicit_cardinality": self.explicit_cardinality,
            "reasons": list(self.reasons),
        }


class EvidenceRequirementPlanner:
    """Convert query features into explicit evidence-role requirements.

    RC8 treats evidence selection as constrained requirement satisfaction before
    the ordinary relevance/diversity fill stage. The planner is deterministic,
    training-free and fully inspectable.
    """

    NUMBER_WORDS = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "一": 1,
        "二": 2,
        "两": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
        "十": 10,
    }

    CARDINALITY_HINTS = {
        "type",
        "types",
        "item",
        "items",
        "reason",
        "reasons",
        "example",
        "examples",
        "memory",
        "memories",
        "each",
        "分别",
        "每个",
        "类型",
        "条",
        "项",
    }

    def __init__(self, config: dict[str, Any]) -> None:
        self.selection = config["selection"]
        self.settings = config.get(
            "evidence_requirements",
            {},
        )

    def plan(
        self,
        *,
        features: QueryFeatures,
        route: RouteDecision,
        conflicts: list[ConflictGroup],
    ) -> EvidencePlan:
        requirements: list[EvidenceRequirement] = []
        reasons: list[str] = []

        def add(
            role: str,
            *,
            min_count: int = 1,
            hard: bool = True,
            distinct: bool = False,
            description: str = "",
        ) -> None:
            existing = next(
                (
                    item
                    for item in requirements
                    if item.role == role
                ),
                None,
            )
            if existing is not None:
                existing.min_count = max(
                    existing.min_count,
                    min_count,
                )
                existing.hard = existing.hard or hard
                existing.distinct = (
                    existing.distinct or distinct
                )
                if description and not existing.description:
                    existing.description = description
                return

            requirements.append(
                EvidenceRequirement(
                    role=role,
                    min_count=min_count,
                    hard=hard,
                    distinct=distinct,
                    description=description,
                )
            )

        # Every answer needs at least one topically direct item.
        add(
            "answer_target",
            description=(
                "Directly addresses the primary information need."
            ),
        )

        if features.query_mode == QueryMode.CURRENT:
            add(
                "current_state",
                description="Current valid answer state.",
            )
            reasons.append("current_state_query")

        elif features.query_mode == QueryMode.HISTORICAL:
            add(
                "historical_state",
                description="Earlier or historical answer state.",
            )
            reasons.append("historical_state_query")

        elif features.query_mode == QueryMode.TIMELINE:
            add(
                "historical_state",
                min_count=int(
                    self.settings.get(
                        "timeline_min_historical_states",
                        2,
                    )
                ),
                description="Earlier timeline states.",
            )
            add(
                "current_state",
                min_count=int(
                    self.settings.get(
                        "timeline_min_current_states",
                        1,
                    )
                ),
                description="Current timeline endpoint.",
            )
            add(
                "transition",
                min_count=int(
                    self.settings.get(
                        "timeline_min_transition_evidence",
                        1,
                    )
                ),
                description="Evidence describing the change or transition.",
            )
            reasons.append("timeline_query")

        if (
            features.query_mode == QueryMode.PROCEDURAL
            or MemoryType.PROCEDURAL in route.selected_types
        ):
            add(
                "procedural_rule",
                min_count=int(
                    self.settings.get(
                        "procedural_min_rules",
                        1,
                    )
                ),
                description="Applicable procedure, rule or response policy.",
            )
            reasons.append("procedural_route")

        if features.asks_conflict or conflicts:
            add(
                "current_state",
                description="Current state used to resolve the comparison.",
            )
            add(
                "alternative_state",
                min_count=int(
                    self.settings.get(
                        "conflict_min_alternatives",
                        1,
                    )
                ),
                description="Competing, previous or alternative state.",
            )
            add(
                "preferred_resolution",
                min_count=int(
                    self.settings.get(
                        "conflict_min_resolutions",
                        1,
                    )
                ),
                description="Preferred or current resolution.",
            )
            reasons.append("conflict_or_comparison_query")

        if features.asks_explanation:
            add(
                "supporting_evidence",
                min_count=int(
                    self.settings.get(
                        "explanation_min_supporting_evidence",
                        2,
                    )
                ),
                description="Evidence supporting or explaining the answer.",
            )
            reasons.append("explanation_query")

        cardinality = self._explicit_cardinality(
            features.normalised_query
        )
        if cardinality is not None:
            add(
                "distinct_item",
                min_count=cardinality,
                distinct=True,
                description=(
                    "Distinct items required by an explicit number in the query."
                ),
            )
            reasons.append(
                f"explicit_cardinality_{cardinality}"
            )

        default_max = int(
            self.settings.get(
                "default_max_evidence",
                self.selection.get("max_evidence", 5),
            )
        )
        hard_max = int(
            self.settings.get(
                "hard_max_evidence",
                max(default_max, 8),
            )
        )
        desired_max = default_max

        if features.query_mode == QueryMode.TIMELINE:
            desired_max = max(
                desired_max,
                int(
                    self.settings.get(
                        "timeline_preferred_evidence",
                        6,
                    )
                ),
            )

        if (
            features.query_mode == QueryMode.PROCEDURAL
            or MemoryType.PROCEDURAL in route.selected_types
        ):
            desired_max = max(
                desired_max,
                int(
                    self.settings.get(
                        "procedural_preferred_evidence",
                        6,
                    )
                ),
            )

        if features.asks_conflict or conflicts:
            desired_max = max(
                desired_max,
                int(
                    self.settings.get(
                        "conflict_preferred_evidence",
                        6,
                    )
                ),
            )

        if features.asks_explanation:
            desired_max = max(
                desired_max,
                int(
                    self.settings.get(
                        "explanation_preferred_evidence",
                        6,
                    )
                ),
            )

        if cardinality is not None:
            desired_max = max(
                desired_max,
                cardinality
                + int(
                    self.settings.get(
                        "cardinality_extra_buffer",
                        1,
                    )
                ),
            )

        return EvidencePlan(
            requirements=requirements,
            max_evidence=min(
                hard_max,
                desired_max,
            ),
            max_per_memory_type=int(
                self.settings.get(
                    "max_per_memory_type",
                    self.selection.get(
                        "max_per_memory_type",
                        3,
                    ),
                )
            ),
            token_budget=int(
                self.settings.get(
                    "evidence_token_budget",
                    self.selection.get(
                        "evidence_token_budget",
                        1200,
                    ),
                )
            ),
            explicit_cardinality=cardinality,
            reasons=reasons,
        )

    def _explicit_cardinality(
        self,
        query: str,
    ) -> int | None:
        query_lower = query.lower()

        has_hint = any(
            hint in query_lower
            for hint in self.CARDINALITY_HINTS
        )
        if not has_hint:
            return None

        digit_match = re.search(
            r"(?<!\d)([2-9]|10)(?!\d)",
            query_lower,
        )
        if digit_match:
            return int(digit_match.group(1))

        for word, value in self.NUMBER_WORDS.items():
            if re.search(
                rf"(?<!\w){re.escape(word)}(?!\w)",
                query_lower,
            ):
                return value

        return None
