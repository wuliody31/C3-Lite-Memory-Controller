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
    """RC8.2 calibrated evidence-requirement planner.

    RC8.2 separates hard answer requirements from soft enrichment roles.
    Requirements are triggered by explicit query intent, not merely by an
    incidental conflict group or an over-inclusive memory route.
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

    EXPLICIT_COMPARISON_PATTERNS = (
        r"\b(?:more current|which is more current)\b",
        r"\b(?:versus|vs\.)\b",
        r"\b(?:rather than|instead of)\b",
        r"\b(?:should|did|is|are|was|were|do|does|can|could|would)\b.+\bor\b.+",
        r"\b(?:prioritise|prioritize|choose|select|prefer)\b.+\bor\b.+",
    )

    HYPOTHETICAL_POLICY_PATTERNS = (
        r"^if\s+.+\b(?:how should|what should|should the system|should i say|should the answer)\b",
        r"\b(?:how should|what should)\b.+\bif\b.+",
    )

    EXPLICIT_CHANGE_SIGNALS = {
        "over time",
        "change",
        "changed",
        "evolve",
        "evolved",
        "evolution",
        "timeline",
        "shifted",
        "moved from",
    }

    def __init__(self, config: dict[str, Any]) -> None:
        self.selection = config["selection"]
        self.settings = config.get("evidence_requirements", {})

    def plan(
        self,
        *,
        features: QueryFeatures,
        route: RouteDecision,
        conflicts: list[ConflictGroup],
    ) -> EvidencePlan:
        del conflicts  # RC8.2 does not turn incidental groups into hard roles.

        requirements: list[EvidenceRequirement] = []
        reasons: list[str] = []
        query = features.normalised_query.lower()

        def add(
            role: str,
            *,
            min_count: int = 1,
            hard: bool = True,
            distinct: bool = False,
            description: str = "",
        ) -> None:
            existing = next(
                (item for item in requirements if item.role == role),
                None,
            )
            if existing is not None:
                existing.min_count = max(existing.min_count, min_count)
                existing.hard = existing.hard or hard
                existing.distinct = existing.distinct or distinct
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

        # Every answer requires a direct answer anchor.
        add(
            "answer_target",
            hard=True,
            description="Directly addresses the primary information need.",
        )

        hypothetical_policy = self._is_hypothetical_policy(query)
        explicit_comparison = self._is_explicit_comparison(query)
        explicit_change = any(
            signal in query for signal in self.EXPLICIT_CHANGE_SIGNALS
        )

        if features.query_mode == QueryMode.CURRENT:
            add(
                "current_state",
                hard=True,
                description="Current valid answer state.",
            )
            reasons.append("current_state_query")

        elif features.query_mode == QueryMode.HISTORICAL:
            add(
                "historical_state",
                hard=True,
                description="Earlier or historical answer state.",
            )
            reasons.append("historical_state_query")

        elif features.query_mode == QueryMode.TIMELINE:
            # One historical endpoint and one current endpoint are sufficient
            # for a minimal timeline answer. Extra historical states and an
            # explicit transition enrich the answer but should not make the
            # whole question fail when absent.
            add(
                "historical_state",
                min_count=int(
                    self.settings.get("timeline_min_historical_states", 1)
                ),
                hard=True,
                description="Earlier timeline endpoint.",
            )
            add(
                "current_state",
                min_count=int(
                    self.settings.get("timeline_min_current_states", 1)
                ),
                hard=True,
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
                hard=bool(
                    self.settings.get(
                        "timeline_transition_hard",
                        False,
                    )
                ) and explicit_change,
                description="Evidence describing the change or transition.",
            )
            reasons.append("timeline_query")

        procedural_intent = (
            features.query_mode == QueryMode.PROCEDURAL
            or features.asks_procedure
            or hypothetical_policy
        )
        routed_procedural_only = (
            MemoryType.PROCEDURAL in route.selected_types
            and not procedural_intent
        )

        if procedural_intent:
            add(
                "procedural_rule",
                min_count=int(
                    self.settings.get("procedural_min_rules", 1)
                ),
                hard=True,
                description="Applicable procedure, rule or response policy.",
            )
            reasons.append("explicit_procedural_intent")
        elif routed_procedural_only:
            add(
                "procedural_rule",
                min_count=1,
                hard=False,
                description="Optional routed procedure supporting the answer.",
            )
            reasons.append("soft_procedural_route")

        # A hypothetical policy question mentions old/new or conflict words,
        # but asks for the rule itself rather than real competing states.
        if features.asks_conflict and not hypothetical_policy:
            if explicit_comparison:
                add(
                    "alternative_state",
                    min_count=int(
                        self.settings.get("conflict_min_alternatives", 1)
                    ),
                    hard=True,
                    description="Competing, previous or alternative state.",
                )
                add(
                    "preferred_resolution",
                    min_count=int(
                        self.settings.get("conflict_min_resolutions", 1)
                    ),
                    hard=True,
                    description="Preferred or current resolution.",
                )
                reasons.append("explicit_conflict_comparison")
            else:
                # Conflict words such as 'outdated' may describe a single
                # historical target. Preserve an alternative only as optional
                # enrichment rather than forcing three state roles.
                add(
                    "alternative_state",
                    min_count=1,
                    hard=False,
                    description="Optional conflicting or superseded state.",
                )
                reasons.append("soft_conflict_context")

        if features.asks_explanation:
            add(
                "supporting_evidence",
                min_count=int(
                    self.settings.get(
                        "explanation_min_supporting_evidence",
                        1,
                    )
                ),
                hard=True,
                description="Evidence supporting or explaining the answer.",
            )
            reasons.append("explanation_query")

        cardinality = self._explicit_cardinality(query)
        if cardinality is not None:
            add(
                "distinct_item",
                min_count=cardinality,
                hard=True,
                distinct=True,
                description=(
                    "Distinct items required by an explicit number in the query."
                ),
            )
            reasons.append(f"explicit_cardinality_{cardinality}")

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
                int(self.settings.get("timeline_preferred_evidence", 5)),
            )

        if procedural_intent:
            desired_max = max(
                desired_max,
                int(self.settings.get("procedural_preferred_evidence", 5)),
            )

        if explicit_comparison and not hypothetical_policy:
            desired_max = max(
                desired_max,
                int(self.settings.get("conflict_preferred_evidence", 5)),
            )

        if features.asks_explanation:
            desired_max = max(
                desired_max,
                int(self.settings.get("explanation_preferred_evidence", 5)),
            )

        if cardinality is not None:
            desired_max = max(desired_max, cardinality)

        return EvidencePlan(
            requirements=requirements,
            max_evidence=min(hard_max, desired_max),
            max_per_memory_type=int(
                self.settings.get(
                    "max_per_memory_type",
                    self.selection.get("max_per_memory_type", 3),
                )
            ),
            token_budget=int(
                self.settings.get(
                    "evidence_token_budget",
                    self.selection.get("evidence_token_budget", 1200),
                )
            ),
            explicit_cardinality=cardinality,
            reasons=reasons,
        )

    @classmethod
    def _is_explicit_comparison(cls, query: str) -> bool:
        return any(
            re.search(pattern, query) is not None
            for pattern in cls.EXPLICIT_COMPARISON_PATTERNS
        )

    @classmethod
    def _is_hypothetical_policy(cls, query: str) -> bool:
        return any(
            re.search(pattern, query) is not None
            for pattern in cls.HYPOTHETICAL_POLICY_PATTERNS
        )

    def _explicit_cardinality(self, query: str) -> int | None:
        query_lower = query.lower()

        has_hint = any(
            hint in query_lower for hint in self.CARDINALITY_HINTS
        )
        if not has_hint:
            return None

        digit_match = re.search(r"(?<!\d)([2-9]|10)(?!\d)", query_lower)
        if digit_match:
            return int(digit_match.group(1))

        for word, value in self.NUMBER_WORDS.items():
            if re.search(
                rf"(?<!\w){re.escape(word)}(?!\w)",
                query_lower,
            ):
                return value

        return None
