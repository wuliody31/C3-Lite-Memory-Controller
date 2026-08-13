from __future__ import annotations

import re
from typing import Any

from .evidence_requirements import (
    EvidenceRequirementPlanner,
)
from .query_analyzer import QueryAnalyzer
from .requirement_spec import (
    RequirementCompilation,
    RequirementSlot,
    RequirementSpec,
)
from .route_planner import RoutePlanner
from .schemas import (
    ConflictGroup,
    QueryFeatures,
    QueryMode,
    RouteDecision,
)


class RequirementCompiler:
    """Compile a query into the unified C3-v3 requirement vector.

    C3-v3 keeps the validated RC8.3 coarse evidence requirements and adds
    query-conditioned semantic requirement slots in shadow mode.

    The slot layer is descriptive only at this stage:
      - it does not change retrieval;
      - it does not change ranking;
      - it does not change conflict resolution;
      - it does not change evidence selection;
      - it does not change repair or generation.

    Paper-level view:

        z_q = (
            T_q,
            tau_q,
            E_q,
            R_q,
            S_q,
            B_q
        )

    where:
        R_q = coarse evidence-role requirements;
        S_q = typed semantic requirement slots.
    """

    GENERIC_INFORMATION_NEEDS = {
        "current valid state",
        "current state",
        "earlier state",
        "historical state",
        "historical event or state",
        "historical evidence",
        "current evidence",
        "supporting evidence",
        "supporting reason or evidence",
        "applicable procedure or rule",
        "alternative or conflicting states",
        "preferred or current resolution",
    }

    MULTI_FACET_PATTERNS = (
        (
            r"\bwhich\s+"
            r"(?:abilities|skills|details|factors|components|"
            r"features|types|aspects|items|things)\b"
        ),
        (
            r"\bwhat\s+"
            r"(?:abilities|skills|details|factors|components|"
            r"features|types|aspects|items|things)\b"
        ),
        r"\bwhat\s+technical\s+details\b",
        r"\blist\b",
        r"\benumerate\b",
    )

    PROVENANCE_PATTERNS = (
        r"\bwhich\s+memor(?:y|ies)\b",
        r"\bwhat\s+memor(?:y|ies)\b",
        r"\bmemory\s+types?\b",
        r"\bcite\b.*\bmemor",
        r"\bwhich\s+evidence\b",
        r"\bwhat\s+evidence\b",
        r"\bcite\b.*\bevidence\b",
    )

    # QueryAnalyzer intentionally treats many A-or-B forms as possible
    # conflicts. Slot compilation is more conservative: only explicit
    # preference/choice questions become CONTRAST + RESOLUTION slots.
    #
    # This avoids misclassifying propositions such as:
    #   "Does my project require model training or fine-tuning?"
    # where "training or fine-tuning" is one semantic proposition rather than
    # two competing alternatives.
    EXPLICIT_CHOICE_PATTERNS = (
        (
            r"\b(?:should|prioritise|prioritize|choose|select|prefer)\b"
            r".+\bor\b.+"
        ),
        r"\brather\s+than\b",
        r"\bversus\b",
        r"\bvs\b",
        r"\binstead\s+of\b",
    )

    # Query/task operators describe how to answer, not the semantic axis
    # that evidence must cover. This deterministic projection prevents
    # over-specific slot targets such as "project scope change over time".
    SEMANTIC_AXIS_OPERATOR_TOKENS = {
        "answer",
        "answers",
        "answered",
        "explain",
        "explains",
        "explained",
        "explanation",
        "cite",
        "cites",
        "cited",
        "citation",
        "memory",
        "memories",
        "evidence",
        "support",
        "supports",
        "supported",
        "supporting",
        "using",
        "used",
        "use",
        "current",
        "currently",
        "historical",
        "history",
        "previous",
        "previously",
        "earlier",
        "latest",
        "change",
        "changed",
        "changes",
        "changing",
        "transition",
        "over",
        "time",
        "what",
        "which",
        "how",
        "why",
        "when",
        "where",
        "who",
        "does",
        "do",
        "did",
        "is",
        "are",
        "was",
        "were",
        "should",
        "would",
        "could",
        "mainly",
        "my",
        "your",
        "our",
        "their",
    }

    def __init__(
        self,
        config: dict[str, Any],
    ) -> None:
        self.analyzer = QueryAnalyzer(config)
        self.router = RoutePlanner(config)
        self.requirement_planner = (
            EvidenceRequirementPlanner(config)
        )

    # =========================================================
    # Slot compilation helpers
    # =========================================================

    @classmethod
    def _is_multi_facet(
        cls,
        query: str,
    ) -> bool:
        text = query.lower()

        return any(
            re.search(
                pattern,
                text,
            )
            is not None
            for pattern
            in cls.MULTI_FACET_PATTERNS
        )

    @classmethod
    def _asks_provenance(
        cls,
        query: str,
    ) -> bool:
        text = query.lower()

        return any(
            re.search(
                pattern,
                text,
            )
            is not None
            for pattern
            in cls.PROVENANCE_PATTERNS
        )

    @classmethod
    def _is_explicit_choice_query(
        cls,
        query: str,
    ) -> bool:
        text = query.lower()

        return any(
            re.search(
                pattern,
                text,
            )
            is not None
            for pattern
            in cls.EXPLICIT_CHOICE_PATTERNS
        )

    @staticmethod
    def _clean_slot_target(
        text: str,
    ) -> str:
        cleaned = " ".join(
            text.strip().split()
        )

        cleaned = cleaned.strip(
            " ?.,;:"
        )

        return cleaned

    @classmethod
    def _semantic_axis(
        cls,
        target: str,
    ) -> str:
        """Project a query phrase onto its semantic subject axis.

        No gold answers or supporting-memory annotations are consulted.
        """

        cleaned = cls._clean_slot_target(
            target
        ).lower()

        tokens = re.findall(
            r"[\w'-]+",
            cleaned,
            flags=re.UNICODE,
        )

        projected = [
            token
            for token in tokens
            if token
            not in cls.SEMANTIC_AXIS_OPERATOR_TOKENS
        ]

        if projected:
            return " ".join(
                projected
            )

        return cleaned

    @classmethod
    def _contrast_options(
        cls,
        query: str,
    ) -> list[str]:
        """Extract explicit A/B options from normative choice questions.

        This parser is deliberately conservative. If it cannot isolate two
        stable options, the compiler will still emit a RESOLUTION slot but
        will not invent benchmark-specific alternatives.
        """

        text = " ".join(
            query.strip().split()
        )

        patterns = (
            (
                r"\b(?:prioritise|prioritize|choose|select|prefer)"
                r"(?:\s+between)?\s+"
                r"(.+?)\s+or\s+(.+?)(?:\?|$)"
            ),
            (
                r"\bshould\b.+?\b(?:be|use)\s+"
                r"(.+?)\s+or\s+(.+?)(?:\?|$)"
            ),
        )

        for pattern in patterns:
            match = re.search(
                pattern,
                text,
                flags=re.IGNORECASE,
            )

            if match is None:
                continue

            options = [
                cls._clean_slot_target(
                    match.group(1)
                ),
                cls._clean_slot_target(
                    match.group(2)
                ),
            ]

            options = [
                option
                for option in options
                if option
            ]

            if len(options) == 2:
                return options

        return []

    @classmethod
    def _primary_target(
        cls,
        *,
        query: str,
        features: QueryFeatures,
    ) -> str:
        """Choose a query-conditioned semantic target without using gold data."""

        for need in features.information_needs:
            cleaned = " ".join(
                str(need)
                .lower()
                .split()
            )

            if (
                cleaned
                and cleaned
                not in cls.GENERIC_INFORMATION_NEEDS
            ):
                return cleaned

        if features.entities:
            return " ".join(
                str(item)
                for item in features.entities
            ).lower()

        if features.normalised_query:
            return features.normalised_query

        return query.lower()

    def _compile_slots(
        self,
        *,
        query: str,
        features: QueryFeatures,
        explicit_cardinality: int | None,
    ) -> list[RequirementSlot]:
        """Compile query-conditioned semantic obligations.

        Important:
        - no gold answer or supporting-memory annotation is consulted;
        - no slot is used for selection yet;
        - this is a shadow representation for later sufficiency auditing.
        """

        slots: list[RequirementSlot] = []

        slot_counter = 0

        def add(
            *,
            kind: str,
            target: str,
            hard: bool = True,
            min_count: int = 1,
            distinct: bool = False,
            temporal_role: str | None = None,
            description: str = "",
        ) -> None:
            nonlocal slot_counter

            cleaned_target = (
                self._clean_slot_target(
                    target
                )
            )

            if not cleaned_target:
                return

            slot_counter += 1

            slots.append(
                RequirementSlot(
                    slot_id=(
                        f"slot_"
                        f"{slot_counter:02d}_"
                        f"{kind.lower()}"
                    ),
                    kind=kind,
                    target=cleaned_target,
                    hard=hard,
                    min_count=max(
                        1,
                        int(min_count),
                    ),
                    distinct=distinct,
                    temporal_role=temporal_role,
                    description=description,
                )
            )

        target = self._primary_target(
            query=query,
            features=features,
        )

        semantic_axis = self._semantic_axis(
            target
        )

        # -------------------------------------------------
        # 1. Explicit preference / choice / conflict query.
        # -------------------------------------------------

        if (
            features.asks_conflict
            and self._is_explicit_choice_query(
                query
            )
        ):
            options = (
                self._contrast_options(
                    query
                )
            )

            for option in options:
                add(
                    kind="CONTRAST",
                    target=option,
                    hard=True,
                    description=(
                        "explicit alternative that should remain "
                        "represented before resolving the query"
                    ),
                )

            add(
                kind="RESOLUTION",
                target=semantic_axis,
                hard=True,
                description=(
                    "preferred, current, or otherwise resolved "
                    "answer between explicit alternatives"
                ),
            )

        # -------------------------------------------------
        # 2. Timeline query.
        # -------------------------------------------------

        elif (
            features.query_mode
            == QueryMode.TIMELINE
        ):
            add(
                kind="HISTORICAL_ENDPOINT",
                target=semantic_axis,
                temporal_role=(
                    "historical_state"
                ),
                description=(
                    "historical endpoint required by a timeline query"
                ),
            )

            add(
                kind="TEMPORAL_TRANSITION",
                target=semantic_axis,
                temporal_role=(
                    "transition_event"
                ),
                description=(
                    "change or transition connecting temporal states"
                ),
            )

            add(
                kind="CURRENT_ENDPOINT",
                target=semantic_axis,
                temporal_role=(
                    "current_state"
                ),
                description=(
                    "current endpoint required by a timeline query"
                ),
            )

        # -------------------------------------------------
        # 3. Multi-facet / enumerative query.
        # -------------------------------------------------

        elif (
            self._is_multi_facet(
                query
            )
            or (
                explicit_cardinality
                is not None
                and explicit_cardinality > 1
            )
        ):
            required_count = (
                explicit_cardinality
                if (
                    explicit_cardinality
                    is not None
                    and explicit_cardinality > 1
                )
                else 2
            )

            add(
                kind="MULTI_FACET",
                target=semantic_axis,
                hard=True,
                min_count=required_count,
                distinct=True,
                description=(
                    "query requests multiple distinct semantic units"
                ),
            )

        # -------------------------------------------------
        # 4. Single current endpoint.
        # -------------------------------------------------

        elif (
            features.query_mode
            == QueryMode.CURRENT
        ):
            add(
                kind="CURRENT_ENDPOINT",
                target=semantic_axis,
                temporal_role=(
                    "current_state"
                ),
                description=(
                    "current query-specific semantic endpoint"
                ),
            )

        # -------------------------------------------------
        # 5. Single historical endpoint.
        # -------------------------------------------------

        elif (
            features.query_mode
            == QueryMode.HISTORICAL
        ):
            add(
                kind="HISTORICAL_ENDPOINT",
                target=semantic_axis,
                temporal_role=(
                    "historical_state"
                ),
                description=(
                    "historical query-specific semantic endpoint"
                ),
            )

        # -------------------------------------------------
        # 6. Procedural query.
        # -------------------------------------------------

        elif (
            features.query_mode
            == QueryMode.PROCEDURAL
        ):
            add(
                kind="PROCEDURE",
                target=semantic_axis,
                temporal_role=(
                    "procedural_rule"
                ),
                description=(
                    "applicable rule or procedure required to answer"
                ),
            )

        # -------------------------------------------------
        # 7. Ordinary factual/content query.
        # -------------------------------------------------

        else:
            add(
                kind="CONTENT",
                target=semantic_axis,
                description=(
                    "primary semantic content required by the query"
                ),
            )

        # -------------------------------------------------
        # 8. Explainability support is orthogonal to base intent.
        # -------------------------------------------------

        if features.asks_explanation:
            add(
                kind="SUPPORT",
                target=semantic_axis,
                hard=True,
                description=(
                    "supporting evidence required by an explanation query"
                ),
            )

        # -------------------------------------------------
        # 9. Provenance is required only when explicitly requested.
        # -------------------------------------------------

        if self._asks_provenance(
            query
        ):
            add(
                kind="PROVENANCE",
                target=semantic_axis,
                hard=True,
                description=(
                    "explicit memory or evidence provenance attribution"
                ),
            )

        return slots

    # =========================================================
    # Public compilation API
    # =========================================================

    def compile(
        self,
        query: str,
        *,
        conflicts: list[ConflictGroup] | None = None,
    ) -> RequirementCompilation:
        """Compile directly from a raw query."""

        features = self.analyzer.analyse(
            query
        )

        route = self.router.plan(
            features
        )

        return self.compile_from_legacy(
            query=query,
            features=features,
            route=route,
            conflicts=conflicts,
        )

    def compile_from_legacy(
        self,
        *,
        query: str,
        features: QueryFeatures,
        route: RouteDecision,
        conflicts: list[ConflictGroup] | None = None,
    ) -> RequirementCompilation:
        """Compile z_q from already validated RC8.3 decisions.

        This avoids running QueryAnalyzer and RoutePlanner twice
        when C3-v3 is integrated into the existing pipeline.

        The legacy role requirements remain unchanged. Semantic slots are
        compiled in parallel and remain shadow-only at this stage.
        """

        evidence_plan = (
            self.requirement_planner.plan(
                features=features,
                route=route,
                conflicts=conflicts or [],
            )
        )

        slots = self._compile_slots(
            query=query,
            features=features,
            explicit_cardinality=(
                evidence_plan.explicit_cardinality
            ),
        )

        spec = RequirementSpec(
            query=query,

            memory_types=list(
                route.selected_types
            ),

            temporal_mode=(
                features.query_mode
            ),

            entities=list(
                features.entities
            ),

            temporal_expressions=list(
                features.temporal_expressions
            ),

            information_needs=list(
                features.information_needs
            ),

            requirements=list(
                evidence_plan.requirements
            ),

            slots=list(
                slots
            ),

            token_budget=(
                evidence_plan.token_budget
            ),

            max_evidence=(
                evidence_plan.max_evidence
            ),

            max_per_memory_type=(
                evidence_plan.max_per_memory_type
            ),

            explicit_cardinality=(
                evidence_plan.explicit_cardinality
            ),

            asks_current_state=(
                features.asks_current_state
            ),

            asks_historical_state=(
                features.asks_historical_state
            ),

            asks_timeline=(
                features.asks_timeline
            ),

            asks_procedure=(
                features.asks_procedure
            ),

            asks_explanation=(
                features.asks_explanation
            ),

            asks_conflict=(
                features.asks_conflict
            ),

            task_type=(
                features.task_type
            ),

            route_scores=dict(
                route.scores
            ),

            route_reasons={
                key: list(values)
                for key, values
                in route.reasons.items()
            },

            compilation_reasons=list(
                evidence_plan.reasons
            ),
        )

        return RequirementCompilation(
            spec=spec,
            features=features,
            route=route,
        )