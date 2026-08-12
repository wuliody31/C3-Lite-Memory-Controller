from __future__ import annotations

from typing import Any

from .evidence_requirements import (
    EvidenceRequirementPlanner,
)
from .query_analyzer import QueryAnalyzer
from .requirement_spec import (
    RequirementCompilation,
    RequirementSpec,
)
from .route_planner import RoutePlanner
from .schemas import ConflictGroup


class RequirementCompiler:
    """Compile a query into the unified C3-v3 requirement vector.

    This first version is deliberately behaviour-preserving:
    it delegates query analysis, routing, and evidence planning
    to the already validated RC8.3 components.
    """

    def __init__(
        self,
        config: dict[str, Any],
    ) -> None:
        self.analyzer = QueryAnalyzer(config)
        self.router = RoutePlanner(config)
        self.requirement_planner = (
            EvidenceRequirementPlanner(config)
        )

    def compile(
        self,
        query: str,
        *,
        conflicts: list[ConflictGroup] | None = None,
    ) -> RequirementCompilation:
        features = self.analyzer.analyse(query)

        route = self.router.plan(features)

        evidence_plan = (
            self.requirement_planner.plan(
                features=features,
                route=route,
                conflicts=conflicts or [],
            )
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