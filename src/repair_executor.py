from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from .candidate_budget import (
    BoundaryAwareCandidateBudget,
)
from .conflict_detector import ConflictDetector
from .conflict_resolver import ConflictResolver
from .evidence_selector import EvidenceSelector
from .repair_planner import RepairPlan
from .requirement_spec import RequirementSpec
from .requirement_sufficiency import (
    RequirementSufficiencyEvaluator,
    SufficiencyResult,
)
from .schemas import (
    ConflictGroup,
    MemoryCandidate,
    MemoryType,
    QueryFeatures,
    QueryState,
    RouteDecision,
    unique_candidates,
)
from .shared_ranker import SharedRanker


@dataclass(slots=True)
class RepairExecutionResult:
    """Result of one bounded C3-v3 repair round."""

    attempted: bool
    accepted: bool

    route: RouteDecision

    selected: list[MemoryCandidate]

    conflicts: list[ConflictGroup]

    sufficiency: SufficiencyResult

    repair_retrieved_ids: list[str] = field(
        default_factory=list
    )

    added_candidate_ids: list[str] = field(
        default_factory=list
    )

    trace: dict[str, Any] = field(
        default_factory=dict
    )


class EvidenceRepairExecutor:
    """Execute one targeted evidence repair round.

    The executor:

        1. expands the route only where required;
        2. performs targeted retrieval;
        3. re-ranks the combined candidate pool;
        4. re-runs conflict resolution;
        5. re-runs requirement-aware selection;
        6. accepts repair only when hard requirement
           coverage strictly improves.

    No model training is used.
    """

    HISTORICAL_REPAIR_ROLES = {
        "historical_state",
        "transition",
        "alternative_state",
    }

    def __init__(
        self,
        *,
        config: dict[str, Any],
        memory_store: Any,
        procedure_store: Any | None,
        ranker: SharedRanker,
        candidate_budget: BoundaryAwareCandidateBudget,
        detector: ConflictDetector,
        resolver: ConflictResolver,
        selector: EvidenceSelector,
        sufficiency: RequirementSufficiencyEvaluator,
    ) -> None:
        self.config = config

        self.memory_store = memory_store
        self.procedure_store = procedure_store

        self.ranker = ranker
        self.candidate_budget = candidate_budget
        self.detector = detector
        self.resolver = resolver
        self.selector = selector
        self.sufficiency = sufficiency

        settings = config.get(
            "repair",
            {},
        )

        self.retrieval_multiplier = int(
            settings.get(
                "retrieval_multiplier",
                2,
            )
        )

        self.candidate_budget_multiplier = int(
            settings.get(
                "candidate_budget_multiplier",
                2,
            )
        )

    def execute(
        self,
        *,
        state: QueryState,
        features: QueryFeatures,
        original_route: RouteDecision,
        spec: RequirementSpec,
        repair_plan: RepairPlan,
        original_raw_candidates: list[MemoryCandidate],
        original_selected: list[MemoryCandidate],
        original_conflicts: list[ConflictGroup],
        original_sufficiency: SufficiencyResult,
    ) -> RepairExecutionResult:
        if not repair_plan.needed:
            return RepairExecutionResult(
                attempted=False,
                accepted=False,
                route=original_route,
                selected=original_selected,
                conflicts=original_conflicts,
                sufficiency=original_sufficiency,
                trace={
                    "reason": "repair_not_needed",
                },
            )

        repair_route = self._expanded_route(
            original_route=original_route,
            repair_plan=repair_plan,
        )

        repair_candidates = self._retrieve_targets(
            state=state,
            features=features,
            repair_plan=repair_plan,
        )

        repair_retrieved_ids = [
            candidate.memory_id
            for candidate in repair_candidates
        ]

        original_ids = {
            candidate.memory_id
            for candidate in original_raw_candidates
        }

        added_candidate_ids = [
            candidate.memory_id
            for candidate in repair_candidates
            if candidate.memory_id not in original_ids
        ]

        # Work on isolated candidate copies because ranking,
        # conflict resolution and evidence selection mutate
        # candidate fields and metadata.
        combined_candidates = [
            deepcopy(candidate)
            for candidate in unique_candidates(
                [
                    *original_raw_candidates,
                    *repair_candidates,
                ]
            )
        ]

        ranked_pool = self.ranker.rank(
            candidates=combined_candidates,
            features=features,
            route=repair_route,
            current_time=state.current_time,
        )

        repair_top_k = self._repair_top_k(
            repair_plan
        )

        ranked_candidates = (
            self.candidate_budget.select(
                ranked_pool=ranked_pool,
                selected_types=(
                    repair_route.selected_types
                ),
                top_k=repair_top_k,
            )
        )

        conflicts = self.detector.detect(
            ranked_candidates
        )

        (
            resolved_candidates,
            conflicts,
        ) = self.resolver.resolve(
            candidates=ranked_candidates,
            conflicts=conflicts,
            features=features,
        )

        selected = self.selector.select(
            candidates=resolved_candidates,
            features=features,
            route=repair_route,
            conflicts=conflicts,
        )

        selector_status = dict(
            self.selector.last_requirement_status
        )

        # C3.9A set-wise repaired requirement status.
        #
        # Repair may still use the validated RC8 selector to construct
        # a candidate repaired set, but acceptance must compare the same
        # requirement semantics before and after repair.  Therefore
        # selector-path credit is replaced by a direct set-wise assessment.
        #
        # candidate_pool=selected is deliberately conservative here:
        # hard coverage depends on satisfaction/completion of the repaired
        # set itself.  Global repair-pool feasibility remains diagnostic only.
        selector_status = (
            self.selector
            .assess_selected_requirement_status(
                selected=selected,
                candidate_pool=selected,
                requirements=list(
                    spec.requirements
                ),
                features=features,
                conflicts=conflicts,
            )
        )

        repaired_sufficiency = (
            self.sufficiency.evaluate(
                spec=spec,
                selected=selected,
                selector_status=selector_status,
            )
        )

        before = (
            original_sufficiency
            .hard_requirement_coverage
        )

        after = (
            repaired_sufficiency
            .hard_requirement_coverage
        )

        accepted = after > before

        trace = {
            "target_memory_types": [
                memory_type.value
                for memory_type
                in repair_plan.target_memory_types
            ],
            "original_route": [
                memory_type.value
                for memory_type
                in original_route.selected_types
            ],
            "repair_route": [
                memory_type.value
                for memory_type
                in repair_route.selected_types
            ],
            "repair_retrieved_ids": (
                repair_retrieved_ids
            ),
            "added_candidate_ids": (
                added_candidate_ids
            ),
            "hard_coverage_before": before,
            "hard_coverage_after": after,
            "accepted": accepted,
        }

        if not accepted:
            trace[
                "rejection_reason"
            ] = (
                "hard_requirement_coverage_"
                "did_not_improve"
            )

            return RepairExecutionResult(
                attempted=True,
                accepted=False,
                route=original_route,
                selected=original_selected,
                conflicts=original_conflicts,
                sufficiency=original_sufficiency,
                repair_retrieved_ids=(
                    repair_retrieved_ids
                ),
                added_candidate_ids=(
                    added_candidate_ids
                ),
                trace=trace,
            )

        return RepairExecutionResult(
            attempted=True,
            accepted=True,
            route=repair_route,
            selected=selected,
            conflicts=conflicts,
            sufficiency=repaired_sufficiency,
            repair_retrieved_ids=(
                repair_retrieved_ids
            ),
            added_candidate_ids=(
                added_candidate_ids
            ),
            trace=trace,
        )

    def _expanded_route(
        self,
        *,
        original_route: RouteDecision,
        repair_plan: RepairPlan,
    ) -> RouteDecision:
        selected_set = set(
            original_route.selected_types
        )

        selected_set.update(
            repair_plan.target_memory_types
        )

        selected_types = [
            memory_type
            for memory_type in MemoryType
            if memory_type in selected_set
        ]

        scores = dict(
            original_route.scores
        )

        reasons = deepcopy(
            original_route.reasons
        )

        thresholds = (
            self.config[
                "routing"
            ][
                "thresholds"
            ]
        )

        for memory_type in (
            repair_plan.target_memory_types
        ):
            key = memory_type.value

            scores[key] = round(
                max(
                    float(
                        scores.get(
                            key,
                            0.0,
                        )
                    ),
                    float(
                        thresholds[
                            key
                        ]
                    ),
                ),
                6,
            )

            reasons.setdefault(
                key,
                [],
            )

            reason = (
                "c3_v3:sufficiency_repair"
            )

            if (
                reason
                not in reasons[key]
            ):
                reasons[key].append(
                    reason
                )

        return RouteDecision(
            selected_types=selected_types,
            scores=scores,
            reasons=reasons,
        )

    def _retrieve_targets(
        self,
        *,
        state: QueryState,
        features: QueryFeatures,
        repair_plan: RepairPlan,
    ) -> list[MemoryCandidate]:
        base_top_k = (
            self.config[
                "retrieval"
            ][
                "top_k"
            ]
        )

        candidate_pool_multiplier = int(
            self.config[
                "retrieval"
            ].get(
                "candidate_pool_multiplier",
                1,
            )
        )

        force_archived = any(
            role
            in self.HISTORICAL_REPAIR_ROLES
            for role
            in repair_plan.missing_roles
        )

        output: list[
            MemoryCandidate
        ] = []

        for memory_type in (
            repair_plan.target_memory_types
        ):
            source = (
                self.procedure_store
                if (
                    memory_type
                    == MemoryType.PROCEDURAL
                    and self.procedure_store
                )
                else self.memory_store
            )

            top_k = (
                int(
                    base_top_k[
                        memory_type.value
                    ]
                )
                * candidate_pool_multiplier
                * self.retrieval_multiplier
            )

            output.extend(
                source.retrieve(
                    memory_type=memory_type,
                    state=state,
                    features=features,
                    top_k=top_k,
                    include_archived=(
                        force_archived
                    ),
                )
            )

        return [
            deepcopy(candidate)
            for candidate
            in unique_candidates(
                output
            )
        ]

    def _repair_top_k(
        self,
        repair_plan: RepairPlan,
    ) -> dict[str, int]:
        base_top_k = {
            str(key): int(value)
            for key, value
            in self.config[
                "retrieval"
            ][
                "top_k"
            ].items()
        }

        for memory_type in (
            repair_plan.target_memory_types
        ):
            key = (
                memory_type.value
            )

            base_top_k[key] = (
                base_top_k[key]
                * self.candidate_budget_multiplier
            )

        return base_top_k