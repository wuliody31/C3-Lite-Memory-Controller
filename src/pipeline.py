from __future__ import annotations

import time
from copy import deepcopy
from datetime import timezone
from pathlib import Path
from typing import Any

from .backbones import Backbone
from .candidate_budget import (
    BoundaryAwareCandidateBudget,
)
from .confidence_controller import (
    ConfidenceController,
)
from .conflict_detector import (
    ConflictDetector,
)
from .conflict_resolver import (
    ConflictResolver,
)
from .coverage_estimator import (
    CoverageEstimator,
)
from .evidence_selector import (
    EvidenceSelector,
)
from .prompt_builder import (
    PromptBuilder,
)
from .query_analyzer import (
    QueryAnalyzer,
)

# ---------------------------------------------------------
# C3-v3 method-level shadow components
# ---------------------------------------------------------
from .repair_planner import (
    TargetedRepairPlanner,
)
from .repair_executor import (
    EvidenceRepairExecutor,
)
from .requirement_compiler import (
    RequirementCompiler,
)
from .requirement_sufficiency import (
    RequirementSufficiencyEvaluator,
)

from .route_planner import (
    RoutePlanner,
)
from .schemas import (
    AnswerDecision,
    C3Result,
    MemoryCandidate,
    MemoryType,
    QueryState,
    RouteDecision,
    unique_candidates,
)
from .shared_ranker import (
    SharedRanker,
)


# =========================================================
# Candidate trace helpers
# =========================================================


def _candidate_score_trace(
    candidate: MemoryCandidate,
) -> dict[str, Any]:
    """Backward-compatible score trace for one candidate."""

    return {
        "memory_id": (
            candidate.memory_id
        ),
        "memory_type": (
            candidate.memory_type.value
        ),
        "status": (
            candidate.status
        ),
        "lexical_score": (
            candidate.lexical_score
        ),
        "graph_entity_score": (
            candidate.graph_entity_score
        ),
        "temporal_task_score": (
            candidate.temporal_task_score
        ),
        "validity_score": (
            candidate.validity_score
        ),
        "source_confidence_score": (
            candidate.source_confidence_score
        ),
        "route_compatibility_score": (
            candidate.route_compatibility_score
        ),
        "conflict_penalty": (
            candidate.conflict_penalty
        ),
        "final_score": (
            candidate.final_score
        ),
        "resolution_action": (
            candidate.resolution_action
        ),
        "ranker_gate_reasons": (
            candidate.metadata.get(
                "ranker_gate_reasons",
                [],
            )
        ),
        "passed_utility_aware_gate": (
            candidate.metadata.get(
                "passed_utility_aware_gate",
                False,
            )
        ),
        "candidate_budget_reason": (
            candidate.metadata.get(
                "candidate_budget_reason"
            )
        ),
        "candidate_budget_cutoff_score": (
            candidate.metadata.get(
                "candidate_budget_cutoff_score"
            )
        ),
        "candidate_budget_score_gap": (
            candidate.metadata.get(
                "candidate_budget_score_gap"
            )
        ),
        "selector_reason": (
            candidate.metadata.get(
                "selector_reason"
            )
        ),
        "evidence_roles": (
            candidate.metadata.get(
                "evidence_roles",
                [],
            )
        ),
        "selector_requirement_gain": (
            candidate.metadata.get(
                "selector_requirement_gain"
            )
        ),
        "selector_topical_score": (
            candidate.metadata.get(
                "selector_topical_score"
            )
        ),
    }


def _candidate_timestamp_value(
    candidate: MemoryCandidate,
) -> float:
    timestamp = (
        candidate.timestamp
    )

    if timestamp is None:
        return float("-inf")

    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(
            tzinfo=timezone.utc
        )

    return timestamp.timestamp()


def _candidate_rank_key(
    candidate: MemoryCandidate,
) -> tuple[
    float,
    float,
    float,
    str,
]:
    """Mirror SharedRanker's deterministic ordering safely."""

    return (
        float(
            candidate.final_score
        ),
        float(
            candidate.confidence
        ),
        _candidate_timestamp_value(
            candidate
        ),
        candidate.memory_id,
    )


def _candidate_text_for_trace(
    candidate: MemoryCandidate,
) -> str:
    triggers = (
        candidate.metadata.get(
            "triggers",
            [],
        )
    )

    trigger_text = (
        " ".join(
            str(item)
            for item in triggers
        )
        if isinstance(
            triggers,
            list,
        )
        else str(
            triggers or ""
        )
    )

    values = [
        candidate.text,
        candidate.subject,
        candidate.predicate,
        candidate.object_value,
        candidate.metadata.get(
            "task_type"
        ),
        trigger_text,
    ]

    return " ".join(
        str(value)
        for value in values
        if value
        not in (
            None,
            "",
        )
    )


def _estimated_token_cost(
    candidate: MemoryCandidate,
) -> int:
    """Cheap trace-only token estimate.

    Formal token-cost experiments should later use
    the backbone tokenizer.
    """

    text = (
        _candidate_text_for_trace(
            candidate
        )
    )

    if not text:
        return 1

    return max(
        1,
        (
            len(text)
            + 3
        )
        // 4,
    )


def _rank_positions(
    candidates: list[
        MemoryCandidate
    ],
) -> tuple[
    dict[str, int],
    dict[str, int],
]:
    ordered = sorted(
        unique_candidates(
            candidates
        ),
        key=_candidate_rank_key,
        reverse=True,
    )

    global_rank = {
        candidate.memory_id: index
        for index, candidate
        in enumerate(
            ordered,
            start=1,
        )
    }

    within_type_rank: dict[
        str,
        int,
    ] = {}

    for memory_type in MemoryType:
        type_candidates = [
            candidate
            for candidate in ordered
            if (
                candidate.memory_type
                == memory_type
            )
        ]

        for index, candidate in enumerate(
            type_candidates,
            start=1,
        ):
            within_type_rank[
                candidate.memory_id
            ] = index

    return (
        global_rank,
        within_type_rank,
    )


def _build_rank_metadata(
    *,
    raw_candidates: list[
        MemoryCandidate
    ],
    ranked_pool: list[
        MemoryCandidate
    ],
    ranked_candidates: list[
        MemoryCandidate
    ],
    top_k: dict[
        str,
        Any,
    ],
) -> dict[
    str,
    dict[str, Any],
]:
    """Build ranking and candidate-budget stage metadata.

    Legacy field names such as ``kept_after_type_top_k`` are
    preserved so the RC5 audit script remains compatible.

    Under RC7+, the field means kept after the boundary-aware
    per-type candidate budget.
    """

    (
        raw_global,
        raw_within_type,
    ) = _rank_positions(
        raw_candidates
    )

    (
        pool_global,
        pool_within_type,
    ) = _rank_positions(
        ranked_pool
    )

    pool_ids = {
        candidate.memory_id
        for candidate
        in ranked_pool
    }

    kept_ids = {
        candidate.memory_id
        for candidate
        in ranked_candidates
    }

    metadata: dict[
        str,
        dict[str, Any],
    ] = {}

    for candidate in raw_candidates:
        memory_id = (
            candidate.memory_id
        )

        passed_gate = (
            memory_id
            in pool_ids
        )

        kept_after_budget = (
            memory_id
            in kept_ids
        )

        if not passed_gate:
            pre_selection_drop_stage = (
                "ranker_gate"
            )

        elif not kept_after_budget:
            pre_selection_drop_stage = (
                "type_top_k"
            )

        else:
            pre_selection_drop_stage = (
                None
            )

        metadata[
            memory_id
        ] = {
            "rank_global_all_scored": (
                raw_global.get(
                    memory_id
                )
            ),
            "rank_within_type_all_scored": (
                raw_within_type.get(
                    memory_id
                )
            ),
            "rank_global_after_gate": (
                pool_global.get(
                    memory_id
                )
            ),
            "rank_within_type_after_gate": (
                pool_within_type.get(
                    memory_id
                )
            ),
            "passed_ranker_gate": (
                passed_gate
            ),
            "top_k_limit_for_type": (
                int(
                    top_k[
                        candidate
                        .memory_type
                        .value
                    ]
                )
            ),

            # Backward-compatible RC5 field.
            "kept_after_type_top_k": (
                kept_after_budget
            ),

            # Explicit candidate-budget field.
            "kept_after_candidate_budget": (
                kept_after_budget
            ),

            "pre_selection_drop_stage": (
                pre_selection_drop_stage
            ),
        }

    return metadata


def _build_full_candidate_score_trace(
    *,
    raw_candidates: list[
        MemoryCandidate
    ],
    score_snapshots: dict[
        str,
        dict[str, Any],
    ],
    rank_metadata: dict[
        str,
        dict[str, Any],
    ],
    route: RouteDecision,
    resolved_candidates: list[
        MemoryCandidate
    ],
    selected: list[
        MemoryCandidate
    ],
) -> list[
    dict[str, Any]
]:
    resolved_ids = {
        candidate.memory_id
        for candidate
        in resolved_candidates
    }

    selected_ids = {
        candidate.memory_id
        for candidate
        in selected
    }

    selected_route_types = {
        memory_type.value
        for memory_type
        in route.selected_types
    }

    output: list[
        dict[str, Any]
    ] = []

    for candidate in raw_candidates:
        memory_id = (
            candidate.memory_id
        )

        snapshot = (
            score_snapshots[
                memory_id
            ]
        )

        rank_info = (
            rank_metadata[
                memory_id
            ]
        )

        survived_resolution = (
            memory_id
            in resolved_ids
        )

        selected_final = (
            memory_id
            in selected_ids
        )

        if rank_info[
            "pre_selection_drop_stage"
        ]:
            drop_stage = (
                rank_info[
                    "pre_selection_drop_stage"
                ]
            )

        elif not survived_resolution:
            drop_stage = (
                "conflict_resolution"
            )

        elif not selected_final:
            drop_stage = (
                "evidence_selector"
            )

        else:
            drop_stage = None

        token_cost = (
            _estimated_token_cost(
                candidate
            )
        )

        utility_v0 = float(
            snapshot[
                "final_score"
            ]
        )

        row = {
            **snapshot,
            **rank_info,

            "candidate_budget_reason": (
                candidate.metadata.get(
                    "candidate_budget_reason"
                )
            ),

            "candidate_budget_cutoff_score": (
                candidate.metadata.get(
                    "candidate_budget_cutoff_score"
                )
            ),

            "candidate_budget_score_gap": (
                candidate.metadata.get(
                    "candidate_budget_score_gap"
                )
            ),

            "selector_reason": (
                candidate.metadata.get(
                    "selector_reason"
                )
            ),

            "evidence_roles": (
                candidate.metadata.get(
                    "evidence_roles",
                    [],
                )
            ),

            "selector_requirement_gain": (
                candidate.metadata.get(
                    "selector_requirement_gain"
                )
            ),

            "selector_topical_score": (
                candidate.metadata.get(
                    "selector_topical_score"
                )
            ),

            "route_selected": (
                candidate.memory_type.value
                in selected_route_types
            ),

            "estimated_token_cost": (
                token_cost
            ),

            "candidate_utility_v0": (
                utility_v0
            ),

            "utility_per_estimated_token": (
                round(
                    utility_v0
                    / max(
                        token_cost,
                        1,
                    ),
                    8,
                )
            ),

            "post_resolution_final_score": (
                candidate.final_score
            ),

            "survived_conflict_resolution": (
                survived_resolution
            ),

            "selected_final": (
                selected_final
            ),

            "drop_stage": (
                drop_stage
            ),
        }

        output.append(
            row
        )

    output.sort(
        key=lambda row: (
            row[
                "rank_global_all_scored"
            ]
            if (
                row[
                    "rank_global_all_scored"
                ]
                is not None
            )
            else 10**9
        )
    )

    return output


# =========================================================
# C3 Pipeline
# =========================================================


class C3Pipeline:
    def __init__(
        self,
        *,
        config: dict[
            str,
            Any,
        ],
        memory_store: Any,
        procedure_store: (
            Any
            | None
        ),
        backbone: Backbone,
        prompt_template: (
            str
            | Path
        ),
    ) -> None:
        self.config = (
            config
        )

        self.memory_store = (
            memory_store
        )

        self.procedure_store = (
            procedure_store
        )

        self.backbone = (
            backbone
        )

        # -------------------------------------------------
        # RC8.3 validated components
        # -------------------------------------------------

        self.analyzer = (
            QueryAnalyzer(
                config
            )
        )

        self.router = (
            RoutePlanner(
                config
            )
        )

        self.ranker = (
            SharedRanker(
                config
            )
        )

        self.candidate_budget = (
            BoundaryAwareCandidateBudget(
                config
            )
        )

        self.detector = (
            ConflictDetector(
                config
            )
        )

        self.resolver = (
            ConflictResolver(
                config
            )
        )

        self.coverage = (
            CoverageEstimator(
                config
            )
        )

        self.selector = (
            EvidenceSelector(
                config,
                self.coverage,
            )
        )

        # -------------------------------------------------
        # C3-v3 shadow method-level components
        # -------------------------------------------------

        self.requirement_compiler = (
            RequirementCompiler(
                config
            )
        )

        self.requirement_sufficiency = (
            RequirementSufficiencyEvaluator(
                config
            )
        )

        self.repair_planner = (
            TargetedRepairPlanner()
        )
        self.repair_executor = (
            EvidenceRepairExecutor(
                config=config,
                memory_store=memory_store,
                procedure_store=procedure_store,
                ranker=self.ranker,
                candidate_budget=(
                    self.candidate_budget
                ),
                detector=self.detector,
                resolver=self.resolver,
                selector=self.selector,
                sufficiency=(
                    self.requirement_sufficiency
                ),
            )
        )
        # -------------------------------------------------
        # Legacy downstream generation/control
        # -------------------------------------------------

        self.confidence = (
            ConfidenceController(
                config
            )
        )

        self.prompt_builder = (
            PromptBuilder(
                config,
                prompt_template,
            )
        )

    def answer(
        self,
        state: QueryState,
    ) -> C3Result:
        start = (
            time.perf_counter()
        )

        # =================================================
        # 1. Query analysis + routing
        # =================================================

        features = (
            self.analyzer.analyse(
                state.query
            )
        )

        route = (
            self.router.plan(
                features
            )
        )

        # =================================================
        # C3-v3 shadow:
        # Compile the legacy query/route decisions into z_q.
        #
        # IMPORTANT:
        # This does NOT replace QueryAnalyzer or RoutePlanner yet.
        # It only exposes their decisions through the unified
        # RequirementSpec representation.
        # =================================================

        c3_v3_compilation = (
            self.requirement_compiler
            .compile_from_legacy(
                query=state.query,
                features=features,
                route=route,
                conflicts=[],
            )
        )

        # =================================================
        # 2. Candidate retrieval
        # =================================================

        include_archived = (
            features.query_mode.value
            in set(
                self.config[
                    "retrieval"
                ][
                    "include_archived_for"
                ]
            )
            or (
                features
                .asks_conflict
            )
            or (
                features
                .asks_explanation
            )
        )

        top_k = (
            self.config[
                "retrieval"
            ][
                "top_k"
            ]
        )

        multiplier = int(
            self.config[
                "retrieval"
            ].get(
                "candidate_pool_multiplier",
                1,
            )
        )

        retrieved_candidates: list[
            MemoryCandidate
        ] = []

        for memory_type in (
            route.selected_types
        ):
            source = (
                self.procedure_store
                if (
                    memory_type
                    == MemoryType.PROCEDURAL
                    and (
                        self.procedure_store
                    )
                )
                else (
                    self.memory_store
                )
            )

            retrieved_candidates.extend(
                source.retrieve(
                    memory_type=(
                        memory_type
                    ),
                    state=state,
                    features=features,
                    top_k=(
                        int(
                            top_k[
                                memory_type
                                .value
                            ]
                        )
                        * multiplier
                    ),
                    include_archived=(
                        include_archived
                    ),
                )
            )

        # Candidate objects may be reused by an in-memory store.
        # Work on per-query deep copies so ranking/conflict/
        # selector metadata cannot leak into later questions.

        raw_candidates = [
            deepcopy(
                candidate
            )
            for candidate
            in unique_candidates(
                retrieved_candidates
            )
        ]

        raw_retrieved_ids = [
            candidate.memory_id
            for candidate
            in raw_candidates
        ]

        # =================================================
        # 3. Shared ranking
        # =================================================

        ranked_pool = (
            self.ranker.rank(
                candidates=(
                    raw_candidates
                ),
                features=features,
                route=route,
                current_time=(
                    state.current_time
                ),
            )
        )

        # Freeze scores before conflict resolution mutates
        # final_score/conflict_penalty.

        score_snapshots = {
            candidate.memory_id: (
                _candidate_score_trace(
                    candidate
                )
            )
            for candidate
            in raw_candidates
        }

        # =================================================
        # 4. Candidate budget
        # =================================================

        ranked_candidates = (
            self.candidate_budget
            .select(
                ranked_pool=(
                    ranked_pool
                ),
                selected_types=(
                    route
                    .selected_types
                ),
                top_k=top_k,
            )
        )

        ranked_candidate_ids = [
            candidate.memory_id
            for candidate
            in ranked_candidates
        ]

        rank_metadata = (
            _build_rank_metadata(
                raw_candidates=(
                    raw_candidates
                ),
                ranked_pool=(
                    ranked_pool
                ),
                ranked_candidates=(
                    ranked_candidates
                ),
                top_k=top_k,
            )
        )

        # =================================================
        # 5. Conflict detection + resolution
        # =================================================

        conflicts = (
            self.detector.detect(
                ranked_candidates
            )
        )

        (
            resolved_candidates,
            conflicts,
        ) = (
            self.resolver.resolve(
                candidates=(
                    ranked_candidates
                ),
                conflicts=conflicts,
                features=features,
            )
        )

        # =================================================
        # 6. Requirement-aware evidence selection
        # =================================================

        selected = (
            self.selector.select(
                candidates=(
                    resolved_candidates
                ),
                features=features,
                route=route,
                conflicts=conflicts,
            )
        )

        evidence_requirement_plan = (
            self.selector
            .last_plan
            .to_dict()
            if (
                self.selector
                .last_plan
                is not None
            )
            else {}
        )

        evidence_requirement_status = (
            dict(
                self.selector
                .last_requirement_status
            )
        )

        # =================================================
        # 7. C3-v3 SHADOW evidence sufficiency
        #
        # This reads the selected evidence and selector
        # requirement status. It does NOT modify selection.
        # =================================================

        c3_v3_sufficiency = (
            self.requirement_sufficiency
            .evaluate(
                spec=(
                    c3_v3_compilation
                    .spec
                ),
                selected=selected,
                selector_status=(
                    evidence_requirement_status
                ),
            )
        )

        # =================================================
        # 8. C3-v3 SHADOW targeted repair planning
        #
        # Only missing hard requirements are considered.
        #
        # IMPORTANT:
        # The repair plan is NOT executed yet.
        # No second retrieval happens in this version.
        # =================================================

        c3_v3_repair_plan = (
            self.repair_planner.plan(
                spec=(
                    c3_v3_compilation
                    .spec
                ),
                missing_requirements=(
                    c3_v3_sufficiency
                    .missing_hard_requirements
                ),
            )
        )
        # =================================================
        # C3-v3 ACTIVE sufficiency-driven repair
        # =================================================

        c3_v3_repair_execution = (
            self.repair_executor.execute(
                state=state,
                features=features,
                original_route=route,
                spec=(
                    c3_v3_compilation
                    .spec
                ),
                repair_plan=(
                    c3_v3_repair_plan
                ),
                original_raw_candidates=(
                    raw_candidates
                ),
                original_selected=selected,
                original_conflicts=conflicts,
                original_sufficiency=(
                    c3_v3_sufficiency
                ),
            )
        )

        # Accept the repaired evidence set only when
        # hard requirement coverage strictly improves.
        if (
            c3_v3_repair_execution
            .accepted
        ):
            route = (
                c3_v3_repair_execution
                .route
            )

            selected = (
                c3_v3_repair_execution
                .selected
            )

            conflicts = (
                c3_v3_repair_execution
                .conflicts
            )

            c3_v3_sufficiency = (
                c3_v3_repair_execution
                .sufficiency
            )

        # NEW: include repair-round retrievals in the
        # total retrieval trace.
        if (
            c3_v3_repair_execution
            .attempted
        ):
            raw_retrieved_ids = list(
                dict.fromkeys(
                    [
                        *raw_retrieved_ids,
                        *(
                            c3_v3_repair_execution
                            .repair_retrieved_ids
                        ),
                    ]
                )
            )

        # =================================================
        # 9. Legacy coverage
        # =================================================

        coverage = (
            self.coverage.compute(
                features.information_needs,
                selected,
            )
        )
        # =================================================
        # 10. Confidence / answer decision
        # =================================================

        confidence = (
            self.confidence.evaluate(
                selected=selected,
                route=route,
                conflicts=conflicts,
                coverage=coverage,
            )
        )

        # =================================================
        # 11. Prompt construction
        # =================================================

        prompt = (
            self.prompt_builder.build(
                query=(
                    state.query
                ),
                features=features,
                selected=selected,
                conflicts=conflicts,
                decision=(
                    confidence
                    .decision
                ),
                coverage=coverage,
                adequacy=(
                    confidence
                    .adequacy
                ),
            )
        )

        # =================================================
        # 12. Generation
        # =================================================

        if (
            confidence.decision
            == AnswerDecision.ABSTAIN
        ):
            answer = str(
                self.config[
                    "decision"
                ][
                    "abstain_message"
                ]
            )

            input_tokens = None
            output_tokens = None

        else:
            generated = (
                self.backbone.generate(
                    prompt,
                    temperature=float(
                        self.config[
                            "generation"
                        ][
                            "temperature"
                        ]
                    ),
                    max_new_tokens=int(
                        self.config[
                            "generation"
                        ][
                            "max_new_tokens"
                        ]
                    ),
                )
            )

            answer = (
                generated.text
            )

            input_tokens = (
                generated
                .input_tokens
            )

            output_tokens = (
                generated
                .output_tokens
            )

            if (
                confidence.decision
                == AnswerDecision.CAVEAT
            ):
                prefix = str(
                    self.config[
                        "decision"
                    ][
                        "caveat_prefix"
                    ]
                ).strip()

                if (
                    prefix
                    and not (
                        answer
                        .lower()
                        .startswith(
                            prefix.lower()
                        )
                    )
                ):
                    answer = (
                        f"{prefix} "
                        f"{answer}"
                    )

        selected_ids = [
            candidate.memory_id
            for candidate
            in selected
        ]

        # =================================================
        # 13. Full audit trace
        # =================================================

        full_candidate_score_trace = (
            _build_full_candidate_score_trace(
                raw_candidates=(
                    raw_candidates
                ),
                score_snapshots=(
                    score_snapshots
                ),
                rank_metadata=(
                    rank_metadata
                ),
                route=route,
                resolved_candidates=(
                    resolved_candidates
                ),
                selected=selected,
            )
        )

        # =================================================
        # 14. Final result
        # =================================================

        return C3Result(
            query=(
                state.query
            ),

            user_id=(
                state.user_id
            ),

            answer=answer,

            decision=(
                confidence.decision
            ),

            query_mode=(
                features.query_mode
            ),

            selected_memory_types=[
                memory_type.value
                for memory_type
                in route.selected_types
            ],

            route_scores=(
                route.scores
            ),

            # Backward-compatible legacy field.
            retrieved_ids=(
                ranked_candidate_ids
            ),

            selected_ids=(
                selected_ids
            ),

            conflict_groups=(
                conflicts
            ),

            coverage=(
                coverage
            ),

            adequacy=(
                confidence.adequacy
            ),

            agreement=(
                confidence.agreement
            ),

            final_prompt=(
                prompt
            ),

            latency_ms=(
                (
                    time.perf_counter()
                    - start
                )
                * 1000
            ),

            input_tokens=(
                input_tokens
            ),

            output_tokens=(
                output_tokens
            ),

            selected_evidence=(
                selected
            ),

            raw_retrieved_ids=(
                raw_retrieved_ids
            ),

            ranked_candidate_ids=(
                ranked_candidate_ids
            ),

            debug={
                # -----------------------------------------
                # Legacy RC8.3 trace
                # -----------------------------------------

                "route_reasons": (
                    route.reasons
                ),

                "information_needs": (
                    features
                    .information_needs
                ),

                "evidence_requirement_plan": (
                    evidence_requirement_plan
                ),

                "evidence_requirement_status": (
                    evidence_requirement_status
                ),

                # -----------------------------------------
                # C3-v3 SHADOW trace
                # -----------------------------------------

                "c3_v3_shadow_only": (
                    False
                ),

                "c3_v3_requirement_spec": (
                    c3_v3_compilation
                    .spec
                    .to_dict()
                ),

                "c3_v3_sufficiency": (
                    c3_v3_sufficiency
                    .to_dict()
                ),

                "c3_v3_repair_plan": (
                    c3_v3_repair_plan
                    .to_dict()
                ),
                "c3_v3_repair_execution": {
                    "attempted": (
                        c3_v3_repair_execution
                        .attempted
                    ),
                    "accepted": (
                        c3_v3_repair_execution
                        .accepted
                    ),
                    "repair_retrieved_ids": (
                        c3_v3_repair_execution
                        .repair_retrieved_ids
                    ),
                    "added_candidate_ids": (
                        c3_v3_repair_execution
                        .added_candidate_ids
                    ),
                    "trace": (
                        c3_v3_repair_execution
                        .trace
                    ),
                },
                "c3_v3_legacy_compatibility": {
                    "query_mode_match": (
                        c3_v3_compilation
                        .spec
                        .temporal_mode
                        == (
                            features
                            .query_mode
                        )
                    ),

                    "route_match": (
                        c3_v3_compilation
                        .spec
                        .memory_types
                        == (
                            route
                            .selected_types
                        )
                    ),

                    "information_needs_match": (
                        c3_v3_compilation
                        .spec
                        .information_needs
                        == (
                            features
                            .information_needs
                        )
                    ),
                },

                # -----------------------------------------
                # Existing diagnostic fields
                # -----------------------------------------

                "entities": (
                    features.entities
                ),

                "confidence_components": (
                    confidence.components
                ),

                "candidate_count_raw_retrieved": (
                    len(
                        raw_candidates
                    )
                ),

                "candidate_count_ranked_pool": (
                    len(
                        ranked_pool
                    )
                ),

                "candidate_count_ranked_candidates": (
                    len(
                        ranked_candidates
                    )
                ),

                "candidate_count_resolved": (
                    len(
                        resolved_candidates
                    )
                ),

                "raw_retrieved_ids": (
                    raw_retrieved_ids
                ),

                "ranked_pool_candidate_ids": [
                    candidate.memory_id
                    for candidate
                    in ranked_pool
                ],

                "ranked_candidate_ids": (
                    ranked_candidate_ids
                ),

                "resolved_candidate_ids": [
                    candidate.memory_id
                    for candidate
                    in resolved_candidates
                ],

                "selected_ids": (
                    selected_ids
                ),

                # Legacy trace over the post-budget set.
                "candidate_score_trace": [
                    _candidate_score_trace(
                        candidate
                    )
                    for candidate
                    in ranked_candidates
                ],

                # Full RC5/RC7/RC8 candidate lifecycle trace.
                "full_candidate_score_trace": (
                    full_candidate_score_trace
                ),
            },
        )

    def close(
        self,
    ) -> None:
        if hasattr(
            self.memory_store,
            "close",
        ):
            self.memory_store.close()

        if (
            self.procedure_store
            and hasattr(
                self.procedure_store,
                "close",
            )
        ):
            self.procedure_store.close()