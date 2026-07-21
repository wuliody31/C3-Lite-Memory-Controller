from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .backbones import Backbone
from .confidence_controller import ConfidenceController
from .conflict_detector import ConflictDetector
from .conflict_resolver import ConflictResolver
from .coverage_estimator import CoverageEstimator
from .evidence_selector import EvidenceSelector
from .prompt_builder import PromptBuilder
from .query_analyzer import QueryAnalyzer
from .route_planner import RoutePlanner
from .schemas import (
    AnswerDecision,
    C3Result,
    MemoryCandidate,
    MemoryType,
    QueryState,
    unique_candidates,
)
from .shared_ranker import SharedRanker


def _candidate_score_trace(
    candidate: MemoryCandidate,
) -> dict[str, Any]:
    return {
        "memory_id": candidate.memory_id,
        "memory_type": candidate.memory_type.value,
        "status": candidate.status,
        "lexical_score": candidate.lexical_score,
        "graph_entity_score": candidate.graph_entity_score,
        "temporal_task_score": (
            candidate.temporal_task_score
        ),
        "validity_score": candidate.validity_score,
        "source_confidence_score": (
            candidate.source_confidence_score
        ),
        "route_compatibility_score": (
            candidate.route_compatibility_score
        ),
        "conflict_penalty": candidate.conflict_penalty,
        "final_score": candidate.final_score,
        "resolution_action": candidate.resolution_action,
    }


class C3Pipeline:
    def __init__(
        self,
        *,
        config: dict[str, Any],
        memory_store: Any,
        procedure_store: Any | None,
        backbone: Backbone,
        prompt_template: str | Path,
    ):
        self.config = config
        self.memory_store = memory_store
        self.procedure_store = procedure_store
        self.backbone = backbone

        self.analyzer = QueryAnalyzer(config)
        self.router = RoutePlanner(config)
        self.ranker = SharedRanker(config)
        self.detector = ConflictDetector(config)
        self.resolver = ConflictResolver(config)
        self.coverage = CoverageEstimator(config)
        self.selector = EvidenceSelector(
            config,
            self.coverage,
        )
        self.confidence = ConfidenceController(config)
        self.prompt_builder = PromptBuilder(
            config,
            prompt_template,
        )

    def answer(self, state: QueryState) -> C3Result:
        start = time.perf_counter()

        features = self.analyzer.analyse(state.query)
        route = self.router.plan(features)

        include_archived = (
            features.query_mode.value
            in set(
                self.config["retrieval"][
                    "include_archived_for"
                ]
            )
            or features.asks_conflict
            or features.asks_explanation
        )

        top_k = self.config["retrieval"]["top_k"]
        multiplier = int(
            self.config["retrieval"].get(
                "candidate_pool_multiplier",
                1,
            )
        )

        retrieved_candidates: list[MemoryCandidate] = []

        for memory_type in route.selected_types:
            source = (
                self.procedure_store
                if (
                    memory_type == MemoryType.PROCEDURAL
                    and self.procedure_store
                )
                else self.memory_store
            )

            retrieved_candidates.extend(
                source.retrieve(
                    memory_type=memory_type,
                    state=state,
                    features=features,
                    top_k=(
                        int(top_k[memory_type.value])
                        * multiplier
                    ),
                    include_archived=include_archived,
                )
            )

        raw_candidates = unique_candidates(
            retrieved_candidates
        )
        raw_retrieved_ids = [
            candidate.memory_id
            for candidate in raw_candidates
        ]

        ranked_pool = self.ranker.rank(
            candidates=raw_candidates,
            features=features,
            route=route,
            current_time=state.current_time,
        )

        limited: list[MemoryCandidate] = []

        for memory_type in route.selected_types:
            limited.extend(
                [
                    candidate
                    for candidate in ranked_pool
                    if candidate.memory_type == memory_type
                ][: int(top_k[memory_type.value])]
            )

        ranked_candidates = sorted(
            unique_candidates(limited),
            key=lambda candidate: candidate.final_score,
            reverse=True,
        )

        ranked_candidate_ids = [
            candidate.memory_id
            for candidate in ranked_candidates
        ]

        conflicts = self.detector.detect(
            ranked_candidates
        )

        resolved_candidates, conflicts = (
            self.resolver.resolve(
                candidates=ranked_candidates,
                conflicts=conflicts,
                features=features,
            )
        )

        selected = self.selector.select(
            candidates=resolved_candidates,
            features=features,
            route=route,
            conflicts=conflicts,
        )

        coverage = self.coverage.compute(
            features.information_needs,
            selected,
        )

        confidence = self.confidence.evaluate(
            selected=selected,
            route=route,
            conflicts=conflicts,
            coverage=coverage,
        )

        prompt = self.prompt_builder.build(
            query=state.query,
            features=features,
            selected=selected,
            conflicts=conflicts,
            decision=confidence.decision,
            coverage=coverage,
            adequacy=confidence.adequacy,
        )

        if confidence.decision == AnswerDecision.ABSTAIN:
            answer = str(
                self.config["decision"]["abstain_message"]
            )
            input_tokens = None
            output_tokens = None
        else:
            generated = self.backbone.generate(
                prompt,
                temperature=float(
                    self.config["generation"]["temperature"]
                ),
                max_new_tokens=int(
                    self.config["generation"][
                        "max_new_tokens"
                    ]
                ),
            )

            answer = generated.text
            input_tokens = generated.input_tokens
            output_tokens = generated.output_tokens

            if confidence.decision == AnswerDecision.CAVEAT:
                prefix = str(
                    self.config["decision"]["caveat_prefix"]
                ).strip()

                if (
                    prefix
                    and not answer.lower().startswith(
                        prefix.lower()
                    )
                ):
                    answer = f"{prefix} {answer}"

        selected_ids = [
            candidate.memory_id
            for candidate in selected
        ]

        return C3Result(
            query=state.query,
            user_id=state.user_id,
            answer=answer,
            decision=confidence.decision,
            query_mode=features.query_mode,
            selected_memory_types=[
                memory_type.value
                for memory_type in route.selected_types
            ],
            route_scores=route.scores,

            # Backward-compatible legacy field.
            retrieved_ids=ranked_candidate_ids,

            selected_ids=selected_ids,
            conflict_groups=conflicts,
            coverage=coverage,
            adequacy=confidence.adequacy,
            agreement=confidence.agreement,
            final_prompt=prompt,
            latency_ms=(
                time.perf_counter() - start
            )
            * 1000,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            selected_evidence=selected,
            raw_retrieved_ids=raw_retrieved_ids,
            ranked_candidate_ids=ranked_candidate_ids,
            debug={
                "route_reasons": route.reasons,
                "information_needs": (
                    features.information_needs
                ),
                "entities": features.entities,
                "confidence_components": (
                    confidence.components
                ),
                "candidate_count_raw_retrieved": len(
                    raw_candidates
                ),
                "candidate_count_ranked_pool": len(
                    ranked_pool
                ),
                "candidate_count_ranked_candidates": len(
                    ranked_candidates
                ),
                "candidate_count_resolved": len(
                    resolved_candidates
                ),
                "raw_retrieved_ids": (
                    raw_retrieved_ids
                ),
                "ranked_candidate_ids": (
                    ranked_candidate_ids
                ),
                "resolved_candidate_ids": [
                    candidate.memory_id
                    for candidate in resolved_candidates
                ],
                "selected_ids": selected_ids,
                "candidate_score_trace": [
                    _candidate_score_trace(candidate)
                    for candidate in ranked_candidates
                ],
            },
        )

    def close(self) -> None:
        if hasattr(self.memory_store, "close"):
            self.memory_store.close()

        if (
            self.procedure_store
            and hasattr(self.procedure_store, "close")
        ):
            self.procedure_store.close()
