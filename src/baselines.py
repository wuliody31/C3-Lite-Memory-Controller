from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .backbones import Backbone
from .prompt_builder import PromptBuilder
from .query_analyzer import QueryAnalyzer
from .schemas import (
    AnswerDecision,
    C3Result,
    MemoryCandidate,
    MemoryType,
    QueryState,
    RouteDecision,
    unique_candidates,
)
from .shared_ranker import SharedRanker


class BaselineRunner:
    """Run the non-C3 comparison methods used in the evaluation pipeline.

    Supported methods:

    - ``no_memory``
    - ``episodic_only``
    - ``semantic_only``
    - ``procedural_only``
    - ``simple_retrieval``
    - ``all_memory``

    The C3 method itself is run by :class:`src.pipeline.C3Pipeline` and is
    therefore intentionally not included in ``METHODS``.
    """

    METHODS = {
        "no_memory",
        "episodic_only",
        "semantic_only",
        "procedural_only",
        "simple_retrieval",
        "all_memory",
    }

    def __init__(
        self,
        *,
        config: dict[str, Any],
        memory_store: Any,
        procedure_store: Any | None,
        backbone: Backbone,
        prompt_template: str | Path,
    ) -> None:
        self.config = config
        self.memory_store = memory_store
        self.procedure_store = procedure_store
        self.backbone = backbone

        self.analyzer = QueryAnalyzer(config)
        self.ranker = SharedRanker(config)
        self.prompt_builder = PromptBuilder(config, prompt_template)

    def answer(
        self,
        method: str,
        state: QueryState,
    ) -> C3Result:
        """Run one baseline method for one query."""
        if method not in self.METHODS:
            supported = ", ".join(sorted(self.METHODS))
            raise ValueError(
                f"Unknown baseline method: {method!r}. "
                f"Supported methods: {supported}."
            )

        started = time.perf_counter()
        features = self.analyzer.analyse(state.query)
        selected_types = self._memory_types_for(method)

        route = RouteDecision(
            selected_types=selected_types,
            scores={
                memory_type.value: float(memory_type in selected_types)
                for memory_type in MemoryType
            },
            reasons={
                memory_type.value: (
                    [f"baseline:{method}"]
                    if memory_type in selected_types
                    else []
                )
                for memory_type in MemoryType
            },
        )

        raw_candidates = self._retrieve(
            method=method,
            state=state,
            features=features,
            selected_types=selected_types,
        )

        ranked = self.ranker.rank(
            candidates=unique_candidates(raw_candidates),
            features=features,
            route=route,
            current_time=state.current_time,
        )

        selected = self._select(
            method=method,
            ranked=ranked,
        )

        coverage = 1.0 if selected else 0.0
        adequacy = max(
            (candidate.final_score for candidate in selected),
            default=0.0,
        )

        # Baselines are intentionally not allowed to use the C3 confidence
        # controller. They pass their selected context directly to the same
        # frozen backbone so the comparison isolates memory-control effects.
        decision = AnswerDecision.DIRECT

        prompt = self.prompt_builder.build(
            query=state.query,
            features=features,
            selected=selected,
            conflicts=[],
            decision=decision,
            coverage=coverage,
            adequacy=adequacy,
        )

        generated = self.backbone.generate(
            prompt,
            temperature=float(
                self.config["generation"]["temperature"]
            ),
            max_new_tokens=int(
                self.config["generation"]["max_new_tokens"]
            ),
        )

        latency_ms = (
            time.perf_counter() - started
        ) * 1000.0

        return C3Result(
            query=state.query,
            user_id=state.user_id,
            answer=generated.text,
            decision=decision,
            query_mode=features.query_mode,
            selected_memory_types=[
                memory_type.value
                for memory_type in selected_types
            ],
            route_scores=route.scores,
            retrieved_ids=[
                candidate.memory_id for candidate in ranked
            ],
            selected_ids=[
                candidate.memory_id for candidate in selected
            ],
            conflict_groups=[],
            coverage=coverage,
            adequacy=adequacy,
            agreement=1.0,
            final_prompt=prompt,
            latency_ms=latency_ms,
            input_tokens=generated.input_tokens,
            output_tokens=generated.output_tokens,
            selected_evidence=selected,
            debug={
                "baseline_method": method,
                "candidate_count_before_ranking": len(
                    unique_candidates(raw_candidates)
                ),
                "candidate_count_after_ranking": len(ranked),
            },
        )

    @staticmethod
    def _memory_types_for(
        method: str,
    ) -> list[MemoryType]:
        mapping = {
            "no_memory": [],
            "episodic_only": [MemoryType.EPISODIC],
            "semantic_only": [MemoryType.SEMANTIC],
            "procedural_only": [MemoryType.PROCEDURAL],
        }
        return mapping.get(method, list(MemoryType))

    def _retrieve(
        self,
        *,
        method: str,
        state: QueryState,
        features: Any,
        selected_types: list[MemoryType],
    ) -> list[MemoryCandidate]:
        candidates: list[MemoryCandidate] = []
        top_k = self.config["retrieval"]["top_k"]

        for memory_type in selected_types:
            if memory_type == MemoryType.PROCEDURAL:
                if self.procedure_store is None:
                    continue
                source = self.procedure_store
            else:
                source = self.memory_store

            candidates.extend(
                source.retrieve(
                    memory_type=memory_type,
                    state=state,
                    features=features,
                    top_k=int(top_k[memory_type.value]),
                    include_archived=(method == "all_memory"),
                )
            )

        return candidates

    def _select(
        self,
        *,
        method: str,
        ranked: list[MemoryCandidate],
    ) -> list[MemoryCandidate]:
        if method == "no_memory":
            return []

        if method == "all_memory":
            top_k = self.config["retrieval"]["top_k"]
            limit = sum(int(value) for value in top_k.values())
        else:
            limit = int(
                self.config["selection"]["max_evidence"]
            )

        return ranked[:limit]
