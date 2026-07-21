from __future__ import annotations

from typing import Any

from src.ranking import token_similarity


ROUTE_TO_MEMORY_TYPE = {
    "neo4j_episodic_graph": "episodic",
    "neo4j_semantic_graph": "semantic",
    "json_procedural_rules": "procedural",
}

HISTORY_AWARE_QUERY_TYPES = {
    "episodic_recall",
    "temporal_update",
    "conflict_resolution",
    "explainability",
}


def adaptive_evidence_budget(
    routes: list[str],
    query_type: str,
) -> int:
    """
    Adaptive budget:
    1 route  -> usually 4 memories
    2 routes -> usually 6-7 memories
    3 routes -> usually 8-9 memories
    """
    history_bonus = (
        1
        if query_type
        in {
            "temporal_update",
            "conflict_resolution",
            "explainability",
        }
        else 0
    )

    return min(
        9,
        2 + 2 * len(routes) + history_bonus,
    )


def select_evidence_mmr(
    resolved: dict[str, Any],
    routes: list[str],
    query_type: str,
    lambda_relevance: float = 0.78,
) -> list[dict[str, Any]]:
    """
    Maximal Marginal Relevance evidence selection.

    MMR(m) =
        lambda * evidence_score
        - (1-lambda) * max_similarity_to_selected

    The selector preserves:
    - memory-type diversity across routed sources;
    - one historical memory for history-aware tasks when relevant;
    - relevance while penalising redundant memories.
    """
    usable = resolved["usable_memories"]

    if not usable:
        return []

    budget = adaptive_evidence_budget(
        routes,
        query_type,
    )

    selected: dict[str, dict[str, Any]] = {}

    top_score = max(
        float(
            memory.get("evidence_score")
            or 0.0
        )
        for memory in usable
    )

    score_floor = max(
        0.22,
        top_score * 0.40,
    )

    # Reserve the strongest useful evidence for each selected route.
    for route in routes:
        memory_type = ROUTE_TO_MEMORY_TYPE[
            route
        ]

        candidates = [
            memory
            for memory in usable
            if (
                memory.get("memory_type")
                == memory_type
                and float(
                    memory.get("evidence_score")
                    or 0.0
                )
                >= score_floor
            )
        ]

        if candidates:
            best = max(
                candidates,
                key=lambda memory: float(
                    memory.get("evidence_score")
                    or 0.0
                ),
            )
            selected[best["id"]] = best

    # Preserve historical context when the task explicitly needs it.
    if (
        query_type
        in HISTORY_AWARE_QUERY_TYPES
        and resolved["historical_memories"]
    ):
        historical_floor = max(
            0.18,
            top_score * 0.30,
        )

        historical_candidates = [
            memory
            for memory
            in resolved["historical_memories"]
            if float(
                memory.get("evidence_score")
                or 0.0
            )
            >= historical_floor
        ]

        if historical_candidates:
            best_historical = max(
                historical_candidates,
                key=lambda memory: float(
                    memory.get("evidence_score")
                    or 0.0
                ),
            )
            selected[
                best_historical["id"]
            ] = best_historical

    candidates = [
        memory
        for memory in usable
        if (
            memory["id"] not in selected
            and float(
                memory.get("evidence_score")
                or 0.0
            )
            >= score_floor
        )
    ]

    while (
        len(selected) < budget
        and candidates
    ):
        scored_candidates: list[
            tuple[float, dict[str, Any]]
        ] = []

        for candidate in candidates:
            redundancy = max(
                (
                    token_similarity(
                        str(
                            candidate.get(
                                "text",
                                "",
                            )
                        ),
                        str(
                            selected_memory.get(
                                "text",
                                "",
                            )
                        ),
                    )
                    for selected_memory
                    in selected.values()
                ),
                default=0.0,
            )

            mmr_score = (
                lambda_relevance
                * float(
                    candidate.get(
                        "evidence_score",
                        0.0,
                    )
                )
                - (
                    1.0 - lambda_relevance
                )
                * redundancy
            )

            scored_candidates.append(
                (mmr_score, candidate)
            )

        best_mmr, best_candidate = max(
            scored_candidates,
            key=lambda item: item[0],
        )

        if best_mmr < 0.20:
            break

        selected[
            best_candidate["id"]
        ] = best_candidate

        candidates = [
            candidate
            for candidate in candidates
            if (
                candidate["id"]
                != best_candidate["id"]
            )
        ]

    return list(selected.values())[:budget]
