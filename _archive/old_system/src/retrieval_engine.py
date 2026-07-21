from __future__ import annotations

from typing import Any

from src.ranking import rank_candidates
EPISODIC_ROUTE = 'neo4j_episodic_graph'
SEMANTIC_ROUTE = 'neo4j_semantic_graph'
PROCEDURAL_ROUTE = 'json_procedural_rules'


class MemoryRetrievalEngine:
    """
    Shared retrieval primitive for C3-Lite and retrieval baselines.
    """

    def __init__(self, neo4j_adapter, procedural_matcher):
        self.neo4j = neo4j_adapter
        self.procedural = procedural_matcher

    def retrieve_episodic(
        self,
        user_id: str,
        question: str,
        limit: int = 8,
    ) -> list[dict[str, Any]]:
        return rank_candidates(
            question,
            self.neo4j.get_episodic_candidates(user_id),
            "episodic",
            limit,
        )

    def retrieve_semantic(
        self,
        user_id: str,
        question: str,
        limit: int = 8,
        include_archived: bool = False,
    ) -> list[dict[str, Any]]:
        return rank_candidates(
            question,
            self.neo4j.get_semantic_candidates(
                user_id,
                include_archived=include_archived,
            ),
            "semantic",
            limit,
        )

    def retrieve_procedural(
        self,
        user_id: str,
        question: str,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        return rank_candidates(
            question,
            self.procedural.get_candidates(user_id),
            "procedural",
            limit,
        )

    def retrieve_routes(
        self,
        user_id: str,
        question: str,
        routes: list[str] | tuple[str, ...],
        include_archived_semantic: bool = False,
        per_route_limit: int = 8,
    ) -> dict[str, list[dict[str, Any]]]:
        result: dict[str, list[dict[str, Any]]] = {}

        if EPISODIC_ROUTE in routes:
            result[EPISODIC_ROUTE] = self.retrieve_episodic(
                user_id,
                question,
                limit=per_route_limit,
            )

        if SEMANTIC_ROUTE in routes:
            result[SEMANTIC_ROUTE] = self.retrieve_semantic(
                user_id,
                question,
                limit=per_route_limit,
                include_archived=include_archived_semantic,
            )

        if PROCEDURAL_ROUTE in routes:
            result[PROCEDURAL_ROUTE] = self.retrieve_procedural(
                user_id,
                question,
                limit=min(5, per_route_limit),
            )

        return result

    @staticmethod
    def flatten(
        bundle: dict[str, list[dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        memories: dict[str, dict[str, Any]] = {}

        for route_items in bundle.values():
            for item in route_items:
                existing = memories.get(item["id"])

                if (
                    existing is None
                    or item.get("evidence_score", 0.0)
                    > existing.get("evidence_score", 0.0)
                ):
                    memories[item["id"]] = item

        return sorted(
            memories.values(),
            key=lambda item: item.get(
                "evidence_score",
                0.0,
            ),
            reverse=True,
        )
