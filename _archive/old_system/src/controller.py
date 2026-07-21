from __future__ import annotations

from typing import Any

from src.query_analyzer import analyse_query
from src.route_planner import plan_routes
from src.conflict_resolver import ConflictResolver
from src.evidence_selector import select_evidence_mmr
from src.confidence_calibrator import calibrate_confidence
from src.answer_generator import generate_answer


class C3LiteController:
    """
    C3-Lite Algorithm v2.1.

    C1 Context-adaptive:
        multi-signal query analysis and route planning.

    C2 Conflict-aware:
        relation-semantic checking and bidirectional graph expansion.

    C3 Confidence-calibrated:
        answerability analysis plus evidence adequacy calibration.

    Evidence selection:
        adaptive budget plus MMR redundancy control.
    """

    def __init__(
        self,
        retrieval_engine,
        neo4j_adapter,
        answer_mode: str = "mock",
        openai_model: str = "gpt-4o-mini",
        openai_api_key: str | None = None,
    ):
        self.retrieval = retrieval_engine
        self.resolver = ConflictResolver(
            neo4j_adapter
        )
        self.answer_mode = answer_mode
        self.openai_model = openai_model
        self.openai_api_key = openai_api_key

    def answer(
        self,
        user_id: str,
        question: str,
    ) -> dict[str, Any]:
        profile = analyse_query(question)
        route_plan = plan_routes(
            question,
            profile,
        )

        bundle = self.retrieval.retrieve_routes(
            user_id=user_id,
            question=question,
            routes=route_plan.routes,
            include_archived_semantic=(
                route_plan
                .include_archived_semantic
            ),
            per_route_limit=8,
        )

        retrieved = self.retrieval.flatten(
            bundle
        )

        resolution = self.resolver.resolve(
            retrieved,
            query_type=profile.primary_intent,
        )

        selected = select_evidence_mmr(
            resolved=resolution,
            routes=list(route_plan.routes),
            query_type=profile.primary_intent,
        )

        confidence = calibrate_confidence(
            question=question,
            selected_memories=selected,
            route_confidence=route_plan.confidence,
            unresolved_conflicts=resolution[
                "unresolved_conflicts"
            ],
            query_type=profile.primary_intent,
            resolution=resolution,
        )

        if confidence.state == "abstain":
            answer = (
                "There is not enough stored evidence "
                "to answer this question confidently. "
                f"Confidence score: "
                f"{confidence.score:.3f}."
            )

        else:
            answer = generate_answer(
                self.answer_mode,
                question,
                selected,
                resolution,
                [
                    memory
                    for memory in selected
                    if (
                        memory.get("memory_type")
                        == "procedural"
                    )
                ],
                profile.primary_intent,
                self.openai_model,
                self.openai_api_key,
            )

            if confidence.state == "caveat":
                answer = (
                    "Based on limited stored evidence, "
                    + answer
                )

        return {
            "question": question,
            "query_type": profile.primary_intent,
            "predicted_route": list(
                route_plan.routes
            ),
            "query_profile": profile.to_dict(),
            "route_plan": route_plan.to_dict(),
            "retrieved_memory_ids": [
                memory["id"]
                for memory in retrieved
            ],
            "used_memory_ids": [
                memory["id"]
                for memory in selected
            ],
            "outdated_memory_ids": resolution[
                "outdated_memory_ids"
            ],
            "historical_memory_ids": [
                memory["id"]
                for memory
                in resolution[
                    "historical_memories"
                ]
            ],
            "newer_memory_ids": resolution[
                "newer_memory_ids"
            ],
            "conflict_notes": resolution[
                "conflict_notes"
            ],
            "supportive_notes": resolution[
                "supportive_notes"
            ],
            "unresolved_conflicts": resolution[
                "unresolved_conflicts"
            ],
            "confidence_score": confidence.score,
            "confidence_state": confidence.state,
            "confidence_reasons": list(
                confidence.reasons
            ),
            "query_coverage": (
                confidence.query_coverage
            ),
            "answerability": (
                confidence.answerability
            ),
            "answer": answer,
        }
