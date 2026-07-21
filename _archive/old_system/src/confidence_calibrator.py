from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Any

from src.answerability import assess_answerability
from src.ranking import query_coverage


@dataclass(frozen=True)
class ConfidenceDecision:
    score: float
    state: str
    reasons: tuple[str, ...]
    answerability: dict[str, Any]
    query_coverage: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "state": self.state,
            "reasons": list(self.reasons),
            "answerability": self.answerability,
            "query_coverage": self.query_coverage,
        }


def calibrate_confidence(
    question: str,
    selected_memories: list[dict[str, Any]],
    route_confidence: float,
    unresolved_conflicts: list[dict[str, Any]],
    query_type: str,
    resolution: dict[str, Any],
) -> ConfidenceDecision:
    if not selected_memories:
        answerability = assess_answerability(
            question,
            [],
        )

        return ConfidenceDecision(
            score=0.0,
            state="abstain",
            reasons=(
                "no relevant memory evidence selected",
            ),
            answerability=answerability.to_dict(),
            query_coverage=0.0,
        )

    evidence_scores = sorted(
        [
            float(
                memory.get("evidence_score")
                or 0.0
            )
            for memory in selected_memories
        ],
        reverse=True,
    )

    top_evidence = evidence_scores[0]
    mean_top_three = mean(
        evidence_scores[:3]
    )

    agreement = max(
        0.0,
        1.0
        - min(
            0.8,
            0.35
            * len(unresolved_conflicts),
        ),
    )

    coverage = query_coverage(
        question,
        [
            str(memory.get("text", ""))
            for memory in selected_memories
        ],
    )

    memory_types = {
        memory.get("memory_type")
        for memory in selected_memories
    }

    evidence_roles = {
        memory.get("evidence_role")
        for memory in selected_memories
    }

    structural_support = 0.0

    if (
        query_type
        in {
            "temporal_update",
            "conflict_resolution",
        }
        and {
            "current",
            "historical",
        }
        <= evidence_roles
    ):
        structural_support = 1.0

    elif (
        query_type == "explainability"
        and len(memory_types) >= 2
    ):
        structural_support = 0.8

    elif (
        query_type == "procedural_following"
        and "procedural" in memory_types
    ):
        structural_support = 1.0

    elif (
        len([
            memory
            for memory in selected_memories
            if float(
                memory.get("evidence_score")
                or 0.0
            )
            >= 0.55
        ])
        >= 2
    ):
        structural_support = 0.6

    answerability = assess_answerability(
        question,
        selected_memories,
    )

    answerability_state = answerability.state

    # For explicit temporal/conflict tasks, a current+historical
    # evidence pair itself establishes structural answerability.
    if (
        query_type
        in {
            "temporal_update",
            "conflict_resolution",
        }
        and structural_support >= 1.0
    ):
        answerability_state = "answerable"

    adequacy = (
        0.30 * top_evidence
        + 0.15 * mean_top_three
        + 0.15 * route_confidence
        + 0.20 * coverage
        + 0.10 * agreement
        + 0.10 * structural_support
    )

    reasons = (
        f"top_evidence={top_evidence:.3f}",
        f"mean_top3={mean_top_three:.3f}",
        (
            "route_confidence="
            f"{route_confidence:.3f}"
        ),
        f"query_coverage={coverage:.3f}",
        f"agreement={agreement:.3f}",
        (
            "structural_support="
            f"{structural_support:.3f}"
        ),
        (
            "answerability="
            f"{answerability_state}"
        ),
    )

    if answerability_state == "insufficient":
        state = "abstain"

    elif answerability_state == "uncertain":
        state = (
            "caveat"
            if adequacy >= 0.45
            else "abstain"
        )

    elif adequacy >= 0.58:
        state = "direct"

    elif adequacy >= 0.45:
        state = "caveat"

    else:
        state = "abstain"

    return ConfidenceDecision(
        score=round(adequacy, 6),
        state=state,
        reasons=reasons,
        answerability=answerability.to_dict(),
        query_coverage=round(coverage, 6),
    )
