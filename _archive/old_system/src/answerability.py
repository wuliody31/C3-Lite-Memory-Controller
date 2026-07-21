from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from src.ranking import candidate_query_coverage


UNCERTAINTY_PATTERNS = (
    "unsupported without evidence",
    "unsupported_without_evidence",
    "state unknown",
    "state_unknown",
    "unknown",
    "uncertain",
    "not stored",
    "check or state unknown",
    "check_or_state_unknown",
    "checking is needed",
    "check needed",
    "avoid exact claims without evidence",
    "avoid_exact_claims_without_evidence",
    "not enough evidence",
    "cannot confirm",
    "without evidence",
    "if evidenced",
    "if_evidenced",
)

NEGATIVE_FACT_PATTERNS = (
    "not required",
    "not_required",
    "no model training",
    "no_model_training",
    "did not",
    "does not",
    "not selected",
    "not_selected",
    "rejected",
    "declined",
)


@dataclass(frozen=True)
class AnswerabilityDecision:
    state: str
    question_mode: str
    direct_support_score: float
    uncertainty_support_score: float
    direct_coverage: float
    uncertainty_coverage: float
    policy_support_score: float
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "question_mode": self.question_mode,
            "direct_support_score": (
                self.direct_support_score
            ),
            "uncertainty_support_score": (
                self.uncertainty_support_score
            ),
            "direct_coverage": self.direct_coverage,
            "uncertainty_coverage": (
                self.uncertainty_coverage
            ),
            "policy_support_score": (
                self.policy_support_score
            ),
            "reasons": list(self.reasons),
        }


def detect_question_mode(
    question: str,
) -> str:
    q = question.lower().strip()

    if (
        q.startswith("should ")
        or "should you" in q
        or "what should" in q
        or "how should" in q
    ):
        return "policy"

    if (
        re.match(
            r"^(did|do|does|is|was|were|have|has|can)\b",
            q,
        )
        or "do you know" in q
    ):
        return "factual_verification"

    return "open"


def classify_memory_role(
    memory: dict[str, Any],
) -> str:
    text = (
        str(memory.get("text", ""))
        .lower()
        .replace("_", " ")
    )

    if any(
        pattern.replace("_", " ") in text
        for pattern in UNCERTAINTY_PATTERNS
    ):
        return "uncertainty"

    if (
        memory.get("memory_type")
        == "procedural"
    ):
        return "policy"

    if any(
        pattern.replace("_", " ") in text
        for pattern in NEGATIVE_FACT_PATTERNS
    ):
        return "direct"

    return "direct"


def assess_answerability(
    question: str,
    selected_memories: list[dict[str, Any]],
) -> AnswerabilityDecision:
    """
    Distinguish evidence that answers a factual proposition from
    evidence that merely states an uncertainty policy.

    This is essential for selective abstention:
    - "model training is not required" can directly answer No;
    - "deployment is unsupported without evidence" supports abstention,
      not a fabricated deployment fact.
    """
    mode = detect_question_mode(question)

    exactness_cue = any(
        cue in question.lower()
        for cue in (
            "exact",
            "definitely",
            "explicitly",
            "confirm",
            "approved",
            "approve",
        )
    )

    direct_memories = [
        memory
        for memory in selected_memories
        if classify_memory_role(memory)
        == "direct"
    ]

    uncertainty_memories = [
        memory
        for memory in selected_memories
        if classify_memory_role(memory)
        == "uncertainty"
    ]

    policy_memories = [
        memory
        for memory in selected_memories
        if classify_memory_role(memory)
        == "policy"
    ]

    def support_score(
        memories: list[dict[str, Any]],
    ) -> float:
        return max(
            (
                float(
                    memory.get("evidence_score")
                    or 0.0
                )
                * candidate_query_coverage(
                    question,
                    str(memory.get("text", "")),
                )
                for memory in memories
            ),
            default=0.0,
        )

    def maximum_coverage(
        memories: list[dict[str, Any]],
    ) -> float:
        return max(
            (
                candidate_query_coverage(
                    question,
                    str(memory.get("text", "")),
                )
                for memory in memories
            ),
            default=0.0,
        )

    direct_support_score = support_score(
        direct_memories
    )
    uncertainty_support_score = support_score(
        uncertainty_memories
    )
    policy_support_score = support_score(
        policy_memories
    )

    direct_coverage = maximum_coverage(
        direct_memories
    )
    uncertainty_coverage = maximum_coverage(
        uncertainty_memories
    )

    top_selected_score = max(
        (
            float(
                memory.get("evidence_score")
                or 0.0
            )
            for memory in selected_memories
        ),
        default=0.0,
    )

    reasons: list[str] = [
        f"question_mode={mode}",
        (
            "direct_support="
            f"{direct_support_score:.3f}"
        ),
        (
            "uncertainty_support="
            f"{uncertainty_support_score:.3f}"
        ),
        (
            "direct_coverage="
            f"{direct_coverage:.3f}"
        ),
        (
            "uncertainty_coverage="
            f"{uncertainty_coverage:.3f}"
        ),
    ]

    if mode == "policy":
        combined_support = max(
            policy_support_score,
            direct_support_score,
            uncertainty_support_score,
        )

        if (
            combined_support >= 0.12
            and top_selected_score >= 0.50
        ):
            state = "answerable"
        else:
            state = "insufficient"

    elif mode == "factual_verification":
        if (
            direct_support_score >= 0.22
            and direct_coverage >= 0.28
        ):
            state = "answerable"

        elif (
            uncertainty_support_score >= 0.16
            and uncertainty_coverage >= 0.20
        ):
            state = "insufficient"

        elif (
            exactness_cue
            and direct_coverage < 0.35
        ):
            state = "insufficient"

        else:
            state = "uncertain"

    else:
        if max(
            direct_support_score,
            policy_support_score,
        ) >= 0.15:
            state = "answerable"

        elif uncertainty_support_score >= 0.15:
            state = "uncertain"

        else:
            state = "insufficient"

    reasons.append(
        f"answerability_state={state}"
    )

    return AnswerabilityDecision(
        state=state,
        question_mode=mode,
        direct_support_score=round(
            direct_support_score,
            6,
        ),
        uncertainty_support_score=round(
            uncertainty_support_score,
            6,
        ),
        direct_coverage=round(
            direct_coverage,
            6,
        ),
        uncertainty_coverage=round(
            uncertainty_coverage,
            6,
        ),
        policy_support_score=round(
            policy_support_score,
            6,
        ),
        reasons=tuple(reasons),
    )
