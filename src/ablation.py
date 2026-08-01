from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from .schemas import (
    AnswerDecision,
    ConfidenceResult,
    MemoryCandidate,
    MemoryType,
    RouteDecision,
)
from .text_utils import approximate_token_count


@dataclass(frozen=True, slots=True)
class AblationSettings:
    """Frozen one-component-at-a-time ablation settings."""

    variant: str = "full_c3"
    disable_route_planner: bool = False
    disable_conflict_handling: bool = False
    disable_coverage_confidence_gate: bool = False
    disable_evidence_selector: bool = False

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
    ) -> "AblationSettings":
        raw = config.get("ablation", {}) or {}

        settings = cls(
            variant=str(
                raw.get("variant", "full_c3")
            ),
            disable_route_planner=bool(
                raw.get(
                    "disable_route_planner",
                    False,
                )
            ),
            disable_conflict_handling=bool(
                raw.get(
                    "disable_conflict_handling",
                    False,
                )
            ),
            disable_coverage_confidence_gate=bool(
                raw.get(
                    "disable_coverage_confidence_gate",
                    False,
                )
            ),
            disable_evidence_selector=bool(
                raw.get(
                    "disable_evidence_selector",
                    False,
                )
            ),
        )

        enabled = settings.enabled_flags()

        if len(enabled) > 1:
            raise ValueError(
                "RC8.5 uses one-component-at-a-time "
                f"ablations, but multiple flags are enabled: {enabled}"
            )

        expected_variants = {
            "disable_route_planner": "no_route_planner",
            "disable_conflict_handling": (
                "no_conflict_handling"
            ),
            "disable_coverage_confidence_gate": (
                "no_coverage_confidence_gate"
            ),
            "disable_evidence_selector": (
                "no_evidence_selector"
            ),
        }

        if enabled:
            expected = expected_variants[enabled[0]]

            if settings.variant != expected:
                raise ValueError(
                    f"Ablation variant is {settings.variant!r}, "
                    f"but enabled flag requires {expected!r}."
                )
        elif settings.variant != "full_c3":
            raise ValueError(
                "A non-default ablation variant was supplied "
                "without an enabled ablation flag."
            )

        return settings

    def enabled_flags(self) -> list[str]:
        return [
            name
            for name in (
                "disable_route_planner",
                "disable_conflict_handling",
                "disable_coverage_confidence_gate",
                "disable_evidence_selector",
            )
            if bool(getattr(self, name))
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "disable_route_planner": (
                self.disable_route_planner
            ),
            "disable_conflict_handling": (
                self.disable_conflict_handling
            ),
            "disable_coverage_confidence_gate": (
                self.disable_coverage_confidence_gate
            ),
            "disable_evidence_selector": (
                self.disable_evidence_selector
            ),
        }


def force_all_memory_route() -> RouteDecision:
    """Replace adaptive routing with a fixed three-memory route."""

    return RouteDecision(
        selected_types=list(MemoryType),
        scores={
            memory_type.value: 1.0
            for memory_type in MemoryType
        },
        reasons={
            memory_type.value: [
                "ablation:no_route_planner"
            ]
            for memory_type in MemoryType
        },
    )


def _candidate_text(
    candidate: MemoryCandidate,
) -> str:
    """Mirror the selector's evidence token accounting."""

    metadata_values: list[str] = []

    for key in (
        "instruction",
        "description",
        "response_policy",
        "task_type",
        "matched_triggers",
        "triggers",
        "steps",
    ):
        value = candidate.metadata.get(key)

        if isinstance(value, list):
            metadata_values.extend(
                str(part)
                for part in value
            )
        elif value not in (None, ""):
            metadata_values.append(str(value))

    values = [
        candidate.text,
        candidate.subject or "",
        candidate.predicate or "",
        candidate.object_value or "",
        *metadata_values,
    ]

    return " ".join(
        value
        for value in values
        if value
    )


def select_ranked_top_k(
    *,
    candidates: list[MemoryCandidate],
    config: dict[str, Any],
) -> list[MemoryCandidate]:
    """Select ranked evidence without structured requirements or MMR.

    Candidate order is preserved. The same global evidence count,
    per-memory-type limit and token budget as C3 are retained.
    """

    selection = config["selection"]

    max_evidence = int(
        selection.get("max_evidence", 5)
    )
    max_per_memory_type = int(
        selection.get(
            "max_per_memory_type",
            max_evidence,
        )
    )
    token_budget = int(
        selection.get(
            "evidence_token_budget",
            1200,
        )
    )

    selected: list[MemoryCandidate] = []
    counts: Counter[MemoryType] = Counter()
    used_tokens = 0

    for candidate in candidates:
        if len(selected) >= max_evidence:
            break

        if (
            counts[candidate.memory_type]
            >= max_per_memory_type
        ):
            continue

        token_cost = approximate_token_count(
            _candidate_text(candidate)
        )

        if used_tokens + token_cost > token_budget:
            continue

        candidate.metadata[
            "selector_reason"
        ] = "ablation_ranked_top_k"
        candidate.metadata["evidence_roles"] = []
        candidate.metadata[
            "selector_requirement_gain"
        ] = 0.0
        candidate.metadata[
            "selector_topical_score"
        ] = None

        selected.append(candidate)
        counts[candidate.memory_type] += 1
        used_tokens += token_cost

    return selected


def bypass_coverage_confidence_gate(
    *,
    confidence: ConfidenceResult,
    selected: list[MemoryCandidate],
) -> ConfidenceResult:
    """Prevent coverage/confidence thresholds from controlling answers.

    Empty evidence still produces abstention so the backbone is not asked
    to answer a memory-dependent question without any retrieved context.
    """

    decision = (
        AnswerDecision.DIRECT
        if selected
        else AnswerDecision.ABSTAIN
    )

    components = dict(confidence.components)
    components[
        "ablation_gate_bypassed"
    ] = 1.0

    return ConfidenceResult(
        adequacy=confidence.adequacy,
        coverage=confidence.coverage,
        agreement=confidence.agreement,
        decision=decision,
        components=components,
    )
