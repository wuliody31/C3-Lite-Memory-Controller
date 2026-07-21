from __future__ import annotations

from pathlib import Path
from typing import Any

from .schemas import (
    AnswerDecision,
    ConflictGroup,
    MemoryCandidate,
    MemoryType,
    QueryFeatures,
)


class PromptBuilder:
    """Build the LLM-facing prompt from controller-selected evidence.

    Internal diagnostics such as ranking scores, confidence values and raw
    controller labels remain available in the full trace, but are deliberately
    omitted from the prompt sent to the frozen language-model backbone.
    """

    INTERNAL_VISIBILITY_POLICY = (
        "Do not expose internal ranking scores, confidence values, route "
        "scores, controller metadata, conflict IDs, or raw status labels "
        "unless the user explicitly requests a technical explanation."
    )

    def __init__(
        self,
        config: dict[str, Any],
        template_path: str | Path,
    ) -> None:
        self.config = config
        self.template = Path(template_path).read_text(encoding="utf-8")

    def build(
        self,
        *,
        query: str,
        features: QueryFeatures,
        selected: list[MemoryCandidate],
        conflicts: list[ConflictGroup],
        decision: AnswerDecision,
        coverage: float,
        adequacy: float,
    ) -> str:
        grouped = {
            memory_type: [
                item
                for item in selected
                if item.memory_type == memory_type
            ]
            for memory_type in MemoryType
        }

        decision_requirement = {
            AnswerDecision.DIRECT: (
                "Answer directly and concisely. Cite selected memory IDs "
                "when an explanation or evidence attribution is requested."
            ),
            AnswerDecision.CAVEAT: (
                "Give only the supported part and explicitly identify "
                "uncertainty or missing evidence."
            ),
            AnswerDecision.ABSTAIN: (
                "Do not guess. State that the stored evidence is "
                "insufficient."
            ),
        }[decision]

        output_requirements = (
            f"{decision_requirement}\n"
            f"{self.INTERNAL_VISIBILITY_POLICY}"
        )

        return self.template.format(
            query=query,
            query_mode=features.query_mode.value,
            information_needs=self._format_list(
                features.information_needs
            ),
            episodic_evidence=self._format_evidence(
                grouped[MemoryType.EPISODIC]
            ),
            semantic_evidence=self._format_evidence(
                grouped[MemoryType.SEMANTIC]
            ),
            procedural_evidence=self._format_evidence(
                grouped[MemoryType.PROCEDURAL]
            ),
            conflict_notes=self._format_conflicts(conflicts),
            decision=decision.value,
            coverage=coverage,
            adequacy=adequacy,
            output_requirements=output_requirements,
        ).strip()

    @staticmethod
    def _format_list(items: list[str]) -> str:
        if not items:
            return "- None identified"
        return "\n".join(f"- {item}" for item in items)

    @staticmethod
    def _format_evidence(
        items: list[MemoryCandidate],
    ) -> str:
        """Format evidence without exposing internal ranking metadata."""
        if not items:
            return "- None"

        output: list[str] = []
        for item in items:
            public_metadata: list[str] = []

            if item.timestamp:
                public_metadata.append(
                    f"time={item.timestamp.isoformat()}"
                )

            # A resolved temporal role helps the model distinguish an older
            # version from the current endpoint. It is not a score or a raw
            # database status label.
            if item.resolution_action in {
                "historical",
                "current_endpoint",
                "preferred_historical",
            }:
                public_metadata.append(
                    f"role={item.resolution_action}"
                )

            suffix = (
                f" ({'; '.join(public_metadata)})"
                if public_metadata
                else ""
            )
            output.append(
                f"[{item.memory_id}]{suffix}\n{item.text}"
            )

        return "\n\n".join(output)

    @staticmethod
    def _format_conflicts(
        items: list[ConflictGroup],
    ) -> str:
        """Provide semantic resolution notes, not internal conflict metadata."""
        if not items:
            return "- No detected conflict."

        notes: list[str] = []
        for item in items:
            if item.unresolved:
                notes.append(
                    "- A retrieved memory conflict remains unresolved. "
                    "Answer cautiously and do not present either version "
                    "as certain."
                )
                continue

            explanation = item.explanation.strip()
            if explanation:
                notes.append(f"- {explanation}")
            else:
                notes.append(
                    "- The controller resolved a memory-version conflict "
                    "and retained the relevant historical context."
                )

        return "\n".join(dict.fromkeys(notes))