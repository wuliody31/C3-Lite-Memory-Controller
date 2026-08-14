from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .schemas import MemoryCandidate


@dataclass(frozen=True, slots=True)
class TransitionCandidateFidelity:
    """Deterministic semantic-bridge trace for one transition candidate."""

    memory_id: str
    historical_bridge_score: float
    current_bridge_score: float
    bridge_fidelity: float
    bridge_mean: float
    axis_alignment: float
    best_historical_memory_id: str | None = None
    best_current_memory_id: str | None = None
    bridge_both_sides: bool = False
    content_tokens: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "historical_bridge_score": self.historical_bridge_score,
            "current_bridge_score": self.current_bridge_score,
            "bridge_fidelity": self.bridge_fidelity,
            "bridge_mean": self.bridge_mean,
            "axis_alignment": self.axis_alignment,
            "best_historical_memory_id": self.best_historical_memory_id,
            "best_current_memory_id": self.best_current_memory_id,
            "bridge_both_sides": self.bridge_both_sides,
            "content_tokens": list(self.content_tokens),
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True, slots=True)
class TransitionSetFidelity:
    """Transition-fidelity summary for one selected evidence set."""

    available: bool
    reason: str
    slot_id: str | None = None
    target: str = ""
    historical_support_ids: tuple[str, ...] = ()
    transition_support_ids: tuple[str, ...] = ()
    current_support_ids: tuple[str, ...] = ()
    candidate_fidelities: tuple[TransitionCandidateFidelity, ...] = ()
    best_transition_id: str | None = None
    best_bridge_fidelity: float | None = None
    best_bridge_mean: float | None = None
    best_axis_alignment: float | None = None
    best_bridge_both_sides: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "reason": self.reason,
            "slot_id": self.slot_id,
            "target": self.target,
            "historical_support_ids": list(self.historical_support_ids),
            "transition_support_ids": list(self.transition_support_ids),
            "current_support_ids": list(self.current_support_ids),
            "candidate_fidelities": [
                item.to_dict()
                for item in self.candidate_fidelities
            ],
            "best_transition_id": self.best_transition_id,
            "best_bridge_fidelity": self.best_bridge_fidelity,
            "best_bridge_mean": self.best_bridge_mean,
            "best_axis_alignment": self.best_axis_alignment,
            "best_bridge_both_sides": self.best_bridge_both_sides,
        }


class TransitionSlotFidelityEvaluator:
    """M2-C3.7A query-conditioned transition bridge fidelity.

    This component is intentionally shadow-only and training-free.  It does
    not change ranking, arbitration, repair, confidence, or generation.

    The binary slot layer answers:

        "Is a TEMPORAL_TRANSITION slot occupied?"

    This evaluator instead asks:

        "Does the selected transition evidence lexically/semantically bridge
         the historical and current endpoint evidence for the same query?"

    The first implementation deliberately avoids learned weights and tuned
    decision thresholds.  It reports continuous, deterministic diagnostics:

        historical_bridge_score
        current_bridge_score
        bridge_fidelity = min(historical, current)
        bridge_mean
        axis_alignment

    The minimum is used for bridge_fidelity because a temporal bridge should
    connect both sides; being close to only one endpoint is insufficient.
    """

    TEMPORAL_FRAMING_TOKENS = {
        "current",
        "currently",
        "latest",
        "historical",
        "history",
        "previous",
        "previously",
        "earlier",
        "before",
        "after",
        "during",
        "timeline",
        "transition",
        "transitions",
        "change",
        "changed",
        "changes",
        "changing",
        "over",
        "time",
        "state",
        "states",
        "endpoint",
        "endpoints",
    }

    QUERY_FRAMING_TOKENS = {
        "what",
        "which",
        "who",
        "when",
        "where",
        "why",
        "how",
        "does",
        "do",
        "did",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "should",
        "would",
        "could",
        "can",
        "my",
        "your",
        "our",
        "their",
        "the",
        "a",
        "an",
        "and",
        "or",
        "to",
        "of",
        "for",
        "in",
        "on",
        "with",
        "as",
        "from",
        "by",
        "at",
        "it",
        "this",
        "that",
    }

    def __init__(
        self,
        config: dict[str, Any],
    ) -> None:
        query_analysis = config.get(
            "query_analysis",
            {},
        )

        configured_stopwords = query_analysis.get(
            "stopwords",
            [],
        )

        self.stopwords = {
            str(item).lower()
            for item in configured_stopwords
        }

        self.stopwords.update(
            self.QUERY_FRAMING_TOKENS
        )

        self.stopwords.update(
            self.TEMPORAL_FRAMING_TOKENS
        )

    # =====================================================
    # Public API
    # =====================================================

    def evaluate_set(
        self,
        *,
        selected: list[MemoryCandidate],
        slot_statuses: list[dict[str, Any]],
    ) -> TransitionSetFidelity:
        """Evaluate transition bridge fidelity for one evidence set."""

        transition_status = self._first_status(
            slot_statuses,
            "TEMPORAL_TRANSITION",
        )

        if transition_status is None:
            return TransitionSetFidelity(
                available=False,
                reason="no_temporal_transition_slot",
            )

        historical_status = self._first_status(
            slot_statuses,
            "HISTORICAL_ENDPOINT",
        )

        current_status = self._first_status(
            slot_statuses,
            "CURRENT_ENDPOINT",
        )

        slot_id = str(
            transition_status.get(
                "slot_id",
                "",
            )
            or ""
        )

        target = str(
            transition_status.get(
                "target",
                "",
            )
            or ""
        )

        selected_by_id = {
            candidate.memory_id: candidate
            for candidate in selected
        }

        historical_ids = self._support_ids(
            historical_status
        )

        transition_ids = self._support_ids(
            transition_status
        )

        current_ids = self._support_ids(
            current_status
        )

        historical_candidates = [
            selected_by_id[memory_id]
            for memory_id in historical_ids
            if memory_id in selected_by_id
        ]

        transition_candidates = [
            selected_by_id[memory_id]
            for memory_id in transition_ids
            if memory_id in selected_by_id
        ]

        current_candidates = [
            selected_by_id[memory_id]
            for memory_id in current_ids
            if memory_id in selected_by_id
        ]

        if not transition_candidates:
            return TransitionSetFidelity(
                available=True,
                reason="transition_slot_has_no_selected_support",
                slot_id=(
                    slot_id
                    or None
                ),
                target=target,
                historical_support_ids=tuple(
                    historical_ids
                ),
                transition_support_ids=tuple(
                    transition_ids
                ),
                current_support_ids=tuple(
                    current_ids
                ),
            )

        axis_tokens = self._token_set(
            target
        )

        candidate_fidelities = [
            self._evaluate_candidate(
                candidate=candidate,
                historical_candidates=(
                    historical_candidates
                ),
                current_candidates=(
                    current_candidates
                ),
                axis_tokens=axis_tokens,
            )
            for candidate in transition_candidates
        ]

        ordered = sorted(
            candidate_fidelities,
            key=lambda item: (
                item.bridge_fidelity,
                item.bridge_mean,
                item.axis_alignment,
                item.memory_id,
            ),
            reverse=True,
        )

        best = ordered[0]

        missing_sides: list[str] = []

        if not historical_candidates:
            missing_sides.append(
                "historical"
            )

        if not current_candidates:
            missing_sides.append(
                "current"
            )

        reason = (
            "transition_bridge_fidelity_computed"
            if not missing_sides
            else (
                "transition_bridge_fidelity_partial_missing_"
                + "_and_".join(
                    missing_sides
                )
                + "_endpoint_support"
            )
        )

        return TransitionSetFidelity(
            available=True,
            reason=reason,
            slot_id=(
                slot_id
                or None
            ),
            target=target,
            historical_support_ids=tuple(
                historical_ids
            ),
            transition_support_ids=tuple(
                transition_ids
            ),
            current_support_ids=tuple(
                current_ids
            ),
            candidate_fidelities=tuple(
                ordered
            ),
            best_transition_id=(
                best.memory_id
            ),
            best_bridge_fidelity=(
                best.bridge_fidelity
            ),
            best_bridge_mean=(
                best.bridge_mean
            ),
            best_axis_alignment=(
                best.axis_alignment
            ),
            best_bridge_both_sides=(
                best.bridge_both_sides
            ),
        )

    def compare(
        self,
        *,
        legacy_selected: list[MemoryCandidate],
        c3_v3_selected: list[MemoryCandidate],
        legacy_slot_statuses: list[dict[str, Any]],
        c3_v3_slot_statuses: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Matched Legacy-vs-C3 transition-fidelity shadow comparison."""

        legacy = self.evaluate_set(
            selected=legacy_selected,
            slot_statuses=legacy_slot_statuses,
        )

        c3_v3 = self.evaluate_set(
            selected=c3_v3_selected,
            slot_statuses=c3_v3_slot_statuses,
        )

        legacy_dict = legacy.to_dict()
        c3_v3_dict = c3_v3.to_dict()

        legacy_fidelity = (
            legacy.best_bridge_fidelity
        )

        c3_v3_fidelity = (
            c3_v3.best_bridge_fidelity
        )

        fidelity_delta: float | None = None
        strict_fidelity_drop = False

        if (
            legacy_fidelity is not None
            and c3_v3_fidelity is not None
        ):
            fidelity_delta = round(
                (
                    c3_v3_fidelity
                    - legacy_fidelity
                ),
                6,
            )

            strict_fidelity_drop = (
                fidelity_delta
                < 0.0
            )

        legacy_transition_set = set(
            legacy.transition_support_ids
        )

        c3_v3_transition_set = set(
            c3_v3.transition_support_ids
        )

        transition_support_replaced = (
            bool(
                legacy_transition_set
            )
            and bool(
                c3_v3_transition_set
            )
            and (
                legacy_transition_set
                != c3_v3_transition_set
            )
        )

        bridge_both_sides_lost = (
            legacy.best_bridge_both_sides
            and not (
                c3_v3.best_bridge_both_sides
            )
        )

        return {
            "available": (
                legacy.available
                or c3_v3.available
            ),
            "active_for_generation": False,
            "method": (
                "deterministic_query_conditioned_"
                "transition_bridge_fidelity"
            ),
            "legacy": legacy_dict,
            "c3_v3": c3_v3_dict,
            "transition_support_replaced": (
                transition_support_replaced
            ),
            "legacy_best_transition_id": (
                legacy.best_transition_id
            ),
            "c3_v3_best_transition_id": (
                c3_v3.best_transition_id
            ),
            "legacy_best_bridge_fidelity": (
                legacy_fidelity
            ),
            "c3_v3_best_bridge_fidelity": (
                c3_v3_fidelity
            ),
            "transition_fidelity_delta": (
                fidelity_delta
            ),
            "strict_transition_fidelity_drop": (
                strict_fidelity_drop
            ),
            "bridge_both_sides_lost": (
                bridge_both_sides_lost
            ),
            "note": (
                "M2-C3.7A shadow diagnostic only; no tuned threshold "
                "and no effect on arbitration or generation."
            ),
        }

    # =====================================================
    # Candidate scoring
    # =====================================================

    def _evaluate_candidate(
        self,
        *,
        candidate: MemoryCandidate,
        historical_candidates: list[MemoryCandidate],
        current_candidates: list[MemoryCandidate],
        axis_tokens: set[str],
    ) -> TransitionCandidateFidelity:
        candidate_tokens = self._candidate_tokens(
            candidate
        )

        (
            historical_score,
            historical_id,
        ) = self._best_endpoint_bridge(
            candidate_tokens=candidate_tokens,
            endpoints=historical_candidates,
        )

        (
            current_score,
            current_id,
        ) = self._best_endpoint_bridge(
            candidate_tokens=candidate_tokens,
            endpoints=current_candidates,
        )

        bridge_fidelity = min(
            historical_score,
            current_score,
        )

        bridge_mean = (
            historical_score
            + current_score
        ) / 2.0

        axis_alignment = self._token_f1(
            candidate_tokens,
            axis_tokens,
        )

        reasons = (
            "bridge_fidelity=min(historical_bridge,current_bridge)",
            "scores_are_deterministic_token_f1_without_tuned_threshold",
        )

        return TransitionCandidateFidelity(
            memory_id=(
                candidate.memory_id
            ),
            historical_bridge_score=round(
                historical_score,
                6,
            ),
            current_bridge_score=round(
                current_score,
                6,
            ),
            bridge_fidelity=round(
                bridge_fidelity,
                6,
            ),
            bridge_mean=round(
                bridge_mean,
                6,
            ),
            axis_alignment=round(
                axis_alignment,
                6,
            ),
            best_historical_memory_id=(
                historical_id
            ),
            best_current_memory_id=(
                current_id
            ),
            bridge_both_sides=(
                historical_score > 0.0
                and current_score > 0.0
            ),
            content_tokens=tuple(
                sorted(
                    candidate_tokens
                )
            ),
            reasons=reasons,
        )

    def _best_endpoint_bridge(
        self,
        *,
        candidate_tokens: set[str],
        endpoints: list[MemoryCandidate],
    ) -> tuple[float, str | None]:
        best_score = 0.0
        best_id: str | None = None

        for endpoint in endpoints:
            score = self._token_f1(
                candidate_tokens,
                self._candidate_tokens(
                    endpoint
                ),
            )

            if (
                score > best_score
                or (
                    score == best_score
                    and best_id is not None
                    and endpoint.memory_id < best_id
                )
                or (
                    score == best_score
                    and best_id is None
                )
            ):
                best_score = score
                best_id = endpoint.memory_id

        return (
            best_score,
            best_id,
        )

    # =====================================================
    # Text helpers
    # =====================================================

    def _candidate_tokens(
        self,
        candidate: MemoryCandidate,
    ) -> set[str]:
        values = [
            candidate.text,
            candidate.subject,
            candidate.predicate,
            candidate.object_value,
        ]

        text = " ".join(
            str(value)
            for value in values
            if value
        )

        return self._token_set(
            text
        )

    def _token_set(
        self,
        text: str,
    ) -> set[str]:
        tokens = re.findall(
            r"[\w'-]+",
            str(text).lower(),
            flags=re.UNICODE,
        )

        return {
            token
            for token in tokens
            if (
                len(token) > 1
                and token not in self.stopwords
            )
        }

    @staticmethod
    def _token_f1(
        left: set[str],
        right: set[str],
    ) -> float:
        if not left or not right:
            return 0.0

        overlap = len(
            left
            & right
        )

        if overlap <= 0:
            return 0.0

        precision = (
            overlap
            / len(left)
        )

        recall = (
            overlap
            / len(right)
        )

        denominator = (
            precision
            + recall
        )

        if denominator <= 0.0:
            return 0.0

        return (
            2.0
            * precision
            * recall
            / denominator
        )

    @staticmethod
    def _first_status(
        statuses: list[dict[str, Any]],
        kind: str,
    ) -> dict[str, Any] | None:
        expected = kind.upper()

        for status in statuses:
            if str(
                status.get(
                    "kind",
                    "",
                )
            ).upper() == expected:
                return status

        return None

    @staticmethod
    def _support_ids(
        status: dict[str, Any] | None,
    ) -> list[str]:
        if status is None:
            return []

        raw = status.get(
            "supporting_memory_ids",
            [],
        )

        if not isinstance(
            raw,
            list,
        ):
            return []

        return [
            str(item)
            for item in raw
        ]
