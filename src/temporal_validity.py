from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .schemas import (
    MemoryCandidate,
    MemoryType,
    QueryFeatures,
    QueryMode,
)


@dataclass(slots=True)
class TemporalValidityResult:
    """Query-relative temporal validity of one memory candidate.

    The score answers:

        How temporally appropriate is this memory
        for the semantics of the current query?

    This is intentionally different from simple recency.
    """

    memory_id: str
    query_mode: QueryMode

    score: float
    temporal_role: str
    compatible: bool

    reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "query_mode": self.query_mode.value,
            "score": self.score,
            "temporal_role": self.temporal_role,
            "compatible": self.compatible,
            "reasons": list(self.reasons),
        }


class QueryRelativeTemporalValidity:
    """Estimate temporal validity relative to query semantics.

    C3-v3 principle:

        Currentness != Recency

    A newer memory is not automatically more valid.

    Examples:

        CURRENT query:
            current version > superseded version

        HISTORICAL query:
            historical/superseded version > current version

        TIMELINE query:
            both historical and current endpoints can be valid,
            but they fulfil different temporal roles.

    M2-B1 is deliberately deterministic and training-free.
    It does not yet modify SharedRanker scores.
    """

    CURRENT_ACTIONS = {
        "preferred",
        "current_endpoint",
    }

    HISTORICAL_ACTIONS = {
        "historical",
        "preferred_historical",
    }

    EXCLUDED_ACTIONS = {
        "excluded",
    }

    def __init__(
        self,
        config: dict[str, Any],
    ) -> None:
        conflict = config["conflict"]

        self.current_statuses = {
            str(value).lower()
            for value
            in conflict[
                "current_status_values"
            ]
        }

        self.outdated_statuses = {
            str(value).lower()
            for value
            in conflict[
                "outdated_status_values"
            ]
        }

    def evaluate(
        self,
        *,
        candidate: MemoryCandidate,
        features: QueryFeatures,
    ) -> TemporalValidityResult:
        mode = features.query_mode

        # Procedural memories are governed primarily by
        # applicability rather than factual version age.
        if (
            candidate.memory_type
            == MemoryType.PROCEDURAL
        ):
            return TemporalValidityResult(
                memory_id=candidate.memory_id,
                query_mode=mode,
                score=1.0,
                temporal_role="procedural_rule",
                compatible=True,
                reasons=[
                    "procedural_memory_temporally_neutral"
                ],
            )

        if mode == QueryMode.CURRENT:
            return self._evaluate_current(
                candidate
            )

        if mode == QueryMode.HISTORICAL:
            return self._evaluate_historical(
                candidate
            )

        if mode == QueryMode.TIMELINE:
            return self._evaluate_timeline(
                candidate
            )

        if mode == QueryMode.PROCEDURAL:
            return TemporalValidityResult(
                memory_id=candidate.memory_id,
                query_mode=mode,
                score=0.60,
                temporal_role="supporting_factual_context",
                compatible=True,
                reasons=[
                    "factual_memory_supports_procedural_query"
                ],
            )

        return self._evaluate_atemporal(
            candidate
        )

    # =====================================================
    # CURRENT
    # =====================================================

    def _evaluate_current(
        self,
        candidate: MemoryCandidate,
    ) -> TemporalValidityResult:
        if self._is_excluded(
            candidate
        ):
            return self._result(
                candidate,
                QueryMode.CURRENT,
                score=0.0,
                role="excluded",
                compatible=False,
                reason="conflict_resolution_excluded",
            )

        if self._is_current_endpoint(
            candidate
        ):
            return self._result(
                candidate,
                QueryMode.CURRENT,
                score=1.0,
                role="current_state",
                compatible=True,
                reason="current_endpoint_matches_current_query",
            )

        if self._is_historical(
            candidate
        ):
            return self._result(
                candidate,
                QueryMode.CURRENT,
                score=0.10,
                role="historical_state",
                compatible=False,
                reason="historical_version_mismatches_current_query",
            )

        if (
            candidate.memory_type
            == MemoryType.EPISODIC
        ):
            return self._result(
                candidate,
                QueryMode.CURRENT,
                score=0.45,
                role="supporting_event",
                compatible=True,
                reason="episodic_event_may_support_current_state",
            )

        return self._result(
            candidate,
            QueryMode.CURRENT,
            score=0.55,
            role="temporally_uncertain",
            compatible=True,
            reason="temporal_status_uncertain",
        )

    # =====================================================
    # HISTORICAL
    # =====================================================

    def _evaluate_historical(
        self,
        candidate: MemoryCandidate,
    ) -> TemporalValidityResult:
        if self._is_excluded(
            candidate
        ):
            return self._result(
                candidate,
                QueryMode.HISTORICAL,
                score=0.0,
                role="excluded",
                compatible=False,
                reason="conflict_resolution_excluded",
            )

        if self._is_historical(
            candidate
        ):
            return self._result(
                candidate,
                QueryMode.HISTORICAL,
                score=1.0,
                role="historical_state",
                compatible=True,
                reason="historical_version_matches_historical_query",
            )

        if (
            candidate.memory_type
            == MemoryType.EPISODIC
        ):
            return self._result(
                candidate,
                QueryMode.HISTORICAL,
                score=0.90,
                role="historical_event",
                compatible=True,
                reason="episodic_event_supports_historical_query",
            )

        if self._is_current_endpoint(
            candidate
        ):
            return self._result(
                candidate,
                QueryMode.HISTORICAL,
                score=0.25,
                role="current_state",
                compatible=False,
                reason="current_version_is_not_historical_target",
            )

        return self._result(
            candidate,
            QueryMode.HISTORICAL,
            score=0.55,
            role="temporally_uncertain",
            compatible=True,
            reason="temporal_status_uncertain",
        )

    # =====================================================
    # TIMELINE
    # =====================================================

    def _evaluate_timeline(
        self,
        candidate: MemoryCandidate,
    ) -> TemporalValidityResult:
        if self._is_excluded(
            candidate
        ):
            return self._result(
                candidate,
                QueryMode.TIMELINE,
                score=0.0,
                role="excluded",
                compatible=False,
                reason="conflict_resolution_excluded",
            )

        if self._is_current_endpoint(
            candidate
        ):
            return self._result(
                candidate,
                QueryMode.TIMELINE,
                score=1.0,
                role="current_endpoint",
                compatible=True,
                reason="current_endpoint_required_for_timeline",
            )

        if self._is_historical(
            candidate
        ):
            return self._result(
                candidate,
                QueryMode.TIMELINE,
                score=1.0,
                role="historical_state",
                compatible=True,
                reason="historical_state_required_for_timeline",
            )

        if (
            candidate.memory_type
            == MemoryType.EPISODIC
        ):
            return self._result(
                candidate,
                QueryMode.TIMELINE,
                score=0.95,
                role="transition_event",
                compatible=True,
                reason="episodic_event_supports_temporal_transition",
            )

        return self._result(
            candidate,
            QueryMode.TIMELINE,
            score=0.60,
            role="temporally_uncertain",
            compatible=True,
            reason="candidate_may_support_timeline",
        )

    # =====================================================
    # ATEMPORAL
    # =====================================================

    def _evaluate_atemporal(
        self,
        candidate: MemoryCandidate,
    ) -> TemporalValidityResult:
        if self._is_excluded(
            candidate
        ):
            return self._result(
                candidate,
                QueryMode.ATEMPORAL,
                score=0.0,
                role="excluded",
                compatible=False,
                reason="conflict_resolution_excluded",
            )

        if self._is_current_endpoint(
            candidate
        ):
            return self._result(
                candidate,
                QueryMode.ATEMPORAL,
                score=0.85,
                role="current_state",
                compatible=True,
                reason="current_fact_preferred_for_atemporal_query",
            )

        if self._is_historical(
            candidate
        ):
            return self._result(
                candidate,
                QueryMode.ATEMPORAL,
                score=0.45,
                role="historical_state",
                compatible=True,
                reason="historical_fact_is_secondary_without_temporal_intent",
            )

        return self._result(
            candidate,
            QueryMode.ATEMPORAL,
            score=0.70,
            role="temporally_neutral",
            compatible=True,
            reason="no_explicit_temporal_constraint",
        )

    # =====================================================
    # Candidate state helpers
    # =====================================================

    def _is_current_endpoint(
        self,
        candidate: MemoryCandidate,
    ) -> bool:
        action = (
            candidate.resolution_action
            or ""
        ).lower()

        if action in self.CURRENT_ACTIONS:
            return True

        if action in (
            self.HISTORICAL_ACTIONS
            | self.EXCLUDED_ACTIONS
        ):
            return False

        return (
            candidate.status.lower()
            in self.current_statuses
        )

    def _is_historical(
        self,
        candidate: MemoryCandidate,
    ) -> bool:
        action = (
            candidate.resolution_action
            or ""
        ).lower()

        if action in self.HISTORICAL_ACTIONS:
            return True

        if action in (
            self.CURRENT_ACTIONS
            | self.EXCLUDED_ACTIONS
        ):
            return False

        return (
            candidate.status.lower()
            in self.outdated_statuses
        )

    def _is_excluded(
        self,
        candidate: MemoryCandidate,
    ) -> bool:
        return (
            (
                candidate.resolution_action
                or ""
            ).lower()
            in self.EXCLUDED_ACTIONS
        )

    # =====================================================
    # Result helper
    # =====================================================

    @staticmethod
    def _result(
        candidate: MemoryCandidate,
        mode: QueryMode,
        *,
        score: float,
        role: str,
        compatible: bool,
        reason: str,
    ) -> TemporalValidityResult:
        bounded_score = max(
            0.0,
            min(
                1.0,
                float(score),
            ),
        )

        return TemporalValidityResult(
            memory_id=candidate.memory_id,
            query_mode=mode,
            score=bounded_score,
            temporal_role=role,
            compatible=compatible,
            reasons=[reason],
        )