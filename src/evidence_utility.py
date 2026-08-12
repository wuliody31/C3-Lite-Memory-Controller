from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .coverage_estimator import CoverageEstimator
from .schemas import MemoryCandidate, QueryFeatures
from .text_utils import (
    approximate_token_count,
    jaccard_similarity,
    tokenize,
)


@dataclass(slots=True)
class EvidenceUtilityBreakdown:
    """Auditable marginal utility for adding one memory to evidence set S."""

    memory_id: str

    relevance: float
    temporal_validity: float
    provenance: float

    requirement_gain: float
    coverage_gain: float

    incompatibility: float
    redundancy: float
    token_cost: int
    normalised_token_cost: float

    utility: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EvidenceUtilityModel:
    """C3-v3 query-conditioned marginal evidence utility.

    This module is initially shadow-only. It does not select evidence.

    Utility is evaluated conditionally on the already selected set S:

        ΔU(m | S, q)
          = + relevance
            + query-relative temporal validity
            + provenance
            + requirement gain
            + information coverage gain
            - incompatibility
            - redundancy
            - token cost

    The defaults are deliberately explicit and inspectable rather than
    learned. They should later be treated as development-set
    hyperparameters and ablated separately.
    """

    def __init__(
        self,
        config: dict[str, Any],
        coverage: CoverageEstimator,
    ) -> None:
        self.config = config
        self.coverage = coverage

        settings = config.get(
            "evidence_utility_v3",
            {},
        )

        self.w_relevance = float(
            settings.get(
                "relevance_weight",
                0.25,
            )
        )

        self.w_validity = float(
            settings.get(
                "temporal_validity_weight",
                0.20,
            )
        )

        self.w_provenance = float(
            settings.get(
                "provenance_weight",
                0.10,
            )
        )

        self.w_requirement = float(
            settings.get(
                "requirement_gain_weight",
                0.20,
            )
        )

        self.w_coverage = float(
            settings.get(
                "coverage_gain_weight",
                0.20,
            )
        )

        self.w_incompatibility = float(
            settings.get(
                "incompatibility_weight",
                0.30,
            )
        )

        self.w_redundancy = float(
            settings.get(
                "redundancy_weight",
                0.10,
            )
        )

        self.w_token_cost = float(
            settings.get(
                "token_cost_weight",
                0.05,
            )
        )

        self.default_token_budget = int(
            settings.get(
                "default_token_budget",
                config.get(
                    "selection",
                    {},
                ).get(
                    "evidence_token_budget",
                    1200,
                ),
            )
        )

        self.stopwords = {
            str(item).lower()
            for item
            in config[
                "query_analysis"
            ].get(
                "stopwords",
                [],
            )
        }

    def evaluate(
        self,
        *,
        candidate: MemoryCandidate,
        selected: list[MemoryCandidate],
        features: QueryFeatures,
        requirement_gain: float = 0.0,
        token_budget: int | None = None,
    ) -> EvidenceUtilityBreakdown:
        """Evaluate ΔU(candidate | selected, query)."""

        relevance = self._relevance(
            candidate
        )

        temporal_validity = max(
            0.0,
            min(
                1.0,
                float(
                    candidate.validity_score
                ),
            ),
        )

        provenance = max(
            0.0,
            min(
                1.0,
                float(
                    candidate.source_confidence_score
                ),
            ),
        )

        requirement_gain = max(
            0.0,
            min(
                1.0,
                float(
                    requirement_gain
                ),
            ),
        )

        coverage_gain = self._coverage_gain(
            candidate=candidate,
            selected=selected,
            features=features,
        )

        incompatibility = (
            self._incompatibility(
                candidate
            )
        )

        redundancy = self._redundancy(
            candidate=candidate,
            selected=selected,
        )

        token_cost = (
            approximate_token_count(
                self._candidate_text(
                    candidate
                )
            )
        )

        effective_budget = max(
            1,
            int(
                token_budget
                if token_budget
                is not None
                else self.default_token_budget
            ),
        )

        normalised_token_cost = min(
            1.0,
            token_cost
            / effective_budget,
        )

        utility = (
            self.w_relevance
            * relevance

            + self.w_validity
            * temporal_validity

            + self.w_provenance
            * provenance

            + self.w_requirement
            * requirement_gain

            + self.w_coverage
            * coverage_gain

            - self.w_incompatibility
            * incompatibility

            - self.w_redundancy
            * redundancy

            - self.w_token_cost
            * normalised_token_cost
        )

        return EvidenceUtilityBreakdown(
            memory_id=(
                candidate.memory_id
            ),
            relevance=round(
                relevance,
                6,
            ),
            temporal_validity=round(
                temporal_validity,
                6,
            ),
            provenance=round(
                provenance,
                6,
            ),
            requirement_gain=round(
                requirement_gain,
                6,
            ),
            coverage_gain=round(
                coverage_gain,
                6,
            ),
            incompatibility=round(
                incompatibility,
                6,
            ),
            redundancy=round(
                redundancy,
                6,
            ),
            token_cost=token_cost,
            normalised_token_cost=round(
                normalised_token_cost,
                6,
            ),
            utility=round(
                utility,
                6,
            ),
        )

    @staticmethod
    def _relevance(
        candidate: MemoryCandidate,
    ) -> float:
        """Relevance excluding validity/provenance to avoid double-counting."""

        score = (
            0.50
            * float(
                candidate.lexical_score
            )
            + 0.30
            * float(
                candidate.graph_entity_score
            )
            + 0.20
            * float(
                candidate.route_compatibility_score
            )
        )

        return max(
            0.0,
            min(
                1.0,
                score,
            ),
        )

    def _coverage_gain(
        self,
        *,
        candidate: MemoryCandidate,
        selected: list[MemoryCandidate],
        features: QueryFeatures,
    ) -> float:
        before = self.coverage.compute(
            features.information_needs,
            selected,
        )

        after = self.coverage.compute(
            features.information_needs,
            [
                *selected,
                candidate,
            ],
        )

        return max(
            0.0,
            min(
                1.0,
                after - before,
            ),
        )

    @staticmethod
    def _incompatibility(
        candidate: MemoryCandidate,
    ) -> float:
        """Combine explicit temporal incompatibility and conflict penalty."""

        temporal_compatible = (
            candidate.metadata.get(
                "query_relative_temporal_compatible"
            )
        )

        temporal_penalty = (
            1.0
            if temporal_compatible is False
            else 0.0
        )

        conflict_penalty = max(
            0.0,
            min(
                1.0,
                float(
                    candidate.conflict_penalty
                ),
            ),
        )

        if (
            candidate.resolution_action
            == "excluded"
        ):
            conflict_penalty = max(
                conflict_penalty,
                1.0,
            )

        return max(
            temporal_penalty,
            conflict_penalty,
        )

    def _redundancy(
        self,
        *,
        candidate: MemoryCandidate,
        selected: list[MemoryCandidate],
    ) -> float:
        if not selected:
            return 0.0

        candidate_tokens = tokenize(
            self._candidate_text(
                candidate
            ),
            self.stopwords,
        )

        return max(
            (
                jaccard_similarity(
                    candidate_tokens,
                    tokenize(
                        self._candidate_text(
                            item
                        ),
                        self.stopwords,
                    ),
                )
                for item
                in selected
            ),
            default=0.0,
        )

    @staticmethod
    def _candidate_text(
        candidate: MemoryCandidate,
    ) -> str:
        triggers = (
            candidate.metadata.get(
                "triggers",
                [],
            )
        )

        trigger_text = (
            " ".join(
                str(item)
                for item
                in triggers
            )
            if isinstance(
                triggers,
                list,
            )
            else str(
                triggers
                or ""
            )
        )

        values = [
            candidate.text,
            candidate.subject,
            candidate.predicate,
            candidate.object_value,
            candidate.metadata.get(
                "task_type"
            ),
            trigger_text,
        ]

        return " ".join(
            str(value)
            for value
            in values
            if value
            not in (
                None,
                "",
            )
        )