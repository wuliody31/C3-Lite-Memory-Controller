from __future__ import annotations

import math
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from .schemas import (
    MemoryCandidate,
    MemoryType,
    QueryFeatures,
    QueryMode,
    RouteDecision,
)
from .temporal_validity import (
    QueryRelativeTemporalValidity,
)
from .text_utils import (
    min_max_normalise,
    overlap_ratio,
    tokenize,
)


class BM25:
    """Small dependency-free BM25 scorer for the shared candidate pool."""

    def __init__(
        self,
        docs: list[list[str]],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.docs = docs
        self.k1 = k1
        self.b = b

        self.term_frequencies = [
            Counter(document)
            for document in docs
        ]

        self.document_lengths = [
            len(document)
            for document in docs
        ]

        self.average_document_length = (
            sum(self.document_lengths)
            / len(self.document_lengths)
            if self.document_lengths
            else 0.0
        )

        self.document_frequencies: Counter[str] = Counter()

        for document in docs:
            self.document_frequencies.update(
                set(document)
            )

    def score(
        self,
        query_tokens: list[str],
        document_index: int,
    ) -> float:
        if (
            not self.docs
            or not self.average_document_length
        ):
            return 0.0

        score = 0.0
        document_count = len(self.docs)

        document_length = (
            self.document_lengths[
                document_index
            ]
        )

        for token in query_tokens:
            frequency = (
                self.term_frequencies[
                    document_index
                ].get(
                    token,
                    0,
                )
            )

            if frequency == 0:
                continue

            document_frequency = (
                self.document_frequencies.get(
                    token,
                    0,
                )
            )

            inverse_document_frequency = math.log(
                1
                + (
                    document_count
                    - document_frequency
                    + 0.5
                )
                / (
                    document_frequency
                    + 0.5
                )
            )

            denominator = (
                frequency
                + self.k1
                * (
                    1
                    - self.b
                    + self.b
                    * document_length
                    / self.average_document_length
                )
            )

            score += (
                inverse_document_frequency
                * (
                    frequency
                    * (
                        self.k1
                        + 1
                    )
                )
                / denominator
            )

        return score


class SharedRanker:
    """Score candidates and apply an RC6.1 query-conditioned gate.

    Candidate score components:
        lexical relevance
        graph/entity compatibility
        temporal/task compatibility
        query-relative temporal validity
        source confidence
        route compatibility

    C3-v3:
        candidate.validity_score is the active query-relative
        temporal validity V(m | q).

        The legacy RC8 status-oriented validity score is preserved
        in candidate.metadata for audit and ablation.

    Admission rules:
        1. procedural memories retain a safety gate;
        2. lexical or entity matches remain valid admission signals;
        3. aggregate utility is accepted only with weak content grounding;
        4. temporal rescue applies only to historical/timeline queries;
        5. explanation rescue applies only to explanation queries.
    """

    def __init__(
        self,
        config: dict[str, Any],
    ) -> None:
        self.config = config
        self.weights = config[
            "ranking"
        ]

        query_analysis = (
            config[
                "query_analysis"
            ]
        )

        self.stopwords = {
            str(item).lower()
            for item
            in query_analysis.get(
                "stopwords",
                [],
            )
        }

        conflict = config[
            "conflict"
        ]

        self.current_statuses = {
            str(item).lower()
            for item
            in conflict[
                "current_status_values"
            ]
        }

        self.outdated_statuses = {
            str(item).lower()
            for item
            in conflict[
                "outdated_status_values"
            ]
        }

        self.authority_scores = {
            str(key): float(value)
            for key, value
            in conflict.get(
                "authority_order",
                {},
            ).items()
        }

        # =================================================
        # C3-v3 active query-relative temporal validity
        #
        # The legacy RC8 validity score remains available
        # through _validity_score() and is stored in
        # candidate.metadata for audit and ablation.
        # =================================================

        self.query_relative_validity = (
            QueryRelativeTemporalValidity(
                config
            )
        )

        retrieval = config[
            "retrieval"
        ]

        self.lexical_gate = float(
            retrieval.get(
                "lexical_relevance_gate",
                0.02,
            )
        )

        self.entity_gate = float(
            retrieval.get(
                "graph_entity_relevance_gate",
                0.20,
            )
        )

        self.utility_gate = float(
            retrieval.get(
                "utility_relevance_gate",
                0.40,
            )
        )

        self.temporal_rescue_enabled = bool(
            retrieval.get(
                "enable_temporal_route_rescue",
                True,
            )
        )

        self.temporal_rescue_min_temporal = float(
            retrieval.get(
                "temporal_route_rescue_min_temporal",
                0.95,
            )
        )

        self.temporal_rescue_min_validity = float(
            retrieval.get(
                "temporal_route_rescue_min_validity",
                0.85,
            )
        )

        self.temporal_rescue_min_route = float(
            retrieval.get(
                "temporal_route_rescue_min_route",
                0.50,
            )
        )

        self.explanation_rescue_enabled = bool(
            retrieval.get(
                "enable_explanation_rescue",
                True,
            )
        )

        self.explanation_rescue_min_utility = float(
            retrieval.get(
                "explanation_rescue_min_utility",
                0.38,
            )
        )

        self.explanation_rescue_min_validity = float(
            retrieval.get(
                "explanation_rescue_min_validity",
                0.85,
            )
        )

        self.explanation_rescue_min_route = float(
            retrieval.get(
                "explanation_rescue_min_route",
                0.50,
            )
        )

    def rank(
        self,
        *,
        candidates: list[
            MemoryCandidate
        ],
        features: QueryFeatures,
        route: RouteDecision,
        current_time: datetime,
    ) -> list[
        MemoryCandidate
    ]:
        if not candidates:
            return []

        documents = [
            tokenize(
                self._candidate_text(
                    candidate
                ),
                self.stopwords,
            )
            for candidate
            in candidates
        ]

        query_tokens = tokenize(
            features.normalised_query,
            self.stopwords,
        )

        bm25 = BM25(
            documents
        )

        raw_lexical_scores = [
            (
                bm25.score(
                    query_tokens,
                    index,
                )
                + float(
                    candidate.metadata.get(
                        "neo4j_fulltext_score",
                        0.0,
                    )
                    or 0.0
                )
            )
            for index, candidate
            in enumerate(
                candidates
            )
        ]

        lexical_scores = (
            min_max_normalise(
                raw_lexical_scores
            )
        )

        for index, candidate in enumerate(
            candidates
        ):
            # =================================================
            # Lexical relevance
            # =================================================

            candidate.lexical_score = (
                lexical_scores[
                    index
                ]
            )

            # =================================================
            # Entity compatibility
            # =================================================

            candidate.graph_entity_score = (
                self._entity_score(
                    candidate,
                    features,
                )
            )

            # =================================================
            # Legacy temporal/task compatibility
            #
            # Kept unchanged for M2-B2.
            # =================================================

            candidate.temporal_task_score = (
                self._temporal_task_score(
                    candidate,
                    features,
                    current_time,
                )
            )

            # =================================================
            # C3-v3 query-relative temporal validity
            # =================================================

            legacy_validity_score = (
                self._validity_score(
                    candidate,
                    features,
                )
            )

            temporal_validity = (
                self.query_relative_validity.evaluate(
                    candidate=candidate,
                    features=features,
                )
            )

            # -------------------------------------------------
            # Preserve RC8 validity for direct audit/ablation.
            # -------------------------------------------------

            candidate.metadata[
                "legacy_validity_score"
            ] = (
                legacy_validity_score
            )

            # -------------------------------------------------
            # Store C3-v3 validity details.
            # -------------------------------------------------

            candidate.metadata[
                "query_relative_validity_score"
            ] = (
                temporal_validity.score
            )

            candidate.metadata[
                "query_relative_temporal_role"
            ] = (
                temporal_validity.temporal_role
            )

            candidate.metadata[
                "query_relative_temporal_compatible"
            ] = (
                temporal_validity.compatible
            )

            candidate.metadata[
                "query_relative_temporal_reasons"
            ] = list(
                temporal_validity.reasons
            )

            # -------------------------------------------------
            # ACTIVE C3-v3 behaviour:
            #
            # validity_weight now multiplies V(m | q)
            # instead of RC8 status-oriented validity.
            # -------------------------------------------------

            candidate.validity_score = (
                temporal_validity.score
            )

            # =================================================
            # Source confidence
            # =================================================

            candidate.source_confidence_score = (
                self._source_confidence_score(
                    candidate
                )
            )

            # =================================================
            # Route compatibility
            # =================================================

            candidate.route_compatibility_score = (
                float(
                    route.scores.get(
                        candidate
                        .memory_type
                        .value,
                        0.0,
                    )
                )
            )

            # =================================================
            # Final weighted score
            # =================================================

            candidate.final_score = (
                self._weighted_final_score(
                    candidate
                )
            )

            # =================================================
            # Candidate admission gate
            # =================================================

            gate_reasons = (
                self._gate_reasons(
                    candidate,
                    features,
                )
            )

            candidate.metadata[
                "ranker_gate_reasons"
            ] = (
                gate_reasons
            )

            candidate.metadata[
                "passed_utility_aware_gate"
            ] = bool(
                gate_reasons
            )

        admitted = [
            candidate
            for candidate
            in candidates
            if bool(
                candidate.metadata.get(
                    "passed_utility_aware_gate",
                    False,
                )
            )
        ]

        admitted.sort(
            key=self._sort_key,
            reverse=True,
        )

        return admitted

    def _weighted_final_score(
        self,
        candidate: MemoryCandidate,
    ) -> float:
        score = (
            float(
                self.weights[
                    "lexical_weight"
                ]
            )
            * candidate.lexical_score

            + float(
                self.weights[
                    "graph_entity_weight"
                ]
            )
            * candidate.graph_entity_score

            + float(
                self.weights[
                    "temporal_task_weight"
                ]
            )
            * candidate.temporal_task_score

            + float(
                self.weights[
                    "validity_weight"
                ]
            )
            * candidate.validity_score

            + float(
                self.weights[
                    "source_confidence_weight"
                ]
            )
            * candidate.source_confidence_score

            + float(
                self.weights[
                    "route_compatibility_weight"
                ]
            )
            * candidate.route_compatibility_score
        )

        return max(
            0.0,
            min(
                1.0,
                score,
            ),
        )

    def _gate_reasons(
        self,
        candidate: MemoryCandidate,
        features: QueryFeatures,
    ) -> list[str]:
        reasons: list[str] = []

        # Procedural constraints remain safety-critical once routed.
        if (
            candidate.memory_type
            == MemoryType.PROCEDURAL
        ):
            reasons.append(
                "procedural_safety_gate"
            )

        if (
            candidate.lexical_score
            >= self.lexical_gate
        ):
            reasons.append(
                "lexical_gate"
            )

        if (
            candidate.graph_entity_score
            >= self.entity_gate
        ):
            reasons.append(
                "entity_gate"
            )

        # A high aggregate score must not admit a completely unrelated
        # current fact. Require at least weak content grounding.
        weak_content_support = (
            candidate.lexical_score
            > 0.0
            or candidate.graph_entity_score
            > 0.0
        )

        if (
            weak_content_support
            and candidate.final_score
            >= self.utility_gate
        ):
            reasons.append(
                "utility_gate"
            )

        temporal_query = (
            features.query_mode
            in {
                QueryMode.HISTORICAL,
                QueryMode.TIMELINE,
            }
        )

        if (
            self.temporal_rescue_enabled
            and temporal_query
            and candidate.memory_type
            in {
                MemoryType.EPISODIC,
                MemoryType.SEMANTIC,
            }
            and candidate.temporal_task_score
            >= self.temporal_rescue_min_temporal
            and candidate.validity_score
            >= self.temporal_rescue_min_validity
            and candidate.route_compatibility_score
            >= self.temporal_rescue_min_route
        ):
            reasons.append(
                "temporal_route_rescue"
            )

        if (
            self.explanation_rescue_enabled
            and features.asks_explanation
            and candidate.memory_type
            in {
                MemoryType.EPISODIC,
                MemoryType.SEMANTIC,
            }
            and candidate.final_score
            >= self.explanation_rescue_min_utility
            and candidate.validity_score
            >= self.explanation_rescue_min_validity
            and candidate.route_compatibility_score
            >= self.explanation_rescue_min_route
        ):
            reasons.append(
                "explanation_rescue"
            )

        return reasons

    def _source_confidence_score(
        self,
        candidate: MemoryCandidate,
    ) -> float:
        authority_score = (
            self.authority_scores.get(
                candidate.authority,
                0.5,
            )
        )

        score = (
            0.75
            * float(
                candidate.confidence
            )
            + 0.25
            * authority_score
        )

        return max(
            0.0,
            min(
                1.0,
                score,
            ),
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

    def _entity_score(
        self,
        candidate: MemoryCandidate,
        features: QueryFeatures,
    ) -> float:
        candidate_text = (
            self._candidate_text(
                candidate
            ).lower()
        )

        if features.entities:
            matched_entities = sum(
                1
                for entity
                in features.entities
                if str(
                    entity
                ).lower()
                in candidate_text
            )

            return (
                matched_entities
                / len(
                    features.entities
                )
            )

        return overlap_ratio(
            features.tokens,
            tokenize(
                candidate_text,
                self.stopwords,
            ),
        )

    def _temporal_task_score(
        self,
        candidate: MemoryCandidate,
        features: QueryFeatures,
        current_time: datetime,
    ) -> float:
        # =================================================
        # Procedural memory:
        # temporal_task_score effectively acts as task
        # applicability.
        # =================================================

        if (
            candidate.memory_type
            == MemoryType.PROCEDURAL
        ):
            task_match = float(
                candidate.metadata.get(
                    "task_match",
                    0.0,
                )
                or 0.0
            )

            scope_score = float(
                candidate.metadata.get(
                    "scope_score",
                    0.5,
                )
                or 0.5
            )

            return min(
                1.0,
                0.65
                * task_match
                + 0.35
                * scope_score,
            )

        # =================================================
        # Timeline query
        # =================================================

        if (
            features.query_mode
            == QueryMode.TIMELINE
        ):
            return (
                1.0
                if candidate.timestamp
                else 0.55
            )

        # =================================================
        # Historical query
        # =================================================

        if (
            features.query_mode
            == QueryMode.HISTORICAL
        ):
            return (
                1.0
                if (
                    candidate.memory_type
                    == MemoryType.EPISODIC
                )
                else 0.70
            )

        # =================================================
        # Current-state query
        # =================================================

        if (
            features.query_mode
            == QueryMode.CURRENT
        ):
            return (
                1.0
                if (
                    candidate.status.lower()
                    in self.current_statuses
                )
                else 0.15
            )

        # =================================================
        # Atemporal query:
        # retain RC8 recency proxy for now.
        # =================================================

        if (
            candidate.timestamp
            is not None
        ):
            aware_now = (
                current_time
                if current_time.tzinfo
                else current_time.replace(
                    tzinfo=timezone.utc
                )
            )

            aware_timestamp = (
                candidate.timestamp
                if candidate.timestamp.tzinfo
                else candidate.timestamp.replace(
                    tzinfo=timezone.utc
                )
            )

            age_days = max(
                0.0,
                (
                    aware_now
                    - aware_timestamp
                ).total_seconds()
                / 86400.0,
            )

            return (
                0.5
                + 0.5
                * math.exp(
                    -age_days
                    / 365.0
                )
            )

        return 0.60

    def _validity_score(
        self,
        candidate: MemoryCandidate,
        features: QueryFeatures,
    ) -> float:
        """Legacy RC8 status-oriented validity.

        C3-v3 no longer uses this value directly as the active
        validity feature.

        It is retained to support:
            - old-vs-new audit;
            - ablation experiments;
            - causal comparison with query-relative validity.
        """

        if (
            candidate.memory_type
            == MemoryType.EPISODIC
        ):
            return 1.0

        status = (
            candidate.status.lower()
        )

        if (
            status
            in self.current_statuses
        ):
            return 1.0

        if (
            status
            in self.outdated_statuses
        ):
            return (
                0.85
                if (
                    features.query_mode
                    in {
                        QueryMode.HISTORICAL,
                        QueryMode.TIMELINE,
                    }
                )
                else 0.10
            )

        return 0.55

    @staticmethod
    def _timestamp_value(
        timestamp: datetime
        | None,
    ) -> float:
        if timestamp is None:
            return float(
                "-inf"
            )

        if (
            timestamp.tzinfo
            is None
        ):
            timestamp = (
                timestamp.replace(
                    tzinfo=timezone.utc
                )
            )

        return (
            timestamp.timestamp()
        )

    def _sort_key(
        self,
        candidate: MemoryCandidate,
    ) -> tuple[
        float,
        float,
        float,
        str,
    ]:
        return (
            float(
                candidate.final_score
            ),
            float(
                candidate.confidence
            ),
            self._timestamp_value(
                candidate.timestamp
            ),
            candidate.memory_id,
        )