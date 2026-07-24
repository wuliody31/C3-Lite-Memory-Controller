from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

from .schemas import (
    MemoryCandidate,
    MemoryType,
)


class BoundaryAwareCandidateBudget:
    """Apply per-memory-type top-k without arbitrarily cutting score ties.

    The nominal top-k remains unchanged. Candidates immediately below the
    boundary are admitted when their score is within ``score_epsilon`` of the
    type-specific cutoff. Extensions are bounded both per memory type and
    globally.

    The output preserves the original ``ranked_pool`` order. This is important
    because SharedRanker has already produced the deterministic ranking, and
    re-sorting here could reverse equal-score candidates.
    """

    def __init__(
        self,
        config: dict[str, Any],
    ) -> None:
        settings = config.get(
            "candidate_budget",
            {},
        )

        self.enabled = bool(
            settings.get(
                "enabled",
                True,
            )
        )
        self.score_epsilon = float(
            settings.get(
                "score_epsilon",
                0.01,
            )
        )
        self.max_extra_per_memory_type = int(
            settings.get(
                "max_extra_per_memory_type",
                5,
            )
        )
        self.max_total_extra = int(
            settings.get(
                "max_total_extra",
                8,
            )
        )

    def select(
        self,
        *,
        ranked_pool: list[MemoryCandidate],
        selected_types: Iterable[MemoryType],
        top_k: dict[str, Any],
    ) -> list[MemoryCandidate]:
        """Return a boundary-aware candidate set in original ranked order."""
        if not ranked_pool:
            return []

        selected_type_list = list(
            selected_types
        )

        if not self.enabled:
            return self._fixed_top_k(
                ranked_pool=ranked_pool,
                selected_types=selected_type_list,
                top_k=top_k,
            )

        grouped: dict[
            MemoryType,
            list[MemoryCandidate],
        ] = defaultdict(list)

        for candidate in ranked_pool:
            grouped[
                candidate.memory_type
            ].append(candidate)

        selected_ids: set[str] = set()
        total_extra = 0

        for memory_type in selected_type_list:
            candidates = grouped.get(
                memory_type,
                [],
            )
            nominal_k = int(
                top_k[memory_type.value]
            )

            if (
                not candidates
                or nominal_k <= 0
            ):
                continue

            base = candidates[:nominal_k]

            for candidate in base:
                candidate.metadata[
                    "candidate_budget_reason"
                ] = "base_top_k"
                selected_ids.add(
                    candidate.memory_id
                )

            if len(candidates) <= nominal_k:
                continue

            cutoff_score = float(
                candidates[
                    nominal_k - 1
                ].final_score
            )
            extra_for_type = 0

            for candidate in candidates[
                nominal_k:
            ]:
                if (
                    extra_for_type
                    >= self.max_extra_per_memory_type
                    or total_extra
                    >= self.max_total_extra
                ):
                    break

                score_gap = (
                    cutoff_score
                    - float(
                        candidate.final_score
                    )
                )

                if (
                    score_gap
                    > self.score_epsilon
                ):
                    break

                candidate.metadata[
                    "candidate_budget_reason"
                ] = "boundary_extension"
                candidate.metadata[
                    "candidate_budget_cutoff_score"
                ] = cutoff_score
                candidate.metadata[
                    "candidate_budget_score_gap"
                ] = score_gap

                selected_ids.add(
                    candidate.memory_id
                )
                extra_for_type += 1
                total_extra += 1

        return self._preserve_ranked_order(
            ranked_pool=ranked_pool,
            selected_ids=selected_ids,
        )

    @staticmethod
    def _fixed_top_k(
        *,
        ranked_pool: list[MemoryCandidate],
        selected_types: Iterable[MemoryType],
        top_k: dict[str, Any],
    ) -> list[MemoryCandidate]:
        selected_ids: set[str] = set()

        for memory_type in selected_types:
            type_candidates = [
                candidate
                for candidate in ranked_pool
                if (
                    candidate.memory_type
                    == memory_type
                )
            ]

            for candidate in type_candidates[
                : int(
                    top_k[
                        memory_type.value
                    ]
                )
            ]:
                selected_ids.add(
                    candidate.memory_id
                )

        return (
            BoundaryAwareCandidateBudget
            ._preserve_ranked_order(
                ranked_pool=ranked_pool,
                selected_ids=selected_ids,
            )
        )

    @staticmethod
    def _preserve_ranked_order(
        *,
        ranked_pool: list[MemoryCandidate],
        selected_ids: set[str],
    ) -> list[MemoryCandidate]:
        """Deduplicate while preserving SharedRanker's original order."""
        output: list[
            MemoryCandidate
        ] = []
        seen: set[str] = set()

        for candidate in ranked_pool:
            memory_id = (
                candidate.memory_id
            )

            if (
                memory_id
                not in selected_ids
                or memory_id in seen
            ):
                continue

            seen.add(memory_id)
            output.append(candidate)

        return output