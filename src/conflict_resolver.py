from __future__ import annotations

from datetime import datetime
from typing import Any

from .schemas import ConflictGroup, MemoryCandidate, QueryFeatures, QueryMode


class ConflictResolver:
    """Resolve factual/version conflicts according to the query's time mode."""

    def __init__(self, config: dict[str, Any]):
        self.config = config["conflict"]
        self.current_statuses = {
            value.lower() for value in self.config["current_status_values"]
        }
        self.outdated_statuses = {
            value.lower() for value in self.config["outdated_status_values"]
        }
        self.authority_order = {
            str(key): float(value)
            for key, value in self.config.get("authority_order", {}).items()
        }

    def resolve(
        self,
        *,
        candidates: list[MemoryCandidate],
        conflicts: list[ConflictGroup],
        features: QueryFeatures,
    ) -> tuple[list[MemoryCandidate], list[ConflictGroup]]:
        by_id = {candidate.memory_id: candidate for candidate in candidates}

        for group in conflicts:
            members = [
                by_id[memory_id]
                for memory_id in group.candidate_ids
                if memory_id in by_id
            ]
            if len(members) < 2:
                group.unresolved = True
                group.explanation += (
                    " Fewer than two retrieved candidates were available "
                    "for resolution."
                )
                continue

            if features.query_mode == QueryMode.TIMELINE:
                self._resolve_timeline(group, members)
            elif features.query_mode == QueryMode.HISTORICAL:
                self._resolve_historical(group, members)
            else:
                self._resolve_current(group, members)

        self._apply_resolution_actions(
            by_id=by_id,
            conflicts=conflicts,
            query_mode=features.query_mode,
        )

        candidates.sort(
            key=lambda item: (
                item.final_score,
                item.timestamp or datetime.min,
                item.memory_id,
            ),
            reverse=True,
        )
        return candidates, conflicts

    def _apply_resolution_actions(
        self,
        *,
        by_id: dict[str, MemoryCandidate],
        conflicts: list[ConflictGroup],
        query_mode: QueryMode,
    ) -> None:
        """Apply labels and score effects without overwriting the endpoint."""

        # Historical/exclusion penalties first.
        for group in conflicts:
            for memory_id in group.historical_ids:
                candidate = by_id.get(memory_id)
                if not candidate:
                    continue
                candidate.resolution_action = "historical"
                candidate.final_score *= 0.90

            for memory_id in group.excluded_ids:
                candidate = by_id.get(memory_id)
                if not candidate:
                    continue
                candidate.resolution_action = "excluded"
                candidate.conflict_penalty = 0.35
                candidate.final_score *= 0.35

            if group.unresolved:
                for memory_id in group.candidate_ids:
                    candidate = by_id.get(memory_id)
                    if not candidate:
                        continue
                    candidate.resolution_action = "unresolved"
                    candidate.conflict_penalty = max(
                        candidate.conflict_penalty, 0.20
                    )
                    candidate.final_score *= 0.80

        # Preferred labels last, so the timeline endpoint cannot be overwritten
        # by "historical".
        for group in conflicts:
            for memory_id in group.preferred_ids:
                candidate = by_id.get(memory_id)
                if not candidate:
                    continue

                if query_mode == QueryMode.TIMELINE:
                    status = candidate.status.lower()

                    # A preferred candidate in a timeline conflict is not
                    # automatically the current endpoint. Preserve lifecycle
                    # identity for archived / superseded historical states.
                    if status in self.outdated_statuses:
                        candidate.resolution_action = "historical"
                    elif status in self.current_statuses:
                        candidate.resolution_action = "current_endpoint"
                    else:
                        # Unknown lifecycle state: keep preference without
                        # inventing a temporal endpoint.
                        candidate.resolution_action = "preferred"

                elif query_mode == QueryMode.HISTORICAL:
                    candidate.resolution_action = "preferred_historical"
                else:
                    candidate.resolution_action = "preferred"

    def _resolve_current(
        self,
        group: ConflictGroup,
        members: list[MemoryCandidate],
    ) -> None:
        ranked = sorted(members, key=self._current_priority, reverse=True)
        preferred = ranked[0]
        top_priority = self._current_priority(preferred)

        tied = [
            item for item in ranked
            if self._current_priority(item) == top_priority
        ]
        if len(tied) > 1 and self._different_values(tied):
            group.unresolved = True
            group.preferred_ids = [item.memory_id for item in tied]
            group.explanation += (
                " The highest-priority current facts remain tied, "
                "so the conflict is unresolved."
            )
            return

        group.preferred_ids = [preferred.memory_id]
        group.historical_ids = []
        group.excluded_ids = []

        for member in ranked[1:]:
            if member.status.lower() in self.outdated_statuses:
                group.historical_ids.append(member.memory_id)
            else:
                group.excluded_ids.append(member.memory_id)

        group.explanation += (
            f" Current-state resolution prefers {preferred.memory_id} "
            "using validity, authority, confidence and recency."
        )

    def _resolve_historical(
        self,
        group: ConflictGroup,
        members: list[MemoryCandidate],
    ) -> None:
        ordered = sorted(
            members,
            key=lambda item: (
                item.timestamp is not None,
                item.timestamp or datetime.min,
                item.confidence,
                item.memory_id,
            ),
        )

        group.historical_ids = [item.memory_id for item in ordered]
        group.preferred_ids = [ordered[-1].memory_id]
        group.excluded_ids = []
        group.explanation += (
            " Historical-state resolution preserves available versions "
            "instead of replacing them with the present fact."
        )

    def _resolve_timeline(
        self,
        group: ConflictGroup,
        members: list[MemoryCandidate],
    ) -> None:
        ordered = sorted(
            members,
            key=lambda item: (
                item.timestamp is not None,
                item.timestamp or datetime.min,
                item.memory_id,
            ),
        )

        current_endpoint = max(ordered, key=self._current_priority)

        # Prior versions and the current endpoint are now explicitly separated.
        group.preferred_ids = [current_endpoint.memory_id]
        group.historical_ids = [
            item.memory_id
            for item in ordered
            if item.memory_id != current_endpoint.memory_id
        ]
        group.excluded_ids = []
        group.unresolved = False
        group.explanation += (
            " Timeline resolution preserves prior versions in chronological "
            f"order and marks {current_endpoint.memory_id} as the current endpoint."
        )

    def _current_priority(
        self,
        item: MemoryCandidate,
    ) -> tuple[float, float, float, float]:
        status_score = (
            1.0 if item.status.lower() in self.current_statuses else 0.0
        )
        authority_score = self.authority_order.get(item.authority, 0.5)
        timestamp_score = item.timestamp.timestamp() if item.timestamp else 0.0
        return (
            status_score,
            authority_score,
            item.confidence,
            timestamp_score,
        )

    @staticmethod
    def _different_values(items: list[MemoryCandidate]) -> bool:
        values = {
            " ".join(str(item.object_value or item.text).lower().split())
            for item in items
        }
        return len(values) > 1
