from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any

from .schemas import ConflictGroup, MemoryCandidate, MemoryType


class ConflictDetector:
    """Detect and merge explicit and implicit memory conflicts.

    A semantic version conflict may be visible both through an explicit graph
    edge (for example SUPERSEDES) and through an implicit slot mismatch. These
    signals are merged into one logical ConflictGroup.
    """

    EXPLICIT_TYPES = {"SUPERSEDES", "CONTRADICTS", "INVALIDATES"}

    def __init__(self, config: dict[str, Any]):
        self.config = config["conflict"]

    def detect(self, candidates: list[MemoryCandidate]) -> list[ConflictGroup]:
        raw_groups: list[ConflictGroup] = []

        if self.config.get("explicit_relation_detection", True):
            raw_groups.extend(self._explicit_groups(candidates))

        if self.config.get("implicit_slot_detection", True):
            raw_groups.extend(self._implicit_groups(candidates))

        return self._merge_groups(raw_groups)

    def _explicit_groups(
        self,
        candidates: list[MemoryCandidate],
    ) -> list[ConflictGroup]:
        by_id = {candidate.memory_id: candidate for candidate in candidates}
        output: list[ConflictGroup] = []

        for candidate in candidates:
            for relation in candidate.relations:
                relation_type = str(relation.get("type", "")).upper()
                target_id = str(relation.get("target_id", ""))

                if (
                    relation_type not in self.EXPLICIT_TYPES
                    or target_id not in by_id
                ):
                    continue

                target = by_id[target_id]
                slot_key = self._slot_key(candidate) or self._slot_key(target)
                if slot_key:
                    candidate.conflict_key = slot_key
                    target.conflict_key = slot_key

                ids = sorted({candidate.memory_id, target_id})
                output.append(
                    ConflictGroup(
                        conflict_id=self._id("raw_explicit", ids),
                        key=slot_key,
                        candidate_ids=ids,
                        conflict_type=relation_type.lower(),
                        explicit=True,
                        explanation=(
                            f"Explicit graph relation {relation_type} connects "
                            f"{candidate.memory_id} and {target_id}."
                        ),
                    )
                )
        return output

    def _implicit_groups(
        self,
        candidates: list[MemoryCandidate],
    ) -> list[ConflictGroup]:
        slots: dict[str, list[MemoryCandidate]] = defaultdict(list)

        for candidate in candidates:
            if candidate.memory_type != MemoryType.SEMANTIC:
                continue

            slot = self._slot_key(candidate)
            if slot:
                slots[slot].append(candidate)

        output: list[ConflictGroup] = []
        for slot, slot_candidates in slots.items():
            objects = {
                self._normalise(candidate.object_value or candidate.text)
                for candidate in slot_candidates
            }
            objects.discard("")

            if len(objects) <= 1:
                continue

            ids = sorted(candidate.memory_id for candidate in slot_candidates)
            for candidate in slot_candidates:
                candidate.conflict_key = slot

            output.append(
                ConflictGroup(
                    conflict_id=self._id("raw_implicit", ids),
                    key=slot,
                    candidate_ids=ids,
                    conflict_type="implicit_version_conflict",
                    explicit=False,
                    explanation=(
                        "Semantic facts share the same subject and predicate "
                        "but contain different object values."
                    ),
                )
            )
        return output

    def _merge_groups(
        self,
        groups: list[ConflictGroup],
    ) -> list[ConflictGroup]:
        """Merge duplicate detection paths into one logical conflict."""
        buckets: dict[
            tuple[str, str | tuple[str, ...]],
            list[ConflictGroup],
        ] = defaultdict(list)

        for group in groups:
            if group.key:
                merge_key = ("slot", group.key)
            else:
                merge_key = ("members", tuple(sorted(group.candidate_ids)))
            buckets[merge_key].append(group)

        merged: list[ConflictGroup] = []
        for bucket_groups in buckets.values():
            candidate_ids = sorted({
                memory_id
                for group in bucket_groups
                for memory_id in group.candidate_ids
            })
            conflict_types = sorted({
                group.conflict_type for group in bucket_groups
            })
            explanations = list(dict.fromkeys(
                group.explanation.strip()
                for group in bucket_groups
                if group.explanation.strip()
            ))
            explicit = any(group.explicit for group in bucket_groups)
            slot_key = next(
                (group.key for group in bucket_groups if group.key),
                "",
            )

            merged.append(
                ConflictGroup(
                    conflict_id=self._id("conflict", candidate_ids),
                    key=slot_key,
                    candidate_ids=candidate_ids,
                    conflict_type="+".join(conflict_types),
                    explicit=explicit,
                    explanation=" ".join(explanations),
                )
            )

        merged.sort(key=lambda group: (group.key, tuple(group.candidate_ids)))
        return merged

    @staticmethod
    def _slot_key(candidate: MemoryCandidate) -> str:
        subject = ConflictDetector._normalise(candidate.subject or "")
        predicate = ConflictDetector._normalise(candidate.predicate or "")
        if subject and predicate:
            return f"{subject}::{predicate}"
        return ""

    @staticmethod
    def _normalise(value: str) -> str:
        return " ".join(str(value).lower().split())

    @staticmethod
    def _id(prefix: str, ids: list[str]) -> str:
        digest = hashlib.sha1("|".join(ids).encode("utf-8")).hexdigest()[:10]
        return f"{prefix}_{digest}"
