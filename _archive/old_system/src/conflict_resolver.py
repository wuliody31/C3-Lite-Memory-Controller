from __future__ import annotations

from collections import defaultdict
from typing import Any


INVALIDATING_RELATIONS = {
    "SUPERSEDES",
    "UPDATES",
    "OVERRIDES",
    "REPLACES",
}

SUPPORTIVE_RELATIONS = {
    "CLARIFIES",
    "ALIGNS_WITH",
    "DERIVED_FROM",
}

HISTORY_AWARE_QUERY_TYPES = {
    "episodic_recall",
    "temporal_update",
    "conflict_resolution",
    "explainability",
}


def _date_key(item: dict[str, Any]) -> str:
    return str(
        item.get("last_updated")
        or item.get("date")
        or ""
    )


class ConflictResolver:
    """
    Relation-aware conflict resolver with bidirectional graph expansion.
    """

    def __init__(self, neo4j_adapter):
        self.neo4j = neo4j_adapter

    def _expand_related_memory(
        self,
        memory_map: dict[str, dict[str, Any]],
        memory_id: str,
        anchor_score: float,
        relation_direction: str,
    ) -> None:
        if memory_id in memory_map:
            return

        memory = self.neo4j.get_memory_by_id(
            memory_id
        )
        if memory is None:
            return

        multiplier = (
            0.95
            if relation_direction == "newer"
            else 0.85
        )

        memory["evidence_score"] = round(
            max(
                0.0,
                min(
                    1.0,
                    anchor_score * multiplier,
                ),
            ),
            6,
        )
        memory["relevance_score"] = 0.15
        memory["graph_expanded"] = True
        memory_map[memory_id] = memory

    def resolve(
        self,
        memories: list[dict[str, Any]],
        query_type: str,
    ) -> dict[str, Any]:
        memory_map = {
            item["id"]: dict(item)
            for item in memories
        }

        relations = (
            self.neo4j
            .find_memory_relations_touching(
                list(memory_map)
            )
        )

        outdated_ids: set[str] = set()
        newer_ids: set[str] = set()

        conflict_notes: list[dict[str, Any]] = []
        supportive_notes: list[dict[str, Any]] = []

        history_aware = (
            query_type
            in HISTORY_AWARE_QUERY_TYPES
        )

        for relation_row in relations:
            relation = str(
                relation_row["relation"]
            ).upper()

            note = {
                "older_memory": relation_row[
                    "older_memory"
                ],
                "newer_memory": relation_row[
                    "newer_memory"
                ],
                "relation": relation,
                "reason": relation_row.get(
                    "reason"
                ),
            }

            if relation in INVALIDATING_RELATIONS:
                older_id = relation_row[
                    "older_memory"
                ]
                newer_id = relation_row[
                    "newer_memory"
                ]

                outdated_ids.add(older_id)
                newer_ids.add(newer_id)
                conflict_notes.append(note)

                anchor_score = max(
                    float(
                        memory_map
                        .get(older_id, {})
                        .get("evidence_score", 0.0)
                    ),
                    float(
                        memory_map
                        .get(newer_id, {})
                        .get("evidence_score", 0.0)
                    ),
                )

                self._expand_related_memory(
                    memory_map,
                    newer_id,
                    anchor_score,
                    "newer",
                )

                if history_aware:
                    self._expand_related_memory(
                        memory_map,
                        older_id,
                        anchor_score,
                        "older",
                    )

            elif relation in SUPPORTIVE_RELATIONS:
                supportive_notes.append(note)

        # Detect implicit version conflicts in semantic memory.
        groups: dict[
            tuple[str, str],
            list[dict[str, Any]],
        ] = defaultdict(list)

        for item in memory_map.values():
            if item.get("memory_type") == "semantic":
                groups[
                    (
                        str(item.get("subject")),
                        str(item.get("relation")),
                    )
                ].append(item)

        unresolved_conflicts: list[
            dict[str, Any]
        ] = []

        for key, group in groups.items():
            objects = {
                str(item.get("object"))
                for item in group
            }

            if len(objects) <= 1:
                continue

            ordered = sorted(
                group,
                key=lambda item: (
                    1
                    if item.get("status") == "active"
                    else 0,
                    _date_key(item),
                    float(
                        item.get("confidence")
                        or 0.0
                    ),
                ),
                reverse=True,
            )

            preferred = ordered[0]

            for older in ordered[1:]:
                if older["id"] in outdated_ids:
                    continue

                if (
                    older.get("status") == "archived"
                    or _date_key(older)
                    < _date_key(preferred)
                ):
                    outdated_ids.add(older["id"])
                    conflict_notes.append({
                        "older_memory": older["id"],
                        "newer_memory": preferred["id"],
                        "relation": (
                            "IMPLICIT_VERSION_CONFLICT"
                        ),
                        "reason": (
                            "Semantic facts share the "
                            f"subject/relation {key} but "
                            "have different objects; the "
                            "active/newer fact was preferred."
                        ),
                    })
                else:
                    unresolved_conflicts.append({
                        "memory_ids": [
                            preferred["id"],
                            older["id"],
                        ],
                        "reason": (
                            "Multiple active semantic facts "
                            f"disagree for {key}."
                        ),
                    })

        current_memories: list[
            dict[str, Any]
        ] = []
        historical_memories: list[
            dict[str, Any]
        ] = []

        for item in memory_map.values():
            enriched = dict(item)

            if item["id"] in outdated_ids:
                enriched["evidence_role"] = (
                    "historical"
                )
                historical_memories.append(
                    enriched
                )
            else:
                enriched["evidence_role"] = (
                    "current"
                )
                current_memories.append(enriched)

        current_memories.sort(
            key=lambda item: item.get(
                "evidence_score",
                0.0,
            ),
            reverse=True,
        )
        historical_memories.sort(
            key=lambda item: item.get(
                "evidence_score",
                0.0,
            ),
            reverse=True,
        )

        usable_memories = (
            current_memories
            + historical_memories
            if history_aware
            else current_memories
        )

        return {
            "usable_memories": usable_memories,
            "current_memories": current_memories,
            "historical_memories": (
                historical_memories
            ),
            "outdated_memory_ids": sorted(
                outdated_ids
            ),
            "newer_memory_ids": sorted(
                newer_ids
            ),
            "conflict_notes": conflict_notes,
            "supportive_notes": supportive_notes,
            "unresolved_conflicts": (
                unresolved_conflicts
            ),
        }
