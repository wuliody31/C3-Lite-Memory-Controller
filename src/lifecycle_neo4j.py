"""Neo4j persistence adapter for the C3 lifecycle control plane.

Only nodes carrying ``lifecycle_managed=true`` are exposed through this
store. This isolates Phase II-B state transitions from existing Dataset A
and LoCoMo graph records.

One ``commit`` call is executed inside one Neo4j write transaction.
Updates, new semantic states, episodic transition events and graph
relations therefore succeed or roll back together.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Iterable

try:
    from neo4j import GraphDatabase
except ImportError:
    GraphDatabase = None

from .lifecycle import (
    LifecycleRecord,
    LifecycleStoreError,
    MemoryStatus,
)
from .schemas import MemoryType


class Neo4jLifecycleStore:
    """Neo4j implementation of the lifecycle persistence protocol."""

    def __init__(
        self,
        *,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
        database: str | None = None,
        driver: Any | None = None,
    ) -> None:
        if driver is None:
            if GraphDatabase is None:
                raise RuntimeError(
                    "Install the neo4j package before "
                    "using Neo4jLifecycleStore."
                )

            if not uri or not user or password is None:
                raise ValueError(
                    "uri, user and password are required "
                    "when driver is not supplied."
                )

            driver = GraphDatabase.driver(
                uri,
                auth=(user, password),
            )
            self._owns_driver = True
        else:
            self._owns_driver = False

        self.driver = driver
        self.database = database

    def close(self) -> None:
        if self._owns_driver:
            self.driver.close()

    def _session(self):
        if self.database:
            return self.driver.session(
                database=self.database
            )

        return self.driver.session()

    def get(
        self,
        memory_id: str,
    ) -> LifecycleRecord | None:
        cypher = """
        /* lifecycle_get */
        MATCH (node)
        WHERE (node:SemanticFact OR node:Episode)
          AND node.lifecycle_managed = true
          AND coalesce(
                node.fact_id,
                node.episode_id,
                node.id
              ) = $memory_id
        RETURN labels(node) AS labels,
               properties(node) AS properties
        LIMIT 1
        """

        with self._session() as session:
            rows = list(
                session.run(
                    cypher,
                    memory_id=memory_id,
                )
            )

        if not rows:
            return None

        return self._record_from_row(rows[0])

    def list(
        self,
        *,
        user_id: str | None = None,
        memory_type: MemoryType | None = None,
        state_key: str | None = None,
        status: MemoryStatus | None = None,
    ) -> list[LifecycleRecord]:
        cypher = """
        /* lifecycle_list */
        MATCH (node)
        WHERE (node:SemanticFact OR node:Episode)
          AND node.lifecycle_managed = true
          AND (
                $user_id IS NULL
                OR node.user_id = $user_id
              )
          AND (
                $memory_type IS NULL
                OR (
                    $memory_type = 'semantic'
                    AND node:SemanticFact
                )
                OR (
                    $memory_type = 'episodic'
                    AND node:Episode
                )
              )
          AND (
                $state_key IS NULL
                OR node.state_key = $state_key
              )
          AND (
                $status IS NULL
                OR toLower(
                    coalesce(node.status, 'current')
                ) = toLower($status)
              )
        RETURN labels(node) AS labels,
               properties(node) AS properties
        ORDER BY coalesce(
                     node.valid_from,
                     node.timestamp
                 ) ASC,
                 coalesce(
                     node.fact_id,
                     node.episode_id,
                     node.id
                 ) ASC
        """

        params = {
            "user_id": user_id,
            "memory_type": (
                memory_type.value
                if memory_type is not None
                else None
            ),
            "state_key": state_key,
            "status": (
                status.value
                if status is not None
                else None
            ),
        }

        with self._session() as session:
            rows = list(
                session.run(
                    cypher,
                    **params,
                )
            )

        return [
            self._record_from_row(row)
            for row in rows
        ]

    def all(self) -> list[LifecycleRecord]:
        return self.list()

    def commit(
        self,
        *,
        add: Iterable[LifecycleRecord] = (),
        update: Iterable[LifecycleRecord] = (),
    ) -> None:
        additions = list(add)
        updates = list(update)

        self._validate_batch(
            additions=additions,
            updates=updates,
        )

        payload = {
            "add_semantic": [
                self._row(record)
                for record in additions
                if (
                    record.memory_type
                    == MemoryType.SEMANTIC
                )
            ],
            "add_episodic": [
                self._row(record)
                for record in additions
                if (
                    record.memory_type
                    == MemoryType.EPISODIC
                )
            ],
            "update_semantic": [
                self._row(record)
                for record in updates
                if (
                    record.memory_type
                    == MemoryType.SEMANTIC
                )
            ],
            "update_episodic": [
                self._row(record)
                for record in updates
                if (
                    record.memory_type
                    == MemoryType.EPISODIC
                )
            ],
            "supersedes": self._supersedes_rows(
                additions
            ),
            "episode_links": self._episode_link_rows(
                additions
            ),
        }

        add_ids = [
            record.memory_id
            for record in additions
        ]

        update_ids = [
            record.memory_id
            for record in updates
        ]

        with self._session() as session:
            if hasattr(session, "execute_write"):
                session.execute_write(
                    self._commit_transaction,
                    add_ids,
                    update_ids,
                    payload,
                )
            elif hasattr(
                session,
                "write_transaction",
            ):
                session.write_transaction(
                    self._commit_transaction,
                    add_ids,
                    update_ids,
                    payload,
                )
            else:
                raise RuntimeError(
                    "Neo4j session does not expose "
                    "execute_write or write_transaction."
                )

    @staticmethod
    def _validate_batch(
        *,
        additions: list[LifecycleRecord],
        updates: list[LifecycleRecord],
    ) -> None:
        add_ids = [
            record.memory_id
            for record in additions
        ]
        update_ids = [
            record.memory_id
            for record in updates
        ]

        if len(add_ids) != len(set(add_ids)):
            raise LifecycleStoreError(
                "Duplicate memory ID inside add batch."
            )

        if len(update_ids) != len(set(update_ids)):
            raise LifecycleStoreError(
                "Duplicate memory ID inside update batch."
            )

        overlap = set(add_ids) & set(update_ids)

        if overlap:
            raise LifecycleStoreError(
                "The same memory cannot be added and "
                "updated in one transaction: "
                f"{sorted(overlap)}"
            )

        unsupported = [
            record.memory_type
            for record in [
                *additions,
                *updates,
            ]
            if record.memory_type not in {
                MemoryType.SEMANTIC,
                MemoryType.EPISODIC,
            }
        ]

        if unsupported:
            raise LifecycleStoreError(
                "Neo4jLifecycleStore only supports "
                "semantic and episodic records."
            )

    @staticmethod
    def _commit_transaction(
        tx: Any,
        add_ids: list[str],
        update_ids: list[str],
        payload: dict[str, Any],
    ) -> None:
        if add_ids:
            existing_add = list(
                tx.run(
                    """
                    /* lifecycle_precheck_add */
                    MATCH (node)
                    WHERE (
                        node:SemanticFact
                        OR node:Episode
                    )
                      AND coalesce(
                            node.fact_id,
                            node.episode_id,
                            node.id
                          ) IN $memory_ids
                    RETURN coalesce(
                        node.fact_id,
                        node.episode_id,
                        node.id
                    ) AS memory_id
                    """,
                    memory_ids=add_ids,
                )
            )

            if existing_add:
                existing = sorted(
                    str(row["memory_id"])
                    for row in existing_add
                )
                raise LifecycleStoreError(
                    "Cannot add existing lifecycle "
                    f"memory IDs: {existing}"
                )

        if update_ids:
            existing_update = list(
                tx.run(
                    """
                    /* lifecycle_precheck_update */
                    MATCH (node)
                    WHERE (
                        node:SemanticFact
                        OR node:Episode
                    )
                      AND node.lifecycle_managed = true
                      AND coalesce(
                            node.fact_id,
                            node.episode_id,
                            node.id
                          ) IN $memory_ids
                    RETURN coalesce(
                        node.fact_id,
                        node.episode_id,
                        node.id
                    ) AS memory_id
                    """,
                    memory_ids=update_ids,
                )
            )

            found = {
                str(row["memory_id"])
                for row in existing_update
            }

            missing = sorted(
                set(update_ids) - found
            )

            if missing:
                raise LifecycleStoreError(
                    "Cannot update missing lifecycle "
                    f"memory IDs: {missing}"
                )

        if payload["update_semantic"]:
            tx.run(
                """
                /* lifecycle_update_semantic */
                UNWIND $rows AS row
                MATCH (node:SemanticFact)
                WHERE coalesce(
                    node.fact_id,
                    node.id
                ) = row.memory_id
                  AND node.lifecycle_managed = true
                SET node += row.properties
                """,
                rows=payload["update_semantic"],
            )

        if payload["update_episodic"]:
            tx.run(
                """
                /* lifecycle_update_episodic */
                UNWIND $rows AS row
                MATCH (node:Episode)
                WHERE coalesce(
                    node.episode_id,
                    node.id
                ) = row.memory_id
                  AND node.lifecycle_managed = true
                SET node += row.properties
                """,
                rows=payload["update_episodic"],
            )

        if payload["add_semantic"]:
            tx.run(
                """
                /* lifecycle_add_semantic */
                UNWIND $rows AS row
                CREATE (node:SemanticFact)
                SET node = row.properties
                """,
                rows=payload["add_semantic"],
            )

        if payload["add_episodic"]:
            tx.run(
                """
                /* lifecycle_add_episodic */
                UNWIND $rows AS row
                CREATE (node:Episode)
                SET node = row.properties
                """,
                rows=payload["add_episodic"],
            )

        if payload["supersedes"]:
            tx.run(
                """
                /* lifecycle_create_supersedes */
                UNWIND $rows AS row
                MATCH (new:SemanticFact)
                WHERE coalesce(
                    new.fact_id,
                    new.id
                ) = row.new_memory_id
                  AND new.lifecycle_managed = true
                MATCH (old:SemanticFact)
                WHERE coalesce(
                    old.fact_id,
                    old.id
                ) = row.old_memory_id
                  AND old.lifecycle_managed = true
                MERGE (new)-[:SUPERSEDES]->(old)
                """,
                rows=payload["supersedes"],
            )

        if payload["episode_links"]:
            tx.run(
                """
                /* lifecycle_create_episode_links */
                UNWIND $rows AS row
                MATCH (event:Episode)
                WHERE coalesce(
                    event.episode_id,
                    event.id
                ) = row.episode_memory_id
                  AND event.lifecycle_managed = true
                MATCH (old:SemanticFact)
                WHERE coalesce(
                    old.fact_id,
                    old.id
                ) = row.old_memory_id
                  AND old.lifecycle_managed = true
                MATCH (new:SemanticFact)
                WHERE coalesce(
                    new.fact_id,
                    new.id
                ) = row.new_memory_id
                  AND new.lifecycle_managed = true
                MERGE (event)-[:FROM_STATE]->(old)
                MERGE (event)-[:TO_STATE]->(new)
                """,
                rows=payload["episode_links"],
            )

    @classmethod
    def _row(
        cls,
        record: LifecycleRecord,
    ) -> dict[str, Any]:
        return {
            "memory_id": record.memory_id,
            "properties": cls._properties(record),
        }

    @staticmethod
    def _properties(
        record: LifecycleRecord,
    ) -> dict[str, Any]:
        metadata = dict(record.metadata)

        source_ids = metadata.get("source_ids")

        if not isinstance(source_ids, list):
            source_ids = (
                [record.source_turn_id]
                if record.source_turn_id
                else []
            )

        common: dict[str, Any] = {
            "id": record.memory_id,
            "memory_type": record.memory_type.value,
            "user_id": record.user_id,
            "text": record.text,
            "status": record.status.value,
            "state_key": record.state_key,
            "value": record.value,
            "valid_from": record.valid_from,
            "valid_to": record.valid_to,
            "source_turn_id": record.source_turn_id,
            "supersedes": record.supersedes,
            "superseded_by": record.superseded_by,
            "source_ids": source_ids,
            "confidence": float(
                metadata.get("confidence", 1.0)
            ),
            "importance": float(
                metadata.get("importance", 0.5)
            ),
            "authority": str(
                metadata.get(
                    "authority",
                    "c3_lifecycle_controller",
                )
            ),
            "lifecycle_role": str(
                metadata.get(
                    "lifecycle_role",
                    "",
                )
            ),
            "lifecycle_managed": True,
            "lifecycle_schema_version": "phase2b_v01",
            "metadata_json": json.dumps(
                metadata,
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            ),
        }

        if record.memory_type == MemoryType.SEMANTIC:
            predicate = (
                metadata.get("predicate")
                or record.state_key
            )

            object_value = (
                metadata.get("object")
                or metadata.get("object_value")
                or record.value
            )

            common.update(
                {
                    "fact_id": record.memory_id,
                    "subject": str(
                        metadata.get(
                            "subject",
                            record.user_id,
                        )
                    ),
                    "predicate": (
                        str(predicate)
                        if predicate is not None
                        else None
                    ),
                    "object": (
                        str(object_value)
                        if object_value is not None
                        else None
                    ),
                    "source_episode_ids": (
                        source_ids
                    ),
                }
            )

        elif record.memory_type == MemoryType.EPISODIC:
            common.update(
                {
                    "episode_id": record.memory_id,
                    "timestamp": record.valid_from,
                }
            )

        return common

    @staticmethod
    def _supersedes_rows(
        additions: list[LifecycleRecord],
    ) -> list[dict[str, str]]:
        return [
            {
                "new_memory_id": record.memory_id,
                "old_memory_id": record.supersedes,
            }
            for record in additions
            if (
                record.memory_type
                == MemoryType.SEMANTIC
                and record.supersedes
            )
        ]

    @staticmethod
    def _episode_link_rows(
        additions: list[LifecycleRecord],
    ) -> list[dict[str, str]]:
        rows = []

        for record in additions:
            if (
                record.memory_type
                != MemoryType.EPISODIC
            ):
                continue

            old_id = record.metadata.get(
                "old_memory_id"
            )
            new_id = record.metadata.get(
                "new_memory_id"
            )

            if old_id and new_id:
                rows.append(
                    {
                        "episode_memory_id": (
                            record.memory_id
                        ),
                        "old_memory_id": str(
                            old_id
                        ),
                        "new_memory_id": str(
                            new_id
                        ),
                    }
                )

        return rows

    @classmethod
    def _record_from_row(
        cls,
        row: Any,
    ) -> LifecycleRecord:
        labels = set(row["labels"])
        properties = dict(row["properties"])

        if "SemanticFact" in labels:
            memory_type = MemoryType.SEMANTIC
            memory_id = str(
                properties.get("fact_id")
                or properties.get("id")
            )
        elif "Episode" in labels:
            memory_type = MemoryType.EPISODIC
            memory_id = str(
                properties.get("episode_id")
                or properties.get("id")
            )
        else:
            raise LifecycleStoreError(
                "Lifecycle node has unsupported labels: "
                f"{sorted(labels)}"
            )

        metadata = cls._decode_metadata(
            properties.get("metadata_json")
        )

        canonical_keys = {
            "id",
            "fact_id",
            "episode_id",
            "memory_type",
            "user_id",
            "text",
            "status",
            "state_key",
            "value",
            "valid_from",
            "valid_to",
            "timestamp",
            "source_turn_id",
            "supersedes",
            "superseded_by",
            "metadata_json",
        }

        for key, value in properties.items():
            if key not in canonical_keys:
                metadata.setdefault(key, value)

        return LifecycleRecord(
            memory_id=memory_id,
            user_id=str(
                properties.get("user_id", "")
            ),
            memory_type=memory_type,
            text=str(
                properties.get("text", "")
            ),
            status=cls._normalise_status(
                properties.get(
                    "status",
                    "current",
                )
            ),
            valid_from=cls._datetime(
                properties.get("valid_from")
                or properties.get("timestamp")
            ),
            state_key=properties.get("state_key"),
            value=properties.get("value"),
            valid_to=cls._optional_datetime(
                properties.get("valid_to")
            ),
            source_turn_id=properties.get(
                "source_turn_id"
            ),
            supersedes=properties.get(
                "supersedes"
            ),
            superseded_by=properties.get(
                "superseded_by"
            ),
            metadata=metadata,
        )

    @staticmethod
    def _decode_metadata(
        value: Any,
    ) -> dict[str, Any]:
        if not value:
            return {}

        if isinstance(value, dict):
            return dict(value)

        try:
            decoded = json.loads(str(value))
        except json.JSONDecodeError:
            return {
                "metadata_decode_error": str(value)
            }

        return (
            decoded
            if isinstance(decoded, dict)
            else {"metadata_value": decoded}
        )

    @staticmethod
    def _normalise_status(
        value: Any,
    ) -> MemoryStatus:
        raw = str(value or "current").lower()

        if raw in {
            "current",
            "active",
            "valid",
        }:
            return MemoryStatus.CURRENT

        if raw in {
            "superseded",
            "outdated",
            "inactive",
            "invalid",
        }:
            return MemoryStatus.SUPERSEDED

        return MemoryStatus.ARCHIVED

    @staticmethod
    def _datetime(
        value: Any,
    ) -> datetime:
        parsed = Neo4jLifecycleStore._optional_datetime(
            value
        )

        if parsed is None:
            raise LifecycleStoreError(
                "Lifecycle node has no valid timestamp."
            )

        return parsed

    @staticmethod
    def _optional_datetime(
        value: Any,
    ) -> datetime | None:
        if value is None or value == "":
            return None

        if isinstance(value, datetime):
            parsed = value
        elif hasattr(value, "to_native"):
            parsed = value.to_native()
        else:
            parsed = datetime.fromisoformat(
                str(value).replace("Z", "+00:00")
            )

        if parsed.tzinfo is None:
            return parsed.replace(
                tzinfo=timezone.utc
            )

        return parsed.astimezone(timezone.utc)
