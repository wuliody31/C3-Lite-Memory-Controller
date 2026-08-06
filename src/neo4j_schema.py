"""Idempotent Neo4j schema bootstrap for C3 lifecycle storage."""

from __future__ import annotations

from typing import Any


class Neo4jSchemaBootstrap:
    """Create and inspect the schema required by C3 lifecycle storage."""

    CONSTRAINT_NAMES = (
        "c3_semantic_fact_id_unique",
        "c3_episode_id_unique",
    )

    INDEX_NAMES = (
        "c3_semantic_lifecycle_lookup",
        "c3_episode_lifecycle_lookup",
        "semantic_fact_text",
        "episode_text",
    )

    STATEMENTS = (
        (
            "c3_semantic_fact_id_unique",
            """
            CREATE CONSTRAINT c3_semantic_fact_id_unique
            IF NOT EXISTS
            FOR (node:SemanticFact)
            REQUIRE node.fact_id IS UNIQUE
            """,
        ),
        (
            "c3_episode_id_unique",
            """
            CREATE CONSTRAINT c3_episode_id_unique
            IF NOT EXISTS
            FOR (node:Episode)
            REQUIRE node.episode_id IS UNIQUE
            """,
        ),
        (
            "c3_semantic_lifecycle_lookup",
            """
            CREATE INDEX c3_semantic_lifecycle_lookup
            IF NOT EXISTS
            FOR (node:SemanticFact)
            ON (
                node.lifecycle_managed,
                node.user_id,
                node.state_key,
                node.status
            )
            """,
        ),
        (
            "c3_episode_lifecycle_lookup",
            """
            CREATE INDEX c3_episode_lifecycle_lookup
            IF NOT EXISTS
            FOR (node:Episode)
            ON (
                node.lifecycle_managed,
                node.user_id,
                node.state_key,
                node.timestamp
            )
            """,
        ),
        (
            "semantic_fact_text",
            """
            CREATE FULLTEXT INDEX semantic_fact_text
            IF NOT EXISTS
            FOR (node:SemanticFact)
            ON EACH [
                node.text,
                node.subject,
                node.predicate,
                node.object
            ]
            """,
        ),
        (
            "episode_text",
            """
            CREATE FULLTEXT INDEX episode_text
            IF NOT EXISTS
            FOR (node:Episode)
            ON EACH [
                node.text
            ]
            """,
        ),
    )

    def __init__(
        self,
        *,
        driver: Any,
        database: str | None = None,
    ) -> None:
        self.driver = driver
        self.database = database

    def _session(self):
        if self.database:
            return self.driver.session(
                database=self.database
            )

        return self.driver.session()

    @staticmethod
    def _consume(result: Any) -> None:
        consume = getattr(result, "consume", None)

        if callable(consume):
            consume()

    def apply(
        self,
        *,
        wait_seconds: int = 60,
    ) -> dict[str, Any]:
        """Apply schema statements and wait for indexes to become online."""

        applied = []

        with self._session() as session:
            for name, statement in self.STATEMENTS:
                result = session.run(statement)
                self._consume(result)
                applied.append(name)

            wait_result = session.run(
                "CALL db.awaitIndexes($wait_seconds)",
                wait_seconds=int(wait_seconds),
            )
            self._consume(wait_result)

        return {
            "applied": applied,
            "constraint_names": list(
                self.CONSTRAINT_NAMES
            ),
            "index_names": list(
                self.INDEX_NAMES
            ),
        }

    def inspect(self) -> dict[str, list[dict[str, Any]]]:
        """Return the installed C3 constraints and indexes."""

        with self._session() as session:
            constraint_rows = list(
                session.run(
                    """
                    SHOW CONSTRAINTS
                    YIELD name, type
                    WHERE name IN $names
                    RETURN name, type
                    ORDER BY name
                    """,
                    names=list(
                        self.CONSTRAINT_NAMES
                    ),
                )
            )

            index_rows = list(
                session.run(
                    """
                    SHOW INDEXES
                    YIELD name, type, state
                    WHERE name IN $names
                    RETURN name, type, state
                    ORDER BY name
                    """,
                    names=list(
                        self.INDEX_NAMES
                    ),
                )
            )

        return {
            "constraints": [
                dict(row)
                for row in constraint_rows
            ],
            "indexes": [
                dict(row)
                for row in index_rows
            ],
        }
