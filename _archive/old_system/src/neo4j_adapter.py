from __future__ import annotations

from typing import Any

from neo4j import GraphDatabase


class Neo4jMemoryAdapter:
    def __init__(
        self,
        uri: str,
        user: str,
        password: str | None,
    ):
        if not password:
            raise ValueError(
                "NEO4J_PASSWORD is missing. Set it in .env."
            )

        self.driver = GraphDatabase.driver(
            uri,
            auth=(user, password),
        )

    def close(self) -> None:
        self.driver.close()

    def ping(self) -> bool:
        with self.driver.session() as session:
            return (
                session.run("RETURN 1 AS ok")
                .single()["ok"]
                == 1
            )

    def smoke_counts(self) -> dict[str, int]:
        query = """
        MATCH (u:User) WITH count(u) AS users
        MATCH (s:Session) WITH users, count(s) AS sessions
        MATCH (e:EpisodicEvent)
        WITH users, sessions, count(e) AS episodic
        MATCH (f:SemanticFact)
        WITH users, sessions, episodic, count(f) AS semantic
        MATCH (en:Entity)
        WITH users, sessions, episodic, semantic, count(en) AS entities
        RETURN users, sessions, episodic, semantic, entities
        """

        with self.driver.session() as session:
            row = session.run(query).single()
            return dict(row) if row else {}

    def get_episodic_candidates(
        self,
        user_id: str,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        query = """
        MATCH (:User {user_id: $user_id})
              -[:HAS_EPISODIC_MEMORY]->
              (e:EpisodicEvent)
        RETURN
            e.memory_id AS id,
            'episodic' AS memory_type,
            toString(e.date) AS date,
            e.event AS text,
            e.importance AS importance,
            e.source_session AS source,
            e.entities AS entities
        ORDER BY e.date DESC, e.importance DESC
        LIMIT $limit
        """

        with self.driver.session() as session:
            return [
                dict(record)
                for record in session.run(
                    query,
                    user_id=user_id,
                    limit=limit,
                )
            ]

    def get_semantic_candidates(
        self,
        user_id: str,
        include_archived: bool = False,
        limit: int = 300,
    ) -> list[dict[str, Any]]:
        status_filter = (
            ""
            if include_archived
            else "WHERE f.status = 'active'"
        )

        query = f"""
        MATCH (:User {{user_id: $user_id}})
              -[:HAS_SEMANTIC_FACT]->
              (f:SemanticFact)
        {status_filter}
        RETURN
            f.triple_id AS id,
            'semantic' AS memory_type,
            f.subject + ' --' + f.relation + '--> ' + f.object AS text,
            f.subject AS subject,
            f.relation AS relation,
            f.object AS object,
            f.status AS status,
            f.confidence AS confidence,
            toString(f.last_updated) AS last_updated,
            f.source AS source
        ORDER BY f.last_updated DESC, f.confidence DESC
        LIMIT $limit
        """

        with self.driver.session() as session:
            return [
                dict(record)
                for record in session.run(
                    query,
                    user_id=user_id,
                    limit=limit,
                )
            ]

    def get_memory_by_id(
        self,
        memory_id: str,
    ) -> dict[str, Any] | None:
        query = """
        OPTIONAL MATCH (
            e:EpisodicEvent {memory_id: $memory_id}
        )
        OPTIONAL MATCH (
            f:SemanticFact {triple_id: $memory_id}
        )
        RETURN
            CASE
                WHEN e IS NOT NULL THEN e.memory_id
                ELSE f.triple_id
            END AS id,
            CASE
                WHEN e IS NOT NULL THEN 'episodic'
                ELSE 'semantic'
            END AS memory_type,
            CASE
                WHEN e IS NOT NULL THEN e.event
                ELSE
                    f.subject
                    + ' --'
                    + f.relation
                    + '--> '
                    + f.object
            END AS text,
            CASE
                WHEN e IS NOT NULL THEN toString(e.date)
                ELSE toString(f.last_updated)
            END AS date,
            CASE
                WHEN f IS NOT NULL THEN f.status
                ELSE null
            END AS status,
            CASE
                WHEN e IS NOT NULL THEN e.importance
                ELSE null
            END AS importance,
            CASE
                WHEN f IS NOT NULL THEN f.confidence
                ELSE null
            END AS confidence,
            CASE
                WHEN f IS NOT NULL THEN f.subject
                ELSE null
            END AS subject,
            CASE
                WHEN f IS NOT NULL THEN f.relation
                ELSE null
            END AS relation,
            CASE
                WHEN f IS NOT NULL THEN f.object
                ELSE null
            END AS object,
            CASE
                WHEN f IS NOT NULL THEN toString(f.last_updated)
                ELSE null
            END AS last_updated
        """

        with self.driver.session() as session:
            row = session.run(
                query,
                memory_id=memory_id,
            ).single()

            if not row or row["id"] is None:
                return None

            return dict(row)

    def find_memory_relations_touching(
        self,
        memory_ids: list[str],
    ) -> list[dict[str, Any]]:
        """
        Return graph memory relations where either endpoint is in the
        retrieved memory set.

        This enables bidirectional graph expansion:
        current -> historical as well as historical -> newer.
        """
        if not memory_ids:
            return []

        query = """
        MATCH (new)-[r:MEMORY_RELATION]->(old)
        WHERE
            coalesce(new.memory_id, new.triple_id) IN $memory_ids
            OR coalesce(old.memory_id, old.triple_id) IN $memory_ids
        RETURN
            coalesce(new.memory_id, new.triple_id) AS newer_memory,
            r.relation AS relation,
            coalesce(old.memory_id, old.triple_id) AS older_memory,
            r.reason AS reason
        """

        with self.driver.session() as session:
            return [
                dict(record)
                for record in session.run(
                    query,
                    memory_ids=memory_ids,
                )
            ]

    def find_memory_relations(
        self,
        memory_ids: list[str],
    ) -> list[dict[str, Any]]:
        # Backward-compatible alias.
        return self.find_memory_relations_touching(
            memory_ids
        )

    def find_superseding_memory(
        self,
        memory_id: str,
    ) -> list[dict[str, Any]]:
        return self.find_memory_relations_touching(
            [memory_id]
        )

    def get_timeline(
        self,
        user_id: str,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        return self.get_episodic_candidates(
            user_id,
            limit=limit,
        )
