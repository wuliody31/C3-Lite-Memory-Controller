from __future__ import annotations

from datetime import datetime
from typing import Any

GraphDatabase: Any
ClientError: Any

try:
    from neo4j import GraphDatabase as _GraphDatabase
    from neo4j.exceptions import ClientError as _Neo4jClientError
except ImportError:
    GraphDatabase = None

    class _FallbackClientError(Exception):
        pass

    ClientError = _FallbackClientError
else:
    GraphDatabase = _GraphDatabase
    ClientError = _Neo4jClientError


from ..errors import RetrievalError

from ..schemas import (
    MemoryCandidate,
    MemoryType,
    QueryFeatures,
    QueryMode,
    QueryState,
)


class Neo4jMemoryStore:
    """Read adapter for Episode and SemanticFact nodes."""

    CURRENT_STATUSES = (
        "current",
        "active",
        "valid",
    )

    def __init__(
        self,
        *,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
        config: dict[str, Any],
        database: str | None = None,
        driver: Any | None = None,
    ) -> None:
        neo = config.get("neo4j", {})

        if driver is None:
            if GraphDatabase is None:
                raise RuntimeError(
                    "Install the neo4j package "
                    "before using --neo4j."
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
        self.database = (
            database
            or neo.get("database")
        )

        self.episode_index = neo.get(
            "episode_fulltext_index",
            "episode_text",
        )

        self.semantic_index = neo.get(
            "semantic_fulltext_index",
            "semantic_fact_text",
        )

    def retrieve(
        self,
        *,
        memory_type: MemoryType,
        state: QueryState,
        features: QueryFeatures,
        top_k: int,
        include_archived: bool = False,
    ) -> list[MemoryCandidate]:
        try:
            if memory_type == MemoryType.EPISODIC:
                return self._episodes(
                    state,
                    features,
                    top_k,
                )

            if memory_type == MemoryType.SEMANTIC:
                effective_include_archived = (
                    self._include_archived(
                        features=features,
                        requested=include_archived,
                    )
                )

                return self._semantic(
                    state,
                    features,
                    top_k,
                    effective_include_archived,
                )

            return []

        except ClientError as exc:
            raise RetrievalError(
                "Neo4j retrieval fallback query failed."
            ) from exc

    @staticmethod
    def _include_archived(
        *,
        features: QueryFeatures,
        requested: bool,
    ) -> bool:
        """Expand historical state without changing current queries."""

        if requested:
            return True

        if features.query_mode in {
            QueryMode.HISTORICAL,
            QueryMode.TIMELINE,
        }:
            return True

        return bool(
            features.asks_historical_state
            or features.asks_timeline
        )

    def _episodes(
        self,
        state: QueryState,
        features: QueryFeatures,
        top_k: int,
    ) -> list[MemoryCandidate]:
        try:
            rows = self._run(
                """
                CALL db.index.fulltext.queryNodes(
                    $index,
                    $search_query
                )
                YIELD node, score
                WHERE node.user_id = $user_id
                RETURN node, score
                ORDER BY
                    score DESC,
                    node.timestamp DESC
                LIMIT $limit
                """,
                index=self.episode_index,
                search_query=self._lucene(
                    features.normalised_query
                ),
                user_id=state.user_id,
                limit=top_k,
            )

        except ClientError:
            rows = self._run(
                """
                MATCH (node:Episode)
                WHERE node.user_id = $user_id
                  AND toLower(
                        coalesce(node.text, '')
                      ) CONTAINS toLower($term)
                RETURN node, 1.0 AS score
                ORDER BY node.timestamp DESC
                LIMIT $limit
                """,
                user_id=state.user_id,
                term=self._term(features),
                limit=top_k,
            )

        output = []

        for row in rows:
            node = dict(row["node"])

            output.append(
                MemoryCandidate(
                    memory_id=str(
                        node.get("episode_id")
                        or node.get("id")
                    ),
                    memory_type=MemoryType.EPISODIC,
                    text=str(
                        node.get("text", "")
                    ),
                    user_id=state.user_id,
                    timestamp=self._date(
                        node.get("timestamp")
                    ),
                    status=str(
                        node.get(
                            "status",
                            "current",
                        )
                    ),
                    confidence=float(
                        node.get(
                            "confidence",
                            1.0,
                        )
                    ),
                    importance=float(
                        node.get(
                            "importance",
                            0.5,
                        )
                    ),
                    authority=str(
                        node.get(
                            "authority",
                            "unknown",
                        )
                    ),
                    source_ids=list(
                        node.get("source_ids")
                        or []
                    ),
                    metadata={
                        **node,
                        "neo4j_fulltext_score": float(
                            row.get("score", 0.0)
                        ),
                    },
                )
            )

        return output

    def _semantic(
        self,
        state: QueryState,
        features: QueryFeatures,
        top_k: int,
        include_archived: bool,
    ) -> list[MemoryCandidate]:
        params = {
            "index": self.semantic_index,
            "search_query": self._lucene(
                features.normalised_query
            ),
            "user_id": state.user_id,
            "include_archived": bool(
                include_archived
            ),
            "current_statuses": list(
                self.CURRENT_STATUSES
            ),
            "limit": top_k,
        }

        try:
            rows = self._run(
                """
                CALL db.index.fulltext.queryNodes(
                    $index,
                    $search_query
                )
                YIELD node, score
                WHERE node.user_id = $user_id
                  AND (
                    $include_archived = true
                    OR toLower(
                        coalesce(
                            node.status,
                            'current'
                        )
                    ) IN $current_statuses
                  )
                OPTIONAL MATCH
                    (node)
                    -[rel:
                        SUPERSEDES
                        |CONTRADICTS
                        |INVALIDATES
                    ]->
                    (other:SemanticFact)
                WITH
                    node,
                    score,
                    collect({
                        type: type(rel),
                        target_id: coalesce(
                            other.fact_id,
                            other.id
                        )
                    }) AS relations
                RETURN node, score, relations
                ORDER BY
                    score DESC,
                    node.valid_from DESC
                LIMIT $limit
                """,
                **params,
            )

        except ClientError:
            rows = self._run(
                """
                MATCH (node:SemanticFact)
                WHERE node.user_id = $user_id
                  AND (
                    $include_archived = true
                    OR toLower(
                        coalesce(
                            node.status,
                            'current'
                        )
                    ) IN $current_statuses
                  )
                  AND (
                    toLower(
                        coalesce(node.text, '')
                    ) CONTAINS toLower($term)
                    OR toLower(
                        coalesce(node.subject, '')
                    ) CONTAINS toLower($term)
                    OR toLower(
                        coalesce(node.object, '')
                    ) CONTAINS toLower($term)
                  )
                OPTIONAL MATCH
                    (node)
                    -[rel:
                        SUPERSEDES
                        |CONTRADICTS
                        |INVALIDATES
                    ]->
                    (other:SemanticFact)
                WITH
                    node,
                    1.0 AS score,
                    collect({
                        type: type(rel),
                        target_id: coalesce(
                            other.fact_id,
                            other.id
                        )
                    }) AS relations
                RETURN node, score, relations
                ORDER BY node.valid_from DESC
                LIMIT $limit
                """,
                user_id=state.user_id,
                include_archived=bool(
                    include_archived
                ),
                current_statuses=list(
                    self.CURRENT_STATUSES
                ),
                term=self._term(features),
                limit=top_k,
            )

        output = []

        for row in rows:
            node = dict(row["node"])

            text = node.get("text") or " ".join(
                str(value)
                for value in (
                    node.get("subject"),
                    node.get("predicate"),
                    node.get("object"),
                )
                if value is not None
            )

            relations = [
                relation
                for relation in (
                    row.get("relations")
                    or []
                )
                if (
                    relation.get("type")
                    and relation.get("target_id")
                )
            ]

            output.append(
                MemoryCandidate(
                    memory_id=str(
                        node.get("fact_id")
                        or node.get("id")
                    ),
                    memory_type=MemoryType.SEMANTIC,
                    text=str(text),
                    user_id=state.user_id,
                    timestamp=self._date(
                        node.get("valid_from")
                    ),
                    subject=node.get("subject"),
                    predicate=node.get("predicate"),
                    object_value=(
                        node.get("object")
                        or node.get("object_value")
                    ),
                    status=str(
                        node.get(
                            "status",
                            "current",
                        )
                    ),
                    confidence=float(
                        node.get(
                            "confidence",
                            1.0,
                        )
                    ),
                    authority=str(
                        node.get(
                            "authority",
                            "unknown",
                        )
                    ),
                    source_ids=list(
                        node.get(
                            "source_episode_ids"
                        )
                        or node.get("source_ids")
                        or []
                    ),
                    relations=relations,
                    metadata={
                        **node,
                        "neo4j_fulltext_score": float(
                            row.get("score", 0.0)
                        ),
                        "include_archived": bool(
                            include_archived
                        ),
                    },
                )
            )

        return output

    def _session(self):
        if self.database:
            return self.driver.session(
                database=self.database
            )

        return self.driver.session()

    def _run(
        self,
        cypher: str,
        **params: Any,
    ) -> list[dict[str, Any]]:
        try:
            with self._session() as session:
                return [
                    dict(record)
                    for record in session.run(
                        cypher,
                        **params,
                    )
                ]

        except ClientError:
            # Full-text ClientError is intentionally allowed to propagate
            # so episodic and semantic retrieval can use their existing
            # fallback MATCH queries.
            raise

        except Exception as exc:
            raise RetrievalError(
                "Neo4j query execution failed."
            ) from exc

    @staticmethod
    def _lucene(query: str) -> str:
        terms = [
            item
            for item in query.replace(
                '"',
                " ",
            ).split()
            if len(item) >= 2
        ]

        return (
            " OR ".join(
                f'"{item}"'
                for item in terms[:12]
            )
            or "*"
        )

    @staticmethod
    def _term(
        features: QueryFeatures,
    ) -> str:
        if features.entities:
            return features.entities[0]

        if features.tokens:
            return max(
                features.tokens,
                key=len,
            )

        return features.normalised_query[:50]

    @staticmethod
    def _date(value: Any) -> datetime | None:
        if not value:
            return None

        if isinstance(value, datetime):
            return value

        if hasattr(value, "to_native"):
            value = value.to_native()

            if isinstance(value, datetime):
                return value

        try:
            return datetime.fromisoformat(
                str(value).replace(
                    "Z",
                    "+00:00",
                )
            )

        except ValueError:
            return None

    def close(self) -> None:
        if self._owns_driver:
            self.driver.close()
