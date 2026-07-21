from __future__ import annotations

from datetime import datetime
from typing import Any

try:
    from neo4j import GraphDatabase
    from neo4j.exceptions import ClientError
except ImportError:  # permits JSON smoke tests before Neo4j is installed
    GraphDatabase = None
    class ClientError(Exception):
        pass

from ..schemas import MemoryCandidate, MemoryType, QueryFeatures, QueryState


class Neo4jMemoryStore:
    """Adapter for Episode and SemanticFact nodes. Change Cypher here if your schema differs."""
    def __init__(self, *, uri: str, user: str, password: str, config: dict[str, Any]):
        if GraphDatabase is None:
            raise RuntimeError("Install the neo4j package before using --neo4j.")
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        neo = config.get("neo4j", {})
        self.episode_index = neo.get("episode_fulltext_index", "episode_text")
        self.semantic_index = neo.get("semantic_fulltext_index", "semantic_fact_text")

    def retrieve(self, *, memory_type: MemoryType, state: QueryState, features: QueryFeatures, top_k: int, include_archived: bool = False) -> list[MemoryCandidate]:
        if memory_type == MemoryType.EPISODIC:
            return self._episodes(state, features, top_k)
        if memory_type == MemoryType.SEMANTIC:
            return self._semantic(state, features, top_k, include_archived)
        return []

    def _episodes(self, state: QueryState, features: QueryFeatures, top_k: int) -> list[MemoryCandidate]:
        try:
            rows = self._run("""
                CALL db.index.fulltext.queryNodes($index, $query) YIELD node, score
                WHERE node.user_id = $user_id
                RETURN node, score ORDER BY score DESC, node.timestamp DESC LIMIT $limit
            """, index=self.episode_index, query=self._lucene(features.normalised_query), user_id=state.user_id, limit=top_k)
        except ClientError:
            rows = self._run("""
                MATCH (node:Episode) WHERE node.user_id = $user_id
                AND toLower(coalesce(node.text,'')) CONTAINS toLower($term)
                RETURN node, 1.0 AS score ORDER BY node.timestamp DESC LIMIT $limit
            """, user_id=state.user_id, term=self._term(features), limit=top_k)
        return [MemoryCandidate(
            memory_id=str(dict(r["node"]).get("episode_id") or dict(r["node"]).get("id")),
            memory_type=MemoryType.EPISODIC, text=str(dict(r["node"]).get("text", "")), user_id=state.user_id,
            timestamp=self._date(dict(r["node"]).get("timestamp")), status=str(dict(r["node"]).get("status", "current")),
            confidence=float(dict(r["node"]).get("confidence", 1.0)), importance=float(dict(r["node"]).get("importance", 0.5)),
            authority=str(dict(r["node"]).get("authority", "unknown")), source_ids=list(dict(r["node"]).get("source_ids") or []),
            metadata={**dict(r["node"]), "neo4j_fulltext_score": float(r.get("score", 0.0))}
        ) for r in rows]

    def _semantic(self, state: QueryState, features: QueryFeatures, top_k: int, include_archived: bool) -> list[MemoryCandidate]:
        statuses = ["current", "active", "valid", "outdated", "superseded", "archived", "invalid"] if include_archived else ["current", "active", "valid"]
        try:
            rows = self._run("""
                CALL db.index.fulltext.queryNodes($index, $query) YIELD node, score
                WHERE node.user_id = $user_id AND toLower(coalesce(node.status,'current')) IN $statuses
                OPTIONAL MATCH (node)-[rel:SUPERSEDES|CONTRADICTS|INVALIDATES]->(other:SemanticFact)
                WITH node, score, collect({type:type(rel), target_id:coalesce(other.fact_id,other.id)}) AS relations
                RETURN node, score, relations ORDER BY score DESC, node.valid_from DESC LIMIT $limit
            """, index=self.semantic_index, query=self._lucene(features.normalised_query), user_id=state.user_id, statuses=statuses, limit=top_k)
        except ClientError:
            rows = self._run("""
                MATCH (node:SemanticFact) WHERE node.user_id = $user_id
                AND toLower(coalesce(node.status,'current')) IN $statuses
                AND (toLower(coalesce(node.text,'')) CONTAINS toLower($term)
                  OR toLower(coalesce(node.subject,'')) CONTAINS toLower($term)
                  OR toLower(coalesce(node.object,'')) CONTAINS toLower($term))
                OPTIONAL MATCH (node)-[rel:SUPERSEDES|CONTRADICTS|INVALIDATES]->(other:SemanticFact)
                WITH node, 1.0 AS score, collect({type:type(rel), target_id:coalesce(other.fact_id,other.id)}) AS relations
                RETURN node, score, relations ORDER BY node.valid_from DESC LIMIT $limit
            """, user_id=state.user_id, statuses=statuses, term=self._term(features), limit=top_k)
        output = []
        for r in rows:
            n = dict(r["node"])
            text = n.get("text") or " ".join(str(x) for x in [n.get("subject"), n.get("predicate"), n.get("object")] if x is not None)
            output.append(MemoryCandidate(
                memory_id=str(n.get("fact_id") or n.get("id")), memory_type=MemoryType.SEMANTIC, text=str(text), user_id=state.user_id,
                timestamp=self._date(n.get("valid_from")), subject=n.get("subject"), predicate=n.get("predicate"), object_value=n.get("object") or n.get("object_value"),
                status=str(n.get("status", "current")), confidence=float(n.get("confidence", 1.0)), authority=str(n.get("authority", "unknown")),
                source_ids=list(n.get("source_episode_ids") or n.get("source_ids") or []),
                relations=[x for x in (r.get("relations") or []) if x.get("type") and x.get("target_id")],
                metadata={**n, "neo4j_fulltext_score": float(r.get("score", 0.0))},
            ))
        return output

    def _run(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        with self.driver.session() as session:
            return [dict(record) for record in session.run(cypher, **params)]

    @staticmethod
    def _lucene(query: str) -> str:
        terms = [x for x in query.replace('"', ' ').split() if len(x) >= 2]
        return " OR ".join(f'"{x}"' for x in terms[:12]) or "*"

    @staticmethod
    def _term(features: QueryFeatures) -> str:
        return features.entities[0] if features.entities else (max(features.tokens, key=len) if features.tokens else features.normalised_query[:50])

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
            return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None

    def close(self) -> None:
        self.driver.close()
