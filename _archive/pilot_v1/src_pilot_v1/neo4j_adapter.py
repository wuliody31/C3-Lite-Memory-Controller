from __future__ import annotations
from typing import Any
from neo4j import GraphDatabase


class Neo4jMemoryAdapter:
    def __init__(self, uri: str, user: str, password: str | None):
        if not password:
            raise ValueError('NEO4J_PASSWORD is missing. Set it in .env.')
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        self.driver.close()

    def ping(self) -> bool:
        with self.driver.session() as session:
            return session.run('RETURN 1 AS ok').single()['ok'] == 1

    def smoke_counts(self) -> dict[str, int]:
        query = '''
        MATCH (u:User) WITH count(u) AS users
        MATCH (s:Session) WITH users, count(s) AS sessions
        MATCH (e:EpisodicEvent) WITH users, sessions, count(e) AS episodic
        MATCH (f:SemanticFact) WITH users, sessions, episodic, count(f) AS semantic
        MATCH (en:Entity) WITH users, sessions, episodic, semantic, count(en) AS entities
        RETURN users, sessions, episodic, semantic, entities
        '''
        with self.driver.session() as session:
            row = session.run(query).single()
            return dict(row) if row else {}

    def search_episodic(self, user_id: str, keyword: str, limit: int = 8) -> list[dict[str, Any]]:
        query = '''
        MATCH (:User {user_id: $user_id})-[:HAS_EPISODIC_MEMORY]->(e:EpisodicEvent)
        WHERE toLower(e.event) CONTAINS toLower($keyword)
           OR toLower(coalesce(e.entities, '')) CONTAINS toLower($keyword)
        RETURN e.memory_id AS id, 'episodic' AS memory_type, toString(e.date) AS date,
               e.event AS text, e.importance AS importance, e.source_session AS source
        ORDER BY e.date DESC, e.importance DESC
        LIMIT $limit
        '''
        with self.driver.session() as session:
            return [dict(r) for r in session.run(query, user_id=user_id, keyword=keyword, limit=limit)]

    def search_semantic(self, user_id: str, keyword: str, limit: int = 8, active_only: bool = True) -> list[dict[str, Any]]:
        status_filter = "AND f.status = 'active'" if active_only else ''
        query = f'''
        MATCH (:User {{user_id: $user_id}})-[:HAS_SEMANTIC_FACT]->(f:SemanticFact)
        WHERE (toLower(f.subject) CONTAINS toLower($keyword)
            OR toLower(f.relation) CONTAINS toLower($keyword)
            OR toLower(f.object) CONTAINS toLower($keyword))
        {status_filter}
        RETURN f.triple_id AS id, 'semantic' AS memory_type,
               f.subject + ' --' + f.relation + '--> ' + f.object AS text,
               f.subject AS subject, f.relation AS relation, f.object AS object,
               f.status AS status, f.confidence AS confidence,
               toString(f.last_updated) AS last_updated, f.source AS source
        ORDER BY f.confidence DESC, f.last_updated DESC
        LIMIT $limit
        '''
        with self.driver.session() as session:
            return [dict(r) for r in session.run(query, user_id=user_id, keyword=keyword, limit=limit)]

    def get_memory_by_id(self, memory_id: str) -> dict[str, Any] | None:
        query = '''
        OPTIONAL MATCH (e:EpisodicEvent {memory_id: $memory_id})
        OPTIONAL MATCH (f:SemanticFact {triple_id: $memory_id})
        RETURN CASE WHEN e IS NOT NULL THEN e.memory_id ELSE f.triple_id END AS id,
               CASE WHEN e IS NOT NULL THEN 'episodic' ELSE 'semantic' END AS memory_type,
               CASE WHEN e IS NOT NULL THEN e.event ELSE f.subject + ' --' + f.relation + '--> ' + f.object END AS text,
               CASE WHEN e IS NOT NULL THEN toString(e.date) ELSE toString(f.last_updated) END AS date,
               CASE WHEN f IS NOT NULL THEN f.status ELSE null END AS status
        '''
        with self.driver.session() as session:
            row = session.run(query, memory_id=memory_id).single()
            if not row or row['id'] is None:
                return None
            return dict(row)

    def find_superseding_memory(self, memory_id: str) -> list[dict[str, Any]]:
        query = '''
        MATCH (new)-[r:MEMORY_RELATION]->(old)
        WHERE coalesce(old.memory_id, old.triple_id) = $memory_id
        RETURN coalesce(new.memory_id, new.triple_id) AS newer_memory,
               r.relation AS relation,
               coalesce(old.memory_id, old.triple_id) AS older_memory,
               r.reason AS reason
        '''
        with self.driver.session() as session:
            return [dict(r) for r in session.run(query, memory_id=memory_id)]
