import csv
from pathlib import Path

from neo4j import GraphDatabase

from src.config import get_config


IMPORT_DIR = Path("../LoCoMo_C3/neo4j_import")


def read_csv(name: str) -> list[dict]:
    path = IMPORT_DIR / name

    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def to_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return default


def to_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def run_batch(session, query: str, rows: list[dict], batch_size: int = 500):
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        session.run(query, rows=batch)


def main():
    config = get_config()

    driver = GraphDatabase.driver(
        config.neo4j_uri,
        auth=(
            config.neo4j_user,
            config.neo4j_password,
        ),
    )

    with driver.session() as session:
        print("Clearing existing graph...")
        session.run("MATCH (n) DETACH DELETE n")

        print("Creating constraints / indexes...")
        session.run(
            """
            CREATE CONSTRAINT user_id_unique IF NOT EXISTS
            FOR (u:User)
            REQUIRE u.user_id IS UNIQUE
            """
        )
        session.run(
            """
            CREATE CONSTRAINT session_id_unique IF NOT EXISTS
            FOR (s:Session)
            REQUIRE s.session_id IS UNIQUE
            """
        )
        session.run(
            """
            CREATE CONSTRAINT episodic_event_id_unique IF NOT EXISTS
            FOR (e:EpisodicEvent)
            REQUIRE e.memory_id IS UNIQUE
            """
        )
        session.run(
            """
            CREATE CONSTRAINT semantic_fact_id_unique IF NOT EXISTS
            FOR (f:SemanticFact)
            REQUIRE f.triple_id IS UNIQUE
            """
        )
        session.run(
            """
            CREATE CONSTRAINT entity_id_unique IF NOT EXISTS
            FOR (e:Entity)
            REQUIRE e.entity_id IS UNIQUE
            """
        )

        # ------------------------------------------------------------------
        # Nodes
        # ------------------------------------------------------------------
        users = read_csv("nodes_users.csv")
        sessions = read_csv("nodes_sessions.csv")
        episodic = read_csv("nodes_episodic_events.csv")
        semantic = read_csv("nodes_semantic_facts.csv")
        entities = read_csv("nodes_entities.csv")

        print("Importing users:", len(users))
        run_batch(
            session,
            """
            UNWIND $rows AS row
            MERGE (u:User {user_id: row.user_id})
            SET u.name = row.name,
                u.domain = row.domain,
                u.long_term_goal = row.long_term_goal,
                u.stable_preferences = row.stable_preferences,
                u.notes = row.notes
            """,
            users,
        )

        print("Importing sessions:", len(sessions))
        run_batch(
            session,
            """
            UNWIND $rows AS row
            MERGE (s:Session {session_id: row.session_id})
            SET s.user_id = row.user_id,
                s.date = row.date,
                s.title = row.title,
                s.turn_count = toInteger(row.turn_count)
            """,
            sessions,
        )

        print("Importing episodic memories:", len(episodic))
        for row in episodic:
            row["importance_int"] = to_int(row.get("importance", 3), 3)

        run_batch(
            session,
            """
            UNWIND $rows AS row
            MERGE (e:EpisodicEvent {memory_id: row.memory_id})
            SET e.user_id = row.user_id,
                e.date = row.date,
                e.event = row.event,
                e.entities = row.entities,
                e.source_session = row.source_session,
                e.importance = row.importance_int,
                e.memory_type = row.memory_type
            """,
            episodic,
        )

        print("Importing semantic facts:", len(semantic))
        for row in semantic:
            row["confidence_float"] = to_float(row.get("confidence", 0.9), 0.9)

        run_batch(
            session,
            """
            UNWIND $rows AS row
            MERGE (f:SemanticFact {triple_id: row.triple_id})
            SET f.user_id = row.user_id,
                f.subject = row.subject,
                f.relation = row.relation,
                f.object = row.object,
                f.confidence = row.confidence_float,
                f.source = row.source,
                f.last_updated = row.last_updated,
                f.status = row.status,
                f.memory_type = row.memory_type
            """,
            semantic,
        )

        print("Importing entities:", len(entities))
        run_batch(
            session,
            """
            UNWIND $rows AS row
            MERGE (ent:Entity {entity_id: row.entity_id})
            SET ent.name = row.name,
                ent.label_hint = row.label_hint
            """,
            entities,
        )

        # ------------------------------------------------------------------
        # Relationships
        # ------------------------------------------------------------------
        rel_user_has_session = read_csv("rel_user_has_session.csv")
        rel_session_has_event = read_csv("rel_session_has_event.csv")
        rel_user_has_event = read_csv("rel_user_has_event.csv")
        rel_user_has_semantic_fact = read_csv("rel_user_has_semantic_fact.csv")
        rel_event_mentions_entity = read_csv("rel_event_mentions_entity.csv")
        rel_semantic_edges = read_csv("rel_semantic_edges.csv")
        rel_memory_relations = read_csv("rel_memory_relations.csv")

        print("Importing HAS_SESSION:", len(rel_user_has_session))
        run_batch(
            session,
            """
            UNWIND $rows AS row
            MATCH (u:User {user_id: row.user_id})
            MATCH (s:Session {session_id: row.session_id})
            MERGE (u)-[:HAS_SESSION]->(s)
            """,
            rel_user_has_session,
        )

        print("Importing CONTAINS_EVENT:", len(rel_session_has_event))
        run_batch(
            session,
            """
            UNWIND $rows AS row
            MATCH (s:Session {session_id: row.session_id})
            MATCH (e:EpisodicEvent {memory_id: row.memory_id})
            MERGE (s)-[:CONTAINS_EVENT]->(e)
            """,
            rel_session_has_event,
        )

        print("Importing HAS_EPISODIC_MEMORY:", len(rel_user_has_event))
        run_batch(
            session,
            """
            UNWIND $rows AS row
            MATCH (u:User {user_id: row.user_id})
            MATCH (e:EpisodicEvent {memory_id: row.memory_id})
            MERGE (u)-[:HAS_EPISODIC_MEMORY]->(e)
            """,
            rel_user_has_event,
        )

        print("Importing HAS_SEMANTIC_FACT:", len(rel_user_has_semantic_fact))
        run_batch(
            session,
            """
            UNWIND $rows AS row
            MATCH (u:User {user_id: row.user_id})
            MATCH (f:SemanticFact {triple_id: row.triple_id})
            MERGE (u)-[:HAS_SEMANTIC_FACT]->(f)
            """,
            rel_user_has_semantic_fact,
        )

        print("Importing MENTIONS:", len(rel_event_mentions_entity))
        run_batch(
            session,
            """
            UNWIND $rows AS row
            MATCH (e:EpisodicEvent {memory_id: row.memory_id})
            MATCH (ent:Entity {entity_id: row.entity_id})
            MERGE (e)-[r:MENTIONS]->(ent)
            SET r.entity_name = row.entity_name
            """,
            rel_event_mentions_entity,
        )

        print("Importing semantic edges:", len(rel_semantic_edges))
        for row in rel_semantic_edges:
            row["confidence_float"] = to_float(row.get("confidence", 0.9), 0.9)

        run_batch(
            session,
            """
            UNWIND $rows AS row
            MATCH (f:SemanticFact {triple_id: row.triple_id})
            MATCH (subj:Entity {entity_id: row.subject_entity_id})
            MATCH (obj:Entity {entity_id: row.object_entity_id})
            MERGE (subj)-[:REPRESENTS_RELATION]->(f)
            MERGE (f)-[r:SEMANTIC_RELATION]->(obj)
            SET r.relation = row.relation,
                r.confidence = row.confidence_float,
                r.source = row.source,
                r.last_updated = row.last_updated,
                r.status = row.status
            """,
            rel_semantic_edges,
        )

        print("Importing memory relations:", len(rel_memory_relations))
        run_batch(
            session,
            """
            UNWIND $rows AS row
            OPTIONAL MATCH (source_e:EpisodicEvent {memory_id: row.source_id})
            OPTIONAL MATCH (source_s:SemanticFact {triple_id: row.source_id})
            OPTIONAL MATCH (target_e:EpisodicEvent {memory_id: row.target_id})
            OPTIONAL MATCH (target_s:SemanticFact {triple_id: row.target_id})
            WITH row,
                 coalesce(source_e, source_s) AS source,
                 coalesce(target_e, target_s) AS target
            WHERE source IS NOT NULL AND target IS NOT NULL
            MERGE (source)-[r:MEMORY_RELATION]->(target)
            SET r.relation = row.relation,
                r.reason = row.reason
            """,
            rel_memory_relations,
        )

        print("\nFinal counts:")
        result = session.run(
            """
            RETURN
              count { MATCH (:User) } AS users,
              count { MATCH (:Session) } AS sessions,
              count { MATCH (:EpisodicEvent) } AS episodic,
              count { MATCH (:SemanticFact) } AS semantic,
              count { MATCH (:Entity) } AS entities
            """
        ).single()

        print(dict(result))

    driver.close()


if __name__ == "__main__":
    main()