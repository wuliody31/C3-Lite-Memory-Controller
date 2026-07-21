from pathlib import Path
import csv


import_dir = Path("../Dataset_A_v0_2/neo4j_import")

files = [
    "nodes_users.csv",
    "nodes_sessions.csv",
    "nodes_episodic_events.csv",
    "nodes_semantic_facts.csv",
    "nodes_entities.csv",
    "rel_user_has_session.csv",
    "rel_session_has_event.csv",
    "rel_user_has_event.csv",
    "rel_user_has_semantic_fact.csv",
    "rel_event_mentions_entity.csv",
    "rel_semantic_edges.csv",
    "rel_memory_relations.csv",
]

for filename in files:
    path = import_dir / filename

    print("\n" + "=" * 80)
    print(filename)

    if not path.exists():
        print("MISSING")
        continue

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        print("HEADERS:")
        print(reader.fieldnames)

        first = next(reader, None)

        print("FIRST ROW:")
        if first is None:
            print("EMPTY")
        else:
            print(first)