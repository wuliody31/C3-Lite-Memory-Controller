import csv
import json
from pathlib import Path


pred_path = Path("outputs/locomo_one_conv_smoke_v2/predictions.jsonl")
import_dir = Path("../LoCoMo_C3/neo4j_import")

episodic_path = import_dir / "nodes_episodic_events.csv"
semantic_path = import_dir / "nodes_semantic_facts.csv"
relations_path = import_dir / "rel_memory_relations.csv"


def read_csv_map(path, key):
    result = {}

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            result[row[key]] = row

    return result


episodic = read_csv_map(episodic_path, "memory_id")
semantic = read_csv_map(semantic_path, "triple_id")

semantic_to_source = {}

with relations_path.open("r", encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)

    for row in reader:
        if row.get("relation") == "DERIVED_FROM":
            semantic_to_source[row["source_id"]] = row["target_id"]


rows = [
    json.loads(line)
    for line in pred_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

rows = [
    r for r in rows
    if r["method"] == "c3_lite_controller"
]


def describe_memory(memory_id):
    if memory_id in episodic:
        row = episodic[memory_id]
        return {
            "id": memory_id,
            "type": "episodic",
            "date": row.get("date"),
            "text": row.get("event"),
        }

    if memory_id in semantic:
        row = semantic[memory_id]
        return {
            "id": memory_id,
            "type": "semantic",
            "source_dialogue_memory": semantic_to_source.get(memory_id, ""),
            "text": row.get("object"),
        }

    return {
        "id": memory_id,
        "type": "unknown",
        "text": "",
    }


for r in rows:
    print("\n" + "=" * 100)
    print("ID:", r["question_id"])
    print("Q:", r["question"])
    print("GOLD ANSWER:", r.get("gold_answer"))
    print("STATE:", r.get("confidence_state"))

    gold = r.get("supporting_memory_ids", [])
    used = r.get("used_memory_ids", [])
    overlap = sorted(set(gold) & set(used))

    print("\nGOLD IDS:", gold)
    print("USED IDS:", used)
    print("OVERLAP:", overlap)

    print("\n--- GOLD MEMORY TEXTS ---")
    for mid in gold:
        item = describe_memory(mid)
        print(json.dumps(item, ensure_ascii=False, indent=2)[:1500])

    print("\n--- USED MEMORY TEXTS ---")
    for mid in used:
        item = describe_memory(mid)
        print(json.dumps(item, ensure_ascii=False, indent=2)[:1500])