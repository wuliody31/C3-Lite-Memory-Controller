import csv
import json
from pathlib import Path
from collections import defaultdict


eval_path = Path("../LoCoMo_C3/data_eval/eval_questions.jsonl")
rel_path = Path("../LoCoMo_C3/neo4j_import/rel_memory_relations.csv")

backup_path = Path("../LoCoMo_C3/data_eval/eval_questions_before_semantic_gold_patch.jsonl")


def read_jsonl(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# Backup original eval file.
backup_path.write_text(
    eval_path.read_text(encoding="utf-8"),
    encoding="utf-8",
)

# Build mapping:
# target episodic memory ID -> semantic observation IDs derived from it.
episodic_to_semantic = defaultdict(list)

with rel_path.open("r", encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)

    for row in reader:
        source_id = row["source_id"]
        target_id = row["target_id"]
        relation = row.get("relation", "")

        if relation == "DERIVED_FROM":
            episodic_to_semantic[target_id].append(source_id)


rows = read_jsonl(eval_path)

patched = []

for row in rows:
    original_supporting = list(row.get("supporting_memory_ids", []))
    expanded_supporting = list(original_supporting)

    for memory_id in original_supporting:
        for semantic_id in episodic_to_semantic.get(memory_id, []):
            if semantic_id not in expanded_supporting:
                expanded_supporting.append(semantic_id)

    row["supporting_memory_ids"] = expanded_supporting

    patched.append({
        "question_id": row["question_id"],
        "before": original_supporting,
        "after": expanded_supporting,
    })

write_jsonl(eval_path, rows)

print("Patched:", eval_path)
print("Backup:", backup_path)
print()

for item in patched:
    print(item["question_id"])
    print("  before:", item["before"])
    print("  after :", item["after"])