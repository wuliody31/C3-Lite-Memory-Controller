import json
import csv
from pathlib import Path


input_path = Path("../Dataset_A_v0_1/data_eval/eval_questions_labeled_conflict.jsonl")
output_path = Path("../Dataset_A_v0_1/data_eval/eval_questions_labeled_conflict_review.csv")

rows = []

for line in input_path.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue

    row = json.loads(line)

    rows.append({
        "question_id": row.get("question_id"),
        "user_id": row.get("user_id"),
        "question_type": row.get("question_type"),
        "question": row.get("question"),
        "supporting_memory_ids": ";".join(row.get("supporting_memory_ids", [])),
        "should_abstain": row.get("should_abstain"),
        "expected_outdated_memory_ids": ";".join(row.get("expected_outdated_memory_ids", [])),
        "conflict_type": row.get("conflict_type"),
        "gold_answer": row.get("gold_answer", ""),
    })

with output_path.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print("Saved:", output_path)