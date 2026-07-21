import json
import csv
from pathlib import Path


pred_path = Path("outputs/heldout_v02_algorithm_v2_1/predictions.jsonl")
out_path = Path("outputs/heldout_v02_algorithm_v2_1/false_abstention_cases.csv")


def stringify(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


rows = []

for line in pred_path.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue

    r = json.loads(line)

    if r["method"] != "c3_lite_controller":
        continue

    expected_abstention = bool(r.get("expected_abstention", False))
    predicted_abstention = r.get("confidence_state") == "abstain"

    if predicted_abstention and not expected_abstention:
        rows.append({
            "question_id": r.get("question_id"),
            "question_type": r.get("question_type"),
            "question": r.get("question"),
            "gold_answer": r.get("gold_answer"),
            "confidence_state": r.get("confidence_state"),
            "confidence_score": r.get("confidence_score"),
            "query_coverage": r.get("query_coverage"),
            "answerability": stringify(r.get("answerability")),
            "confidence_reasons": stringify(r.get("confidence_reasons")),
            "used_memory_ids": ";".join(r.get("used_memory_ids", [])),
            "retrieved_memory_ids": ";".join(r.get("retrieved_memory_ids", [])),
            "predicted_answer": r.get("predicted_answer"),
        })


out_path.parent.mkdir(parents=True, exist_ok=True)

with out_path.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print("Saved:", out_path)
print("False abstentions:", len(rows))

for row in rows:
    print()
    print("ID:", row["question_id"])
    print("TYPE:", row["question_type"])
    print("Q:", row["question"])
    print("confidence_score:", row["confidence_score"])
    print("query_coverage:", row["query_coverage"])
    print("used:", row["used_memory_ids"])
    print("reasons:", row["confidence_reasons"])