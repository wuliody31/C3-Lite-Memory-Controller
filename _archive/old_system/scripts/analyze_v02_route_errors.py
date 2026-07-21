import json
import csv
from pathlib import Path
from collections import Counter


pred_path = Path("outputs/heldout_v02_algorithm_v2_1/predictions.jsonl")
out_path = Path("outputs/heldout_v02_algorithm_v2_1/route_error_analysis.csv")


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

    gold = set(r.get("expected_route", []))
    pred = set(r.get("predicted_route", []))

    if gold == pred:
        continue

    missing = gold - pred
    extra = pred - gold

    rows.append({
        "question_id": r.get("question_id"),
        "question_type": r.get("question_type"),
        "question": r.get("question"),
        "predicted_query_type": r.get("predicted_query_type"),
        "gold_route": ";".join(sorted(gold)),
        "predicted_route": ";".join(sorted(pred)),
        "missing_route": ";".join(sorted(missing)),
        "extra_route": ";".join(sorted(extra)),
        "confidence_state": r.get("confidence_state"),
        "confidence_score": r.get("confidence_score"),
        "query_coverage": r.get("query_coverage"),
        "used_memory_ids": ";".join(r.get("used_memory_ids", [])),
        "route_plan": stringify(r.get("route_plan")),
        "query_profile": stringify(r.get("query_profile")),
    })


out_path.parent.mkdir(parents=True, exist_ok=True)

with out_path.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print("Saved:", out_path)
print("Route errors:", len(rows))

print("\nBy question type:")
print(Counter(r["question_type"] for r in rows))

print("\nMissing route counts:")
missing_counter = Counter()
for r in rows:
    for item in r["missing_route"].split(";"):
        if item:
            missing_counter[item] += 1
print(missing_counter)

print("\nExtra route counts:")
extra_counter = Counter()
for r in rows:
    for item in r["extra_route"].split(";"):
        if item:
            extra_counter[item] += 1
print(extra_counter)