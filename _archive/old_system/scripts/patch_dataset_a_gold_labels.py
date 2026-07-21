import json
from pathlib import Path


PATCH = {
    "q_user01_003": {
        "expected_outdated_memory_ids": ["e_user01_006", "s_user01_005"],
        "conflict_type": "supersedes",
    },
    "q_user01_004": {
        "expected_outdated_memory_ids": ["e_user01_001", "e_user01_006", "s_user01_005"],
        "conflict_type": "supersedes",
    },
    "q_user01_007": {
        "expected_outdated_memory_ids": ["e_user01_014", "s_user01_016"],
        "conflict_type": "updates",
    },
    "q_user02_003": {
        "expected_outdated_memory_ids": ["e_user02_001", "s_user02_001"],
        "conflict_type": "updates",
    },
    "q_user02_004": {
        "expected_outdated_memory_ids": ["e_user02_001", "s_user02_001"],
        "conflict_type": "updates",
    },
    "q_user02_020": {
        "expected_outdated_memory_ids": ["e_user02_001", "s_user02_001"],
        "conflict_type": "updates",
    },
    "q_user03_003": {
        "expected_outdated_memory_ids": ["e_user03_001", "s_user03_001"],
        "conflict_type": "updates",
    },
    "q_user03_004": {
        "expected_outdated_memory_ids": ["e_user03_001", "s_user03_001"],
        "conflict_type": "updates",
    },
    "q_user03_006": {
        "expected_outdated_memory_ids": ["e_user03_002", "s_user03_004"],
        "conflict_type": "updates",
    },
    "q_user03_014": {
        "expected_outdated_memory_ids": ["e_user03_001", "s_user03_001"],
        "conflict_type": "updates",
    },
    "q_user03_019": {
        "expected_outdated_memory_ids": ["e_user03_002", "s_user03_004"],
        "conflict_type": "updates",
    },
}


input_path = Path("../Dataset_A_v0_1/data_eval/eval_questions_labeled.jsonl")
output_path = Path("../Dataset_A_v0_1/data_eval/eval_questions_labeled_conflict.jsonl")

rows = []

for line in input_path.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue

    row = json.loads(line)

    if row["question_id"] in PATCH:
        row.update(PATCH[row["question_id"]])

    rows.append(row)

with output_path.open("w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print("Saved:", output_path)
print("Total:", len(rows))
print("should_abstain=True:", sum(1 for r in rows if r.get("should_abstain") is True))
print("conflict labeled:", sum(1 for r in rows if r.get("conflict_type") != "none"))
print("outdated labeled:", sum(1 for r in rows if r.get("expected_outdated_memory_ids")))