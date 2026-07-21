import json
import shutil
from pathlib import Path


v01_path = Path("../Dataset_A_v0_1/data_eval/eval_questions.jsonl")
v02_path = Path("../Dataset_A_v0_2/data_eval/eval_questions.jsonl")

heldout_dir = Path("../Dataset_A_v0_2_heldout")
heldout_eval_dir = heldout_dir / "data_eval"
heldout_raw_dir = heldout_dir / "data_raw"

heldout_eval_dir.mkdir(parents=True, exist_ok=True)
heldout_raw_dir.mkdir(parents=True, exist_ok=True)

v01_ids = set()

for line in v01_path.read_text(encoding="utf-8").splitlines():
    if line.strip():
        row = json.loads(line)
        v01_ids.add(row["question_id"])

heldout_rows = []

for line in v02_path.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue

    row = json.loads(line)

    if row["question_id"] not in v01_ids:
        row["should_abstain"] = row.get(
            "should_abstain",
            not bool(row.get("supporting_memory_ids", [])),
        )

        row["expected_outdated_memory_ids"] = row.get(
            "expected_outdated_memory_ids",
            [],
        )

        row["conflict_type"] = row.get(
            "conflict_type",
            "none",
        )

        heldout_rows.append(row)

output_path = heldout_eval_dir / "eval_questions.jsonl"

with output_path.open("w", encoding="utf-8") as f:
    for row in heldout_rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

src_rules = Path("../Dataset_A_v0_2/data_raw/procedural_memory.json")
dst_rules = heldout_raw_dir / "procedural_memory.json"

if src_rules.exists():
    shutil.copy(src_rules, dst_rules)

print("Saved held-out file:", output_path)
print("Number of held-out questions:", len(heldout_rows))
print("Should abstain:", sum(1 for r in heldout_rows if r["should_abstain"]))
print()
print("Question IDs:")

for row in heldout_rows:
    print(
        row["question_id"],
        "|",
        row.get("question_type"),
        "|",
        row.get("question"),
    )