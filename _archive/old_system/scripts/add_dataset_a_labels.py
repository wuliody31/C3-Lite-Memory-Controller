import json
from pathlib import Path


def add_labels(input_path: Path, output_path: Path) -> None:
    rows = []

    for line in input_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue

        row = json.loads(line)

        # If no supporting memory is annotated, the system should abstain.
        row["should_abstain"] = not bool(row.get("supporting_memory_ids", []))

        # Default conflict labels. These should be manually reviewed later.
        row["expected_outdated_memory_ids"] = row.get("expected_outdated_memory_ids", [])
        row["conflict_type"] = row.get("conflict_type", "none")

        rows.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("Saved:", output_path)
    print("Total questions:", len(rows))
    print("Should abstain:", sum(1 for r in rows if r["should_abstain"]))
    print("Conflict candidates:", sum(1 for r in rows if r.get("question_type") in ["conflict_resolution", "temporal_update"]))


if __name__ == "__main__":
    add_labels(
        Path("../Dataset_A_v0_1/data_eval/eval_questions.jsonl"),
        Path("../Dataset_A_v0_1/data_eval/eval_questions_labeled.jsonl"),
    )