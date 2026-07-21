from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-path",
        default=(
            "../Dataset_A_v0_1/"
            "data_eval/eval_questions.jsonl"
        ),
    )
    args = parser.parse_args()

    path = Path(args.eval_path)

    rows = [
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    print(
        "question_id | question_type | "
        "should_abstain | label_source"
    )
    print("-" * 90)

    count = 0

    for row in rows:
        explicit = row.get("should_abstain")

        if explicit is None:
            should_abstain = not bool(
                row.get(
                    "supporting_memory_ids",
                    [],
                )
            )
            source = "derived_from_empty_support"
        else:
            should_abstain = bool(explicit)
            source = "explicit"

        if should_abstain:
            count += 1

        print(
            f"{row['question_id']} | "
            f"{row['question_type']} | "
            f"{should_abstain} | "
            f"{source}"
        )

    print()
    print(
        f"Questions requiring abstention: "
        f"{count}/{len(rows)}"
    )


if __name__ == "__main__":
    main()
