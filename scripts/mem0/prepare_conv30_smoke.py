from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        return list(csv.DictReader(handle))


def write_csv(
    path: Path,
    rows: list[dict[str, str]],
) -> None:
    if not rows:
        raise ValueError("No rows selected.")

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def is_answerable(row: dict[str, str]) -> bool:
    should_abstain = (
        row.get("should_abstain", "")
        .strip()
        .lower()
    )

    return (
        should_abstain
        in {"", "false", "0", "no"}
        and bool(
            row.get(
                "supporting_memory_ids",
                "",
            ).strip()
        )
    )


def round_robin_select(
    rows: list[dict[str, str]],
    count: int,
) -> list[dict[str, str]]:
    buckets: dict[
        str,
        list[dict[str, str]],
    ] = defaultdict(list)

    for row in rows:
        buckets[
            row["question_type"]
        ].append(row)

    for bucket in buckets.values():
        bucket.sort(
            key=lambda item: item[
                "question_id"
            ]
        )

    types = sorted(buckets)
    offsets = {
        question_type: 0
        for question_type in types
    }

    selected: list[
        dict[str, str]
    ] = []

    while len(selected) < count:
        added = False

        for question_type in types:
            index = offsets[
                question_type
            ]

            bucket = buckets[
                question_type
            ]

            if index >= len(bucket):
                continue

            selected.append(
                bucket[index]
            )

            offsets[
                question_type
            ] += 1

            added = True

            if len(selected) == count:
                break

        if not added:
            break

    if len(selected) != count:
        raise ValueError(
            f"Expected {count} questions, "
            f"selected {len(selected)}."
        )

    return selected


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--questions",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--conversation-id",
        default="locomo_conv_30",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10,
    )

    args = parser.parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_rows = read_csv(
        args.questions
    )

    conversation_rows = [
        row
        for row in all_rows
        if row["user_id"]
        == args.conversation_id
    ]

    eligible_rows = [
        row
        for row in conversation_rows
        if is_answerable(row)
    ]

    selected = round_robin_select(
        eligible_rows,
        args.count,
    )

    output_path = (
        args.output_dir
        / "questions_smoke10.csv"
    )

    write_csv(
        output_path,
        selected,
    )

    category_counts = Counter(
        row["question_type"]
        for row in selected
    )

    manifest = {
        "conversation_id": (
            args.conversation_id
        ),
        "selection_strategy": (
            "deterministic_category_round_robin"
        ),
        "conversation_questions": len(
            conversation_rows
        ),
        "eligible_questions": len(
            eligible_rows
        ),
        "selected_questions": len(
            selected
        ),
        "category_counts": dict(
            sorted(category_counts.items())
        ),
        "question_ids": [
            row["question_id"]
            for row in selected
        ],
        "source_sha256": sha256(
            args.questions
        ),
        "output_sha256": sha256(
            output_path
        ),
    }

    manifest_path = (
        args.output_dir
        / "selection_manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        )
    )

    print(
        "CONV30 SMOKE SELECTION: PASSED"
    )


if __name__ == "__main__":
    main()
