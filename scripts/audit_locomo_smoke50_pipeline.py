from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_jsonl(
    path: Path,
) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


def load_csv(
    path: Path,
) -> list[dict[str, str]]:
    with path.open(
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        return list(csv.DictReader(handle))


def as_ids(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, (list, tuple, set)):
        output: list[str] = []

        for item in value:
            output.extend(as_ids(item))

        return output

    if isinstance(value, str):
        text = value.strip()

        if not text:
            return []

        return [
            item.strip()
            for item in re.split(
                r"[;,]",
                text,
            )
            if item.strip()
        ]

    return [str(value).strip()]


def first_ids(
    row: dict[str, Any],
    debug: dict[str, Any],
    keys: tuple[str, ...],
) -> list[str]:
    for mapping in (row, debug):
        for key in keys:
            values = as_ids(
                mapping.get(key)
            )

            if values:
                return values

    return []


def expand_sources(
    ids: list[str],
    memory_map: dict[str, dict[str, Any]],
) -> set[str]:
    expanded = set(ids)

    for memory_id in ids:
        item = memory_map.get(memory_id)

        if not item:
            continue

        expanded.update(
            as_ids(item.get("source_ids"))
        )

    return expanded


def recall(
    predicted: set[str],
    gold: set[str],
) -> float | None:
    if not gold:
        return None

    return len(
        predicted & gold
    ) / len(gold)


def precision(
    predicted: set[str],
    gold: set[str],
) -> float | None:
    if not predicted:
        return 0.0 if gold else None

    return len(
        predicted & gold
    ) / len(predicted)


def mean(
    values: list[float | None],
) -> float | None:
    valid = [
        value
        for value in values
        if value is not None
    ]

    if not valid:
        return None

    return sum(valid) / len(valid)


def format_value(
    value: float | None,
) -> str:
    return (
        "NA"
        if value is None
        else f"{value:.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--predictions",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--questions",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--memories",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--answer-scores",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
    )

    args = parser.parse_args()

    predictions = load_jsonl(
        args.predictions
    )

    questions = {
        row["question_id"]: row
        for row in load_csv(args.questions)
    }

    answer_scores = {
        (
            row["method"],
            row["question_id"],
        ): float(row["answer_token_f1"])
        for row in load_csv(
            args.answer_scores
        )
    }

    memories = json.loads(
        args.memories.read_text(
            encoding="utf-8"
        )
    )["memories"]

    memory_map = {
        item["memory_id"]: item
        for item in memories
    }

    output_rows: list[dict[str, Any]] = []

    for row in predictions:
        method = row["method"]
        question_id = row["question_id"]
        question = questions[question_id]

        debug = row.get("debug", {})

        if not isinstance(debug, dict):
            debug = {}

        gold_ids = as_ids(
            question[
                "supporting_memory_ids"
            ]
        )

        raw_ids = first_ids(
            row,
            debug,
            (
                "raw_retrieved_ids",
                "raw_candidate_ids",
                "retrieved_ids",
            ),
        )

        ranked_ids = first_ids(
            row,
            debug,
            (
                "ranked_candidate_ids",
                "ranked_ids",
            ),
        )

        selected_ids = first_ids(
            row,
            debug,
            (
                "selected_ids",
                "used_ids",
                "selected_memory_ids",
                "used_memory_ids",
            ),
        )

        gold_set = set(gold_ids)

        raw_strict = set(raw_ids)
        ranked_strict = set(ranked_ids)
        selected_strict = set(
            selected_ids
        )

        raw_source = expand_sources(
            raw_ids,
            memory_map,
        )

        ranked_source = expand_sources(
            ranked_ids,
            memory_map,
        )

        selected_source = expand_sources(
            selected_ids,
            memory_map,
        )

        selected_types = Counter(
            memory_map.get(
                memory_id,
                {},
            ).get(
                "memory_type",
                "unknown",
            )
            for memory_id in selected_ids
        )

        route_types = (
            row.get(
                "selected_memory_types"
            )
            or debug.get(
                "selected_memory_types"
            )
            or debug.get(
                "route_selected_types"
            )
            or []
        )

        output_rows.append({
            "question_id": question_id,
            "method": method,
            "question_type": question[
                "question_type"
            ],
            "should_abstain": question[
                "should_abstain"
            ],
            "answer_token_f1": (
                answer_scores.get(
                    (method, question_id),
                    0.0,
                )
            ),
            "decision": row.get(
                "decision"
            ),
            "route_types": json.dumps(
                route_types,
                ensure_ascii=False,
            ),
            "gold_count": len(
                gold_ids
            ),
            "raw_count": len(raw_ids),
            "ranked_count": len(
                ranked_ids
            ),
            "selected_count": len(
                selected_ids
            ),
            "selected_type_counts": (
                json.dumps(
                    dict(selected_types),
                    ensure_ascii=False,
                    sort_keys=True,
                )
            ),
            "raw_strict_recall": recall(
                raw_strict,
                gold_set,
            ),
            "raw_source_recall": recall(
                raw_source,
                gold_set,
            ),
            "ranked_strict_recall": recall(
                ranked_strict,
                gold_set,
            ),
            "ranked_source_recall": recall(
                ranked_source,
                gold_set,
            ),
            "selected_strict_precision": precision(
                selected_strict,
                gold_set,
            ),
            "selected_strict_recall": recall(
                selected_strict,
                gold_set,
            ),
            "selected_source_precision": precision(
                selected_source,
                gold_set,
            ),
            "selected_source_recall": recall(
                selected_source,
                gold_set,
            ),
            "gold_ids": json.dumps(
                gold_ids,
                ensure_ascii=False,
            ),
            "selected_ids": json.dumps(
                selected_ids,
                ensure_ascii=False,
            ),
            "selected_expanded_sources": json.dumps(
                sorted(selected_source),
                ensure_ascii=False,
            ),
        })

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                output_rows[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(output_rows)

    print("=" * 88)
    print("LOCOMO SMOKE-50 PIPELINE AUDIT")
    print("=" * 88)

    for method in (
        "no_memory",
        "simple_retrieval",
        "all_memory",
        "c3",
    ):
        rows = [
            item
            for item in output_rows
            if item["method"] == method
        ]

        gold_rows = [
            item
            for item in rows
            if int(item["gold_count"]) > 0
        ]

        print()
        print("METHOD:", method)
        print("Questions:", len(rows))
        print(
            "Answer F1:",
            format_value(mean([
                float(item[
                    "answer_token_f1"
                ])
                for item in rows
                if (
                    item[
                        "should_abstain"
                    ].lower()
                    == "false"
                )
            ])),
        )
        print(
            "Raw strict recall:",
            format_value(mean([
                item["raw_strict_recall"]
                for item in gold_rows
            ])),
        )
        print(
            "Raw source recall:",
            format_value(mean([
                item["raw_source_recall"]
                for item in gold_rows
            ])),
        )
        print(
            "Ranked strict recall:",
            format_value(mean([
                item[
                    "ranked_strict_recall"
                ]
                for item in gold_rows
            ])),
        )
        print(
            "Ranked source recall:",
            format_value(mean([
                item[
                    "ranked_source_recall"
                ]
                for item in gold_rows
            ])),
        )
        print(
            "Selected strict recall:",
            format_value(mean([
                item[
                    "selected_strict_recall"
                ]
                for item in gold_rows
            ])),
        )
        print(
            "Selected source recall:",
            format_value(mean([
                item[
                    "selected_source_recall"
                ]
                for item in gold_rows
            ])),
        )
        print(
            "Mean selected:",
            format_value(mean([
                float(item[
                    "selected_count"
                ])
                for item in rows
            ])),
        )

    print()
    print("=" * 88)
    print("C3 BY CATEGORY")
    print("=" * 88)

    c3_rows = [
        item
        for item in output_rows
        if item["method"] == "c3"
    ]

    by_category: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(list)

    for item in c3_rows:
        by_category[
            item["question_type"]
        ].append(item)

    for category in sorted(by_category):
        rows = by_category[category]

        gold_rows = [
            item
            for item in rows
            if int(item["gold_count"]) > 0
        ]

        print(
            category,
            "| n =",
            len(rows),
            "| answer F1 =",
            format_value(mean([
                float(item[
                    "answer_token_f1"
                ])
                for item in rows
                if (
                    item[
                        "should_abstain"
                    ].lower()
                    == "false"
                )
            ])),
            "| raw source recall =",
            format_value(mean([
                item[
                    "raw_source_recall"
                ]
                for item in gold_rows
            ])),
            "| selected source recall =",
            format_value(mean([
                item[
                    "selected_source_recall"
                ]
                for item in gold_rows
            ])),
        )

    print()
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
