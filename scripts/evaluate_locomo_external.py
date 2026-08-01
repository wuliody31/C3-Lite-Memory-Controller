from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_csv(
    path: Path,
) -> list[dict[str, str]]:
    with path.open(
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        return list(csv.DictReader(handle))


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


def parse_ids(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, list):
        output: list[str] = []

        for item in value:
            output.extend(parse_ids(item))

        return output

    text = str(value).strip()

    if not text:
        return []

    return [
        item.strip()
        for item in text.replace(
            ",",
            ";",
        ).split(";")
        if item.strip()
    ]


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
    }


def safe_divide(
    numerator: float,
    denominator: float,
) -> float:
    if denominator == 0:
        return 0.0

    return numerator / denominator


def precision_recall_f1(
    predicted: set[str],
    gold: set[str],
) -> tuple[float, float, float]:
    overlap = len(predicted & gold)

    precision = safe_divide(
        overlap,
        len(predicted),
    )

    recall = safe_divide(
        overlap,
        len(gold),
    )

    f1 = (
        0.0
        if precision + recall == 0
        else (
            2
            * precision
            * recall
            / (precision + recall)
        )
    )

    return precision, recall, f1


def project_to_sources(
    selected_ids: list[str],
    memory_map: dict[str, dict[str, Any]],
) -> set[str]:
    projected: set[str] = set()

    for memory_id in selected_ids:
        memory = memory_map.get(memory_id)

        if memory is None:
            projected.add(memory_id)
            continue

        memory_type = str(
            memory.get(
                "memory_type",
                "",
            )
        )

        sources = set(
            parse_ids(
                memory.get("source_ids")
            )
        )

        if memory_type == "episodic":
            projected.add(memory_id)

        elif sources:
            projected.update(sources)

        else:
            projected.add(memory_id)

    return projected


def mean(
    values: list[float],
) -> float | None:
    if not values:
        return None

    return sum(values) / len(values)


def format_optional(
    value: float | None,
) -> str:
    return (
        ""
        if value is None
        else str(value)
    )


def summarise(
    rows: list[dict[str, Any]],
    *,
    method: str,
    question_type: str,
) -> dict[str, Any]:
    answerable = [
        row
        for row in rows
        if not row["should_abstain"]
    ]

    unanswerable = [
        row
        for row in rows
        if row["should_abstain"]
    ]

    evidence_rows = [
        row
        for row in rows
        if row["has_gold_evidence"]
    ]

    return {
        "method": method,
        "question_type": question_type,
        "num_questions": len(rows),
        "num_answerable": len(answerable),
        "num_unanswerable": len(unanswerable),
        "num_evidence_scored": len(
            evidence_rows
        ),
        "answer_f1_answerable": (
            format_optional(mean([
                row["answer_token_f1"]
                for row in answerable
            ]))
        ),
        "abstention_accuracy": (
            format_optional(mean([
                float(
                    row[
                        "abstention_correct"
                    ]
                )
                for row in rows
            ]))
        ),
        "unanswerable_abstain_recall": (
            format_optional(mean([
                float(
                    row[
                        "predicted_abstain"
                    ]
                )
                for row in unanswerable
            ]))
        ),
        "answerable_response_rate": (
            format_optional(mean([
                float(
                    not row[
                        "predicted_abstain"
                    ]
                )
                for row in answerable
            ]))
        ),
        "strict_evidence_precision": (
            format_optional(mean([
                row[
                    "strict_evidence_precision"
                ]
                for row in evidence_rows
            ]))
        ),
        "strict_evidence_recall": (
            format_optional(mean([
                row[
                    "strict_evidence_recall"
                ]
                for row in evidence_rows
            ]))
        ),
        "strict_evidence_f1": (
            format_optional(mean([
                row["strict_evidence_f1"]
                for row in evidence_rows
            ]))
        ),
        "source_evidence_precision": (
            format_optional(mean([
                row[
                    "source_evidence_precision"
                ]
                for row in evidence_rows
            ]))
        ),
        "source_evidence_recall": (
            format_optional(mean([
                row[
                    "source_evidence_recall"
                ]
                for row in evidence_rows
            ]))
        ),
        "source_evidence_f1": (
            format_optional(mean([
                row["source_evidence_f1"]
                for row in evidence_rows
            ]))
        ),
        "mean_selected_evidence": (
            format_optional(mean([
                float(
                    row[
                        "selected_evidence_count"
                    ]
                )
                for row in rows
            ]))
        ),
        "mean_projected_sources": (
            format_optional(mean([
                float(
                    row[
                        "projected_source_count"
                    ]
                )
                for row in rows
            ]))
        ),
    }


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                rows[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(rows)


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
        "--output-dir",
        required=True,
        type=Path,
    )

    args = parser.parse_args()

    predictions = load_jsonl(
        args.predictions
    )

    questions = {
        row["question_id"]: row
        for row in load_csv(
            args.questions
        )
    }

    answer_scores = {
        (
            row["method"],
            row["question_id"],
        ): float(
            row["answer_token_f1"]
        )
        for row in load_csv(
            args.answer_scores
        )
    }

    memory_data = json.loads(
        args.memories.read_text(
            encoding="utf-8"
        )
    )

    memory_map = {
        memory["memory_id"]: memory
        for memory in memory_data[
            "memories"
        ]
    }

    per_question: list[
        dict[str, Any]
    ] = []

    methods = sorted({
        row["method"]
        for row in predictions
    })

    for prediction in predictions:
        method = str(
            prediction["method"]
        )

        question_id = str(
            prediction["question_id"]
        )

        question = questions[
            question_id
        ]

        selected_ids = parse_ids(
            prediction.get(
                "selected_ids"
            )
        )

        gold_ids = set(
            parse_ids(
                question[
                    "supporting_memory_ids"
                ]
            )
        )

        selected_strict = set(
            selected_ids
        )

        selected_sources = (
            project_to_sources(
                selected_ids,
                memory_map,
            )
        )

        if gold_ids:
            (
                strict_precision,
                strict_recall,
                strict_f1,
            ) = precision_recall_f1(
                selected_strict,
                gold_ids,
            )

            (
                source_precision,
                source_recall,
                source_f1,
            ) = precision_recall_f1(
                selected_sources,
                gold_ids,
            )
        else:
            strict_precision = 0.0
            strict_recall = 0.0
            strict_f1 = 0.0
            source_precision = 0.0
            source_recall = 0.0
            source_f1 = 0.0

        should_abstain = as_bool(
            question["should_abstain"]
        )

        predicted_abstain = (
            prediction.get("decision")
            == "abstain"
        )

        per_question.append({
            "question_id": question_id,
            "method": method,
            "user_id": question[
                "user_id"
            ],
            "question_type": question[
                "question_type"
            ],
            "should_abstain": (
                should_abstain
            ),
            "predicted_abstain": (
                predicted_abstain
            ),
            "abstention_correct": (
                should_abstain
                == predicted_abstain
            ),
            "answer_token_f1": (
                answer_scores.get(
                    (
                        method,
                        question_id,
                    ),
                    0.0,
                )
            ),
            "has_gold_evidence": bool(
                gold_ids
            ),
            "gold_evidence_count": len(
                gold_ids
            ),
            "selected_evidence_count": (
                len(selected_ids)
            ),
            "projected_source_count": (
                len(selected_sources)
            ),
            "strict_evidence_precision": (
                strict_precision
            ),
            "strict_evidence_recall": (
                strict_recall
            ),
            "strict_evidence_f1": (
                strict_f1
            ),
            "source_evidence_precision": (
                source_precision
            ),
            "source_evidence_recall": (
                source_recall
            ),
            "source_evidence_f1": (
                source_f1
            ),
            "decision": prediction.get(
                "decision"
            ),
            "route_types": json.dumps(
                prediction.get(
                    "selected_memory_types",
                    [],
                ),
                ensure_ascii=False,
            ),
            "gold_ids": json.dumps(
                sorted(gold_ids),
                ensure_ascii=False,
            ),
            "selected_ids": json.dumps(
                selected_ids,
                ensure_ascii=False,
            ),
            "projected_sources": (
                json.dumps(
                    sorted(
                        selected_sources
                    ),
                    ensure_ascii=False,
                )
            ),
        })

    summary_rows = []

    for method in methods:
        method_rows = [
            row
            for row in per_question
            if row["method"] == method
        ]

        summary_rows.append(
            summarise(
                method_rows,
                method=method,
                question_type="all",
            )
        )

    category_rows = []

    categories = sorted({
        row["question_type"]
        for row in per_question
    })

    for method in methods:
        for category in categories:
            rows = [
                row
                for row in per_question
                if (
                    row["method"]
                    == method
                    and row[
                        "question_type"
                    ]
                    == category
                )
            ]

            if not rows:
                continue

            category_rows.append(
                summarise(
                    rows,
                    method=method,
                    question_type=category,
                )
            )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_csv(
        args.output_dir
        / "locomo_external_per_question.csv",
        per_question,
    )

    write_csv(
        args.output_dir
        / "locomo_external_summary.csv",
        summary_rows,
    )

    write_csv(
        args.output_dir
        / "locomo_external_by_category.csv",
        category_rows,
    )

    manifest = {
        "predictions": str(
            args.predictions
        ),
        "questions": str(
            args.questions
        ),
        "memories": str(
            args.memories
        ),
        "answer_scores": str(
            args.answer_scores
        ),
        "strict_evidence_definition": (
            "Exact selected memory IDs "
            "against gold episodic IDs."
        ),
        "source_equivalent_definition": (
            "Episodic memories map to "
            "themselves; semantic memories "
            "map to their official LoCoMo "
            "dialogue source_ids."
        ),
        "evidence_aggregation": (
            "Macro average over questions "
            "with non-empty gold evidence."
        ),
        "answer_aggregation": (
            "Macro Token F1 over questions "
            "with non-empty gold answers."
        ),
    }

    (
        args.output_dir
        / "locomo_external_manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        args.output_dir
        / "locomo_external_summary.csv"
    )

    print(
        args.output_dir
        / "locomo_external_by_category.csv"
    )

    print(
        args.output_dir
        / "locomo_external_per_question.csv"
    )


if __name__ == "__main__":
    main()
