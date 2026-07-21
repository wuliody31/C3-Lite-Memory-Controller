from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .dataset_loader import load_evaluation_csv
from .metrics import aggregate, per_question_metrics
from src.schemas import QueryState


def run_benchmark(
    *,
    dataset_path: str,
    methods: list[str],
    c3_pipeline: Any,
    baseline_runner: Any,
    output_dir: str,
    config_snapshot: dict[str, Any],
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    questions = load_evaluation_csv(dataset_path)
    rows: list[dict[str, Any]] = []

    predictions_path = (
        output_path / "predictions.jsonl"
    )

    with predictions_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for method in methods:
            for question in questions:
                query_state = QueryState(
                    question.question,
                    question.user_id,
                    datetime.now(timezone.utc),
                )

                if method == "c3":
                    result = c3_pipeline.answer(
                        query_state
                    )
                else:
                    result = baseline_runner.answer(
                        method,
                        query_state,
                    )

                legacy_retrieved_ids = list(
                    result.retrieved_ids
                )

                raw_retrieved_ids = list(
                    getattr(
                        result,
                        "raw_retrieved_ids",
                        None,
                    )
                    or legacy_retrieved_ids
                )

                ranked_candidate_ids = list(
                    getattr(
                        result,
                        "ranked_candidate_ids",
                        None,
                    )
                    or legacy_retrieved_ids
                )

                metrics = per_question_metrics(
                    selected_memory_types=(
                        result.selected_memory_types
                    ),
                    expected_memory_types=(
                        question.expected_memory_types
                    ),
                    route_metric_applicable=(
                        question.route_metric_applicable
                    ),
                    used_ids=result.selected_ids,
                    gold_ids=(
                        question.supporting_memory_ids
                    ),
                    decision=result.decision.value,
                    should_abstain=(
                        question.should_abstain
                    ),
                    expected_outdated_ids=(
                        question.expected_outdated_memory_ids
                    ),
                    retrieved_ids=legacy_retrieved_ids,
                    raw_retrieved_ids=raw_retrieved_ids,
                    ranked_candidate_ids=(
                        ranked_candidate_ids
                    ),
                )

                rows.append(
                    {
                        "method": method,
                        "question_id": (
                            question.question_id
                        ),
                        "question_type": (
                            question.question_type
                        ),
                        "expected_memory_types": (
                            ";".join(
                                question.expected_memory_types
                            )
                        ),
                        "selected_memory_types": (
                            ";".join(
                                result.selected_memory_types
                            )
                        ),
                        "route_metric_applicable": (
                            question.route_metric_applicable
                        ),
                        "raw_retrieved_ids": (
                            ";".join(raw_retrieved_ids)
                        ),
                        "ranked_candidate_ids": (
                            ";".join(ranked_candidate_ids)
                        ),
                        "selected_ids": (
                            ";".join(result.selected_ids)
                        ),
                        **metrics,
                        "coverage": result.coverage,
                        "confidence_score": (
                            result.adequacy
                        ),
                        "latency_ms": result.latency_ms,
                    }
                )

                prediction = {
                    "method": method,
                    "question_id": (
                        question.question_id
                    ),
                    "question_type": (
                        question.question_type
                    ),
                    "gold_answer": (
                        question.gold_answer
                    ),
                    "supporting_memory_ids": (
                        question.supporting_memory_ids
                    ),
                    "should_abstain": (
                        question.should_abstain
                    ),
                    "expected_memory_types": (
                        question.expected_memory_types
                    ),
                    "route_metric_applicable": (
                        question.route_metric_applicable
                    ),
                    "summary_expected_memory_types": (
                        question.summary_expected_memory_types
                    ),
                    "route_label_source": (
                        question.route_label_source
                    ),
                    "route_label_note": (
                        question.route_label_note
                    ),
                    **result.to_dict(),
                    "metrics": metrics,
                }

                handle.write(
                    json.dumps(
                        prediction,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    _csv(
        output_path / "per_question_metrics.csv",
        rows,
    )

    by_method: dict[
        str,
        list[dict[str, float]],
    ] = defaultdict(list)

    by_type: dict[
        tuple[str, str],
        list[dict[str, float]],
    ] = defaultdict(list)

    for row in rows:
        numeric = {
            key: float(value)
            for key, value in row.items()
            if isinstance(value, (int, float))
            and not isinstance(value, bool)
        }

        by_method[row["method"]].append(numeric)
        by_type[
            (
                row["method"],
                row["question_type"],
            )
        ].append(numeric)

    method_summary = {
        method: _aggregate_group(group_rows)
        for method, group_rows in by_method.items()
    }

    (
        output_path / "summary.json"
    ).write_text(
        json.dumps(
            method_summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    type_summary = [
        {
            "method": method,
            "question_type": question_type,
            **_aggregate_group(group_rows),
        }
        for (
            method,
            question_type,
        ), group_rows in sorted(by_type.items())
    ]

    _csv(
        output_path
        / "summary_by_question_type.csv",
        type_summary,
    )

    (
        output_path / "experiment_config.json"
    ).write_text(
        json.dumps(
            config_snapshot,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _aggregate_group(
    rows: list[dict[str, float]],
) -> dict[str, float | int]:
    summary: dict[str, float | int] = aggregate(
        rows
    )

    summary["num_questions"] = len(rows)
    summary["num_route_scored"] = sum(
        1
        for row in rows
        if row.get("route_scored") == 1.0
    )

    return summary


def _csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fields = list(
        dict.fromkeys(
            key
            for row in rows
            for key in row
        )
    )

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
        )
        writer.writeheader()
        writer.writerows(rows)
