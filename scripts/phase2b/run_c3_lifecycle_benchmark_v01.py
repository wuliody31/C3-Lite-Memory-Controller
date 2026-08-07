#!/usr/bin/env python3
"""Run Controlled Lifecycle Benchmark v0.1 on C3 LifecycleService."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

from src.lifecycle import (
    InMemoryLifecycleStore,
    LifecycleService,
    MemoryStatus,
)
from src.schemas import MemoryType


ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CASES = (
    ROOT
    / "data"
    / "lifecycle_benchmark_v01"
    / "cases.jsonl"
)

DEFAULT_OUTPUT = (
    ROOT
    / "experiments"
    / "phase2b_b5_c3_lifecycle_v01"
)


def normalise(value: Any) -> str:
    return " ".join(
        str(value).strip().lower().split()
    )


def load_cases(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


def multiset_recall(
    actual: list[str],
    expected: list[str],
) -> float | None:
    if not expected:
        return None

    actual_counter = Counter(
        normalise(x)
        for x in actual
    )

    expected_counter = Counter(
        normalise(x)
        for x in expected
    )

    matched = sum(
        min(
            actual_counter[value],
            count,
        )
        for value, count
        in expected_counter.items()
    )

    return matched / sum(
        expected_counter.values()
    )


def record_dict(record) -> dict[str, Any]:
    return {
        "memory_id": record.memory_id,
        "memory_type": record.memory_type.value,
        "text": record.text,
        "status": record.status.value,
        "valid_from": record.valid_from.isoformat(),
        "valid_to": (
            record.valid_to.isoformat()
            if record.valid_to
            else None
        ),
        "state_key": record.state_key,
        "value": record.value,
        "source_turn_id": record.source_turn_id,
        "supersedes": record.supersedes,
        "superseded_by": record.superseded_by,
        "metadata": record.metadata,
    }


def run_case(
    case: dict[str, Any],
) -> dict[str, Any]:
    store = InMemoryLifecycleStore()
    service = LifecycleService(store)

    actual_operations: list[str] = []
    transitions: list[dict[str, Any]] = []

    for event in case["events"]:
        transition = service.ingest_state(
            user_id=case["user_id"],
            state_key=case["state_key"],
            value=event["value"],
            text=event["text"],
            observed_at=datetime.fromisoformat(
                event["observed_at"]
            ),
            source_turn_id=(
                f"{case['case_id']}:"
                f"{event['event_id']}"
            ),
            metadata={
                "benchmark": (
                    case["benchmark_version"]
                ),
                "case_id": case["case_id"],
                "pattern": case["pattern"],
                "surface_variant": (
                    case["surface_variant"]
                ),
                "ingestion_index": (
                    event["ingestion_index"]
                ),
            },
        )

        operation = transition.operation.value

        actual_operations.append(operation)

        transitions.append(
            {
                "operation": operation,
                "previous_memory_id": (
                    transition.previous_memory_id
                ),
                "current_memory_id": (
                    transition.current_memory_id
                ),
                "episodic_memory_id": (
                    transition.episodic_memory_id
                ),
                "reason": transition.reason,
            }
        )

    service.validate_invariants()

    current = service.current_state(
        user_id=case["user_id"],
        state_key=case["state_key"],
    )

    history = service.state_history(
        user_id=case["user_id"],
        state_key=case["state_key"],
    )

    semantic_records = [
        record
        for record in history
        if record.memory_type
        == MemoryType.SEMANTIC
    ]

    episodic_records = [
        record
        for record in history
        if record.memory_type
        == MemoryType.EPISODIC
    ]

    current_records = [
        record
        for record in semantic_records
        if record.status
        == MemoryStatus.CURRENT
    ]

    superseded_records = [
        record
        for record in semantic_records
        if record.status
        == MemoryStatus.SUPERSEDED
    ]

    current_values = [
        record.value
        for record in current_records
        if record.value is not None
    ]

    historical_values = [
        record.value
        for record in superseded_records
        if record.value is not None
    ]

    previous_values = (
        [historical_values[-1]]
        if historical_values
        else []
    )

    stale_norm = {
        normalise(value)
        for value in case["stale_values"]
    }

    current_norm = {
        normalise(value)
        for value in current_values
    }

    exposed_stale_values = [
        value
        for value in case["stale_values"]
        if normalise(value) in current_norm
    ]

    current_gold_present = (
        normalise(case["current_gold"])
        in current_norm
    )

    contradiction = bool(
        current_gold_present
        and any(
            value in current_norm
            for value in stale_norm
        )
    )

    previous_recall = multiset_recall(
        previous_values,
        case["historical_gold"],
    )

    history_recall = multiset_recall(
        historical_values,
        case["history_values_expected"],
    )

    expected_operations = (
        case["expected_operations"]
    )

    max_steps = max(
        len(expected_operations),
        len(actual_operations),
        1,
    )

    step_matches = sum(
        actual == expected
        for actual, expected in zip(
            actual_operations,
            expected_operations,
        )
    )

    result = {
        "case_id": case["case_id"],
        "state_key": case["state_key"],
        "pattern": case["pattern"],
        "surface_variant": (
            case["surface_variant"]
        ),
        "user_id": case["user_id"],
        "expected_operations": (
            expected_operations
        ),
        "actual_operations": (
            actual_operations
        ),
        "operation_exact_match": (
            actual_operations
            == expected_operations
        ),
        "operation_step_accuracy": (
            step_matches / max_steps
        ),
        "current_gold": (
            case["current_gold"]
        ),
        "current_values": current_values,
        "current_state_correct": (
            len(current_values) == 1
            and current_gold_present
        ),
        "stale_values": (
            case["stale_values"]
        ),
        "exposed_stale_values": (
            exposed_stale_values
        ),
        "stale_exposure": bool(
            exposed_stale_values
        ),
        "contradiction": contradiction,
        "previous_gold": (
            case["historical_gold"]
        ),
        "previous_values": (
            previous_values
        ),
        "previous_state_recall": (
            previous_recall
        ),
        "history_gold": (
            case["history_values_expected"]
        ),
        "history_values": (
            historical_values
        ),
        "history_retention_recall": (
            history_recall
        ),
        "single_current_invariant": (
            len(current_records) == 1
        ),
        "memory_counts": {
            "semantic": len(
                semantic_records
            ),
            "semantic_current": len(
                current_records
            ),
            "semantic_superseded": len(
                superseded_records
            ),
            "episodic_transition": len(
                episodic_records
            ),
            "total": len(history),
        },
        "transitions": transitions,
        "records": [
            record_dict(record)
            for record in history
        ],
    }

    return result


def average_defined(
    rows: list[dict[str, Any]],
    field: str,
) -> float | None:
    values = [
        float(row[field])
        for row in rows
        if row[field] is not None
    ]

    if not values:
        return None

    return mean(values)


def aggregate(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if not rows:
        raise ValueError(
            "Cannot aggregate empty rows."
        )

    stale_cases = [
        row
        for row in rows
        if row["stale_values"]
    ]

    stale_value_total = sum(
        len(row["stale_values"])
        for row in stale_cases
    )

    stale_value_exposed = sum(
        len(row["exposed_stale_values"])
        for row in stale_cases
    )

    operation_counts = Counter()

    for row in rows:
        operation_counts.update(
            row["actual_operations"]
        )

    return {
        "case_count": len(rows),
        "operation_exact_match_rate": mean(
            int(
                row[
                    "operation_exact_match"
                ]
            )
            for row in rows
        ),
        "operation_step_accuracy": mean(
            row[
                "operation_step_accuracy"
            ]
            for row in rows
        ),
        "current_state_accuracy": mean(
            int(
                row[
                    "current_state_correct"
                ]
            )
            for row in rows
        ),
        "single_current_invariant_rate": mean(
            int(
                row[
                    "single_current_invariant"
                ]
            )
            for row in rows
        ),
        "stale_memory_exposure_rate": (
            stale_value_exposed
            / stale_value_total
            if stale_value_total
            else 0.0
        ),
        "stale_memory_exposure_case_rate": (
            mean(
                int(row["stale_exposure"])
                for row in stale_cases
            )
            if stale_cases
            else 0.0
        ),
        "previous_state_recall": (
            average_defined(
                rows,
                "previous_state_recall",
            )
        ),
        "history_retention_recall": (
            average_defined(
                rows,
                "history_retention_recall",
            )
        ),
        "contradiction_rate": (
            mean(
                int(row["contradiction"])
                for row in stale_cases
            )
            if stale_cases
            else 0.0
        ),
        "mean_semantic_records": mean(
            row["memory_counts"][
                "semantic"
            ]
            for row in rows
        ),
        "mean_current_semantic_records": mean(
            row["memory_counts"][
                "semantic_current"
            ]
            for row in rows
        ),
        "mean_superseded_semantic_records": mean(
            row["memory_counts"][
                "semantic_superseded"
            ]
            for row in rows
        ),
        "mean_episodic_transition_records": mean(
            row["memory_counts"][
                "episodic_transition"
            ]
            for row in rows
        ),
        "mean_total_records": mean(
            row["memory_counts"]["total"]
            for row in rows
        ),
        "operation_counts": dict(
            sorted(operation_counts.items())
        ),
    }


def build_report(
    summary: dict[str, Any],
) -> str:
    overall = summary["overall"]

    lines = [
        "# C3 Lifecycle Benchmark v0.1",
        "",
        "## Scope",
        "",
        (
            "This run evaluates deterministic "
            "lifecycle-contract conformance. "
            "It does not compare C3 against "
            "Mem0 and does not measure LLM "
            "answer quality."
        ),
        "",
        "## Overall results",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        (
            "| Operation exact-match rate | "
            f"{overall['operation_exact_match_rate']:.4f} |"
        ),
        (
            "| Operation step accuracy | "
            f"{overall['operation_step_accuracy']:.4f} |"
        ),
        (
            "| Current-state accuracy | "
            f"{overall['current_state_accuracy']:.4f} |"
        ),
        (
            "| Single-current invariant rate | "
            f"{overall['single_current_invariant_rate']:.4f} |"
        ),
        (
            "| Stale-memory exposure rate | "
            f"{overall['stale_memory_exposure_rate']:.4f} |"
        ),
        (
            "| Stale-memory exposure case rate | "
            f"{overall['stale_memory_exposure_case_rate']:.4f} |"
        ),
        (
            "| Previous-state recall | "
            f"{overall['previous_state_recall']:.4f} |"
        ),
        (
            "| History-retention recall | "
            f"{overall['history_retention_recall']:.4f} |"
        ),
        (
            "| Contradiction rate | "
            f"{overall['contradiction_rate']:.4f} |"
        ),
        "",
        "## Memory growth",
        "",
        (
            "- Mean semantic records: "
            f"{overall['mean_semantic_records']:.2f}"
        ),
        (
            "- Mean current semantic records: "
            f"{overall['mean_current_semantic_records']:.2f}"
        ),
        (
            "- Mean superseded semantic records: "
            f"{overall['mean_superseded_semantic_records']:.2f}"
        ),
        (
            "- Mean episodic transition records: "
            f"{overall['mean_episodic_transition_records']:.2f}"
        ),
        (
            "- Mean total records: "
            f"{overall['mean_total_records']:.2f}"
        ),
        "",
        "## Operation counts",
        "",
    ]

    for operation, count in (
        overall["operation_counts"].items()
    ):
        lines.append(
            f"- {operation}: {count}"
        )

    lines += [
        "",
        "## Interpretation boundary",
        "",
        (
            "A perfect score here would show "
            "that the implemented lifecycle "
            "kernel conforms to the controlled "
            "benchmark contract. It would not "
            "establish superiority over an "
            "external memory system."
        ),
        "",
    ]

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cases",
        type=Path,
        default=DEFAULT_CASES,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    args = parser.parse_args()

    cases = load_cases(args.cases)

    results = [
        run_case(case)
        for case in cases
    ]

    patterns = sorted(
        {
            row["pattern"]
            for row in results
        }
    )

    state_keys = sorted(
        {
            row["state_key"]
            for row in results
        }
    )

    summary = {
        "experiment": (
            "phase2b_b5_c3_lifecycle_v01"
        ),
        "benchmark": (
            "lifecycle_benchmark_v01"
        ),
        "system": "c3_lifecycle",
        "execution_mode": (
            "deterministic_in_memory"
        ),
        "overall": aggregate(results),
        "by_pattern": {
            pattern: aggregate(
                [
                    row
                    for row in results
                    if row["pattern"]
                    == pattern
                ]
            )
            for pattern in patterns
        },
        "by_state_key": {
            state_key: aggregate(
                [
                    row
                    for row in results
                    if row["state_key"]
                    == state_key
                ]
            )
            for state_key in state_keys
        },
    }

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_path = (
        args.output_dir
        / "results.jsonl"
    )

    with results_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for row in results:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

    (
        args.output_dir
        / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    (
        args.output_dir
        / "report.md"
    ).write_text(
        build_report(summary),
        encoding="utf-8",
    )

    print(
        json.dumps(
            summary["overall"],
            ensure_ascii=False,
            indent=2,
        )
    )

    print(
        "Results:",
        results_path,
    )


if __name__ == "__main__":
    main()
