from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise TypeError(f"{path}:{line_no} is not a JSON object")
            rows.append(obj)
    return rows


def as_ids(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value if str(x)]
    if isinstance(value, tuple):
        return [str(x) for x in value if str(x)]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if ";" in text:
            return [x.strip() for x in text.split(";") if x.strip()]
        if "," in text:
            return [x.strip() for x in text.split(",") if x.strip()]
        return [text]
    return [str(value)]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze C3-v3 strict information-need shadow instrumentation "
            "on Dataset A Full60."
        )
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    args = parser.parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = [
        row
        for row in read_jsonl(args.predictions)
        if str(row.get("method", "")).lower() == "c3"
    ]

    if not rows:
        raise AssertionError("No C3 predictions found.")

    per_question: list[dict[str, Any]] = []
    removed_gold_cases: list[dict[str, Any]] = []
    positive_gain_cases: list[dict[str, Any]] = []

    removed_total = 0
    removed_gold_total = 0
    removed_gold_positive_gain = 0
    removed_gold_zero_gain = 0
    removed_gold_missing_audit = 0

    coverage_loss_questions = 0
    coverage_gain_questions = 0
    coverage_same_questions = 0

    positive_gain_question_ids: set[str] = set()
    removed_gold_question_ids: set[str] = set()

    query_mode_counts: Counter[str] = Counter()
    positive_gain_by_mode: Counter[str] = Counter()

    for row in rows:
        question_id = str(row.get("question_id", ""))
        question_type = str(row.get("question_type", ""))
        query_mode = str(row.get("query_mode", ""))
        query = str(row.get("query", ""))

        shadow = (
            (row.get("debug") or {})
            .get("c3_v3_arbitration_shadow")
        )

        if not isinstance(shadow, dict):
            raise AssertionError(
                f"{question_id}: missing c3_v3_arbitration_shadow"
            )

        required_fields = [
            "information_needs",
            "legacy_information_need_coverage",
            "c3_v3_information_need_coverage",
            "information_need_coverage_delta",
            "removed_information_need_gain",
        ]
        missing = [
            field
            for field in required_fields
            if field not in shadow
        ]
        if missing:
            raise AssertionError(
                f"{question_id}: missing information-need shadow fields: {missing}"
            )

        gold_ids = set(
            as_ids(
                row.get("supporting_memory_ids")
            )
        )
        removed_ids = as_ids(
            shadow.get("removed_by_c3_v3")
        )

        removed_total += len(removed_ids)

        removed_gold_ids = [
            memory_id
            for memory_id in removed_ids
            if memory_id in gold_ids
        ]

        removed_gold_total += len(
            removed_gold_ids
        )

        if removed_gold_ids:
            removed_gold_question_ids.add(
                question_id
            )

        need_gain_map = (
            shadow.get(
                "removed_information_need_gain"
            )
            or {}
        )

        legacy_cov = float(
            shadow.get(
                "legacy_information_need_coverage",
                0.0,
            )
            or 0.0
        )
        c3_cov = float(
            shadow.get(
                "c3_v3_information_need_coverage",
                0.0,
            )
            or 0.0
        )
        cov_delta = float(
            shadow.get(
                "information_need_coverage_delta",
                c3_cov - legacy_cov,
            )
            or 0.0
        )

        if cov_delta < 0:
            coverage_loss_questions += 1
        elif cov_delta > 0:
            coverage_gain_questions += 1
        else:
            coverage_same_questions += 1

        query_mode_counts[
            query_mode
        ] += len(
            removed_gold_ids
        )

        question_positive_gold = 0
        question_zero_gold = 0
        question_missing_gold = 0

        for memory_id in removed_gold_ids:
            audit = (
                need_gain_map.get(memory_id)
                if isinstance(
                    need_gain_map,
                    dict,
                )
                else None
            )

            if not isinstance(audit, dict):
                gain = None
                gained_needs: list[str] = []
                question_missing_gold += 1
                removed_gold_missing_audit += 1
                classification = "missing_audit"
            else:
                gain = float(
                    audit.get("gain", 0.0)
                    or 0.0
                )
                gained_needs = [
                    str(x)
                    for x in (
                        audit.get(
                            "gained_needs"
                        )
                        or []
                    )
                ]

                if gain > 0.0:
                    question_positive_gold += 1
                    removed_gold_positive_gain += 1
                    positive_gain_question_ids.add(
                        question_id
                    )
                    positive_gain_by_mode[
                        query_mode
                    ] += 1
                    classification = (
                        "positive_information_need_gain"
                    )
                else:
                    question_zero_gold += 1
                    removed_gold_zero_gain += 1
                    classification = (
                        "zero_information_need_gain"
                    )

            case = {
                "question_id": question_id,
                "question_type": question_type,
                "query_mode": query_mode,
                "query": query,
                "memory_id": memory_id,
                "classification": classification,
                "gain": (
                    "" if gain is None else gain
                ),
                "gained_needs": ";".join(
                    gained_needs
                ),
                "information_needs": ";".join(
                    str(x)
                    for x in (
                        shadow.get(
                            "information_needs"
                        )
                        or []
                    )
                ),
                "legacy_information_need_coverage": (
                    legacy_cov
                ),
                "c3_v3_information_need_coverage": (
                    c3_cov
                ),
                "information_need_coverage_delta": (
                    cov_delta
                ),
                "legacy_selected_ids": ";".join(
                    as_ids(
                        shadow.get(
                            "legacy_selected_ids"
                        )
                    )
                ),
                "c3_v3_selected_ids": ";".join(
                    as_ids(
                        shadow.get(
                            "c3_v3_selected_ids"
                        )
                    )
                ),
                "removed_by_c3_v3": ";".join(
                    removed_ids
                ),
                "hard_complete": bool(
                    shadow.get(
                        "hard_complete"
                    )
                ),
            }

            removed_gold_cases.append(
                case
            )

            if classification == (
                "positive_information_need_gain"
            ):
                positive_gain_cases.append(
                    case
                )

        per_question.append(
            {
                "question_id": question_id,
                "question_type": question_type,
                "query_mode": query_mode,
                "query": query,
                "legacy_information_need_coverage": (
                    legacy_cov
                ),
                "c3_v3_information_need_coverage": (
                    c3_cov
                ),
                "information_need_coverage_delta": (
                    cov_delta
                ),
                "removed_count": len(
                    removed_ids
                ),
                "removed_gold_count": len(
                    removed_gold_ids
                ),
                "removed_gold_positive_gain_count": (
                    question_positive_gold
                ),
                "removed_gold_zero_gain_count": (
                    question_zero_gold
                ),
                "removed_gold_missing_audit_count": (
                    question_missing_gold
                ),
                "hard_complete": bool(
                    shadow.get(
                        "hard_complete"
                    )
                ),
            }
        )

    legacy_coverages = [
        float(
            row[
                "legacy_information_need_coverage"
            ]
        )
        for row in per_question
    ]
    c3_coverages = [
        float(
            row[
                "c3_v3_information_need_coverage"
            ]
        )
        for row in per_question
    ]

    summary = {
        "questions": len(rows),
        "mean_legacy_information_need_coverage": (
            mean(
                legacy_coverages
            )
        ),
        "mean_c3_v3_information_need_coverage": (
            mean(
                c3_coverages
            )
        ),
        "mean_information_need_coverage_delta": (
            mean(
                [
                    c3 - legacy
                    for legacy, c3
                    in zip(
                        legacy_coverages,
                        c3_coverages,
                    )
                ]
            )
        ),
        "coverage_outcomes": {
            "loss_questions": (
                coverage_loss_questions
            ),
            "same_questions": (
                coverage_same_questions
            ),
            "gain_questions": (
                coverage_gain_questions
            ),
        },
        "removed_evidence": {
            "removed_total": (
                removed_total
            ),
            "removed_gold_total": (
                removed_gold_total
            ),
            "questions_with_removed_gold": len(
                removed_gold_question_ids
            ),
        },
        "removed_gold_information_need_gain": {
            "positive_gain": (
                removed_gold_positive_gain
            ),
            "zero_gain": (
                removed_gold_zero_gain
            ),
            "missing_audit": (
                removed_gold_missing_audit
            ),
            "positive_gain_rate_among_removed_gold": (
                removed_gold_positive_gain
                / removed_gold_total
                if removed_gold_total
                else 0.0
            ),
            "questions_with_positive_gain_removed_gold": len(
                positive_gain_question_ids
            ),
        },
        "removed_gold_by_query_mode": (
            dict(
                query_mode_counts
            )
        ),
        "positive_gain_removed_gold_by_query_mode": (
            dict(
                positive_gain_by_mode
            )
        ),
        "decision_support": {
            "supports_information_need_aware_arbitration": (
                removed_gold_positive_gain > 0
            ),
            "note": (
                "A high positive-gain rate means current arbitration "
                "is discarding evidence that contributes distinct "
                "query-conditioned semantic information. Zero-gain "
                "removed gold requires separate treatment such as "
                "cardinality, provenance, or benchmark-semantic analysis."
            ),
        },
    }

    (
        args.output_dir
        / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    write_csv(
        args.output_dir
        / "per_question_information_need.csv",
        per_question,
    )
    write_csv(
        args.output_dir
        / "removed_gold_information_need_cases.csv",
        removed_gold_cases,
    )
    write_csv(
        args.output_dir
        / "positive_gain_removed_gold_cases.csv",
        positive_gain_cases,
    )

    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
    )
    print()
    print(
        "Saved:",
        args.output_dir
        / "summary.json",
    )
    print(
        "Saved:",
        args.output_dir
        / "removed_gold_information_need_cases.csv",
    )
    print(
        "Saved:",
        args.output_dir
        / "positive_gain_removed_gold_cases.csv",
    )


if __name__ == "__main__":
    main()
