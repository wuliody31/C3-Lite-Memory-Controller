from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

EVAL_PATH = (
    ROOT.parent
    / "Dataset_A_v0_1"
    / "c3_compatible"
    / "eval_questions_c3.csv"
)

OUTPUT_PATH = (
    ROOT.parent
    / "Dataset_A_v0_1"
    / "c3_compatible"
    / "route_label_audit.csv"
)

TYPE_ORDER = [
    "episodic",
    "semantic",
    "procedural",
]


def parse_semicolon_list(raw_value: str) -> list[str]:
    return [
        item.strip()
        for item in (raw_value or "").split(";")
        if item.strip()
    ]


def normalise_types(values: list[str]) -> list[str]:
    cleaned = {
        value.strip().lower()
        for value in values
        if value.strip()
    }

    return [
        memory_type
        for memory_type in TYPE_ORDER
        if memory_type in cleaned
    ]


def infer_types_from_gold_ids(
    supporting_memory_ids: str,
) -> list[str]:
    inferred: set[str] = set()

    for memory_id in parse_semicolon_list(
        supporting_memory_ids
    ):
        lowered = memory_id.lower()

        if lowered.startswith("e_"):
            inferred.add("episodic")

        elif lowered.startswith("s_"):
            inferred.add("semantic")

        elif lowered.startswith("p_"):
            inferred.add("procedural")

        else:
            print(
                "Warning: unknown memory ID prefix: "
                f"{memory_id}"
            )

    return [
        memory_type
        for memory_type in TYPE_ORDER
        if memory_type in inferred
    ]


def parse_bool(raw_value: str) -> bool:
    return str(raw_value).strip().lower() in {
        "true",
        "1",
        "yes",
        "y",
    }


def main() -> None:
    if not EVAL_PATH.exists():
        raise FileNotFoundError(
            f"Evaluation file not found: {EVAL_PATH}"
        )

    with EVAL_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    audit_rows: list[dict[str, str]] = []
    mismatch_count = 0
    empty_gold_count = 0

    for row in rows:
        question_id = (
            row.get("question_id") or ""
        ).strip()

        question_type = (
            row.get("question_type") or ""
        ).strip()

        should_abstain = parse_bool(
            row.get("should_abstain") or ""
        )

        supporting_ids = (
            row.get("supporting_memory_ids") or ""
        ).strip()

        summary_types = normalise_types(
            parse_semicolon_list(
                row.get("expected_memory_types") or ""
            )
        )

        gold_types = infer_types_from_gold_ids(
            supporting_ids
        )

        has_gold_evidence = bool(
            parse_semicolon_list(supporting_ids)
        )

        if not has_gold_evidence:
            empty_gold_count += 1

        labels_match = summary_types == gold_types

        if not labels_match:
            mismatch_count += 1

        if not has_gold_evidence and should_abstain:
            recommended_policy = (
                "exclude_route_metric"
            )
        elif has_gold_evidence:
            recommended_policy = (
                "use_gold_derived_types"
            )
        else:
            recommended_policy = (
                "manual_review"
            )

        audit_rows.append(
            {
                "question_id": question_id,
                "question_type": question_type,
                "should_abstain": str(
                    should_abstain
                ),
                "supporting_memory_ids": supporting_ids,
                "summary_expected_types": ";".join(
                    summary_types
                ),
                "gold_derived_types": ";".join(
                    gold_types
                ),
                "labels_match": str(labels_match),
                "recommended_policy": (
                    recommended_policy
                ),
            }
        )

    fieldnames = [
        "question_id",
        "question_type",
        "should_abstain",
        "supporting_memory_ids",
        "summary_expected_types",
        "gold_derived_types",
        "labels_match",
        "recommended_policy",
    ]

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(audit_rows)

    print(f"Questions audited:       {len(rows)}")
    print(f"Label mismatches:        {mismatch_count}")
    print(f"Questions with no gold:  {empty_gold_count}")
    print(f"Audit report:            {OUTPUT_PATH}")

    print()
    print("Mismatched route labels:")
    print()

    for row in audit_rows:
        if row["labels_match"] == "False":
            print(
                f"{row['question_id']}: "
                f"summary=[{row['summary_expected_types']}] "
                f"gold=[{row['gold_derived_types']}] "
                f"abstain={row['should_abstain']}"
            )


if __name__ == "__main__":
    main()