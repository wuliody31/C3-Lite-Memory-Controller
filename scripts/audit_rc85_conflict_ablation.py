from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def load_predictions(
    path: Path,
) -> dict[str, dict[str, Any]]:
    output = {}

    for line in path.read_text(
        encoding="utf-8"
    ).splitlines():
        if not line.strip():
            continue

        row = json.loads(line)

        if row.get("method") != "c3":
            continue

        output[str(row["question_id"])] = row

    return output


def evidence_f1(row: dict[str, Any]) -> float:
    value = row.get("evidence_f1")

    if value is not None:
        return float(value)

    metrics = row.get("metrics", {})

    if isinstance(metrics, dict):
        value = metrics.get("evidence_f1")

    return float(value or 0.0)


def conflict_trace(
    row: dict[str, Any],
) -> list[dict[str, Any]]:
    groups = row.get("conflict_groups", [])

    if not isinstance(groups, list):
        return []

    return [
        group
        for group in groups
        if isinstance(group, dict)
    ]


def conflict_debug(
    row: dict[str, Any],
) -> dict[str, Any]:
    debug = row.get("debug", {})

    if not isinstance(debug, dict):
        return {}

    return {
        key: value
        for key, value in debug.items()
        if "conflict" in key.lower()
        or "resolution" in key.lower()
    }


def compact_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--full-predictions",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--no-conflict-predictions",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
    )

    args = parser.parse_args()

    full = load_predictions(
        args.full_predictions
    )
    ablated = load_predictions(
        args.no_conflict_predictions
    )

    question_ids = [
        question_id
        for question_id, row in full.items()
        if row.get("question_type")
        == "conflict_resolution"
    ]

    output_rows = []

    detected_count = 0
    selected_changed_count = 0
    answer_changed_count = 0
    decision_changed_count = 0

    for question_id in sorted(question_ids):
        full_row = full[question_id]
        ablated_row = ablated[question_id]

        full_groups = conflict_trace(full_row)
        ablated_groups = conflict_trace(
            ablated_row
        )

        full_selected = list(
            full_row.get("selected_ids", [])
        )
        ablated_selected = list(
            ablated_row.get("selected_ids", [])
        )

        selected_changed = (
            full_selected != ablated_selected
        )
        answer_changed = (
            str(full_row.get("answer", "")).strip()
            != str(
                ablated_row.get("answer", "")
            ).strip()
        )
        decision_changed = (
            full_row.get("decision")
            != ablated_row.get("decision")
        )

        detected_count += int(
            bool(full_groups)
        )
        selected_changed_count += int(
            selected_changed
        )
        answer_changed_count += int(
            answer_changed
        )
        decision_changed_count += int(
            decision_changed
        )

        output_rows.append(
            {
                "question_id": question_id,
                "question": full_row.get(
                    "question",
                    full_row.get("query", ""),
                ),
                "full_conflict_count": len(
                    full_groups
                ),
                "ablated_conflict_count": len(
                    ablated_groups
                ),
                "selected_changed": (
                    selected_changed
                ),
                "answer_changed": answer_changed,
                "decision_changed": (
                    decision_changed
                ),
                "full_decision": full_row.get(
                    "decision"
                ),
                "ablated_decision": (
                    ablated_row.get("decision")
                ),
                "full_evidence_f1": evidence_f1(
                    full_row
                ),
                "ablated_evidence_f1": (
                    evidence_f1(ablated_row)
                ),
                "full_selected_ids": (
                    compact_json(full_selected)
                ),
                "ablated_selected_ids": (
                    compact_json(
                        ablated_selected
                    )
                ),
                "full_conflict_groups": (
                    compact_json(full_groups)
                ),
                "full_conflict_debug": (
                    compact_json(
                        conflict_debug(full_row)
                    )
                ),
            }
        )

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

    print("=" * 72)
    print("RC8.6 CONFLICT ABLATION AUDIT")
    print("=" * 72)
    print("Conflict-labelled questions:", len(question_ids))
    print("Questions with detected conflicts:", detected_count)
    print("Questions with changed selected evidence:", selected_changed_count)
    print("Questions with changed answers:", answer_changed_count)
    print("Questions with changed decisions:", decision_changed_count)
    print("Saved:", args.output)

    print()
    for row in output_rows:
        print(
            row["question_id"],
            "| detected =",
            row["full_conflict_count"],
            "| selected changed =",
            row["selected_changed"],
            "| answer changed =",
            row["answer_changed"],
            "| evidence F1 =",
            f"{row['full_evidence_f1']:.4f}",
            "→",
            f"{row['ablated_evidence_f1']:.4f}",
        )


if __name__ == "__main__":
    main()
