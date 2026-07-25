from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate RC8 evidence-requirement plans, completion and selected roles."
        )
    )
    parser.add_argument(
        "--predictions",
        required=True,
        help="Path to RC8 predictions.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for requirement audit outputs",
    )
    return parser.parse_args()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL line {line_number}: {exc}"
                ) from exc
            if isinstance(row, dict):
                output.append(row)
    return output


def question_id(row: dict[str, Any]) -> str:
    for key in ("question_id", "query_id", "id"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    debug = row.get("debug")
    if isinstance(debug, dict):
        for key in ("question_id", "query_id", "id"):
            value = debug.get(key)
            if value not in (None, ""):
                return str(value)
    return ""


def stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
        )
    return str(value)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    predictions = read_jsonl(args.predictions)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    question_rows: list[dict[str, Any]] = []
    role_required: Counter[str] = Counter()
    role_satisfied: Counter[str] = Counter()
    role_complete: Counter[str] = Counter()
    role_questions: Counter[str] = Counter()
    selected_role_counts: Counter[str] = Counter()
    selector_reason_counts: Counter[str] = Counter()

    for row in predictions:
        debug = row.get("debug")
        if not isinstance(debug, dict):
            debug = {}

        plan = debug.get("evidence_requirement_plan")
        if not isinstance(plan, dict):
            plan = {}
        status = debug.get("evidence_requirement_status")
        if not isinstance(status, dict):
            status = {}

        incomplete_roles: list[str] = []
        for role, values in status.items():
            if not isinstance(values, dict):
                continue
            required = int(values.get("required", 0) or 0)
            satisfied = int(values.get("satisfied", 0) or 0)
            complete = bool(values.get("complete", False))

            role_required[role] += required
            role_satisfied[role] += satisfied
            role_questions[role] += 1
            if complete:
                role_complete[role] += 1
            else:
                incomplete_roles.append(role)

        full_trace = debug.get("full_candidate_score_trace")
        if not isinstance(full_trace, list):
            full_trace = []

        selected_roles: list[str] = []
        selected_reasons: list[str] = []
        for candidate in full_trace:
            if not isinstance(candidate, dict):
                continue
            if not bool(candidate.get("selected_final", False)):
                continue

            roles = candidate.get("evidence_roles", [])
            if isinstance(roles, str):
                roles = [roles]
            if isinstance(roles, list):
                for role in roles:
                    role_text = str(role)
                    selected_roles.append(role_text)
                    selected_role_counts[role_text] += 1

            reason = candidate.get("selector_reason")
            if reason not in (None, ""):
                reason_text = str(reason)
                selected_reasons.append(reason_text)
                selector_reason_counts[reason_text] += 1

        question_rows.append(
            {
                "question_id": question_id(row),
                "query": row.get("query", ""),
                "query_mode": row.get("query_mode", ""),
                "decision": row.get("decision", ""),
                "max_evidence": plan.get("max_evidence", ""),
                "explicit_cardinality": plan.get(
                    "explicit_cardinality",
                    "",
                ),
                "plan_reasons": stringify(
                    plan.get("reasons", [])
                ),
                "requirement_status": stringify(status),
                "all_requirements_complete": (
                    len(incomplete_roles) == 0
                    if status
                    else False
                ),
                "incomplete_roles": ";".join(incomplete_roles),
                "selected_ids": ";".join(
                    str(value)
                    for value in row.get("selected_ids", [])
                ),
                "selected_roles": ";".join(selected_roles),
                "selector_reasons": ";".join(selected_reasons),
            }
        )

    role_rows: list[dict[str, Any]] = []
    for role in sorted(role_questions):
        questions = role_questions[role]
        required = role_required[role]
        satisfied = role_satisfied[role]
        complete_questions = role_complete[role]
        role_rows.append(
            {
                "role": role,
                "questions": questions,
                "required_total": required,
                "satisfied_total": satisfied,
                "item_satisfaction_rate": (
                    satisfied / required
                    if required
                    else 0.0
                ),
                "complete_questions": complete_questions,
                "question_completion_rate": (
                    complete_questions / questions
                    if questions
                    else 0.0
                ),
                "selected_role_assignments": (
                    selected_role_counts[role]
                ),
            }
        )

    write_csv(
        output_dir / "requirement_question_audit.csv",
        question_rows,
    )
    write_csv(
        output_dir / "requirement_role_summary.csv",
        role_rows,
    )

    questions_with_status = sum(
        1
        for row in question_rows
        if row["requirement_status"] not in ("", "{}")
    )
    complete_questions = sum(
        1
        for row in question_rows
        if row["all_requirements_complete"] is True
    )

    report = {
        "num_questions": len(question_rows),
        "questions_with_requirement_status": questions_with_status,
        "complete_questions": complete_questions,
        "question_completion_rate": (
            complete_questions / questions_with_status
            if questions_with_status
            else 0.0
        ),
        "selector_reason_counts": dict(selector_reason_counts),
        "selected_role_counts": dict(selected_role_counts),
        "outputs": {
            "question_audit": str(
                output_dir / "requirement_question_audit.csv"
            ),
            "role_summary": str(
                output_dir / "requirement_role_summary.csv"
            ),
        },
    }

    with (
        output_dir / "requirement_report.json"
    ).open("w", encoding="utf-8") as handle:
        json.dump(
            report,
            handle,
            ensure_ascii=False,
            indent=2,
        )

    print("=" * 72)
    print("RC8 EVIDENCE REQUIREMENT AUDIT")
    print("=" * 72)
    print("Questions:", report["num_questions"])
    print(
        "Requirement completion:",
        f"{complete_questions}/{questions_with_status}",
        f"({report['question_completion_rate']:.3f})",
    )
    print(
        "Selector reasons:",
        report["selector_reason_counts"],
    )
    print(
        "Saved:",
        report["outputs"]["question_audit"],
    )
    print(
        "Saved:",
        report["outputs"]["role_summary"],
    )
    print(
        "Saved:",
        output_dir / "requirement_report.json",
    )


if __name__ == "__main__":
    main()
