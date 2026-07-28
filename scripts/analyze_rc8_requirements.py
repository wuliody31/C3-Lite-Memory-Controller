from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate RC8.2 hard/soft evidence requirements, feasibility, "
            "completion and selected roles."
        )
    )
    parser.add_argument(
        "--predictions",
        required=True,
        help="Path to RC8.2 predictions.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for requirement audit outputs",
    )
    return parser.parse_args()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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
                rows.append(row)
    return rows


def get_question_id(row: dict[str, Any]) -> str:
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
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
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

    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    predictions = read_jsonl(args.predictions)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    question_rows: list[dict[str, Any]] = []

    role_questions: Counter[str] = Counter()
    role_required: Counter[str] = Counter()
    role_satisfied: Counter[str] = Counter()
    role_eligible: Counter[str] = Counter()
    role_feasible_questions: Counter[str] = Counter()
    role_complete_questions: Counter[str] = Counter()
    role_feasible_complete_questions: Counter[str] = Counter()
    selected_role_counts: Counter[str] = Counter()
    selector_reason_counts: Counter[str] = Counter()

    hard_required_total = 0
    hard_satisfied_total = 0
    soft_required_total = 0
    soft_satisfied_total = 0
    feasible_required_total = 0
    feasible_satisfied_total = 0
    infeasible_requirement_count = 0

    hard_complete_questions = 0
    all_complete_questions = 0
    evaluable_questions = 0
    no_candidate_questions = 0

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

        hard_incomplete: list[str] = []
        soft_incomplete: list[str] = []
        infeasible_roles: list[str] = []
        feasible_incomplete: list[str] = []

        for role, values in status.items():
            if not isinstance(values, dict):
                continue

            required = int(values.get("required", 0) or 0)
            satisfied = int(values.get("satisfied", 0) or 0)
            hard = bool(values.get("hard", False))
            complete = bool(values.get("complete", False))
            eligible_count = int(values.get("eligible_count", 0) or 0)
            feasible = bool(values.get("feasible", False))

            role_questions[role] += 1
            role_required[role] += required
            role_satisfied[role] += satisfied
            role_eligible[role] += eligible_count

            if complete:
                role_complete_questions[role] += 1
            if feasible:
                role_feasible_questions[role] += 1
                feasible_required_total += required
                feasible_satisfied_total += min(satisfied, required)
                if complete:
                    role_feasible_complete_questions[role] += 1
                else:
                    feasible_incomplete.append(role)
            else:
                infeasible_requirement_count += 1
                infeasible_roles.append(role)

            if hard:
                hard_required_total += required
                hard_satisfied_total += min(satisfied, required)
                if not complete:
                    hard_incomplete.append(role)
            else:
                soft_required_total += required
                soft_satisfied_total += min(satisfied, required)
                if not complete:
                    soft_incomplete.append(role)

        full_trace = debug.get("full_candidate_score_trace")
        if not isinstance(full_trace, list):
            full_trace = []

        selected_roles: list[str] = []
        selected_reasons: list[str] = []
        resolved_candidate_count = 0
        eligible_candidate_count = 0

        for candidate in full_trace:
            if not isinstance(candidate, dict):
                continue
            if bool(candidate.get("survived_conflict_resolution", False)):
                resolved_candidate_count += 1

            eligible_roles = candidate.get("eligible_evidence_roles", [])
            if isinstance(eligible_roles, str):
                eligible_roles = [eligible_roles]
            if isinstance(eligible_roles, list) and eligible_roles:
                eligible_candidate_count += 1

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

        has_status = bool(status)
        has_resolved_candidates = resolved_candidate_count > 0
        if has_status:
            evaluable_questions += 1
        if not has_resolved_candidates:
            no_candidate_questions += 1

        hard_complete = has_status and not hard_incomplete
        all_complete = has_status and not hard_incomplete and not soft_incomplete
        if hard_complete:
            hard_complete_questions += 1
        if all_complete:
            all_complete_questions += 1

        if not has_resolved_candidates:
            evaluation_status = "not_evaluable_no_candidates"
        elif not has_status:
            evaluation_status = "not_evaluable_no_status"
        elif hard_incomplete:
            evaluation_status = "hard_incomplete"
        elif soft_incomplete:
            evaluation_status = "hard_complete_soft_incomplete"
        else:
            evaluation_status = "complete"

        question_rows.append(
            {
                "question_id": get_question_id(row),
                "query": row.get("query", ""),
                "query_mode": row.get("query_mode", ""),
                "decision": row.get("decision", ""),
                "evaluation_status": evaluation_status,
                "resolved_candidate_count": resolved_candidate_count,
                "eligible_candidate_count": eligible_candidate_count,
                "max_evidence": plan.get("max_evidence", ""),
                "explicit_cardinality": plan.get("explicit_cardinality", ""),
                "plan_reasons": stringify(plan.get("reasons", [])),
                "requirement_status": stringify(status),
                "all_hard_requirements_complete": hard_complete,
                "all_requirements_complete": all_complete,
                "hard_incomplete_roles": ";".join(hard_incomplete),
                "soft_incomplete_roles": ";".join(soft_incomplete),
                "infeasible_roles": ";".join(infeasible_roles),
                "feasible_incomplete_roles": ";".join(feasible_incomplete),
                "selected_ids": ";".join(
                    str(value) for value in row.get("selected_ids", [])
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
        feasible_questions = role_feasible_questions[role]
        role_rows.append(
            {
                "role": role,
                "questions": questions,
                "required_total": required,
                "satisfied_total": satisfied,
                "eligible_candidate_total": role_eligible[role],
                "item_satisfaction_rate": (
                    satisfied / required if required else 0.0
                ),
                "feasible_questions": feasible_questions,
                "feasibility_rate": (
                    feasible_questions / questions if questions else 0.0
                ),
                "complete_questions": role_complete_questions[role],
                "question_completion_rate": (
                    role_complete_questions[role] / questions
                    if questions
                    else 0.0
                ),
                "feasible_complete_questions": (
                    role_feasible_complete_questions[role]
                ),
                "feasible_completion_rate": (
                    role_feasible_complete_questions[role]
                    / feasible_questions
                    if feasible_questions
                    else 0.0
                ),
                "selected_role_assignments": selected_role_counts[role],
            }
        )

    question_path = output_dir / "requirement_question_audit.csv"
    role_path = output_dir / "requirement_role_summary.csv"
    write_csv(question_path, question_rows)
    write_csv(role_path, role_rows)

    report = {
        "num_questions": len(question_rows),
        "questions_with_requirement_status": evaluable_questions,
        "no_candidate_questions": no_candidate_questions,
        "all_hard_complete_questions": hard_complete_questions,
        "hard_question_completion_rate": (
            hard_complete_questions / evaluable_questions
            if evaluable_questions
            else 0.0
        ),
        "all_complete_questions": all_complete_questions,
        "all_question_completion_rate": (
            all_complete_questions / evaluable_questions
            if evaluable_questions
            else 0.0
        ),
        "hard_required_total": hard_required_total,
        "hard_satisfied_total": hard_satisfied_total,
        "hard_item_completion_rate": (
            hard_satisfied_total / hard_required_total
            if hard_required_total
            else 0.0
        ),
        "soft_required_total": soft_required_total,
        "soft_satisfied_total": soft_satisfied_total,
        "soft_item_completion_rate": (
            soft_satisfied_total / soft_required_total
            if soft_required_total
            else 0.0
        ),
        "feasible_required_total": feasible_required_total,
        "feasible_satisfied_total": feasible_satisfied_total,
        "feasible_requirement_completion_rate": (
            feasible_satisfied_total / feasible_required_total
            if feasible_required_total
            else 0.0
        ),
        "infeasible_requirement_count": infeasible_requirement_count,
        "selector_reason_counts": dict(selector_reason_counts),
        "selected_role_counts": dict(selected_role_counts),
        "outputs": {
            "question_audit": str(question_path),
            "role_summary": str(role_path),
        },
    }

    report_path = output_dir / "requirement_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    print("=" * 72)
    print("RC8.2 REQUIREMENT FEASIBILITY AUDIT")
    print("=" * 72)
    print("Questions:", report["num_questions"])
    print(
        "Hard question completion:",
        f"{hard_complete_questions}/{evaluable_questions}",
        f"({report['hard_question_completion_rate']:.3f})",
    )
    print(
        "Feasible requirement completion:",
        f"{feasible_satisfied_total}/{feasible_required_total}",
        f"({report['feasible_requirement_completion_rate']:.3f})",
    )
    print("No-candidate questions:", no_candidate_questions)
    print("Selector reasons:", report["selector_reason_counts"])
    print("Saved:", question_path)
    print("Saved:", role_path)
    print("Saved:", report_path)


if __name__ == "__main__":
    main()
