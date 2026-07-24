from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


TRUE_VALUES = {"1", "true", "yes", "y", "t"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyse gold evidence lost by EvidenceSelector and join each "
            "candidate with its memory content and selected competitors."
        )
    )
    parser.add_argument(
        "--candidate-scores",
        required=True,
        help="Path to all_candidate_scores.csv",
    )
    parser.add_argument(
        "--gold-diagnostics",
        required=True,
        help="Path to gold_candidate_diagnostics.csv",
    )
    parser.add_argument(
        "--predictions",
        required=True,
        help="Path to predictions.jsonl",
    )
    parser.add_argument(
        "--memory-json",
        required=True,
        help="Path to memories_c3.json",
    )
    parser.add_argument(
        "--procedure-json",
        default=None,
        help="Optional path to procedures_c3.json",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for selector-loss reports",
    )
    return parser.parse_args()


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        return list(csv.DictReader(handle))


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL at line {line_number}: {exc}"
                ) from exc
            if isinstance(value, dict):
                rows.append(value)
    return rows


def iter_dicts(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for nested in value.values():
            yield from iter_dicts(nested)
    elif isinstance(value, list):
        for item in value:
            yield from iter_dicts(item)


def memory_identifier(item: dict[str, Any]) -> str | None:
    for key in (
        "memory_id",
        "id",
        "episode_id",
        "semantic_id",
        "procedure_id",
    ):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def build_memory_map(*documents: Any) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for document in documents:
        if document is None:
            continue
        for item in iter_dicts(document):
            identifier = memory_identifier(item)
            if identifier:
                output.setdefault(identifier, item)
    return output


def first_nonempty(
    mapping: dict[str, Any],
    keys: Iterable[str],
    default: Any = "",
) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value not in (None, "", [], {}):
            return value
    return default


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


def memory_text(item: dict[str, Any] | None) -> str:
    if not item:
        return ""

    direct = first_nonempty(
        item,
        (
            "text",
            "content",
            "description",
            "statement",
            "fact",
            "event",
            "instruction",
            "procedure",
        ),
    )
    if direct:
        return stringify(direct)

    subject = stringify(
        first_nonempty(item, ("subject", "source"))
    )
    predicate = stringify(
        first_nonempty(
            item,
            ("predicate", "relation", "property"),
        )
    )
    object_value = stringify(
        first_nonempty(
            item,
            ("object_value", "object", "value", "target"),
        )
    )

    return " | ".join(
        part
        for part in (subject, predicate, object_value)
        if part
    )


def normalise_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in TRUE_VALUES


def question_id_from_prediction(
    row: dict[str, Any],
) -> str:
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


def prediction_summary(
    row: dict[str, Any],
) -> dict[str, Any]:
    debug = row.get("debug")
    if not isinstance(debug, dict):
        debug = {}

    return {
        "query": stringify(row.get("query", "")),
        "query_mode": stringify(row.get("query_mode", "")),
        "decision": stringify(row.get("decision", "")),
        "selected_ids": row.get(
            "selected_ids",
            debug.get("selected_ids", []),
        ),
        "selected_memory_types": row.get(
            "selected_memory_types",
            [],
        ),
        "information_needs": debug.get(
            "information_needs",
            [],
        ),
    }


def row_question_id(row: dict[str, str]) -> str:
    return (
        row.get("question_id")
        or row.get("query_id")
        or row.get("id")
        or ""
    )


def row_memory_id(row: dict[str, str]) -> str:
    return (
        row.get("memory_id")
        or row.get("gold_id")
        or row.get("evidence_id")
        or ""
    )


def numeric_text(value: Any) -> str:
    if value in (None, ""):
        return ""
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
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
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidate_rows = read_csv(args.candidate_scores)
    gold_rows = read_csv(args.gold_diagnostics)
    prediction_rows = read_jsonl(args.predictions)

    memory_documents = [read_json(args.memory_json)]
    if args.procedure_json:
        memory_documents.append(read_json(args.procedure_json))
    memories = build_memory_map(*memory_documents)

    predictions: dict[str, dict[str, Any]] = {}
    for row in prediction_rows:
        question_id = question_id_from_prediction(row)
        if question_id:
            predictions[question_id] = prediction_summary(row)

    candidates_by_question: dict[
        str,
        list[dict[str, str]],
    ] = defaultdict(list)
    candidate_lookup: dict[
        tuple[str, str],
        dict[str, str],
    ] = {}

    for row in candidate_rows:
        question_id = row_question_id(row)
        memory_id = row_memory_id(row)
        candidates_by_question[question_id].append(row)
        candidate_lookup[(question_id, memory_id)] = row

    selector_losses = [
        row
        for row in gold_rows
        if row.get("drop_stage") == "evidence_selector"
    ]

    details: list[dict[str, Any]] = []
    summary_groups: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(list)

    for gold in selector_losses:
        question_id = row_question_id(gold)
        gold_id = (
            gold.get("gold_id")
            or row_memory_id(gold)
        )
        candidate = candidate_lookup.get(
            (question_id, gold_id),
            gold,
        )
        prediction = predictions.get(
            question_id,
            {},
        )

        selected_rows = [
            row
            for row in candidates_by_question.get(
                question_id,
                [],
            )
            if normalise_bool(
                row.get("selected_final")
            )
        ]
        selected_rows.sort(
            key=lambda row: float(
                row.get(
                    "candidate_utility_v0",
                    row.get("final_score", 0.0),
                )
                or 0.0
            ),
            reverse=True,
        )

        selected_ids = [
            row_memory_id(row)
            for row in selected_rows
        ]
        selected_texts = [
            memory_text(
                memories.get(row_memory_id(row))
            )
            for row in selected_rows
        ]
        selected_scores = [
            numeric_text(
                row.get(
                    "candidate_utility_v0",
                    row.get("final_score"),
                )
            )
            for row in selected_rows
        ]

        memory_item = memories.get(gold_id, {})
        detail = {
            "question_id": question_id,
            "query": (
                prediction.get("query")
                or gold.get("query", "")
            ),
            "query_mode": prediction.get(
                "query_mode",
                "",
            ),
            "decision": prediction.get(
                "decision",
                "",
            ),
            "information_needs": stringify(
                prediction.get(
                    "information_needs",
                    [],
                )
            ),
            "lost_gold_id": gold_id,
            "memory_type": (
                candidate.get("memory_type")
                or gold.get("memory_type", "")
            ),
            "gold_text": memory_text(memory_item),
            "gold_status": stringify(
                first_nonempty(
                    memory_item,
                    ("status", "validity_status"),
                )
            ),
            "gold_timestamp": stringify(
                first_nonempty(
                    memory_item,
                    ("timestamp", "time", "date"),
                )
            ),
            "gold_subject": stringify(
                first_nonempty(
                    memory_item,
                    ("subject", "source"),
                )
            ),
            "gold_predicate": stringify(
                first_nonempty(
                    memory_item,
                    ("predicate", "relation", "property"),
                )
            ),
            "gold_object": stringify(
                first_nonempty(
                    memory_item,
                    ("object_value", "object", "value", "target"),
                )
            ),
            "gold_utility": numeric_text(
                candidate.get(
                    "candidate_utility_v0",
                    candidate.get("final_score"),
                )
            ),
            "gold_rank_after_gate": (
                candidate.get(
                    "rank_global_after_gate",
                    "",
                )
            ),
            "gold_type_rank_after_gate": (
                candidate.get(
                    "rank_within_type_after_gate",
                    "",
                )
            ),
            "budget_reason": candidate.get(
                "candidate_budget_reason",
                "",
            ),
            "selected_ids": ";".join(selected_ids),
            "selected_scores": ";".join(selected_scores),
            "selected_texts": " || ".join(selected_texts),
        }

        details.append(detail)
        summary_groups[question_id].append(detail)

    summaries: list[dict[str, Any]] = []
    for question_id, group in summary_groups.items():
        first = group[0]
        selected_ids = first["selected_ids"]
        lost_ids = [
            row["lost_gold_id"]
            for row in group
        ]
        types = Counter(
            row["memory_type"]
            for row in group
        )
        summaries.append(
            {
                "question_id": question_id,
                "query": first["query"],
                "query_mode": first["query_mode"],
                "decision": first["decision"],
                "information_needs": first["information_needs"],
                "num_gold_lost_by_selector": len(group),
                "lost_gold_ids": ";".join(lost_ids),
                "lost_memory_type_counts": json.dumps(
                    types,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "selected_ids": selected_ids,
            }
        )

    summaries.sort(
        key=lambda row: (
            -int(row["num_gold_lost_by_selector"]),
            row["question_id"],
        )
    )
    details.sort(
        key=lambda row: (
            row["question_id"],
            row["memory_type"],
            row["lost_gold_id"],
        )
    )

    write_csv(
        output_dir / "selector_loss_summary.csv",
        summaries,
    )
    write_csv(
        output_dir / "selector_loss_details.csv",
        details,
    )

    mode_counts = Counter(
        row["query_mode"] or "unknown"
        for row in details
    )
    question_loss_counts = Counter(
        row["question_id"]
        for row in details
    )

    report = {
        "num_questions_with_selector_loss": len(
            summary_groups
        ),
        "num_gold_lost_by_selector": len(details),
        "losses_by_query_mode": dict(mode_counts),
        "top_questions": question_loss_counts.most_common(),
        "outputs": {
            "summary_csv": str(
                output_dir / "selector_loss_summary.csv"
            ),
            "details_csv": str(
                output_dir / "selector_loss_details.csv"
            ),
        },
    }

    with (
        output_dir / "selector_loss_report.json"
    ).open("w", encoding="utf-8") as handle:
        json.dump(
            report,
            handle,
            ensure_ascii=False,
            indent=2,
        )

    print("=" * 72)
    print("RC8 SELECTOR LOSS ANALYSIS")
    print("=" * 72)
    print(
        "Questions with selector loss:",
        report["num_questions_with_selector_loss"],
    )
    print(
        "Gold lost by selector:",
        report["num_gold_lost_by_selector"],
    )
    print(
        "Losses by query mode:",
        report["losses_by_query_mode"],
    )
    print(
        "Saved:",
        report["outputs"]["summary_csv"],
    )
    print(
        "Saved:",
        report["outputs"]["details_csv"],
    )
    print(
        "Saved:",
        output_dir / "selector_loss_report.json",
    )


if __name__ == "__main__":
    main()
