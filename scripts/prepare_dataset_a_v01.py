from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Iterable


REQUIRED_EVAL_COLUMNS = {
    "question_id",
    "user_id",
    "question_type",
    "question",
}

OPTIONAL_EVAL_COLUMNS = {
    "supporting_memory_ids",
    "should_abstain",
    "expected_outdated_memory_ids",
    "conflict_type",
    "gold_answer",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert Dataset_A_v0_1 into the JSON/CSV formats expected by "
            "C3-Lite run_experiment.py."
        )
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Path to Dataset_A_v0_1.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Output directory. Defaults to "
            "<dataset-root>/c3_compatible."
        ),
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=20,
        help=(
            "Number of questions in the deterministic stratified pilot CSV. "
            "Use 0 to skip pilot creation."
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Exit with an error when gold supporting IDs are not present in "
            "the converted memories/procedures."
        ),
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(
                    f"Each JSONL record in {path} must be an object; "
                    f"line {line_number} is {type(value).__name__}."
                )
            records.append(value)
    return records


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = [
            {
                str(key): (value or "")
                for key, value in row.items()
            }
            for row in reader
        ]
    return fieldnames, rows


def write_csv(
    path: Path,
    fieldnames: list[str],
    rows: Iterable[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def readable_identifier(value: Any) -> str:
    text = clean_text(value)
    return text.replace("_", " ")


def normalise_importance(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.5

    if number > 1.0:
        number = number / 5.0

    return max(0.0, min(1.0, number))


def normalise_confidence(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 1.0
    return max(0.0, min(1.0, number))


def normalise_status(value: Any) -> str:
    status = clean_text(value).lower()
    mapping = {
        "": "current",
        "active": "current",
        "valid": "current",
        "current": "current",
        "inactive": "outdated",
        "deprecated": "outdated",
        "obsolete": "outdated",
        "old": "outdated",
        "superseded": "superseded",
        "archived": "archived",
        "invalid": "invalid",
        "outdated": "outdated",
    }
    return mapping.get(status, status or "current")


def split_ids(value: Any) -> list[str]:
    if isinstance(value, list):
        return [
            clean_text(item)
            for item in value
            if clean_text(item)
        ]

    text = clean_text(value)
    if not text:
        return []

    return [
        part.strip()
        for part in text.replace(",", ";").split(";")
        if part.strip()
    ]


def convert_episodic(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []

    for index, record in enumerate(records, start=1):
        memory_id = clean_text(
            record.get("memory_id") or record.get("id")
        )
        if not memory_id:
            raise ValueError(
                f"Episodic record {index} has no memory_id/id."
            )

        text = clean_text(
            record.get("event")
            or record.get("text")
            or record.get("description")
        )
        if not text:
            raise ValueError(
                f"Episodic record {memory_id} has no event/text."
            )

        source_session = clean_text(
            record.get("source_session")
            or record.get("session_id")
            or record.get("source")
        )

        converted.append(
            {
                "memory_id": memory_id,
                "memory_type": "episodic",
                "text": text,
                "user_id": clean_text(record.get("user_id")),
                "timestamp": clean_text(
                    record.get("timestamp")
                    or record.get("date")
                )
                or None,
                "status": normalise_status(
                    record.get("status", "current")
                ),
                "confidence": normalise_confidence(
                    record.get("confidence", 1.0)
                ),
                "importance": normalise_importance(
                    record.get("importance", 0.5)
                ),
                "authority": clean_text(
                    record.get("authority", "user_confirmed")
                )
                or "user_confirmed",
                "source_ids": (
                    [source_session] if source_session else []
                ),
                "relations": list(record.get("relations") or []),
                "metadata": {
                    "entities": list(record.get("entities") or []),
                    "source_session": source_session or None,
                    "original_record": record,
                },
            }
        )

    return converted


def convert_semantic(
    rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []

    for index, row in enumerate(rows, start=1):
        memory_id = clean_text(
            row.get("triple_id")
            or row.get("memory_id")
            or row.get("id")
        )
        if not memory_id:
            raise ValueError(
                f"Semantic row {index} has no triple_id/memory_id/id."
            )

        subject_raw = clean_text(row.get("subject"))
        predicate_raw = clean_text(
            row.get("relation") or row.get("predicate")
        )
        object_raw = clean_text(
            row.get("object") or row.get("object_value")
        )

        if not subject_raw or not predicate_raw or not object_raw:
            raise ValueError(
                f"Semantic row {memory_id} must contain "
                "subject, relation/predicate and object."
            )

        subject_text = readable_identifier(subject_raw)
        predicate_text = readable_identifier(predicate_raw)
        object_text = readable_identifier(object_raw)

        source = clean_text(
            row.get("source")
            or row.get("source_session")
            or row.get("source_id")
        )

        converted.append(
            {
                "memory_id": memory_id,
                "memory_type": "semantic",
                "text": (
                    f"{subject_text} {predicate_text} is {object_text}."
                ),
                "user_id": clean_text(row.get("user_id")),
                "timestamp": clean_text(
                    row.get("last_updated")
                    or row.get("timestamp")
                    or row.get("date")
                )
                or None,
                "subject": subject_raw,
                "predicate": predicate_raw,
                "object": object_raw,
                "status": normalise_status(row.get("status")),
                "confidence": normalise_confidence(
                    row.get("confidence", 1.0)
                ),
                "importance": normalise_importance(
                    row.get("importance", 0.7)
                ),
                "authority": clean_text(
                    row.get("authority", "user_confirmed")
                )
                or "user_confirmed",
                "source_ids": [source] if source else [],
                "relations": [],
                "metadata": {
                    "source": source or None,
                    "original_subject": subject_raw,
                    "original_relation": predicate_raw,
                    "original_object": object_raw,
                    "original_record": row,
                },
            }
        )

    return converted



def first_non_empty_text(
    record: dict[str, Any],
    keys: list[str],
) -> str:
    """Return the first usable string/list value from candidate keys."""
    for key in keys:
        if key not in record:
            continue

        value = record.get(key)

        if isinstance(value, str):
            cleaned = clean_text(value)
            if cleaned:
                return cleaned

        elif isinstance(value, list):
            parts = [
                clean_text(item)
                for item in value
                if clean_text(item)
            ]
            if parts:
                return " ".join(parts)

        elif isinstance(value, dict):
            # Common nested representations:
            # {"text": "..."}, {"instruction": "..."}, {"action": "..."}
            nested = first_non_empty_text(
                value,
                [
                    "instruction",
                    "text",
                    "rule",
                    "content",
                    "description",
                    "action",
                    "guideline",
                    "policy",
                    "procedure",
                    "directive",
                    "value",
                ],
            )
            if nested:
                return nested

    return ""


def to_string_list(value: Any) -> list[str]:
    """Normalise list/string/dict trigger and task fields."""
    if value is None:
        return []

    if isinstance(value, list):
        return [
            clean_text(item)
            for item in value
            if clean_text(item)
        ]

    if isinstance(value, dict):
        output: list[str] = []
        for key, item in value.items():
            if isinstance(item, bool):
                if item:
                    output.append(clean_text(key))
            elif isinstance(item, list):
                output.extend(to_string_list(item))
            elif clean_text(item):
                output.append(clean_text(item))
        return list(dict.fromkeys(item for item in output if item))

    return split_ids(value)


def load_and_normalise_procedures(
    path: Path,
) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        data = json.load(handle)

    if isinstance(data, dict):
        rules = (
            data.get("procedures")
            or data.get("rules")
            or data.get("procedural_memories")
            or data.get("items")
            or data.get("data")
            or []
        )
    else:
        rules = data

    if not isinstance(rules, list):
        raise ValueError(
            "procedural_memory.json must contain a list, or an object "
            "with procedures/rules/procedural_memories/items/data."
        )

    output: list[dict[str, Any]] = []

    instruction_keys = [
        "instruction",
        "text",
        "rule",
        "rule_text",
        "content",
        "description",
        "guideline",
        "policy",
        "procedure",
        "directive",
        "action",
        "response_rule",
        "response_policy",
        "recommended_action",
        "steps",
    ]

    for index, rule in enumerate(rules, start=1):
        if not isinstance(rule, dict):
            raise ValueError(
                f"Procedure {index} must be a JSON object."
            )

        rule_id = clean_text(
            rule.get("rule_id")
            or rule.get("memory_id")
            or rule.get("procedure_id")
            or rule.get("policy_id")
            or rule.get("id")
        )

        instruction = first_non_empty_text(
            rule,
            instruction_keys,
        )

        if not rule_id:
            raise ValueError(
                f"Procedure {index} has no "
                "rule_id/memory_id/procedure_id/policy_id/id. "
                f"Available keys: {sorted(rule.keys())}"
            )

        if not instruction:
            raise ValueError(
                f"Procedure {rule_id} has no recognised instruction field. "
                f"Checked: {instruction_keys}. "
                f"Available keys: {sorted(rule.keys())}"
            )

        raw_task_types = (
            rule.get("task_types")
            or rule.get("tasks")
            or rule.get("applicable_tasks")
            or []
        )
        task_types_clean = to_string_list(raw_task_types)

        task_type = clean_text(
            rule.get("task_type")
            or rule.get("task")
            or rule.get("category")
            or rule.get("rule_type")
            or rule.get("procedure_type")
        )

        if task_type and task_type not in task_types_clean:
            task_types_clean.append(task_type)

        raw_triggers = (
            rule.get("triggers")
            or rule.get("trigger")
            or rule.get("keywords")
            or rule.get("trigger_keywords")
            or rule.get("conditions")
            or rule.get("when")
            or []
        )
        triggers = to_string_list(raw_triggers)

        priority_raw = (
            rule.get("priority")
            if rule.get("priority") is not None
            else rule.get("importance", 5)
        )
        try:
            priority = float(priority_raw)
        except (TypeError, ValueError):
            priority = 5.0

        normalised = dict(rule)
        normalised.update(
            {
                "rule_id": rule_id,
                "instruction": instruction,
                "user_id": clean_text(
                    rule.get("user_id")
                    or rule.get("owner_id")
                    or "*"
                )
                or "*",
                "task_type": task_type or (
                    task_types_clean[0]
                    if task_types_clean
                    else None
                ),
                "task_types": task_types_clean,
                "triggers": triggers,
                "scope": clean_text(
                    rule.get("scope")
                    or rule.get("domain")
                    or "project"
                )
                or "project",
                "priority": priority,
                "enabled": bool(rule.get("enabled", True)),
                "confidence": normalise_confidence(
                    rule.get("confidence", 1.0)
                ),
                "authority": clean_text(
                    rule.get("authority")
                    or rule.get("source_authority")
                    or "user_confirmed"
                )
                or "user_confirmed",
                "metadata": {
                    "original_record": rule,
                    "instruction_source_field": next(
                        (
                            key
                            for key in instruction_keys
                            if key in rule
                            and first_non_empty_text(rule, [key])
                        ),
                        None,
                    ),
                },
            }
        )
        output.append(normalised)

    return output

def validate_eval(
    fieldnames: list[str],
    rows: list[dict[str, str]],
) -> None:
    missing = REQUIRED_EVAL_COLUMNS - set(fieldnames)
    if missing:
        raise ValueError(
            f"Evaluation CSV is missing required columns: "
            f"{sorted(missing)}"
        )

    seen_ids: set[str] = set()
    duplicates: list[str] = []

    for row_number, row in enumerate(rows, start=2):
        question_id = clean_text(row.get("question_id"))
        user_id = clean_text(row.get("user_id"))
        question = clean_text(row.get("question"))

        if not question_id:
            raise ValueError(
                f"Evaluation CSV row {row_number} has no question_id."
            )
        if not user_id:
            raise ValueError(
                f"Evaluation CSV row {row_number} has no user_id."
            )
        if not question:
            raise ValueError(
                f"Evaluation CSV row {row_number} has no question."
            )

        if question_id in seen_ids:
            duplicates.append(question_id)
        seen_ids.add(question_id)

    if duplicates:
        raise ValueError(
            f"Duplicate question IDs: {sorted(set(duplicates))}"
        )


def make_stratified_sample(
    rows: list[dict[str, str]],
    sample_size: int,
) -> list[dict[str, str]]:
    if sample_size <= 0:
        return []

    if sample_size >= len(rows):
        return list(rows)

    grouped: dict[str, deque[dict[str, str]]] = defaultdict(deque)
    order: list[str] = []

    for row in rows:
        question_type = clean_text(
            row.get("question_type")
        ) or "unknown"
        if question_type not in grouped:
            order.append(question_type)
        grouped[question_type].append(row)

    selected: list[dict[str, str]] = []

    while len(selected) < sample_size:
        added = False
        for question_type in order:
            if grouped[question_type]:
                selected.append(grouped[question_type].popleft())
                added = True
                if len(selected) >= sample_size:
                    break
        if not added:
            break

    return selected


def main() -> None:
    args = parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else dataset_root / "c3_compatible"
    )

    eval_source = (
        dataset_root
        / "data_eval"
        / "eval_questions_labeled_conflict_review.csv"
    )
    episodic_source = (
        dataset_root
        / "data_raw"
        / "episodic_memory_source.jsonl"
    )
    semantic_source = (
        dataset_root
        / "data_raw"
        / "semantic_triples_source.csv"
    )
    procedural_source = (
        dataset_root
        / "data_raw"
        / "procedural_memory.json"
    )

    required_sources = [
        eval_source,
        episodic_source,
        semantic_source,
        procedural_source,
    ]
    missing_sources = [
        str(path)
        for path in required_sources
        if not path.exists()
    ]
    if missing_sources:
        raise FileNotFoundError(
            "Required source files are missing:\n- "
            + "\n- ".join(missing_sources)
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    eval_fields, eval_rows = read_csv_rows(eval_source)
    validate_eval(eval_fields, eval_rows)

    episodic_records = read_jsonl(episodic_source)
    semantic_fields, semantic_rows = read_csv_rows(
        semantic_source
    )
    del semantic_fields

    episodic_memories = convert_episodic(episodic_records)
    semantic_memories = convert_semantic(semantic_rows)
    procedures = load_and_normalise_procedures(
        procedural_source
    )

    all_memory_ids = {
        item["memory_id"]
        for item in [
            *episodic_memories,
            *semantic_memories,
        ]
    }
    all_procedure_ids = {
        item["rule_id"]
        for item in procedures
    }
    all_evidence_ids = all_memory_ids | all_procedure_ids

    missing_gold_by_question: dict[str, list[str]] = {}
    for row in eval_rows:
        gold_ids = split_ids(
            row.get("supporting_memory_ids", "")
        )
        missing = [
            memory_id
            for memory_id in gold_ids
            if memory_id not in all_evidence_ids
        ]
        if missing:
            missing_gold_by_question[
                clean_text(row.get("question_id"))
            ] = missing

    memories_path = output_dir / "memories_c3.json"
    procedures_path = output_dir / "procedures_c3.json"
    eval_path = output_dir / "eval_questions_c3.csv"
    pilot_path = output_dir / (
        f"eval_questions_pilot_{args.sample_size}.csv"
    )
    report_path = output_dir / "conversion_report.json"

    memories_path.write_text(
        json.dumps(
            {
                "memories": [
                    *episodic_memories,
                    *semantic_memories,
                ]
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    procedures_path.write_text(
        json.dumps(
            {"procedures": procedures},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    shutil.copyfile(eval_source, eval_path)

    pilot_rows = make_stratified_sample(
        eval_rows,
        args.sample_size,
    )
    if pilot_rows:
        write_csv(pilot_path, eval_fields, pilot_rows)

    report = {
        "dataset_root": str(dataset_root),
        "output_dir": str(output_dir),
        "sources": {
            "evaluation": str(eval_source),
            "episodic": str(episodic_source),
            "semantic": str(semantic_source),
            "procedural": str(procedural_source),
        },
        "outputs": {
            "evaluation": str(eval_path),
            "pilot_evaluation": (
                str(pilot_path) if pilot_rows else None
            ),
            "memories": str(memories_path),
            "procedures": str(procedures_path),
        },
        "counts": {
            "evaluation_questions": len(eval_rows),
            "pilot_questions": len(pilot_rows),
            "episodic_memories": len(episodic_memories),
            "semantic_memories": len(semantic_memories),
            "procedural_rules": len(procedures),
            "all_evidence_items": len(all_evidence_ids),
        },
        "question_types": dict(
            sorted(
                Counter(
                    clean_text(row.get("question_type"))
                    or "unknown"
                    for row in eval_rows
                ).items()
            )
        ),
        "pilot_question_types": dict(
            sorted(
                Counter(
                    clean_text(row.get("question_type"))
                    or "unknown"
                    for row in pilot_rows
                ).items()
            )
        ),
        "semantic_statuses": dict(
            sorted(
                Counter(
                    item["status"]
                    for item in semantic_memories
                ).items()
            )
        ),
        "missing_gold_supporting_ids": (
            missing_gold_by_question
        ),
        "missing_gold_question_count": len(
            missing_gold_by_question
        ),
    }

    report_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Dataset conversion completed.")
    print(f"Evaluation questions: {len(eval_rows)}")
    print(f"Pilot questions:      {len(pilot_rows)}")
    print(f"Episodic memories:    {len(episodic_memories)}")
    print(f"Semantic memories:    {len(semantic_memories)}")
    print(f"Procedural rules:     {len(procedures)}")
    print(
        "Gold-ID warnings:     "
        f"{len(missing_gold_by_question)} questions"
    )
    print()
    print(f"Evaluation CSV:  {eval_path}")
    if pilot_rows:
        print(f"Pilot CSV:       {pilot_path}")
    print(f"Memory JSON:     {memories_path}")
    print(f"Procedure JSON:  {procedures_path}")
    print(f"Report:          {report_path}")

    if args.strict and missing_gold_by_question:
        raise RuntimeError(
            "Strict validation failed because some gold supporting "
            "memory IDs are missing. See conversion_report.json."
        )


if __name__ == "__main__":
    main()