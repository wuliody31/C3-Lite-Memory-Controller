from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


MEMORY_TYPE_BY_PREFIX = {
    "e_": "episodic",
    "s_": "semantic",
    "p_": "procedural",
}

ROUTE_KEYS = {
    "selected_memory_types",
    "predicted_memory_types",
    "memory_types",
    "selected_types",
    "route_types",
    "required_memory_types",
}

TEXT_KEYS = (
    "text",
    "content",
    "memory",
    "description",
    "value",
    "statement",
    "rule",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose whether gold evidence missing from raw retrieval was caused "
            "by route omission or by retrieval failure after a correct route."
        )
    )
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--gold-diagnostics", required=True, type=Path)
    parser.add_argument("--memory-json", required=True, type=Path)
    parser.add_argument("--procedure-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL at {path}:{line_number}: {exc}"
                ) from exc
            if isinstance(value, dict):
                rows.append(value)
    return rows


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def infer_memory_type(memory_id: str) -> str:
    lowered = memory_id.strip().lower()
    for prefix, memory_type in MEMORY_TYPE_BY_PREFIX.items():
        if lowered.startswith(prefix):
            return memory_type
    return ""


def normalise_memory_types(value: Any) -> list[str]:
    valid = {"episodic", "semantic", "procedural"}
    result: list[str] = []

    def add(token: Any) -> None:
        if token is None:
            return
        if isinstance(token, str):
            parts = (
                token.replace("|", ";")
                .replace(",", ";")
                .replace("[", "")
                .replace("]", "")
                .replace("'", "")
                .replace('"', "")
                .split(";")
            )
            for part in parts:
                candidate = part.strip().lower()
                if candidate in valid and candidate not in result:
                    result.append(candidate)
            return
        if isinstance(token, (list, tuple, set)):
            for item in token:
                add(item)
            return
        if isinstance(token, dict):
            for item in token.values():
                add(item)

    add(value)
    return result


def recursive_values_for_keys(
    value: Any,
    keys: set[str],
) -> Iterable[Any]:
    if isinstance(value, dict):
        for key, nested in value.items():
            if key in keys:
                yield nested
            yield from recursive_values_for_keys(nested, keys)
    elif isinstance(value, list):
        for nested in value:
            yield from recursive_values_for_keys(nested, keys)


def extract_route_types(prediction: dict[str, Any]) -> list[str]:
    result: list[str] = []

    for value in recursive_values_for_keys(prediction, ROUTE_KEYS):
        for memory_type in normalise_memory_types(value):
            if memory_type not in result:
                result.append(memory_type)

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                key_lower = str(key).lower()
                if key_lower in {"episodic", "semantic", "procedural"}:
                    if bool(nested) and key_lower not in result:
                        result.append(key_lower)
                walk(nested)
        elif isinstance(value, list):
            for nested in value:
                walk(nested)

    if not result:
        walk(prediction)

    return result


def first_present(row: dict[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def extract_question_id(row: dict[str, Any]) -> str:
    return first_present(row, ("question_id", "id", "qid"))


def extract_gold_id(row: dict[str, Any]) -> str:
    return first_present(
        row,
        ("gold_id", "gold_memory_id", "memory_id", "lost_gold_id"),
    )


def collect_memory_records(
    value: Any,
    output: dict[str, dict[str, str]],
) -> None:
    if isinstance(value, dict):
        memory_id = first_present(
            value,
            ("memory_id", "id", "evidence_id", "procedure_id", "rule_id"),
        )
        if memory_id:
            text = first_present(value, TEXT_KEYS)
            memory_type = first_present(
                value,
                ("memory_type", "type", "kind"),
            ).lower()
            if memory_type not in {"episodic", "semantic", "procedural"}:
                memory_type = infer_memory_type(memory_id)
            output[memory_id] = {
                "text": text,
                "memory_type": memory_type,
            }
        for nested in value.values():
            collect_memory_records(nested, output)
    elif isinstance(value, list):
        for nested in value:
            collect_memory_records(nested, output)


def build_memory_index(paths: Iterable[Path]) -> dict[str, dict[str, str]]:
    output: dict[str, dict[str, str]] = {}
    for path in paths:
        collect_memory_records(load_json(path), output)
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    predictions = load_jsonl(args.predictions)
    dataset_rows = load_csv(args.dataset)
    diagnostic_rows = load_csv(args.gold_diagnostics)
    memory_index = build_memory_index(
        (args.memory_json, args.procedure_json)
    )

    prediction_by_id = {
        extract_question_id(row): row
        for row in predictions
        if extract_question_id(row)
    }
    dataset_by_id = {
        extract_question_id(row): row
        for row in dataset_rows
        if extract_question_id(row)
    }

    raw_missing = [
        row
        for row in diagnostic_rows
        if row.get("drop_stage", "").strip() == "not_in_raw_trace"
    ]

    output_rows: list[dict[str, Any]] = []
    cause_counts: Counter[str] = Counter()
    type_counts: Counter[str] = Counter()

    for row in raw_missing:
        question_id = extract_question_id(row)
        gold_id = extract_gold_id(row)
        prediction = prediction_by_id.get(question_id, {})
        dataset_row = dataset_by_id.get(question_id, {})
        memory_record = memory_index.get(gold_id, {})

        gold_type = (
            row.get("memory_type", "").strip().lower()
            or memory_record.get("memory_type", "")
            or infer_memory_type(gold_id)
        )
        route_types = extract_route_types(prediction)

        if not route_types:
            cause = "unknown_route_trace"
        elif gold_type not in route_types:
            cause = "route_failure"
        else:
            cause = "retrieval_failure_after_correct_route"

        cause_counts[cause] += 1
        type_counts[gold_type or "unknown"] += 1

        query = first_present(
            dataset_row,
            ("query", "question", "text"),
        ) or first_present(
            prediction,
            ("query", "question", "input"),
        ) or row.get("query", "").strip()

        question_type = first_present(
            dataset_row,
            ("question_type", "type", "task_type"),
        )
        query_mode = first_present(
            prediction,
            ("query_mode",),
        )
        if not query_mode:
            debug = prediction.get("debug", {})
            if isinstance(debug, dict):
                query_mode = first_present(debug, ("query_mode",))

        output_rows.append(
            {
                "question_id": question_id,
                "question_type": question_type,
                "query_mode": query_mode,
                "query": query,
                "gold_id": gold_id,
                "gold_memory_type": gold_type,
                "predicted_memory_types": ";".join(route_types),
                "route_included_gold_type": (
                    gold_type in route_types if route_types else ""
                ),
                "cause": cause,
                "gold_text": (
                    memory_record.get("text", "")
                    or row.get("gold_text", "").strip()
                ),
            }
        )

    csv_path = args.output_dir / "raw_missing_route_causality_fixed.csv"
    summary_path = args.output_dir / "raw_missing_route_causality_summary.json"
    write_csv(csv_path, output_rows)

    summary = {
        "raw_missing_gold_count": len(output_rows),
        "cause_counts": dict(cause_counts),
        "gold_memory_type_counts": dict(type_counts),
        "route_failure_count": cause_counts.get("route_failure", 0),
        "retrieval_failure_after_correct_route_count": cause_counts.get(
            "retrieval_failure_after_correct_route",
            0,
        ),
        "unknown_route_trace_count": cause_counts.get(
            "unknown_route_trace",
            0,
        ),
        "outputs": {
            "details_csv": str(csv_path),
        },
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("=" * 72)
    print("RC8.3 RAW-MISSING ROUTE CAUSALITY AUDIT")
    print("=" * 72)
    print(f"Raw-missing gold evidence: {len(output_rows)}")
    for cause, count in sorted(cause_counts.items()):
        print(f"{cause}: {count}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()