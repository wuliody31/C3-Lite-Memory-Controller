from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path


VALID_MEMORY_TYPES = {
    "episodic",
    "semantic",
    "procedural",
}


@dataclass(slots=True)
class EvaluationQuestion:
    question_id: str
    user_id: str
    question_type: str
    question: str
    supporting_memory_ids: list[str]
    should_abstain: bool
    expected_outdated_memory_ids: list[str]
    conflict_type: str
    gold_answer: str

    # Per-question route labels used by route evaluation.
    # Defaults preserve compatibility with older CSV files and callers.
    expected_memory_types: list[str] = field(default_factory=list)
    route_metric_applicable: bool = True
    summary_expected_memory_types: list[str] = field(default_factory=list)
    route_label_source: str = ""
    route_label_note: str = ""


def _ids(value: str | None) -> list[str]:
    return [
        item.strip()
        for item in (value or "").split(";")
        if item.strip()
    ]


def _bool(value: str | bool | None, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value

    text = str(value or "").strip().lower()

    if not text:
        return default

    if text in {"true", "1", "yes", "y"}:
        return True

    if text in {"false", "0", "no", "n"}:
        return False

    raise ValueError(f"Invalid boolean value in evaluation CSV: {value!r}")


def _memory_types(value: str | None) -> list[str]:
    memory_types = [
        item.lower()
        for item in _ids(value)
    ]

    unknown = [
        item
        for item in memory_types
        if item not in VALID_MEMORY_TYPES
    ]

    if unknown:
        raise ValueError(
            "Unknown expected memory type(s): "
            f"{sorted(set(unknown))}. "
            f"Allowed values: {sorted(VALID_MEMORY_TYPES)}"
        )

    # Preserve canonical order and remove duplicates.
    return [
        memory_type
        for memory_type in (
            "episodic",
            "semantic",
            "procedural",
        )
        if memory_type in memory_types
    ]


def load_evaluation_csv(
    path: str | Path,
) -> list[EvaluationQuestion]:
    output: list[EvaluationQuestion] = []

    with Path(path).open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)

        required = {
            "question_id",
            "user_id",
            "question_type",
            "question",
        }

        missing = required - set(reader.fieldnames or [])

        if missing:
            raise ValueError(
                "Dataset is missing columns: "
                f"{sorted(missing)}"
            )

        for row_number, row in enumerate(reader, start=2):
            try:
                output.append(
                    EvaluationQuestion(
                        question_id=(
                            row.get("question_id") or ""
                        ).strip(),
                        user_id=(
                            row.get("user_id") or ""
                        ).strip(),
                        question_type=(
                            row.get("question_type") or ""
                        ).strip(),
                        question=(
                            row.get("question") or ""
                        ).strip(),
                        supporting_memory_ids=_ids(
                            row.get("supporting_memory_ids")
                        ),
                        should_abstain=_bool(
                            row.get("should_abstain"),
                            default=False,
                        ),
                        expected_outdated_memory_ids=_ids(
                            row.get(
                                "expected_outdated_memory_ids"
                            )
                        ),
                        conflict_type=(
                            row.get("conflict_type", "none")
                            or "none"
                        ).strip(),
                        gold_answer=(
                            row.get("gold_answer", "")
                            or ""
                        ).strip(),
                        expected_memory_types=_memory_types(
                            row.get("expected_memory_types")
                        ),
                        route_metric_applicable=_bool(
                            row.get(
                                "route_metric_applicable"
                            ),
                            default=True,
                        ),
                        summary_expected_memory_types=(
                            _memory_types(
                                row.get(
                                    "summary_expected_memory_types"
                                )
                            )
                        ),
                        route_label_source=(
                            row.get("route_label_source", "")
                            or ""
                        ).strip(),
                        route_label_note=(
                            row.get("route_label_note", "")
                            or ""
                        ).strip(),
                    )
                )
            except ValueError as exc:
                raise ValueError(
                    f"{Path(path).name}, row {row_number}: {exc}"
                ) from exc

    return output
