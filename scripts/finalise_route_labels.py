from __future__ import annotations

import csv
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT.parent / "Dataset_A_v0_1" / "c3_compatible"

TARGET_FILES = [
    DATASET_DIR / "eval_questions_c3.csv",
    DATASET_DIR / "eval_questions_pilot_20.csv",
]

BACKUP_DIR = DATASET_DIR / "_backup_before_route_finalisation"

TYPE_ORDER = ["episodic", "semantic", "procedural"]


def parse_bool(value: str) -> bool:
    return str(value or "").strip().lower() in {
        "true",
        "1",
        "yes",
        "y",
    }


def split_semicolon(value: str) -> list[str]:
    return [
        item.strip()
        for item in str(value or "").split(";")
        if item.strip()
    ]


def normalise_types(values: list[str]) -> list[str]:
    selected = {
        value.strip().lower()
        for value in values
        if value.strip()
    }
    return [
        memory_type
        for memory_type in TYPE_ORDER
        if memory_type in selected
    ]


def infer_types_from_gold_ids(value: str) -> list[str]:
    inferred: set[str] = set()

    for memory_id in split_semicolon(value):
        lowered = memory_id.lower()

        if lowered.startswith("e_"):
            inferred.add("episodic")
        elif lowered.startswith("s_"):
            inferred.add("semantic")
        elif lowered.startswith("p_"):
            inferred.add("procedural")
        else:
            raise ValueError(
                f"Unknown supporting-memory ID prefix: {memory_id}"
            )

    return [
        memory_type
        for memory_type in TYPE_ORDER
        if memory_type in inferred
    ]


def update_file(path: Path) -> dict[str, int]:
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    required = {
        "question_id",
        "supporting_memory_ids",
        "should_abstain",
        "expected_memory_types",
    }
    missing = required - set(fieldnames)
    if missing:
        raise ValueError(
            f"{path.name} is missing columns: {sorted(missing)}"
        )

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    backup_path = BACKUP_DIR / path.name
    if not backup_path.exists():
        shutil.copy2(path, backup_path)

    stats = {
        "rows": 0,
        "gold_derived": 0,
        "route_not_applicable": 0,
        "manual_review": 0,
        "summary_gold_mismatches": 0,
    }

    for row in rows:
        stats["rows"] += 1

        summary_types = normalise_types(
            split_semicolon(
                row.get("expected_memory_types", "")
            )
        )
        gold_ids = row.get("supporting_memory_ids", "")
        gold_types = infer_types_from_gold_ids(gold_ids)
        should_abstain = parse_bool(
            row.get("should_abstain", "")
        )

        row["summary_expected_memory_types"] = ";".join(
            summary_types
        )

        if gold_types:
            row["expected_memory_types"] = ";".join(gold_types)
            row["route_metric_applicable"] = "True"
            row["route_label_source"] = "gold_supporting_ids"

            if summary_types != gold_types:
                stats["summary_gold_mismatches"] += 1
                row["route_label_note"] = (
                    "Summary required_memory disagreed with the "
                    "memory types implied by supporting_memory_ids; "
                    "gold-derived types were used."
                )
            else:
                row["route_label_note"] = ""

            stats["gold_derived"] += 1

        elif should_abstain:
            row["expected_memory_types"] = ""
            row["route_metric_applicable"] = "False"
            row["route_label_source"] = (
                "not_applicable_no_gold_evidence"
            )
            row["route_label_note"] = (
                "The question should abstain and has no gold "
                "supporting evidence, so route quality is not scored."
            )
            stats["route_not_applicable"] += 1

        else:
            row["expected_memory_types"] = ""
            row["route_metric_applicable"] = "False"
            row["route_label_source"] = "manual_review_required"
            row["route_label_note"] = (
                "No gold supporting evidence is present although "
                "should_abstain is False."
            )
            stats["manual_review"] += 1

    extra_fields = [
        "summary_expected_memory_types",
        "route_metric_applicable",
        "route_label_source",
        "route_label_note",
    ]

    for field in extra_fields:
        if field not in fieldnames:
            fieldnames.append(field)

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)

    return stats


def main() -> None:
    totals = {
        "rows": 0,
        "gold_derived": 0,
        "route_not_applicable": 0,
        "manual_review": 0,
        "summary_gold_mismatches": 0,
    }

    for path in TARGET_FILES:
        stats = update_file(path)

        print()
        print(f"Updated: {path}")
        for key, value in stats.items():
            print(f"{key:26}: {value}")

        for key in totals:
            totals[key] += stats[key]

    print()
    print(f"Backups: {BACKUP_DIR}")

    if totals["manual_review"] > 0:
        raise RuntimeError(
            "At least one non-abstention question has no gold "
            "supporting evidence. Review the updated CSV."
        )

    print()
    print("Route labels finalised successfully.")


if __name__ == "__main__":
    main()
