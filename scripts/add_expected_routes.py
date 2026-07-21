from __future__ import annotations

import ast
import csv
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = (
    ROOT.parent
    / "Dataset_A_v0_1"
    / "c3_compatible"
)

SUMMARY_PATH = (
    ROOT.parent
    / "Dataset_A_v0_1"
    / "data_eval"
    / "eval_questions_summary.csv"
)

TARGET_FILES = [
    DATASET_DIR / "eval_questions_c3.csv",
    DATASET_DIR / "eval_questions_pilot_20.csv",
]

BACKUP_DIR = DATASET_DIR / "_backup_before_route_labels"


def parse_required_memory(raw_value: str) -> list[str]:
    """Convert values such as ['episodic', 'semantic'] into clean labels."""

    text = (raw_value or "").strip()

    if not text:
        return []

    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        parsed = [
            part.strip()
            for part in text.replace(",", ";").split(";")
            if part.strip()
        ]

    if isinstance(parsed, str):
        parsed = [parsed]

    if not isinstance(parsed, (list, tuple, set)):
        return []

    normalised: list[str] = []

    for item in parsed:
        value = str(item).strip().lower()

        mapping = {
            "neo4j_episodic_graph": "episodic",
            "episodic_memory": "episodic",
            "episodic": "episodic",

            "neo4j_semantic_graph": "semantic",
            "semantic_memory": "semantic",
            "semantic": "semantic",

            "procedural_json": "procedural",
            "procedural_memory": "procedural",
            "procedural": "procedural",
        }

        mapped = mapping.get(value)

        if mapped and mapped not in normalised:
            normalised.append(mapped)

    return normalised


def load_expected_routes() -> dict[str, list[str]]:
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"Summary file not found: {SUMMARY_PATH}"
        )

    mapping: dict[str, list[str]] = {}

    with SUMMARY_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)

        required_columns = {
            "question_id",
            "required_memory",
        }

        missing = required_columns - set(
            reader.fieldnames or []
        )

        if missing:
            raise ValueError(
                "eval_questions_summary.csv is missing columns: "
                f"{sorted(missing)}"
            )

        for row in reader:
            question_id = (
                row.get("question_id") or ""
            ).strip()

            expected = parse_required_memory(
                row.get("required_memory") or ""
            )

            if question_id:
                mapping[question_id] = expected

    return mapping


def update_csv(
    path: Path,
    expected_routes: dict[str, list[str]],
) -> tuple[int, list[str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Evaluation file not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        original_fields = list(
            reader.fieldnames or []
        )

    if "question_id" not in original_fields:
        raise ValueError(
            f"{path.name} has no question_id column."
        )

    missing_question_ids: list[str] = []

    for row in rows:
        question_id = (
            row.get("question_id") or ""
        ).strip()

        expected = expected_routes.get(question_id)

        if expected is None:
            missing_question_ids.append(question_id)
            row["expected_memory_types"] = ""
        else:
            row["expected_memory_types"] = ";".join(
                expected
            )

    output_fields = list(original_fields)

    if "expected_memory_types" not in output_fields:
        output_fields.append(
            "expected_memory_types"
        )

    BACKUP_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    backup_path = BACKUP_DIR / path.name

    if not backup_path.exists():
        shutil.copy2(path, backup_path)

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=output_fields,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)

    return len(rows), missing_question_ids


def main() -> None:
    expected_routes = load_expected_routes()

    print(
        f"Loaded expected routes for "
        f"{len(expected_routes)} questions."
    )

    total_missing: list[str] = []

    for target_path in TARGET_FILES:
        row_count, missing_ids = update_csv(
            target_path,
            expected_routes,
        )

        print()
        print(f"Updated: {target_path}")
        print(f"Rows:    {row_count}")
        print(
            f"Missing expected routes: "
            f"{len(missing_ids)}"
        )

        total_missing.extend(missing_ids)

    print()
    print(
        f"Backups saved to: {BACKUP_DIR}"
    )

    if total_missing:
        print()
        print("Question IDs missing from summary:")
        for question_id in sorted(
            set(total_missing)
        ):
            print(f"- {question_id}")
    else:
        print()
        print(
            "All evaluation questions received "
            "per-question expected route labels."
        )


if __name__ == "__main__":
    main()