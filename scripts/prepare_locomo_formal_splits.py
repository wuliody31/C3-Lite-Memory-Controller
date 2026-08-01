from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def slug(value: str) -> str:
    return re.sub(
        r"[^a-z0-9]+",
        "_",
        value.lower(),
    ).strip("_")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--questions",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--expected-splits",
        type=int,
        default=10,
    )

    args = parser.parse_args()

    with args.questions.open(
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)

    if not fieldnames:
        raise RuntimeError(
            "Question CSV has no header."
        )

    question_ids = [
        row["question_id"]
        for row in rows
    ]

    if len(question_ids) != len(
        set(question_ids)
    ):
        raise RuntimeError(
            "Duplicate question IDs in source."
        )

    grouped: dict[
        str,
        list[dict[str, str]],
    ] = defaultdict(list)

    for row in rows:
        grouped[row["user_id"]].append(
            row
        )

    if len(grouped) != args.expected_splits:
        raise RuntimeError(
            f"Expected {args.expected_splits} "
            f"conversations, found {len(grouped)}."
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for old_file in args.output_dir.glob(
        "split_*.csv"
    ):
        old_file.unlink()

    manifest_rows: list[
        dict[str, Any]
    ] = []

    written_ids: list[str] = []

    for split_index, user_id in enumerate(
        sorted(grouped)
    ):
        split_rows = sorted(
            grouped[user_id],
            key=lambda row: row[
                "question_id"
            ],
        )

        filename = (
            f"split_{split_index:02d}_"
            f"{slug(user_id)}.csv"
        )

        path = args.output_dir / filename

        with path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=fieldnames,
            )
            writer.writeheader()
            writer.writerows(split_rows)

        written_ids.extend(
            row["question_id"]
            for row in split_rows
        )

        manifest_rows.append({
            "split_index": split_index,
            "filename": filename,
            "user_id": user_id,
            "num_questions": len(
                split_rows
            ),
            "num_answerable": sum(
                row["should_abstain"].lower()
                == "false"
                for row in split_rows
            ),
            "num_unanswerable": sum(
                row["should_abstain"].lower()
                == "true"
                for row in split_rows
            ),
            "category_counts": dict(
                sorted(
                    Counter(
                        row["question_type"]
                        for row in split_rows
                    ).items()
                )
            ),
            "sha256": sha256_file(path),
        })

    if len(written_ids) != len(rows):
        raise RuntimeError(
            "Written row count does not "
            "match source."
        )

    if set(written_ids) != set(
        question_ids
    ):
        raise RuntimeError(
            "Split question IDs do not "
            "match source."
        )

    manifest = {
        "split_version": (
            "locomo_formal_splits_v01"
        ),
        "source_questions": str(
            args.questions
        ),
        "source_sha256": sha256_file(
            args.questions
        ),
        "num_splits": len(
            manifest_rows
        ),
        "total_questions": len(rows),
        "expected_predictions": (
            len(rows) * 4
        ),
        "methods": [
            "no_memory",
            "simple_retrieval",
            "all_memory",
            "c3",
        ],
        "splits": manifest_rows,
    }

    manifest_path = (
        args.output_dir
        / "splits_manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    checksum_paths = sorted(
        path
        for path in args.output_dir.iterdir()
        if (
            path.is_file()
            and path.name
            != "SHA256SUMS.txt"
        )
    )

    with (
        args.output_dir
        / "SHA256SUMS.txt"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        for path in checksum_paths:
            handle.write(
                f"{sha256_file(path)}  "
                f"{path.name}\n"
            )

    print("=" * 80)
    print("LOCOMO FORMAL SPLITS")
    print("=" * 80)
    print("Questions:", len(rows))
    print("Splits:", len(manifest_rows))
    print(
        "Expected predictions:",
        len(rows) * 4,
    )

    for row in manifest_rows:
        print(
            row["split_index"],
            row["user_id"],
            row["num_questions"],
        )

    print("Output:", args.output_dir)


if __name__ == "__main__":
    main()
