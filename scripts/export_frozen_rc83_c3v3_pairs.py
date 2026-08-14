from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(
    path: Path,
) -> list[dict[str, Any]]:
    rows = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line_number, raw in enumerate(
            handle,
            start=1,
        ):
            if not raw.strip():
                continue

            row = json.loads(raw)

            if not isinstance(row, dict):
                raise TypeError(
                    f"{path}:{line_number} "
                    "is not an object."
                )

            rows.append(row)

    return rows


def unique_by_question(
    rows: list[dict[str, Any]],
    label: str,
) -> dict[str, dict[str, Any]]:
    selected = [
        row
        for row in rows
        if str(
            row.get("method", "")
        ).lower()
        in {
            "c3",
            "c3_lite_controller",
        }
    ]

    output = {}

    for row in selected:
        qid = str(
            row["question_id"]
        )

        if qid in output:
            raise AssertionError(
                f"Duplicate {label} "
                f"question_id={qid}"
            )

        output[qid] = row

    return output


def list_ids(
    value: Any,
) -> list[str]:
    if value is None:
        return []

    if isinstance(value, list):
        return [
            str(item)
            for item in value
        ]

    if isinstance(value, tuple):
        return [
            str(item)
            for item in value
        ]

    if isinstance(value, str):
        text = value.strip()

        if not text:
            return []

        return [
            item.strip()
            for item in text.split(";")
            if item.strip()
        ]

    return [
        str(value)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--rc83",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--c3v3",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    rc83 = unique_by_question(
        read_jsonl(args.rc83),
        "RC8.3",
    )

    c3v3 = unique_by_question(
        read_jsonl(args.c3v3),
        "C3-v3",
    )

    if len(rc83) != 60:
        raise AssertionError(
            f"Expected 60 RC8.3 rows, "
            f"got {len(rc83)}."
        )

    if len(c3v3) != 60:
        raise AssertionError(
            f"Expected 60 C3-v3 rows, "
            f"got {len(c3v3)}."
        )

    if set(rc83) != set(c3v3):
        missing_rc83 = sorted(
            set(c3v3) - set(rc83)
        )

        missing_c3 = sorted(
            set(rc83) - set(c3v3)
        )

        raise AssertionError(
            "Question-ID mismatch. "
            f"Missing RC8.3={missing_rc83[:5]}, "
            f"missing C3-v3={missing_c3[:5]}"
        )

    output_rows = []

    for qid in sorted(rc83):
        old = rc83[qid]
        new = c3v3[qid]

        for field in (
            "query",
            "question_type",
            "gold_answer",
        ):
            if (
                str(old.get(field, ""))
                != str(new.get(field, ""))
            ):
                raise AssertionError(
                    f"{qid}: field mismatch "
                    f"for {field}"
                )

        old_prompt = str(
            old.get(
                "final_prompt",
                "",
            )
        ).strip()

        new_prompt = str(
            new.get(
                "final_prompt",
                "",
            )
        ).strip()

        if not old_prompt:
            raise AssertionError(
                f"{qid}: RC8.3 final_prompt "
                "is empty."
            )

        if not new_prompt:
            raise AssertionError(
                f"{qid}: C3-v3 final_prompt "
                "is empty."
            )

        old_ids = list_ids(
            old.get("selected_ids")
        )

        new_ids = list_ids(
            new.get("selected_ids")
        )

        removed = [
            memory_id
            for memory_id in old_ids
            if memory_id not in set(
                new_ids
            )
        ]

        added = [
            memory_id
            for memory_id in new_ids
            if memory_id not in set(
                old_ids
            )
        ]

        gold_ids = list_ids(
            old.get(
                "supporting_memory_ids"
            )
        )

        c3_debug = (
            new.get("debug")
            if isinstance(
                new.get("debug"),
                dict,
            )
            else {}
        )

        arbitration = (
            c3_debug.get(
                "c3_v3_arbitration_shadow"
            )
            if isinstance(
                c3_debug.get(
                    "c3_v3_arbitration_shadow"
                ),
                dict,
            )
            else {}
        )

        row = {
            "question_id": qid,
            "question_type": old.get(
                "question_type"
            ),
            "query_mode": new.get(
                "query_mode"
            ),
            "query": old.get(
                "query",
                "",
            ),
            "gold_answer": old.get(
                "gold_answer",
                "",
            ),
            "should_abstain": bool(
                old.get(
                    "should_abstain",
                    False,
                )
            ),
            "gold_memory_ids": gold_ids,

            # Existing Qwen runner names these
            # variants "legacy" and "c3_v3".
            # In this final experiment:
            #
            # legacy == exact frozen RC8.3
            # c3_v3  == exact frozen C3-v3 Active
            "legacy_selected_ids": old_ids,
            "c3_v3_selected_ids": new_ids,
            "removed_by_c3_v3": removed,
            "added_by_c3_v3": added,

            "legacy_prompt": old_prompt,
            "c3_v3_prompt": new_prompt,

            "hard_complete": bool(
                arbitration.get(
                    "hard_complete",
                    False,
                )
            ),

            "comparison_definition": {
                "legacy_variant":
                    "c3-rc8.3-paper-baseline",
                "c3_v3_variant":
                    "c3-v3-active-final-candidate",
            },
        }

        output_rows.append(row)

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for row in output_rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

    changed = sum(
        set(
            row["legacy_selected_ids"]
        )
        != set(
            row["c3_v3_selected_ids"]
        )
        for row in output_rows
    )

    rc83_total = sum(
        len(
            row[
                "legacy_selected_ids"
            ]
        )
        for row in output_rows
    )

    c3_total = sum(
        len(
            row[
                "c3_v3_selected_ids"
            ]
        )
        for row in output_rows
    )

    print(
        "questions:",
        len(output_rows),
    )

    print(
        "changed_evidence_questions:",
        changed,
    )

    print(
        "rc83_selected_total:",
        rc83_total,
    )

    print(
        "c3v3_selected_total:",
        c3_total,
    )

    print(
        "selected_reduction:",
        rc83_total - c3_total,
    )

    print(
        "output:",
        args.output,
    )


if __name__ == "__main__":
    main()
