#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--results",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--cases",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    return parser.parse_args()


def read_jsonl(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


def contains_value(
    texts: list[str],
    value: str,
) -> bool:
    joined = "\n".join(
        str(text)
        for text in texts
    ).casefold()

    target = str(value).strip().casefold()

    if not target:
        return False

    pattern = (
        r"(?<!\w)"
        + re.escape(target)
        + r"(?!\w)"
    )

    return bool(
        re.search(pattern, joined)
    )


def top1_contains(
    texts: list[str],
    value: str,
) -> bool:
    if not texts:
        return False

    return contains_value(
        [texts[0]],
        value,
    )


def score_case(
    result: dict[str, Any],
    gold: dict[str, Any],
) -> dict[str, Any]:
    current_gold = gold["current_gold"]

    stale_values = list(
        gold["stale_values"]
    )

    previous_values = list(
        gold["historical_gold"]
    )

    history_values = list(
        gold["history_values_expected"]
    )

    current_texts = list(
        result["current_search_texts"]
    )

    historical_texts = list(
        result["historical_search_texts"]
    )

    final_texts = list(
        result["final_memory_texts"]
    )

    current_available = (
        contains_value(
            current_texts,
            current_gold,
        )
    )

    current_top1 = (
        top1_contains(
            current_texts,
            current_gold,
        )
    )

    exposed_stale = [
        value
        for value in stale_values
        if contains_value(
            current_texts,
            value,
        )
    ]

    final_stale = [
        value
        for value in stale_values
        if contains_value(
            final_texts,
            value,
        )
    ]

    previous_found = [
        value
        for value in previous_values
        if contains_value(
            historical_texts,
            value,
        )
    ]

    history_found = [
        value
        for value in history_values
        if contains_value(
            historical_texts,
            value,
        )
    ]

    previous_recall = (
        len(previous_found)
        / len(previous_values)
        if previous_values
        else None
    )

    history_recall = (
        len(history_found)
        / len(history_values)
        if history_values
        else None
    )

    current_contradiction = bool(
        current_available
        and exposed_stale
    )

    final_contradiction = bool(
        contains_value(
            final_texts,
            current_gold,
        )
        and final_stale
    )

    add_events = []

    for turn in result["turns"]:
        add_result = turn.get(
            "add_result",
            {}
        )

        if not isinstance(
            add_result,
            dict,
        ):
            continue

        for item in add_result.get(
            "results",
            [],
        ):
            if not isinstance(
                item,
                dict,
            ):
                continue

            event = item.get("event")

            if event:
                add_events.append(
                    str(event)
                )

    return {
        "case_id": result["case_id"],
        "pattern": result["pattern"],
        "state_key": result["state_key"],
        "surface_variant": (
            result["surface_variant"]
        ),

        "current_gold": current_gold,
        "current_search_texts": (
            current_texts
        ),
        "current_value_available": (
            current_available
        ),
        "current_top1_correct": (
            current_top1
        ),

        "stale_values": stale_values,
        "exposed_stale_values": (
            exposed_stale
        ),
        "stale_exposure": bool(
            exposed_stale
        ),

        "previous_gold": (
            previous_values
        ),
        "previous_found": (
            previous_found
        ),
        "previous_value_recall": (
            previous_recall
        ),

        "history_gold": history_values,
        "history_found": history_found,
        "history_value_recall": (
            history_recall
        ),

        "current_contradiction": (
            current_contradiction
        ),
        "final_contradiction": (
            final_contradiction
        ),

        "final_memory_count": len(
            result["final_memory_ids"]
        ),
        "final_memory_texts": (
            final_texts
        ),

        "add_events": add_events,

        "temporal_symmetry": (
            result["pattern"]
            != "out_of_order"
        ),
    }


def average_defined(
    rows,
    field,
):
    values = [
        float(row[field])
        for row in rows
        if row[field] is not None
    ]

    if not values:
        return None

    return mean(values)


def aggregate(rows):
    if not rows:
        return None

    stale_rows = [
        row
        for row in rows
        if row["stale_values"]
    ]

    stale_value_total = sum(
        len(row["stale_values"])
        for row in stale_rows
    )

    stale_exposed_total = sum(
        len(
            row[
                "exposed_stale_values"
            ]
        )
        for row in stale_rows
    )

    return {
        "case_count": len(rows),

        "current_value_availability": mean(
            int(
                row[
                    "current_value_available"
                ]
            )
            for row in rows
        ),

        "current_top1_accuracy": mean(
            int(
                row[
                    "current_top1_correct"
                ]
            )
            for row in rows
        ),

        "stale_exposure_case_rate": (
            mean(
                int(
                    row[
                        "stale_exposure"
                    ]
                )
                for row in stale_rows
            )
            if stale_rows
            else None
        ),

        "stale_exposure_value_rate": (
            stale_exposed_total
            / stale_value_total
            if stale_value_total
            else None
        ),

        "previous_value_recall": (
            average_defined(
                rows,
                "previous_value_recall",
            )
        ),

        "history_value_recall": (
            average_defined(
                rows,
                "history_value_recall",
            )
        ),

        "current_contradiction_rate": (
            mean(
                int(
                    row[
                        "current_contradiction"
                    ]
                )
                for row in stale_rows
            )
            if stale_rows
            else None
        ),

        "final_contradiction_rate": (
            mean(
                int(
                    row[
                        "final_contradiction"
                    ]
                )
                for row in stale_rows
            )
            if stale_rows
            else None
        ),

        "mean_final_memory_count": (
            mean(
                row[
                    "final_memory_count"
                ]
                for row in rows
            )
        ),
    }


def main():
    args = parse_args()

    results = read_jsonl(
        args.results
    )

    cases = read_jsonl(
        args.cases
    )

    gold_by_id = {
        row["case_id"]: row
        for row in cases
    }

    scored = []

    for result in results:
        case_id = result["case_id"]

        if case_id not in gold_by_id:
            raise RuntimeError(
                f"Missing gold case: "
                f"{case_id}"
            )

        scored.append(
            score_case(
                result,
                gold_by_id[case_id],
            )
        )

    by_pattern_rows = defaultdict(
        list
    )

    by_state_rows = defaultdict(
        list
    )

    for row in scored:
        by_pattern_rows[
            row["pattern"]
        ].append(row)

        by_state_rows[
            row["state_key"]
        ].append(row)

    symmetric = [
        row
        for row in scored
        if row["temporal_symmetry"]
    ]

    temporal_stress = [
        row
        for row in scored
        if not row[
            "temporal_symmetry"
        ]
    ]

    summary = {
        "experiment": (
            "mem0_lifecycle_"
            "behaviour_evaluation"
        ),

        "scope": {
            "system": (
                "Mem0 OSS infer=True"
            ),
            "comparison_level": (
                "observable memory "
                "behaviour"
            ),
            "out_of_order_policy": (
                "reported separately "
                "as temporal stress "
                "because structured "
                "observation timestamps "
                "are unavailable in the "
                "evaluated OSS API"
            ),
        },

        "overall_all_cases": (
            aggregate(scored)
        ),

        "symmetric_cases": (
            aggregate(symmetric)
        ),

        "temporal_stress_cases": (
            aggregate(
                temporal_stress
            )
        ),

        "by_pattern": {
            key: aggregate(value)
            for key, value
            in sorted(
                by_pattern_rows.items()
            )
        },

        "by_state_key": {
            key: aggregate(value)
            for key, value
            in sorted(
                by_state_rows.items()
            )
        },
    }

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with (
        args.output_dir
        / "per_case.jsonl"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        for row in scored:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

    (
        args.output_dir
        / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
