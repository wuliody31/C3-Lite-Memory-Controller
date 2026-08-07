#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
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
    text: str,
    value: str,
) -> bool:
    text = str(text).casefold()
    value = str(value).strip().casefold()

    if not value:
        return False

    pattern = (
        r"(?<!\w)"
        + re.escape(value)
        + r"(?!\w)"
    )

    return bool(
        re.search(pattern, text)
    )


def any_contains(
    texts: list[str],
    value: str,
) -> bool:
    return any(
        contains_value(text, value)
        for text in texts
    )


def classify_memory(
    text: str,
    current_gold: str,
    stale_values: list[str],
) -> dict[str, Any]:
    current_match = contains_value(
        text,
        current_gold,
    )

    matched_stale = [
        value
        for value in stale_values
        if contains_value(
            text,
            value,
        )
    ]

    stale_match = bool(
        matched_stale
    )

    if current_match and stale_match:
        label = "mixed_transition"
    elif current_match:
        label = "current_only"
    elif stale_match:
        label = "stale_only"
    else:
        label = "other"

    return {
        "text": text,
        "label": label,
        "current_match": current_match,
        "matched_stale_values": (
            matched_stale
        ),
    }


def mean_defined(
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


def score_case(
    result: dict[str, Any],
    gold: dict[str, Any],
) -> dict[str, Any]:
    current_gold = str(
        gold["current_gold"]
    )

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

    current_items = [
        classify_memory(
            text,
            current_gold,
            stale_values,
        )
        for text in current_texts
    ]

    final_items = [
        classify_memory(
            text,
            current_gold,
            stale_values,
        )
        for text in final_texts
    ]

    top1_label = (
        current_items[0]["label"]
        if current_items
        else "missing"
    )

    current_available = any_contains(
        current_texts,
        current_gold,
    )

    current_top1_correct = (
        bool(current_items)
        and current_items[0][
            "current_match"
        ]
    )

    stale_only_exposure = any(
        item["label"] == "stale_only"
        for item in current_items
    )

    mixed_transition_exposure = any(
        item["label"]
        == "mixed_transition"
        for item in current_items
    )

    historical_signal_exposure = (
        stale_only_exposure
        or mixed_transition_exposure
    )

    previous_found = [
        value
        for value in previous_values
        if any_contains(
            historical_texts,
            value,
        )
    ]

    previous_available = (
        len(previous_found)
        == len(previous_values)
        if previous_values
        else None
    )

    previous_top1_correct = (
        any(
            contains_value(
                historical_texts[0],
                value,
            )
            for value in previous_values
        )
        if (
            previous_values
            and historical_texts
        )
        else None
    )

    history_found = [
        value
        for value in history_values
        if any_contains(
            historical_texts,
            value,
        )
    ]

    history_recall = (
        len(history_found)
        / len(history_values)
        if history_values
        else None
    )

    current_label_counts = Counter(
        item["label"]
        for item in current_items
    )

    final_label_counts = Counter(
        item["label"]
        for item in final_items
    )

    return {
        "case_id": result["case_id"],
        "pattern": result["pattern"],
        "state_key": result["state_key"],
        "surface_variant": (
            result["surface_variant"]
        ),

        "temporal_symmetry": (
            result["pattern"]
            != "out_of_order"
        ),

        "current_gold": current_gold,
        "stale_values": stale_values,

        "current_value_available": (
            current_available
        ),

        "current_top1_correct": (
            current_top1_correct
        ),

        "current_top1_label": (
            top1_label
        ),

        "current_search_items": (
            current_items
        ),

        "current_label_counts": dict(
            current_label_counts
        ),

        "stale_only_exposure": (
            stale_only_exposure
        ),

        "mixed_transition_exposure": (
            mixed_transition_exposure
        ),

        "historical_signal_exposure": (
            historical_signal_exposure
        ),

        "previous_gold": (
            previous_values
        ),

        "previous_found": (
            previous_found
        ),

        "previous_value_available": (
            previous_available
        ),

        "previous_top1_correct": (
            previous_top1_correct
        ),

        "history_gold": (
            history_values
        ),

        "history_found": (
            history_found
        ),

        "history_value_recall": (
            history_recall
        ),

        "final_memory_count": len(
            final_texts
        ),

        "final_memory_items": (
            final_items
        ),

        "final_label_counts": dict(
            final_label_counts
        ),
    }


def aggregate(rows):
    if not rows:
        return None

    stale_rows = [
        row
        for row in rows
        if row["stale_values"]
    ]

    previous_rows = [
        row
        for row in rows
        if row[
            "previous_value_available"
        ]
        is not None
    ]

    top1_counts = Counter(
        row["current_top1_label"]
        for row in rows
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

        "current_only_top1_rate": mean(
            int(
                row[
                    "current_top1_label"
                ]
                == "current_only"
            )
            for row in rows
        ),

        "stale_only_top1_rate": mean(
            int(
                row[
                    "current_top1_label"
                ]
                == "stale_only"
            )
            for row in rows
        ),

        "mixed_transition_top1_rate": mean(
            int(
                row[
                    "current_top1_label"
                ]
                == "mixed_transition"
            )
            for row in rows
        ),

        "other_or_missing_top1_rate": mean(
            int(
                row[
                    "current_top1_label"
                ]
                in {
                    "other",
                    "missing",
                }
            )
            for row in rows
        ),

        "stale_only_exposure_case_rate": (
            mean(
                int(
                    row[
                        "stale_only_exposure"
                    ]
                )
                for row in stale_rows
            )
            if stale_rows
            else None
        ),

        "mixed_transition_exposure_case_rate": (
            mean(
                int(
                    row[
                        "mixed_transition_exposure"
                    ]
                )
                for row in stale_rows
            )
            if stale_rows
            else None
        ),

        "historical_signal_exposure_case_rate": (
            mean(
                int(
                    row[
                        "historical_signal_exposure"
                    ]
                )
                for row in stale_rows
            )
            if stale_rows
            else None
        ),

        "previous_value_availability": (
            mean(
                int(
                    row[
                        "previous_value_available"
                    ]
                )
                for row in previous_rows
            )
            if previous_rows
            else None
        ),

        "previous_top1_accuracy": (
            mean_defined(
                rows,
                "previous_top1_correct",
            )
        ),

        "history_value_recall": (
            mean_defined(
                rows,
                "history_value_recall",
            )
        ),

        "mean_final_memory_count": mean(
            row["final_memory_count"]
            for row in rows
        ),

        "current_top1_label_counts": dict(
            sorted(
                top1_counts.items()
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
                gold_by_id[
                    case_id
                ],
            )
        )

    by_pattern = defaultdict(list)
    by_state_key = defaultdict(list)

    for row in scored:
        by_pattern[
            row["pattern"]
        ].append(row)

        by_state_key[
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
        if not row["temporal_symmetry"]
    ]

    summary = {
        "experiment":
            "mem0_lifecycle_behaviour_evaluation_v02",

        "metric_semantics": {
            "current_only":
                "retrieved item contains current gold "
                "and no stale gold value",

            "stale_only":
                "retrieved item contains stale gold "
                "value(s) and not current gold",

            "mixed_transition":
                "retrieved item contains both current "
                "and stale gold values; this may encode "
                "a transition and is not automatically "
                "treated as a contradiction",

            "other":
                "retrieved item contains neither "
                "current nor stale gold value",

            "historical_signal_exposure":
                "current retrieval contains at least "
                "one stale-only or mixed-transition item",
        },

        "scope": {
            "system":
                "Mem0 OSS infer=True",

            "comparison_level":
                "observable retrieval behaviour",

            "out_of_order_policy":
                "reported separately as temporal "
                "stress because structured observation "
                "timestamps are unavailable in the "
                "evaluated OSS API",
        },

        "overall_all_cases":
            aggregate(scored),

        "symmetric_cases":
            aggregate(symmetric),

        "temporal_stress_cases":
            aggregate(
                temporal_stress
            ),

        "by_pattern": {
            key: aggregate(value)
            for key, value in sorted(
                by_pattern.items()
            )
        },

        "by_state_key_all": {
            key: aggregate(value)
            for key, value in sorted(
                by_state_key.items()
            )
        },

        "by_state_key_symmetric": {
            key: aggregate(
                [
                    row
                    for row in value
                    if row["temporal_symmetry"]
                ]
            )
            for key, value in sorted(
                by_state_key.items()
            )
        },

        "by_state_key_temporal_stress": {
            key: aggregate(
                [
                    row
                    for row in value
                    if not row["temporal_symmetry"]
                ]
            )
            for key, value in sorted(
                by_state_key.items()
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
