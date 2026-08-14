from __future__ import annotations

import json
import math
import random
import statistics
from pathlib import Path


ROOT = Path(
    "/data/alyjw80/c3_lite/evaluation/"
    "frozen_rc83_vs_c3v3_final"
)

PREDICTIONS = (
    ROOT
    / "qwen3_full60_job39861"
    / "predictions.jsonl"
)

PAIRS = (
    ROOT
    / "frozen_rc83_vs_c3v3_pairs.jsonl"
)

SEED = 20260814
BOOTSTRAPS = 50000
PERMUTATIONS = 50000


def read_jsonl(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


predictions = read_jsonl(
    PREDICTIONS
)

pair_rows = read_jsonl(
    PAIRS
)

pair_meta = {
    row["question_id"]: row
    for row in pair_rows
}

by_qid = {}

for row in predictions:
    by_qid.setdefault(
        row["question_id"],
        {},
    )[row["variant"]] = row


records = []

for qid in sorted(by_qid):
    variants = by_qid[qid]

    if (
        "legacy" not in variants
        or "c3_v3" not in variants
    ):
        raise AssertionError(
            f"Missing paired variant: {qid}"
        )

    legacy = variants["legacy"]
    c3 = variants["c3_v3"]

    meta = pair_meta[qid]

    legacy_ids = set(
        meta.get(
            "legacy_selected_ids",
            [],
        )
    )

    c3_ids = set(
        meta.get(
            "c3_v3_selected_ids",
            [],
        )
    )

    removed = set(
        meta.get(
            "removed_by_c3_v3",
            [],
        )
    )

    gold = set(
        meta.get(
            "gold_memory_ids",
            [],
        )
    )

    changed = (
        legacy_ids != c3_ids
    )

    removed_gold = bool(
        removed & gold
    )

    if not changed:
        group = "A_unchanged"
    elif removed_gold:
        group = "C_removed_gold"
    else:
        group = (
            "B_changed_no_gold_removed"
        )

    records.append({
        "qid": qid,
        "group": group,

        "legacy_f1": float(
            legacy["answer_f1"]
        ),

        "c3_f1": float(
            c3["answer_f1"]
        ),

        "delta": (
            float(c3["answer_f1"])
            - float(
                legacy["answer_f1"]
            )
        ),

        "legacy_input": float(
            legacy["input_tokens"]
        ),

        "c3_input": float(
            c3["input_tokens"]
        ),

        "legacy_output": float(
            legacy["output_tokens"]
        ),

        "c3_output": float(
            c3["output_tokens"]
        ),

        "legacy_latency": float(
            legacy["latency_ms"]
        ),

        "c3_latency": float(
            c3["latency_ms"]
        ),
    })


def bootstrap_ci(
    values,
):
    if not values:
        return (
            math.nan,
            math.nan,
        )

    rng = random.Random(SEED)

    n = len(values)

    means = []

    for _ in range(
        BOOTSTRAPS
    ):
        sample = [
            values[
                rng.randrange(n)
            ]
            for _ in range(n)
        ]

        means.append(
            statistics.fmean(sample)
        )

    means.sort()

    low = means[
        int(
            0.025
            * len(means)
        )
    ]

    high = means[
        int(
            0.975
            * len(means)
        )
    ]

    return low, high


def sign_flip_p(
    values,
):
    if not values:
        return math.nan

    observed = abs(
        statistics.fmean(
            values
        )
    )

    rng = random.Random(
        SEED
    )

    extreme = 0

    for _ in range(
        PERMUTATIONS
    ):
        permuted = [
            value
            * (
                1
                if rng.random()
                < 0.5
                else -1
            )
            for value in values
        ]

        if (
            abs(
                statistics.fmean(
                    permuted
                )
            )
            >= observed
        ):
            extreme += 1

    return (
        extreme + 1
    ) / (
        PERMUTATIONS + 1
    )


def reduction(
    old,
    new,
):
    if old == 0:
        return math.nan

    return (
        old - new
    ) / old


groups = [
    "ALL",
    "A_unchanged",
    "B_changed_no_gold_removed",
    "C_removed_gold",
]


summary = {}

for group in groups:

    if group == "ALL":
        subset = records
    else:
        subset = [
            row
            for row in records
            if row["group"]
            == group
        ]

    deltas = [
        row["delta"]
        for row in subset
    ]

    if not subset:
        continue

    legacy_f1 = (
        statistics.fmean(
            row["legacy_f1"]
            for row in subset
        )
    )

    c3_f1 = (
        statistics.fmean(
            row["c3_f1"]
            for row in subset
        )
    )

    legacy_input = (
        statistics.fmean(
            row["legacy_input"]
            for row in subset
        )
    )

    c3_input = (
        statistics.fmean(
            row["c3_input"]
            for row in subset
        )
    )

    legacy_output = (
        statistics.fmean(
            row["legacy_output"]
            for row in subset
        )
    )

    c3_output = (
        statistics.fmean(
            row["c3_output"]
            for row in subset
        )
    )

    legacy_latency = (
        statistics.fmean(
            row["legacy_latency"]
            for row in subset
        )
    )

    c3_latency = (
        statistics.fmean(
            row["c3_latency"]
            for row in subset
        )
    )

    ci_low, ci_high = (
        bootstrap_ci(
            deltas
        )
    )

    better = sum(
        row["delta"] > 1e-12
        for row in subset
    )

    worse = sum(
        row["delta"] < -1e-12
        for row in subset
    )

    tied = (
        len(subset)
        - better
        - worse
    )

    result = {
        "questions":
            len(subset),

        "legacy_f1":
            legacy_f1,

        "c3_f1":
            c3_f1,

        "delta_f1":
            statistics.fmean(
                deltas
            ),

        "ci_low":
            ci_low,

        "ci_high":
            ci_high,

        "sign_flip_p":
            sign_flip_p(
                deltas
            ),

        "better":
            better,

        "worse":
            worse,

        "tied":
            tied,

        "legacy_input":
            legacy_input,

        "c3_input":
            c3_input,

        "input_reduction":
            reduction(
                legacy_input,
                c3_input,
            ),

        "legacy_output":
            legacy_output,

        "c3_output":
            c3_output,

        "output_reduction":
            reduction(
                legacy_output,
                c3_output,
            ),

        "legacy_latency":
            legacy_latency,

        "c3_latency":
            c3_latency,

        "latency_reduction":
            reduction(
                legacy_latency,
                c3_latency,
            ),
    }

    summary[group] = result


for group, result in (
    summary.items()
):
    print()
    print("=" * 80)
    print(group)

    for key, value in (
        result.items()
    ):
        if isinstance(
            value,
            float,
        ):
            print(
                f"{key}: "
                f"{value:.6f}"
            )
        else:
            print(
                f"{key}: {value}"
            )


out = (
    ROOT
    / "final_three_group_analysis.json"
)

out.write_text(
    json.dumps(
        summary,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)

print()
print(
    "Saved:",
    out,
)
