from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


METRICS = [
    "route_exact_match",
    "route_f1",
    "evidence_recall",
    "evidence_precision",
    "evidence_density",
    "abstention_correctness",
    "num_retrieved",
    "num_used",
    "num_outdated",
    "latency_seconds",
]


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--v2",
        required=True,
    )
    parser.add_argument(
        "--v21",
        required=True,
    )
    parser.add_argument(
        "--out",
        default=None,
    )

    args = parser.parse_args()

    before = pd.read_csv(args.v2)
    after = pd.read_csv(args.v21)

    merged = before.merge(
        after,
        on="method",
        suffixes=("_v2", "_v21"),
    )

    delta_columns: list[str] = []

    for metric in METRICS:
        before_column = f"{metric}_v2"
        after_column = f"{metric}_v21"

        if (
            before_column in merged
            and after_column in merged
        ):
            delta_column = f"{metric}_delta"
            merged[delta_column] = (
                merged[after_column]
                - merged[before_column]
            )
            delta_columns.append(delta_column)

    output = (
        Path(args.out)
        if args.out
        else Path(args.v21).parent
        / "v2_vs_v21.csv"
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    merged.to_csv(
        output,
        index=False,
    )

    print(
        merged[
            ["method"] + delta_columns
        ].to_string(index=False)
    )

    print()
    print(f"Saved comparison to {output}")


if __name__ == "__main__":
    main()
