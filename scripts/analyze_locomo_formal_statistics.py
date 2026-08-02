from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np


QUALITY_COMPARISONS = (
    (
        "answer_token_f1",
        "c3",
        "simple_retrieval",
        "answerable",
    ),
    (
        "answer_token_f1",
        "c3",
        "all_memory",
        "answerable",
    ),
    (
        "answer_token_f1",
        "c3",
        "no_memory",
        "answerable",
    ),
    (
        "source_evidence_f1",
        "c3",
        "simple_retrieval",
        "evidence",
    ),
    (
        "source_evidence_f1",
        "c3",
        "all_memory",
        "evidence",
    ),
    (
        "source_evidence_f1",
        "c3",
        "no_memory",
        "evidence",
    ),
)

EFFICIENCY_COMPARISONS = (
    (
        "selected_evidence_count",
        "c3",
        "simple_retrieval",
        "all",
    ),
    (
        "selected_evidence_count",
        "c3",
        "all_memory",
        "all",
    ),
)


def read_csv(
    path: Path,
) -> list[dict[str, str]]:
    with path.open(
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        return list(csv.DictReader(handle))


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
    }


def holm_adjust(
    p_values: list[float],
) -> list[float]:
    count = len(p_values)

    order = sorted(
        range(count),
        key=lambda index: p_values[index],
    )

    adjusted = [1.0] * count
    running_max = 0.0

    for rank, index in enumerate(order):
        multiplier = count - rank

        candidate = min(
            1.0,
            p_values[index] * multiplier,
        )

        running_max = max(
            running_max,
            candidate,
        )

        adjusted[index] = running_max

    return adjusted


def paired_bootstrap(
    differences: np.ndarray,
    *,
    resamples: int,
    seed: int,
    batch_size: int = 1000,
) -> dict[str, float | int]:
    if differences.ndim != 1:
        raise ValueError(
            "Differences must be one-dimensional."
        )

    if differences.size == 0:
        raise ValueError(
            "No paired differences supplied."
        )

    rng = np.random.default_rng(seed)

    observed = float(
        differences.mean()
    )

    bootstrap_means = np.empty(
        resamples,
        dtype=np.float64,
    )

    written = 0

    while written < resamples:
        current = min(
            batch_size,
            resamples - written,
        )

        indices = rng.integers(
            low=0,
            high=differences.size,
            size=(
                current,
                differences.size,
            ),
        )

        bootstrap_means[
            written:written + current
        ] = differences[indices].mean(
            axis=1
        )

        written += current

    lower, upper = np.percentile(
        bootstrap_means,
        [2.5, 97.5],
    )

    centered = (
        bootstrap_means - observed
    )

    raw_p = (
        np.count_nonzero(
            np.abs(centered)
            >= abs(observed)
        )
        + 1
    ) / (resamples + 1)

    tolerance = 1e-12

    wins = int(
        np.count_nonzero(
            differences > tolerance
        )
    )

    ties = int(
        np.count_nonzero(
            np.abs(differences)
            <= tolerance
        )
    )

    losses = int(
        np.count_nonzero(
            differences < -tolerance
        )
    )

    return {
        "n_pairs": int(
            differences.size
        ),
        "mean_difference": observed,
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "raw_p": float(raw_p),
        "wins": wins,
        "ties": ties,
        "losses": losses,
    }


def select_ids(
    rows: list[dict[str, str]],
    scope: str,
) -> set[str]:
    selected = set()

    for row in rows:
        if scope == "answerable":
            if as_bool(
                row["should_abstain"]
            ):
                continue

        elif scope == "evidence":
            if not as_bool(
                row["has_gold_evidence"]
            ):
                continue

        elif scope != "all":
            raise ValueError(
                f"Unknown scope: {scope}"
            )

        selected.add(
            row["question_id"]
        )

    return selected


def run_family(
    *,
    rows: list[dict[str, str]],
    comparisons: tuple[
        tuple[str, str, str, str],
        ...,
    ],
    resamples: int,
    seed: int,
    family: str,
) -> list[dict[str, Any]]:
    by_method: dict[
        str,
        dict[str, dict[str, str]],
    ] = {}

    for row in rows:
        by_method.setdefault(
            row["method"],
            {},
        )[row["question_id"]] = row

    results = []

    for comparison_index, (
        metric,
        first_method,
        second_method,
        scope,
    ) in enumerate(comparisons):
        first_rows = list(
            by_method[first_method].values()
        )

        eligible_ids = select_ids(
            first_rows,
            scope,
        )

        eligible_ids &= set(
            by_method[second_method]
        )

        question_ids = sorted(
            eligible_ids
        )

        first_values = np.asarray(
            [
                float(
                    by_method[first_method][
                        question_id
                    ][metric]
                )
                for question_id
                in question_ids
            ],
            dtype=np.float64,
        )

        second_values = np.asarray(
            [
                float(
                    by_method[second_method][
                        question_id
                    ][metric]
                )
                for question_id
                in question_ids
            ],
            dtype=np.float64,
        )

        differences = (
            first_values - second_values
        )

        result = paired_bootstrap(
            differences,
            resamples=resamples,
            seed=seed + comparison_index,
        )

        result.update({
            "family": family,
            "metric": metric,
            "comparison": (
                f"{first_method}_minus_"
                f"{second_method}"
            ),
            "scope": scope,
            "first_method_mean": float(
                first_values.mean()
            ),
            "second_method_mean": float(
                second_values.mean()
            ),
        })

        results.append(result)

    adjusted = holm_adjust(
        [
            float(result["raw_p"])
            for result in results
        ]
    )

    for result, adjusted_p in zip(
        results,
        adjusted,
    ):
        result["holm_p"] = adjusted_p
        result[
            "significant_holm_0_05"
        ] = adjusted_p < 0.05

    return results


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                rows[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--per-question",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--resamples",
        type=int,
        default=50000,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260802,
    )

    args = parser.parse_args()

    rows = read_csv(
        args.per_question
    )

    quality_results = run_family(
        rows=rows,
        comparisons=QUALITY_COMPARISONS,
        resamples=args.resamples,
        seed=args.seed,
        family="primary_quality",
    )

    efficiency_results = run_family(
        rows=rows,
        comparisons=EFFICIENCY_COMPARISONS,
        resamples=args.resamples,
        seed=args.seed + 100,
        family="efficiency",
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    quality_path = (
        args.output_dir
        / "locomo_formal_quality_bootstrap.csv"
    )

    efficiency_path = (
        args.output_dir
        / "locomo_formal_efficiency_bootstrap.csv"
    )

    write_csv(
        quality_path,
        quality_results,
    )

    write_csv(
        efficiency_path,
        efficiency_results,
    )

    print(quality_path)
    print(efficiency_path)


if __name__ == "__main__":
    main()
