from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


BASELINES = (
    "simple_retrieval",
    "all_memory",
)


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    return str(value).strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
    }


def extract_evidence_f1(row: dict[str, Any]) -> float:
    """Read stored evidence F1, or reconstruct it as a fallback."""
    for key in (
        "evidence_f1",
        "evidence_set_f1",
    ):
        if row.get(key) is not None:
            return float(row[key])

    for container_key in (
        "metrics",
        "scores",
        "evaluation",
    ):
        container = row.get(container_key)

        if isinstance(container, dict):
            for key in (
                "evidence_f1",
                "evidence_set_f1",
            ):
                if container.get(key) is not None:
                    return float(container[key])

    selected = {
        str(value)
        for value in (
            row.get("selected_ids", [])
            or []
        )
    }

    gold = {
        str(value)
        for value in (
            row.get("supporting_memory_ids", [])
            or row.get("gold_memory_ids", [])
            or []
        )
    }

    overlap = len(selected & gold)

    precision = (
        overlap / len(selected)
        if selected
        else 0.0
    )
    recall = (
        overlap / len(gold)
        if gold
        else 0.0
    )

    if precision + recall == 0.0:
        return 0.0

    return (
        2.0
        * precision
        * recall
        / (precision + recall)
    )


def load_answer_scores(
    path: Path,
) -> tuple[
    dict[tuple[str, str], float],
    dict[str, bool],
]:
    scores: dict[
        tuple[str, str],
        float,
    ] = {}

    should_abstain: dict[str, bool] = {}

    with path.open(
        encoding="utf-8",
        newline="",
    ) as handle:
        for row in csv.DictReader(handle):
            question_id = row["question_id"]
            method = row["method"]

            scores[
                (question_id, method)
            ] = float(
                row["answer_token_f1"]
            )

            should_abstain[
                question_id
            ] = parse_bool(
                row.get(
                    "should_abstain",
                    False,
                )
            )

    return scores, should_abstain


def load_evidence_scores(
    path: Path,
) -> dict[tuple[str, str], float]:
    scores: dict[
        tuple[str, str],
        float,
    ] = {}

    for line in path.read_text(
        encoding="utf-8"
    ).splitlines():
        if not line.strip():
            continue

        row = json.loads(line)

        key = (
            str(row["question_id"]),
            str(row["method"]),
        )

        scores[key] = extract_evidence_f1(
            row
        )

    return scores


def paired_bootstrap(
    differences: np.ndarray,
    *,
    resamples: int,
    seed: int,
) -> dict[str, float]:
    if differences.ndim != 1:
        raise ValueError(
            "Differences must be one-dimensional."
        )

    n = len(differences)

    if n < 2:
        raise ValueError(
            "At least two paired observations are required."
        )

    observed = float(
        differences.mean()
    )

    rng = np.random.default_rng(seed)

    indices = rng.integers(
        0,
        n,
        size=(resamples, n),
    )

    bootstrap_means = differences[
        indices
    ].mean(axis=1)

    ci_low, ci_high = np.quantile(
        bootstrap_means,
        [0.025, 0.975],
    )

    standard_error = float(
        bootstrap_means.std(ddof=1)
    )

    superiority_probability = float(
        np.mean(bootstrap_means > 0.0)
    )

    # Center the paired differences under H0: mean difference = 0.
    centered = (
        differences - observed
    )

    null_indices = rng.integers(
        0,
        n,
        size=(resamples, n),
    )

    null_means = centered[
        null_indices
    ].mean(axis=1)

    p_value = (
        np.count_nonzero(
            np.abs(null_means)
            >= abs(observed)
        )
        + 1
    ) / (resamples + 1)

    return {
        "mean_difference": observed,
        "standard_error": standard_error,
        "ci_95_low": float(ci_low),
        "ci_95_high": float(ci_high),
        "bootstrap_superiority_probability": (
            superiority_probability
        ),
        "p_value_raw": float(p_value),
    }


def holm_adjust(
    rows: list[dict[str, Any]],
) -> None:
    families = sorted({
        str(row["test_family"])
        for row in rows
    })

    for family in families:
        indices = [
            index
            for index, row in enumerate(rows)
            if row["test_family"] == family
        ]

        ordered = sorted(
            indices,
            key=lambda index: float(
                rows[index]["p_value_raw"]
            ),
        )

        running_max = 0.0
        total = len(ordered)

        for rank, index in enumerate(
            ordered
        ):
            multiplier = total - rank

            adjusted = min(
                1.0,
                multiplier
                * float(
                    rows[index][
                        "p_value_raw"
                    ]
                ),
            )

            running_max = max(
                running_max,
                adjusted,
            )

            rows[index][
                "p_value_holm"
            ] = running_max

    for row in rows:
        row["significant_holm_0_05"] = (
            float(row["p_value_holm"])
            < 0.05
        )
        row["ci_excludes_zero"] = (
            float(row["ci_95_low"]) > 0.0
            or float(row["ci_95_high"]) < 0.0
        )


def paired_rows(
    scores: dict[tuple[str, str], float],
    baseline: str,
    *,
    allowed_ids: set[str] | None = None,
) -> tuple[
    list[str],
    np.ndarray,
    np.ndarray,
]:
    c3_ids = {
        question_id
        for question_id, method in scores
        if method == "c3"
    }

    baseline_ids = {
        question_id
        for question_id, method in scores
        if method == baseline
    }

    question_ids = (
        c3_ids & baseline_ids
    )

    if allowed_ids is not None:
        question_ids &= allowed_ids

    ordered_ids = sorted(question_ids)

    c3 = np.asarray(
        [
            scores[
                (question_id, "c3")
            ]
            for question_id in ordered_ids
        ],
        dtype=float,
    )

    comparison = np.asarray(
        [
            scores[
                (question_id, baseline)
            ]
            for question_id in ordered_ids
        ],
        dtype=float,
    )

    return ordered_ids, c3, comparison


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--predictions",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--answer-quality",
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
        default=50_000,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20_260_801,
    )

    args = parser.parse_args()

    answer_scores, should_abstain = (
        load_answer_scores(
            args.answer_quality
        )
    )

    evidence_scores = (
        load_evidence_scores(
            args.predictions
        )
    )

    answerable_ids = {
        question_id
        for question_id, value
        in should_abstain.items()
        if not value
    }

    specifications = [
        {
            "metric": "answer_token_f1",
            "subset": "all_questions",
            "family": "primary",
            "scores": answer_scores,
            "allowed_ids": None,
        },
        {
            "metric": "evidence_f1",
            "subset": "all_questions",
            "family": "primary",
            "scores": evidence_scores,
            "allowed_ids": None,
        },
        {
            "metric": "answer_token_f1",
            "subset": "answerable_only",
            "family": "sensitivity",
            "scores": answer_scores,
            "allowed_ids": answerable_ids,
        },
    ]

    output_rows: list[
        dict[str, Any]
    ] = []

    test_index = 0

    for specification in specifications:
        for baseline in BASELINES:
            (
                question_ids,
                c3_values,
                baseline_values,
            ) = paired_rows(
                specification["scores"],
                baseline,
                allowed_ids=specification[
                    "allowed_ids"
                ],
            )

            differences = (
                c3_values
                - baseline_values
            )

            result = paired_bootstrap(
                differences,
                resamples=args.resamples,
                seed=args.seed + test_index,
            )

            test_index += 1

            wins = int(
                np.sum(differences > 1e-12)
            )
            ties = int(
                np.sum(
                    np.abs(differences)
                    <= 1e-12
                )
            )
            losses = int(
                np.sum(differences < -1e-12)
            )

            baseline_mean = float(
                baseline_values.mean()
            )
            c3_mean = float(
                c3_values.mean()
            )

            relative_improvement = (
                100.0
                * (
                    c3_mean
                    - baseline_mean
                )
                / baseline_mean
                if baseline_mean != 0.0
                else float("nan")
            )

            output_rows.append(
                {
                    "test_family": (
                        specification[
                            "family"
                        ]
                    ),
                    "metric": (
                        specification[
                            "metric"
                        ]
                    ),
                    "subset": (
                        specification[
                            "subset"
                        ]
                    ),
                    "comparison": (
                        f"c3_minus_{baseline}"
                    ),
                    "n_pairs": len(
                        question_ids
                    ),
                    "c3_mean": c3_mean,
                    "baseline_mean": (
                        baseline_mean
                    ),
                    "mean_difference": (
                        result[
                            "mean_difference"
                        ]
                    ),
                    "relative_improvement_pct": (
                        relative_improvement
                    ),
                    "standard_error": (
                        result[
                            "standard_error"
                        ]
                    ),
                    "ci_95_low": (
                        result[
                            "ci_95_low"
                        ]
                    ),
                    "ci_95_high": (
                        result[
                            "ci_95_high"
                        ]
                    ),
                    "bootstrap_superiority_probability": (
                        result[
                            "bootstrap_superiority_probability"
                        ]
                    ),
                    "p_value_raw": (
                        result[
                            "p_value_raw"
                        ]
                    ),
                    "wins": wins,
                    "ties": ties,
                    "losses": losses,
                }
            )

    holm_adjust(output_rows)

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        args.output_dir
        / "paired_bootstrap_results.csv"
    )

    with csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                output_rows[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(output_rows)

    manifest = {
        "predictions": str(
            args.predictions
        ),
        "answer_quality": str(
            args.answer_quality
        ),
        "resamples": args.resamples,
        "seed": args.seed,
        "confidence_interval": (
            "95% percentile paired bootstrap"
        ),
        "hypothesis_test": (
            "two-sided centered paired "
            "bootstrap under H0 mean difference=0"
        ),
        "multiple_testing": (
            "Holm-Bonferroni within each "
            "test family"
        ),
        "primary_family_tests": 4,
        "sensitivity_family_tests": 2,
    }

    manifest_path = (
        args.output_dir
        / "paired_bootstrap_manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(csv_path)
    print(manifest_path)
    print()

    header = (
        f"{'Metric':20} "
        f"{'Subset':16} "
        f"{'Comparison':28} "
        f"{'N':>3} "
        f"{'Delta':>9} "
        f"{'95% CI':>23} "
        f"{'p(raw)':>10} "
        f"{'p(Holm)':>10}"
    )

    print(header)
    print("-" * len(header))

    for row in output_rows:
        ci = (
            f"[{float(row['ci_95_low']):.4f}, "
            f"{float(row['ci_95_high']):.4f}]"
        )

        print(
            f"{row['metric']:20} "
            f"{row['subset']:16} "
            f"{row['comparison']:28} "
            f"{int(row['n_pairs']):3d} "
            f"{float(row['mean_difference']):9.4f} "
            f"{ci:>23} "
            f"{float(row['p_value_raw']):10.6f} "
            f"{float(row['p_value_holm']):10.6f}"
        )


if __name__ == "__main__":
    main()
