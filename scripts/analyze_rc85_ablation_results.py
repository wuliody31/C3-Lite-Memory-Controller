from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


VARIANTS = (
    "no_route_planner",
    "no_conflict_handling",
    "no_coverage_confidence_gate",
    "no_evidence_selector",
)


def load_json(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        return list(csv.DictReader(handle))


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
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def answer_summary(
    directory: Path,
) -> dict[str, str]:
    rows = load_csv(
        directory
        / "answer_quality_summary.csv"
    )

    return next(
        row
        for row in rows
        if row["method"] == "c3"
    )


def answer_per_question(
    directory: Path,
) -> dict[str, float]:
    rows = load_csv(
        directory
        / "answer_quality_per_question.csv"
    )

    return {
        row["question_id"]: float(
            row["answer_token_f1"]
        )
        for row in rows
        if row["method"] == "c3"
    }


def evidence_per_question(
    prediction_path: Path,
) -> dict[str, float]:
    rows = load_jsonl(prediction_path)

    output: dict[str, float] = {}

    for row in rows:
        if row.get("method") != "c3":
            continue

        value = row.get("evidence_f1")

        if value is None:
            metrics = row.get("metrics", {})

            if isinstance(metrics, dict):
                value = metrics.get(
                    "evidence_f1"
                )

        if value is None:
            raise KeyError(
                "No per-question evidence_f1 "
                f"for {row.get('question_id')}"
            )

        output[
            str(row["question_id"])
        ] = float(value)

    return output


def system_summary(
    directory: Path,
) -> dict[str, Any]:
    summary = load_json(
        directory / "summary.json"
    )
    return summary["c3"]


def paired_bootstrap(
    differences: np.ndarray,
    *,
    resamples: int,
    seed: int,
) -> dict[str, float]:
    n = len(differences)

    if n < 2:
        raise ValueError(
            "At least two pairs are required."
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

    low, high = np.quantile(
        bootstrap_means,
        [0.025, 0.975],
    )

    centered = differences - observed

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
        "standard_error": float(
            bootstrap_means.std(ddof=1)
        ),
        "ci_95_low": float(low),
        "ci_95_high": float(high),
        "p_value_raw": float(p_value),
        "superiority_probability": float(
            np.mean(bootstrap_means > 0.0)
        ),
    }


def holm_adjust(
    rows: list[dict[str, Any]],
) -> None:
    order = sorted(
        range(len(rows)),
        key=lambda index: float(
            rows[index]["p_value_raw"]
        ),
    )

    total = len(order)
    running = 0.0

    for rank, index in enumerate(order):
        adjusted = min(
            1.0,
            (total - rank)
            * float(
                rows[index]["p_value_raw"]
            ),
        )

        running = max(
            running,
            adjusted,
        )

        rows[index][
            "p_value_holm"
        ] = running

    for row in rows:
        row["ci_excludes_zero"] = (
            float(row["ci_95_low"]) > 0.0
            or float(row["ci_95_high"]) < 0.0
        )

        row["significant_holm_0_05"] = (
            float(row["p_value_holm"])
            < 0.05
            and bool(
                row["ci_excludes_zero"]
            )
        )


def paired_values(
    full: dict[str, float],
    ablation: dict[str, float],
) -> tuple[
    list[str],
    np.ndarray,
    np.ndarray,
]:
    question_ids = sorted(
        set(full) & set(ablation)
    )

    full_values = np.asarray(
        [
            full[question_id]
            for question_id in question_ids
        ],
        dtype=float,
    )

    ablation_values = np.asarray(
        [
            ablation[question_id]
            for question_id in question_ids
        ],
        dtype=float,
    )

    return (
        question_ids,
        full_values,
        ablation_values,
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--full-c3-dir",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--ablation-dir",
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

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    experiment_dirs = {
        "full_c3": args.full_c3_dir,
        **{
            variant: (
                args.ablation_dir / variant
            )
            for variant in VARIANTS
        },
    }

    main_rows: list[
        dict[str, Any]
    ] = []

    for name, directory in (
        experiment_dirs.items()
    ):
        system = system_summary(directory)
        answers = answer_summary(directory)

        main_rows.append(
            {
                "variant": name,
                "answer_token_f1_all": float(
                    answers[
                        "answer_f1_all"
                    ]
                ),
                "answer_token_f1_answerable": float(
                    answers[
                        "answer_f1_answerable"
                    ]
                ),
                "evidence_f1": float(
                    system["evidence_f1"]
                ),
                "evidence_precision": float(
                    system[
                        "evidence_precision"
                    ]
                ),
                "evidence_recall": float(
                    system[
                        "evidence_recall"
                    ]
                ),
                "route_exact": float(
                    system["route_exact"]
                ),
                "route_f1": float(
                    system["route_f1"]
                ),
                "abstention_correctness": float(
                    system[
                        "abstention_correctness"
                    ]
                ),
                "coverage": float(
                    system["coverage"]
                ),
                "num_used": float(
                    system["num_used"]
                ),
                "latency_ms": float(
                    system["latency_ms"]
                ),
            }
        )

    write_csv(
        args.output_dir
        / "rc85_ablation_main_results.csv",
        main_rows,
    )

    full_answer = answer_per_question(
        args.full_c3_dir
    )
    full_evidence = evidence_per_question(
        args.full_c3_dir
        / "predictions.jsonl"
    )

    test_rows: list[
        dict[str, Any]
    ] = []

    test_index = 0

    for variant in VARIANTS:
        variant_dir = (
            args.ablation_dir / variant
        )

        score_sets = (
            (
                "answer_token_f1",
                full_answer,
                answer_per_question(
                    variant_dir
                ),
            ),
            (
                "evidence_f1",
                full_evidence,
                evidence_per_question(
                    variant_dir
                    / "predictions.jsonl"
                ),
            ),
        )

        for (
            metric,
            full_scores,
            ablation_scores,
        ) in score_sets:
            (
                question_ids,
                full_values,
                ablation_values,
            ) = paired_values(
                full_scores,
                ablation_scores,
            )

            differences = (
                full_values
                - ablation_values
            )

            result = paired_bootstrap(
                differences,
                resamples=args.resamples,
                seed=(
                    args.seed + test_index
                ),
            )
            test_index += 1

            test_rows.append(
                {
                    "metric": metric,
                    "comparison": (
                        f"full_c3_minus_{variant}"
                    ),
                    "n_pairs": len(
                        question_ids
                    ),
                    "full_c3_mean": float(
                        full_values.mean()
                    ),
                    "ablation_mean": float(
                        ablation_values.mean()
                    ),
                    "mean_difference": (
                        result[
                            "mean_difference"
                        ]
                    ),
                    "standard_error": (
                        result[
                            "standard_error"
                        ]
                    ),
                    "ci_95_low": (
                        result["ci_95_low"]
                    ),
                    "ci_95_high": (
                        result["ci_95_high"]
                    ),
                    "p_value_raw": (
                        result["p_value_raw"]
                    ),
                    "superiority_probability": (
                        result[
                            "superiority_probability"
                        ]
                    ),
                    "wins": int(
                        np.sum(
                            differences > 1e-12
                        )
                    ),
                    "ties": int(
                        np.sum(
                            np.abs(differences)
                            <= 1e-12
                        )
                    ),
                    "losses": int(
                        np.sum(
                            differences < -1e-12
                        )
                    ),
                }
            )

    holm_adjust(test_rows)

    write_csv(
        args.output_dir
        / "rc85_ablation_paired_bootstrap.csv",
        test_rows,
    )

    manifest = {
        "full_c3_dir": str(
            args.full_c3_dir
        ),
        "ablation_dir": str(
            args.ablation_dir
        ),
        "variants": list(VARIANTS),
        "resamples": args.resamples,
        "seed": args.seed,
        "confidence_interval": (
            "95% percentile paired bootstrap"
        ),
        "hypothesis_test": (
            "two-sided centered paired "
            "bootstrap"
        ),
        "multiple_testing": (
            "Holm-Bonferroni across "
            "8 primary comparisons"
        ),
    }

    (
        args.output_dir
        / "rc85_ablation_analysis_manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        args.output_dir
        / "rc85_ablation_main_results.csv"
    )
    print(
        args.output_dir
        / "rc85_ablation_paired_bootstrap.csv"
    )


if __name__ == "__main__":
    main()
