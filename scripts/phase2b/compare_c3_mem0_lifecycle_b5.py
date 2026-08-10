#!/usr/bin/env python3
"""Paired cluster-aware C3 vs Mem0 lifecycle comparison.

Primary comparison:
- symmetric lifecycle cases only;
- canonical scenario is the resampling unit;
- four surface variants remain within the same cluster.

Temporal out-of-order stress cases are reported descriptively
and excluded from the primary bootstrap analysis.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[2]


DEFAULT_C3 = (
    ROOT
    / "experiments"
    / "phase2b_b5_c3_observable_formal100_eval_v02"
    / "per_case.jsonl"
)

DEFAULT_MEM0 = (
    ROOT
    / "experiments"
    / "phase2b_b5_mem0_lifecycle_formal100_eval_v02"
    / "per_case.jsonl"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument(
        "--c3",
        type=Path,
        default=DEFAULT_C3,
    )

    p.add_argument(
        "--mem0",
        type=Path,
        default=DEFAULT_MEM0,
    )

    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    p.add_argument(
        "--bootstrap",
        type=int,
        default=50000,
    )

    p.add_argument(
        "--seed",
        type=int,
        default=20260807,
    )

    return p.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


def canonical_id(case_id: str) -> str:
    return re.sub(
        r"_v\d+$",
        "",
        case_id,
    )


def percentile(
    values: list[float],
    q: float,
) -> float:
    if not values:
        raise ValueError("Empty percentile input.")

    ordered = sorted(values)

    if len(ordered) == 1:
        return ordered[0]

    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)

    if lo == hi:
        return ordered[lo]

    weight = pos - lo

    return (
        ordered[lo] * (1.0 - weight)
        + ordered[hi] * weight
    )


def paired_cluster_bootstrap(
    cluster_diffs: dict[str, float],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    ids = sorted(cluster_diffs)

    if not ids:
        raise ValueError(
            "No eligible canonical clusters."
        )

    values = [
        cluster_diffs[cid]
        for cid in ids
    ]

    point = mean(values)

    rng = random.Random(seed)

    samples: list[float] = []

    n = len(values)

    for _ in range(iterations):
        draw = [
            values[
                rng.randrange(n)
            ]
            for _ in range(n)
        ]

        samples.append(
            mean(draw)
        )

    return {
        "cluster_count": n,
        "paired_difference":
            point,
        "bootstrap_iterations":
            iterations,
        "bootstrap_ci95": [
            percentile(samples, 0.025),
            percentile(samples, 0.975),
        ],
        "min_cluster_difference":
            min(values),
        "max_cluster_difference":
            max(values),
    }


def bool_num(value: Any) -> float:
    return 1.0 if bool(value) else 0.0


def current_only(row: dict[str, Any]) -> float:
    return (
        1.0
        if row["current_top1_label"]
        == "current_only"
        else 0.0
    )


def stale_only_top1(row: dict[str, Any]) -> float:
    return (
        1.0
        if row["current_top1_label"]
        == "stale_only"
        else 0.0
    )


def mixed_top1(row: dict[str, Any]) -> float:
    return (
        1.0
        if row["current_top1_label"]
        == "mixed_transition"
        else 0.0
    )


METRICS = {
    "current_only_top1_rate": {
        "fn": current_only,
        "eligible":
            lambda row: True,
        "higher_is_better": True,
    },

    "stale_only_top1_rate": {
        "fn": stale_only_top1,
        "eligible":
            lambda row: True,
        "higher_is_better": False,
    },

    "mixed_transition_top1_rate": {
        "fn": mixed_top1,
        "eligible":
            lambda row: True,
        "higher_is_better": False,
    },

    "stale_only_exposure_case_rate": {
        "fn":
            lambda row:
                bool_num(
                    row["stale_only_exposure"]
                ),
        "eligible":
            lambda row:
                bool(row["stale_values"]),
        "higher_is_better": False,
    },

    "mixed_transition_exposure_case_rate": {
        "fn":
            lambda row:
                bool_num(
                    row[
                        "mixed_transition_exposure"
                    ]
                ),
        "eligible":
            lambda row:
                bool(row["stale_values"]),
        "higher_is_better": False,
    },

    "historical_signal_exposure_case_rate": {
        "fn":
            lambda row:
                bool_num(
                    row[
                        "historical_signal_exposure"
                    ]
                ),
        "eligible":
            lambda row:
                bool(row["stale_values"]),
        "higher_is_better": False,
    },

    "previous_value_availability": {
        "fn":
            lambda row:
                bool_num(
                    row[
                        "previous_value_available"
                    ]
                ),
        "eligible":
            lambda row:
                bool(row["previous_gold"]),
        "higher_is_better": True,
    },

    "previous_top1_accuracy": {
        "fn":
            lambda row:
                bool_num(
                    row[
                        "previous_top1_correct"
                    ]
                ),
        "eligible":
            lambda row:
                bool(row["previous_gold"]),
        "higher_is_better": True,
    },

    "history_value_recall": {
        "fn":
            lambda row:
                float(
                    row[
                        "history_value_recall"
                    ]
                ),
        "eligible":
            lambda row:
                bool(row["history_gold"]),
        "higher_is_better": True,
    },
}


def validate_pairs(
    c3_rows: list[dict[str, Any]],
    mem_rows: list[dict[str, Any]],
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    c3 = {
        row["case_id"]: row
        for row in c3_rows
    }

    mem = {
        row["case_id"]: row
        for row in mem_rows
    }

    if set(c3) != set(mem):
        only_c3 = sorted(
            set(c3) - set(mem)
        )
        only_mem = sorted(
            set(mem) - set(c3)
        )

        raise RuntimeError(
            "Case ID mismatch. "
            f"only_c3={only_c3}, "
            f"only_mem0={only_mem}"
        )

    if len(c3) != 100:
        raise RuntimeError(
            f"Expected 100 paired cases, got {len(c3)}."
        )

    invariant_fields = [
        "pattern",
        "state_key",
        "surface_variant",
        "temporal_symmetry",
        "current_gold",
        "stale_values",
        "previous_gold",
        "history_gold",
    ]

    for cid in sorted(c3):
        for field in invariant_fields:
            if c3[cid][field] != mem[cid][field]:
                raise RuntimeError(
                    "Paired benchmark metadata mismatch: "
                    f"{cid} field={field} "
                    f"c3={c3[cid][field]!r} "
                    f"mem0={mem[cid][field]!r}"
                )

    return c3, mem


def build_metric(
    *,
    name: str,
    spec: dict[str, Any],
    c3: dict[str, dict[str, Any]],
    mem: dict[str, dict[str, Any]],
    symmetric: bool,
    bootstrap_iterations: int,
    seed: int,
) -> dict[str, Any]:
    fn: Callable[
        [dict[str, Any]],
        float,
    ] = spec["fn"]

    eligible = spec["eligible"]

    case_ids = [
        cid
        for cid in sorted(c3)
        if (
            bool(
                c3[cid]["temporal_symmetry"]
            )
            == symmetric
        )
        and eligible(c3[cid])
    ]

    c3_values = {
        cid: fn(c3[cid])
        for cid in case_ids
    }

    mem_values = {
        cid: fn(mem[cid])
        for cid in case_ids
    }

    clusters: dict[
        str,
        list[str],
    ] = defaultdict(list)

    for cid in case_ids:
        clusters[
            canonical_id(cid)
        ].append(cid)

    for cluster_id, members in clusters.items():
        variants = sorted(
            int(
                c3[cid][
                    "surface_variant"
                ]
            )
            for cid in members
        )

        if variants != [1, 2, 3, 4]:
            raise RuntimeError(
                "Cluster does not contain exactly "
                "surface variants 1..4: "
                f"{cluster_id}: {variants}"
            )

    c3_cluster_means = {
        cluster:
            mean(
                c3_values[cid]
                for cid in members
            )
        for cluster, members
        in clusters.items()
    }

    mem_cluster_means = {
        cluster:
            mean(
                mem_values[cid]
                for cid in members
            )
        for cluster, members
        in clusters.items()
    }

    cluster_diffs = {
        cluster:
            (
                c3_cluster_means[cluster]
                - mem_cluster_means[cluster]
            )
        for cluster in clusters
    }

    c3_mean = mean(
        c3_values.values()
    )

    mem_mean = mean(
        mem_values.values()
    )

    result = {
        "metric": name,
        "case_count":
            len(case_ids),
        "canonical_cluster_count":
            len(clusters),
        "c3":
            c3_mean,
        "mem0":
            mem_mean,
        "c3_minus_mem0":
            c3_mean - mem_mean,
        "higher_is_better":
            spec["higher_is_better"],
    }

    if symmetric:
        result.update(
            paired_cluster_bootstrap(
                cluster_diffs,
                iterations=(
                    bootstrap_iterations
                ),
                seed=seed,
            )
        )

        result[
            "effect_favouring_c3"
        ] = (
            result["c3_minus_mem0"]
            if spec[
                "higher_is_better"
            ]
            else -result[
                "c3_minus_mem0"
            ]
        )

        ci = result[
            "bootstrap_ci95"
        ]

        if spec["higher_is_better"]:
            fav_ci = ci
        else:
            fav_ci = [
                -ci[1],
                -ci[0],
            ]

        result[
            "effect_favouring_c3_ci95"
        ] = fav_ci

    return result


def main() -> None:
    args = parse_args()

    c3_rows = read_jsonl(
        args.c3
    )

    mem_rows = read_jsonl(
        args.mem0
    )

    c3, mem = validate_pairs(
        c3_rows,
        mem_rows,
    )

    symmetric_results = {}

    temporal_results = {}

    for index, (
        name,
        spec,
    ) in enumerate(
        METRICS.items()
    ):
        symmetric_results[name] = (
            build_metric(
                name=name,
                spec=spec,
                c3=c3,
                mem=mem,
                symmetric=True,
                bootstrap_iterations=(
                    args.bootstrap
                ),
                seed=(
                    args.seed
                    + index
                ),
            )
        )

        temporal_results[name] = (
            build_metric(
                name=name,
                spec=spec,
                c3=c3,
                mem=mem,
                symmetric=False,
                bootstrap_iterations=(
                    args.bootstrap
                ),
                seed=(
                    args.seed
                    + 1000
                    + index
                ),
            )
        )

    symmetric_case_ids = [
        cid
        for cid, row in c3.items()
        if row["temporal_symmetry"]
    ]

    temporal_case_ids = [
        cid
        for cid, row in c3.items()
        if not row["temporal_symmetry"]
    ]

    final_memory = {
        "note":
            "Descriptive only. C3 final_memory_count "
            "covers semantic lifecycle records and does "
            "not include episodic transition records.",

        "symmetric": {
            "c3":
                mean(
                    c3[cid][
                        "final_memory_count"
                    ]
                    for cid
                    in symmetric_case_ids
                ),
            "mem0":
                mean(
                    mem[cid][
                        "final_memory_count"
                    ]
                    for cid
                    in symmetric_case_ids
                ),
        },

        "temporal_stress": {
            "c3":
                mean(
                    c3[cid][
                        "final_memory_count"
                    ]
                    for cid
                    in temporal_case_ids
                ),
            "mem0":
                mean(
                    mem[cid][
                        "final_memory_count"
                    ]
                    for cid
                    in temporal_case_ids
                ),
        },
    }

    summary = {
        "experiment":
            "phase2b_b5_c3_vs_mem0_paired_comparison",

        "primary_analysis":
            "symmetric lifecycle cases",

        "paired_case_count":
            100,

        "symmetric_case_count":
            len(symmetric_case_ids),

        "symmetric_canonical_clusters":
            len(
                {
                    canonical_id(cid)
                    for cid
                    in symmetric_case_ids
                }
            ),

        "temporal_stress_case_count":
            len(temporal_case_ids),

        "temporal_stress_canonical_clusters":
            len(
                {
                    canonical_id(cid)
                    for cid
                    in temporal_case_ids
                }
            ),

        "surface_variants_are_independent":
            False,

        "bootstrap_unit":
            "canonical lifecycle scenario",

        "bootstrap_iterations":
            args.bootstrap,

        "bootstrap_seed":
            args.seed,

        "c3_protocol":
            "phase2b-b5-c3-observable-protocol-v03",

        "mem0_protocol":
            "Mem0 OSS infer=True additive "
            "memory-formation baseline",

        "evaluator":
            "phase2b-b5-lifecycle-observable-evaluator-v02",

        "symmetric":
            symmetric_results,

        "temporal_stress_descriptive":
            temporal_results,

        "final_memory_count_descriptive":
            final_memory,

        "interpretation_constraints": [
            "The primary comparison is symmetric_cases; out-of-order cases are temporal capability stress tests.",
            "The four surface variants within a canonical scenario are treated as correlated observations.",
            "C3 and Mem0 use different upstream memory-formation interfaces.",
            "C3 is evaluated with benchmark-role-conditioned CURRENT/HISTORICAL retrieval.",
            "The comparison therefore supports observable lifecycle-retrieval conclusions, not a claim of fully identical end-to-end inputs.",
            "Confidence intervals quantify variation across the benchmark's canonical scenarios; they should not be interpreted as population-level guarantees.",
        ],
    }

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        args.output_dir
        / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    csv_path = (
        args.output_dir
        / "symmetric_metrics.csv"
    )

    with csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.writer(
            handle,
            lineterminator="\n",
        )

        writer.writerow([
            "metric",
            "cases",
            "clusters",
            "c3",
            "mem0",
            "c3_minus_mem0",
            "higher_is_better",
            "effect_favouring_c3",
            "ci95_low_favouring_c3",
            "ci95_high_favouring_c3",
        ])

        for name, row in (
            symmetric_results.items()
        ):
            writer.writerow([
                name,
                row["case_count"],
                row[
                    "canonical_cluster_count"
                ],
                row["c3"],
                row["mem0"],
                row["c3_minus_mem0"],
                row[
                    "higher_is_better"
                ],
                row[
                    "effect_favouring_c3"
                ],
                row[
                    "effect_favouring_c3_ci95"
                ][0],
                row[
                    "effect_favouring_c3_ci95"
                ][1],
            ])

    report = []

    report.append(
        "# Phase II-B5 C3 vs Mem0 "
        "Lifecycle Retrieval Comparison"
    )

    report.append("")
    report.append(
        "Primary analysis: 80 symmetric observations "
        "grouped into 20 canonical lifecycle-scenario clusters."
    )

    report.append("")
    report.append(
        "| Metric | C3 | Mem0 | C3-Mem0 | "
        "Effect favouring C3 | Cluster-bootstrap 95% CI |"
    )

    report.append(
        "|---|---:|---:|---:|---:|---:|"
    )

    for name, row in (
        symmetric_results.items()
    ):
        ci = row[
            "effect_favouring_c3_ci95"
        ]

        report.append(
            f"| {name} "
            f"| {row['c3']:.4f} "
            f"| {row['mem0']:.4f} "
            f"| {row['c3_minus_mem0']:+.4f} "
            f"| {row['effect_favouring_c3']:+.4f} "
            f"| [{ci[0]:+.4f}, {ci[1]:+.4f}] |"
        )

    report.append("")
    report.append(
        "Temporal out-of-order cases are excluded "
        "from the primary inferential comparison "
        "and retained as descriptive stress tests."
    )

    (
        args.output_dir
        / "report.md"
    ).write_text(
        "\n".join(report)
        + "\n",
        encoding="utf-8",
    )

    print(
        "========================================"
    )
    print(
        "B5.4 C3 vs MEM0 PAIRED COMPARISON"
    )
    print(
        "========================================"
    )

    print(
        "Paired cases:",
        summary["paired_case_count"],
    )

    print(
        "Primary symmetric cases:",
        summary[
            "symmetric_case_count"
        ],
    )

    print(
        "Primary canonical clusters:",
        summary[
            "symmetric_canonical_clusters"
        ],
    )

    print()

    for name, row in (
        symmetric_results.items()
    ):
        ci = row[
            "effect_favouring_c3_ci95"
        ]

        print(name)
        print(
            f"  C3               = "
            f"{row['c3']:.6f}"
        )
        print(
            f"  Mem0              = "
            f"{row['mem0']:.6f}"
        )
        print(
            f"  C3-Mem0           = "
            f"{row['c3_minus_mem0']:+.6f}"
        )
        print(
            f"  effect favour C3  = "
            f"{row['effect_favouring_c3']:+.6f}"
        )
        print(
            "  cluster CI95      = "
            f"[{ci[0]:+.6f}, "
            f"{ci[1]:+.6f}]"
        )
        print(
            f"  cases/clusters    = "
            f"{row['case_count']}/"
            f"{row['canonical_cluster_count']}"
        )
        print()

    print(
        "OUTPUT:",
        args.output_dir,
    )


if __name__ == "__main__":
    main()
