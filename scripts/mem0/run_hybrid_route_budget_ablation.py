from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

from src.backbones import MockBackbone
from src.config import load_config
from src.pipeline import C3Pipeline
from src.schemas import QueryState


VARIANTS = (
    "base_hybrid",
    "all_route",
    "expanded_budget",
    "all_route_expanded_budget",
)


def load_runner(path: Path):
    spec = importlib.util.spec_from_file_location(
        "hybrid_runner_module",
        path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load runner module: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_jsonl(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )


def configure_variant(
    base_config: dict[str, Any],
    variant: str,
    budget_multiplier: int,
) -> dict[str, Any]:
    config = deepcopy(base_config)

    force_all_route = (
        variant
        in {
            "all_route",
            "all_route_expanded_budget",
        }
    )

    ablation = config.setdefault(
        "ablation",
        {},
    )
    ablation.update(
        {
            "variant": (
                "no_route_planner"
                if force_all_route
                else "full_c3"
            ),
            "disable_route_planner": force_all_route,
            "disable_conflict_handling": False,
            "disable_coverage_confidence_gate": False,
            "disable_evidence_selector": False,
        }
    )

    if variant in {
        "expanded_budget",
        "all_route_expanded_budget",
    }:
        top_k = config["retrieval"]["top_k"]

        for memory_type, value in list(
            top_k.items()
        ):
            top_k[memory_type] = (
                int(value)
                * budget_multiplier
            )

    return config


def aggregate(
    rows: list[dict[str, Any]],
    variant: str,
) -> dict[str, Any]:
    metrics = [
        row["variants"][variant]["metrics"]
        for row in rows
    ]

    question_count = len(metrics)

    return {
        "questions": question_count,
        "mean_precision": sum(
            float(item["precision"])
            for item in metrics
        )
        / question_count,
        "mean_recall": sum(
            float(item["recall"])
            for item in metrics
        )
        / question_count,
        "mean_f1": sum(
            float(item["f1"])
            for item in metrics
        )
        / question_count,
        "hit_rate": sum(
            int(bool(item["hit"]))
            for item in metrics
        )
        / question_count,
        "mean_selected": sum(
            len(
                row["variants"][variant][
                    "selected_ids"
                ]
            )
            for row in rows
        )
        / question_count,
        "mean_ranked": sum(
            len(
                row["variants"][variant][
                    "ranked_ids"
                ]
            )
            for row in rows
        )
        / question_count,
        "mean_actual_raw": sum(
            len(
                row["variants"][variant][
                    "actual_raw_ids"
                ]
            )
            for row in rows
        )
        / question_count,
    }


def paired_vs_base(
    rows: list[dict[str, Any]],
    variant: str,
) -> dict[str, Any]:
    both_hit = 0
    base_only_hit = 0
    variant_only_hit = 0
    neither_hit = 0

    for row in rows:
        base_hit = bool(
            row["variants"]["base_hybrid"][
                "metrics"
            ]["hit"]
        )
        variant_hit = bool(
            row["variants"][variant][
                "metrics"
            ]["hit"]
        )

        if base_hit and variant_hit:
            both_hit += 1
        elif base_hit:
            base_only_hit += 1
        elif variant_hit:
            variant_only_hit += 1
        else:
            neither_hit += 1

    return {
        "both_hit": both_hit,
        "base_only_hit": base_only_hit,
        "variant_only_hit": variant_only_hit,
        "neither_hit": neither_hit,
        "net_rescues": (
            variant_only_hit
            - base_only_hit
        ),
    }


def category_breakdown(
    rows: list[dict[str, Any]],
    variant: str,
) -> dict[str, Any]:
    grouped: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(list)

    for row in rows:
        grouped[
            str(row["question_type"])
        ].append(row)

    output: dict[str, Any] = {}

    for category, items in sorted(
        grouped.items()
    ):
        output[category] = aggregate(
            items,
            variant,
        )

    return output


def build_report(
    summary: dict[str, Any],
) -> str:
    lines = [
        "# Hybrid Route–Budget Ablation — LoCoMo Pilot-200",
        "",
        "## Purpose",
        "",
        (
            "This controller-only ablation tests whether complementary "
            "Mem0 candidates are lost mainly because of memory-type routing "
            "or the fixed per-type candidate budget."
        ),
        "",
        "## Variants",
        "",
        "- `base_hybrid`: current C3 route and current candidate budget.",
        "- `all_route`: all memory types enabled; current budget retained.",
        "- `expanded_budget`: current route retained; per-type top-k multiplied.",
        "- `all_route_expanded_budget`: all memory types plus expanded budget.",
        "",
        "The evidence selector, conflict handling, confidence controller, "
        "maximum selected evidence and token budget remain unchanged.",
        "",
        "## Main results",
        "",
        "| Variant | Precision | Recall | F1 | Hit | Mean selected | Mean ranked |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for variant in VARIANTS:
        row = summary["variants"][variant]

        lines.append(
            f"| {variant} | "
            f"{row['mean_precision']:.4f} | "
            f"{row['mean_recall']:.4f} | "
            f"{row['mean_f1']:.4f} | "
            f"{row['hit_rate']:.4f} | "
            f"{row['mean_selected']:.2f} | "
            f"{row['mean_ranked']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Paired outcomes versus base hybrid",
            "",
            "| Variant | Base only | Variant only | Net rescues |",
            "|---|---:|---:|---:|",
        ]
    )

    for variant in VARIANTS[1:]:
        row = summary[
            "paired_vs_base"
        ][variant]

        lines.append(
            f"| {variant} | "
            f"{row['base_only_hit']} | "
            f"{row['variant_only_hit']} | "
            f"{row['net_rescues']} |"
        )

    lines.extend(
        [
            "",
            "## Recovery on the 25 backend-rescue questions",
            "",
            "| Variant | Hit questions | Hit rate |",
            "|---|---:|---:|",
        ]
    )

    for variant in VARIANTS:
        row = summary[
            "backend_rescue_subset"
        ][variant]

        lines.append(
            f"| {variant} | "
            f"{row['hit_questions']} | "
            f"{row['hit_rate']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Decision rule",
            "",
            (
                "- If `all_route` produces most of the recovery, implement "
                "confidence-aware route expansion rather than permanently "
                "forcing all memory types."
            ),
            (
                "- If `expanded_budget` produces most of the recovery, "
                "implement dynamic per-type budget allocation."
            ),
            (
                "- If the combined variant is substantially better than "
                "either single change, implement both mechanisms jointly."
            ),
            (
                "- This is a diagnostic upper-bound study. The final policy "
                "must be query-adaptive and must not use gold evidence."
            ),
            "",
        ]
    )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a controller-only 2x2 route and candidate-budget "
            "ablation over the C3+Mem0 Pilot-200 union pool."
        )
    )

    parser.add_argument(
        "--runner-module",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--prompt-template",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--questions",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--memories",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--c3-predictions",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--mem0-top20",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--budget-multiplier",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    if args.budget_multiplier < 2:
        raise ValueError(
            "--budget-multiplier must be at least 2"
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    runner = load_runner(
        args.runner_module
    )
    base_config = load_config(
        args.config
    )

    records = runner.load_memory_records(
        args.memories
    )
    catalog, source_map = (
        runner.build_catalog(records)
    )

    questions = runner.read_csv(
        args.questions
    )
    questions_by_id = {
        str(row["question_id"]): row
        for row in questions
    }

    c3_rows = [
        row
        for row in runner.read_jsonl(
            args.c3_predictions
        )
        if str(
            row.get("method", "")
        ).lower()
        in {
            "c3",
            "c3_lite_controller",
        }
    ]
    c3_by_id = {
        str(row["question_id"]): row
        for row in c3_rows
    }

    mem0_rows = runner.read_jsonl(
        args.mem0_top20
    )
    mem0_by_id = {
        str(row["question_id"]): row
        for row in mem0_rows
    }

    question_ids = set(
        questions_by_id
    )

    if len(question_ids) != 200:
        raise AssertionError(
            "Expected exactly 200 questions."
        )

    if set(c3_by_id) != question_ids:
        raise AssertionError(
            "C3 IDs do not match canonical questions."
        )

    if set(mem0_by_id) != question_ids:
        raise AssertionError(
            "Mem0 IDs do not match canonical questions."
        )

    hybrid_pools = {}
    backend_rescue_ids = set()

    for question_id in sorted(
        question_ids
    ):
        c3_ids = runner.split_ids(
            c3_by_id[question_id].get(
                "raw_retrieved_ids"
            )
        )
        mem0_ids = runner.split_ids(
            mem0_by_id[question_id].get(
                "retrieved_memory_ids"
            )
        )

        hybrid_pools[question_id] = (
            runner.make_pool(
                c3_ids,
                mem0_ids,
            )
        )

        canonical = questions_by_id[
            question_id
        ]
        gold_source_ids = set(
            runner.project_ids(
                runner.split_ids(
                    canonical.get(
                        "supporting_memory_ids",
                        "",
                    )
                ),
                source_map,
            )
        )
        c3_source_ids = set(
            runner.project_ids(
                c3_ids,
                source_map,
            )
        )
        mem0_source_ids = set(
            runner.project_ids(
                mem0_ids,
                source_map,
            )
        )

        if (
            not (
                c3_source_ids
                & gold_source_ids
            )
            and (
                mem0_source_ids
                & gold_source_ids
            )
        ):
            backend_rescue_ids.add(
                question_id
            )

    if len(backend_rescue_ids) != 25:
        raise AssertionError(
            "Expected 25 backend-rescue questions, "
            f"found {len(backend_rescue_ids)}."
        )

    pipelines = {}

    for variant in VARIANTS:
        config = configure_variant(
            base_config,
            variant,
            args.budget_multiplier,
        )

        pipelines[variant] = C3Pipeline(
            config=config,
            memory_store=(
                runner.FrozenCandidateStore(
                    catalog=catalog,
                    pools=hybrid_pools,
                    label=variant,
                )
            ),
            procedure_store=None,
            backbone=MockBackbone(),
            prompt_template=(
                args.prompt_template
            ),
        )

    output_rows: list[
        dict[str, Any]
    ] = []

    try:
        for question_id in sorted(
            question_ids
        ):
            c3 = c3_by_id[
                question_id
            ]
            canonical = (
                questions_by_id[
                    question_id
                ]
            )

            gold_memory_ids = (
                runner.split_ids(
                    canonical.get(
                        "supporting_memory_ids",
                        "",
                    )
                )
            )
            gold_source_ids = (
                runner.project_ids(
                    gold_memory_ids,
                    source_map,
                )
            )

            variant_outputs = {}

            for variant, pipeline in (
                pipelines.items()
            ):
                result = pipeline.answer(
                    QueryState(
                        query=str(
                            c3["query"]
                        ),
                        user_id=str(
                            c3["user_id"]
                        ),
                        session_id=(
                            question_id
                        ),
                    )
                )

                selected_ids = list(
                    result.selected_ids
                )
                selected_source_ids = (
                    runner.project_ids(
                        selected_ids,
                        source_map,
                    )
                )

                variant_outputs[
                    variant
                ] = {
                    "route_types": (
                        result.selected_memory_types
                    ),
                    "decision": (
                        result.decision.value
                    ),
                    "actual_raw_ids": list(
                        result.raw_retrieved_ids
                    ),
                    "ranked_ids": list(
                        result.ranked_candidate_ids
                    ),
                    "selected_ids": (
                        selected_ids
                    ),
                    "selected_source_ids": (
                        selected_source_ids
                    ),
                    "metrics": (
                        runner.evidence_metrics(
                            selected_source_ids,
                            gold_source_ids,
                        )
                    ),
                }

            output_rows.append(
                {
                    "question_id": (
                        question_id
                    ),
                    "question_type": (
                        c3.get(
                            "question_type",
                            canonical.get(
                                "question_type",
                                "unknown",
                            ),
                        )
                    ),
                    "query": c3["query"],
                    "user_id": c3["user_id"],
                    "gold_memory_ids": (
                        gold_memory_ids
                    ),
                    "gold_source_ids": (
                        gold_source_ids
                    ),
                    "is_backend_rescue_question": (
                        question_id
                        in backend_rescue_ids
                    ),
                    "union_pool_ids": (
                        hybrid_pools[
                            question_id
                        ].ordered_ids
                    ),
                    "variants": (
                        variant_outputs
                    ),
                }
            )
    finally:
        for pipeline in (
            pipelines.values()
        ):
            pipeline.close()

    variant_summaries = {
        variant: aggregate(
            output_rows,
            variant,
        )
        for variant in VARIANTS
    }

    paired_summaries = {
        variant: paired_vs_base(
            output_rows,
            variant,
        )
        for variant in VARIANTS[1:]
    }

    rescue_rows = [
        row
        for row in output_rows
        if row[
            "is_backend_rescue_question"
        ]
    ]

    rescue_summary = {}

    for variant in VARIANTS:
        hit_questions = sum(
            int(
                bool(
                    row["variants"][
                        variant
                    ]["metrics"]["hit"]
                )
            )
            for row in rescue_rows
        )

        rescue_summary[variant] = {
            "questions": len(
                rescue_rows
            ),
            "hit_questions": (
                hit_questions
            ),
            "hit_rate": (
                hit_questions
                / len(rescue_rows)
            ),
        }

    summary = {
        "experiment": (
            "rc8_8b7_hybrid_route_budget_ablation"
        ),
        "questions": len(
            output_rows
        ),
        "budget_multiplier": (
            args.budget_multiplier
        ),
        "backend_rescue_questions": (
            len(
                backend_rescue_ids
            )
        ),
        "protocol": {
            "candidate_pool": (
                "frozen C3 raw union Mem0 top-20"
            ),
            "generation": (
                "MockBackbone; controller-only"
            ),
            "selector": (
                "unchanged"
            ),
            "evidence_token_budget": (
                base_config[
                    "selection"
                ][
                    "evidence_token_budget"
                ]
            ),
        },
        "variants": (
            variant_summaries
        ),
        "paired_vs_base": (
            paired_summaries
        ),
        "backend_rescue_subset": (
            rescue_summary
        ),
        "category_breakdown": {
            variant: (
                category_breakdown(
                    output_rows,
                    variant,
                )
            )
            for variant in VARIANTS
        },
    }

    write_jsonl(
        args.output_dir
        / "route_budget_ablation.jsonl",
        output_rows,
    )

    (
        args.output_dir
        / "route_budget_ablation_summary.json"
    ).write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    (
        args.output_dir
        / "route_budget_ablation_report.md"
    ).write_text(
        build_report(summary),
        encoding="utf-8",
    )

    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
    )
    print()
    print(
        "HYBRID ROUTE-BUDGET "
        "ABLATION: PASSED"
    )


if __name__ == "__main__":
    main()
