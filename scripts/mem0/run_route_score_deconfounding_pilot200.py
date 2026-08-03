from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

from src.backbones import MockBackbone
from src.config import load_config
from src.pipeline import C3Pipeline
from src.route_planner import RoutePlanner
from src.schemas import MemoryType, QueryFeatures, QueryState, RouteDecision


VARIANTS = (
    "base_hybrid",
    "adaptive_preserve_scores",
    "adaptive_threshold_floor",
    "all_types_preserve_scores",
    "all_route_unit_scores",
)


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(
        module_name,
        path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        value = json.loads(line)

        if not isinstance(value, dict):
            raise TypeError(
                f"{path}:{line_number} is not a JSON object"
            )

        rows.append(value)

    return rows


def write_jsonl(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False) + "\n"
            )


def configure_all_route(
    base_config: dict[str, Any],
) -> dict[str, Any]:
    config = deepcopy(base_config)
    ablation = config.setdefault("ablation", {})
    ablation.update(
        {
            "variant": "no_route_planner",
            "disable_route_planner": True,
            "disable_conflict_handling": False,
            "disable_coverage_confidence_gate": False,
            "disable_evidence_selector": False,
        }
    )
    return config


class AllTypesPreserveScoresPlanner:
    """Expose all memory types while preserving original route scores."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.base = RoutePlanner(config)
        self.last_trace: dict[str, Any] = {}

    def plan(
        self,
        features: QueryFeatures,
    ) -> RouteDecision:
        original = self.base.plan(features)
        original_types = list(original.selected_types)
        expanded_types = list(MemoryType)
        reasons = deepcopy(original.reasons)

        for memory_type in MemoryType:
            if memory_type not in original_types:
                reasons.setdefault(memory_type.value, [])
                reasons[memory_type.value].append(
                    "deconfounding:all_types_preserve_scores"
                )

        self.last_trace = {
            "original_route_types": [
                item.value for item in original_types
            ],
            "expanded_route_types": [
                item.value for item in expanded_types
            ],
            "added_memory_types": [
                item.value
                for item in expanded_types
                if item not in original_types
            ],
            "original_scores": dict(original.scores),
            "final_scores": dict(original.scores),
        }

        return RouteDecision(
            selected_types=expanded_types,
            scores=dict(original.scores),
            reasons=reasons,
        )


class AdaptiveThresholdFloorPlanner:
    """Apply the B9 adaptive policy and floor added-type scores at threshold."""

    def __init__(
        self,
        config: dict[str, Any],
        policy_module: Any,
    ) -> None:
        self.inner = policy_module.ConfidenceAwareRoutePlanner(
            config
        )
        self.thresholds = {
            str(key): float(value)
            for key, value in config["routing"][
                "thresholds"
            ].items()
        }
        self.last_trace: dict[str, Any] = {}

    def plan(
        self,
        features: QueryFeatures,
    ) -> RouteDecision:
        decision = self.inner.plan(features)
        trace = deepcopy(self.inner.last_trace)
        scores = dict(decision.scores)

        for memory_type_name in trace.get(
            "added_memory_types",
            [],
        ):
            scores[memory_type_name] = round(
                max(
                    float(scores.get(memory_type_name, 0.0)),
                    float(self.thresholds[memory_type_name]),
                ),
                6,
            )

        trace["score_policy"] = "threshold_floor"
        trace["final_scores"] = dict(scores)
        self.last_trace = trace

        return RouteDecision(
            selected_types=list(decision.selected_types),
            scores=scores,
            reasons=deepcopy(decision.reasons),
        )


def aggregate(
    rows: list[dict[str, Any]],
    variant: str,
) -> dict[str, Any]:
    subset = [
        row for row in rows
        if row["variant"] == variant
    ]

    if not subset:
        raise AssertionError(f"No rows for {variant}")

    def mean(key: str) -> float:
        return sum(
            float(row[key]) for row in subset
        ) / len(subset)

    return {
        "questions": len(subset),
        "mean_precision": mean("evidence_precision"),
        "mean_recall": mean("evidence_recall"),
        "mean_f1": mean("evidence_f1"),
        "hit_rate": mean("evidence_hit"),
        "mean_selected": mean("selected_count"),
        "mean_ranked": mean("ranked_count"),
        "mean_raw": mean("raw_count"),
    }


def paired_hits(
    rows: list[dict[str, Any]],
    comparison_variant: str,
) -> dict[str, int]:
    base = {
        row["question_id"]: bool(row["evidence_hit"])
        for row in rows
        if row["variant"] == "base_hybrid"
    }
    comparison = {
        row["question_id"]: bool(row["evidence_hit"])
        for row in rows
        if row["variant"] == comparison_variant
    }

    both = 0
    base_only = 0
    comparison_only = 0
    neither = 0

    for question_id in sorted(base):
        base_hit = base[question_id]
        comparison_hit = comparison[question_id]

        if base_hit and comparison_hit:
            both += 1
        elif base_hit:
            base_only += 1
        elif comparison_hit:
            comparison_only += 1
        else:
            neither += 1

    return {
        "both_hit": both,
        "base_only_hit": base_only,
        "comparison_only_hit": comparison_only,
        "neither_hit": neither,
        "net_rescues": comparison_only - base_only,
    }


def category_breakdown(
    rows: list[dict[str, Any]],
    variant: str,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        if row["variant"] == variant:
            grouped[str(row["question_type"])].append(row)

    output: dict[str, Any] = {}

    for category, items in sorted(grouped.items()):
        output[category] = {
            "questions": len(items),
            "mean_f1": sum(
                float(item["evidence_f1"])
                for item in items
            )
            / len(items),
            "hit_rate": sum(
                int(bool(item["evidence_hit"]))
                for item in items
            )
            / len(items),
        }

    return output


def metric_delta(
    variants: dict[str, dict[str, Any]],
    *,
    left: str,
    right: str,
    metric: str,
) -> float:
    return (
        float(variants[right][metric])
        - float(variants[left][metric])
    )


def build_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Route-Type Access and Route-Score Deconfounding — Pilot-200",
        "",
        "## Why this experiment is necessary",
        "",
        (
            "The prior `all_route` ablation simultaneously exposed all memory "
            "types and replaced every route score with 1.0. This experiment "
            "separates type-access effects from route-score effects."
        ),
        "",
        "## Variants",
        "",
        "- `base_hybrid`: unchanged C3 route.",
        "- `adaptive_preserve_scores`: B9 one-type expansion, original scores.",
        "- `adaptive_threshold_floor`: same expansion, added type floored at its route threshold.",
        "- `all_types_preserve_scores`: all types exposed, original scores retained.",
        "- `all_route_unit_scores`: all types exposed and all scores set to 1.0.",
        "",
        "## Results",
        "",
        "| Variant | Precision | Recall | F1 | Hit | Selected | Ranked | Raw |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
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
            f"{row['mean_ranked']:.2f} | "
            f"{row['mean_raw']:.2f} |"
        )

    effects = summary["causal_contrasts"]

    lines.extend(
        [
            "",
            "## Deconfounded contrasts",
            "",
            "| Contrast | Evidence F1 delta | Hit delta |",
            "|---|---:|---:|",
        ]
    )

    for label, values in effects.items():
        lines.append(
            f"| {label} | "
            f"{values['delta_f1']:+.4f} | "
            f"{values['delta_hit']:+.4f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation rule",
            "",
            (
                "- A positive `type_access_effect` means excluded memory types "
                "contain useful evidence even without score inflation."
            ),
            (
                "- A positive `score_inflation_effect` means the previous "
                "all-route upper bound was partly driven by replacing route "
                "scores with 1.0."
            ),
            (
                "- A positive `adaptive_floor_effect` supports assigning an "
                "admitted expansion type at least its route threshold."
            ),
            "",
        ]
    )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Separate memory-type access from route-score inflation "
            "in the C3+Mem0 Pilot-200 route experiment."
        )
    )

    parser.add_argument("--runner-module", type=Path, required=True)
    parser.add_argument("--policy-module", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prompt-template", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--memories", type=Path, required=True)
    parser.add_argument("--c3-predictions", type=Path, required=True)
    parser.add_argument("--mem0-top20", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    runner = load_module(
        args.runner_module,
        "hybrid_runner_module",
    )
    policy_module = load_module(
        args.policy_module,
        "confidence_route_policy_module",
    )
    base_config = load_config(args.config)

    records = runner.load_memory_records(args.memories)
    catalog, source_map = runner.build_catalog(records)

    questions = runner.read_csv(args.questions)
    questions_by_id = {
        str(row["question_id"]): row
        for row in questions
    }

    c3_rows = [
        row
        for row in read_jsonl(args.c3_predictions)
        if str(row.get("method", "")).lower()
        in {"c3", "c3_lite_controller"}
    ]
    c3_by_id = {
        str(row["question_id"]): row
        for row in c3_rows
    }

    mem0_rows = read_jsonl(args.mem0_top20)
    mem0_by_id = {
        str(row["question_id"]): row
        for row in mem0_rows
    }

    question_ids = sorted(questions_by_id)

    if len(question_ids) != 200:
        raise AssertionError(
            f"Expected 200 questions, found {len(question_ids)}"
        )

    if set(c3_by_id) != set(question_ids):
        raise AssertionError("C3 question IDs do not align")

    if set(mem0_by_id) != set(question_ids):
        raise AssertionError("Mem0 question IDs do not align")

    hybrid_pools = {}

    for question_id in question_ids:
        c3_ids = runner.split_ids(
            c3_by_id[question_id].get("raw_retrieved_ids")
        )
        mem0_ids = runner.split_ids(
            mem0_by_id[question_id].get("retrieved_memory_ids")
        )
        hybrid_pools[question_id] = runner.make_pool(
            c3_ids,
            mem0_ids,
        )

    base_pipeline = C3Pipeline(
        config=deepcopy(base_config),
        memory_store=runner.FrozenCandidateStore(
            catalog=catalog,
            pools=hybrid_pools,
            label="base_hybrid",
        ),
        procedure_store=None,
        backbone=MockBackbone(),
        prompt_template=args.prompt_template,
    )

    adaptive_preserve_router = (
        policy_module.ConfidenceAwareRoutePlanner(
            base_config
        )
    )
    adaptive_preserve_pipeline = C3Pipeline(
        config=deepcopy(base_config),
        memory_store=runner.FrozenCandidateStore(
            catalog=catalog,
            pools=hybrid_pools,
            label="adaptive_preserve_scores",
        ),
        procedure_store=None,
        backbone=MockBackbone(),
        prompt_template=args.prompt_template,
    )
    adaptive_preserve_pipeline.router = adaptive_preserve_router

    adaptive_floor_router = AdaptiveThresholdFloorPlanner(
        base_config,
        policy_module,
    )
    adaptive_floor_pipeline = C3Pipeline(
        config=deepcopy(base_config),
        memory_store=runner.FrozenCandidateStore(
            catalog=catalog,
            pools=hybrid_pools,
            label="adaptive_threshold_floor",
        ),
        procedure_store=None,
        backbone=MockBackbone(),
        prompt_template=args.prompt_template,
    )
    adaptive_floor_pipeline.router = adaptive_floor_router

    all_preserve_router = AllTypesPreserveScoresPlanner(
        base_config
    )
    all_preserve_pipeline = C3Pipeline(
        config=deepcopy(base_config),
        memory_store=runner.FrozenCandidateStore(
            catalog=catalog,
            pools=hybrid_pools,
            label="all_types_preserve_scores",
        ),
        procedure_store=None,
        backbone=MockBackbone(),
        prompt_template=args.prompt_template,
    )
    all_preserve_pipeline.router = all_preserve_router

    all_unit_pipeline = C3Pipeline(
        config=configure_all_route(base_config),
        memory_store=runner.FrozenCandidateStore(
            catalog=catalog,
            pools=hybrid_pools,
            label="all_route_unit_scores",
        ),
        procedure_store=None,
        backbone=MockBackbone(),
        prompt_template=args.prompt_template,
    )

    pipelines = {
        "base_hybrid": base_pipeline,
        "adaptive_preserve_scores": adaptive_preserve_pipeline,
        "adaptive_threshold_floor": adaptive_floor_pipeline,
        "all_types_preserve_scores": all_preserve_pipeline,
        "all_route_unit_scores": all_unit_pipeline,
    }

    routers = {
        "adaptive_preserve_scores": adaptive_preserve_router,
        "adaptive_threshold_floor": adaptive_floor_router,
        "all_types_preserve_scores": all_preserve_router,
    }

    rows: list[dict[str, Any]] = []
    expansion_counts = Counter()
    added_type_counts: dict[str, Counter[str]] = defaultdict(Counter)

    try:
        for question_index, question_id in enumerate(
            question_ids,
            start=1,
        ):
            c3 = c3_by_id[question_id]
            canonical = questions_by_id[question_id]

            gold_memory_ids = runner.split_ids(
                canonical.get("supporting_memory_ids", "")
            )
            gold_source_ids = runner.project_ids(
                gold_memory_ids,
                source_map,
            )

            for variant in VARIANTS:
                result = pipelines[variant].answer(
                    QueryState(
                        query=str(c3["query"]),
                        user_id=str(c3["user_id"]),
                        session_id=question_id,
                    )
                )

                selected_ids = list(result.selected_ids)
                selected_source_ids = runner.project_ids(
                    selected_ids,
                    source_map,
                )
                evidence = runner.evidence_metrics(
                    selected_source_ids,
                    gold_source_ids,
                )

                trace: dict[str, Any] = {}
                if variant in routers:
                    trace = deepcopy(
                        routers[variant].last_trace
                    )
                    if trace.get("triggered"):
                        expansion_counts[variant] += 1
                    added_type_counts[variant].update(
                        trace.get("added_memory_types", [])
                    )

                rows.append(
                    {
                        "question_id": question_id,
                        "question_index": question_index,
                        "question_type": str(
                            c3.get(
                                "question_type",
                                canonical.get(
                                    "question_type",
                                    "unknown",
                                ),
                            )
                        ),
                        "variant": variant,
                        "query": c3["query"],
                        "gold_memory_ids": gold_memory_ids,
                        "gold_source_ids": gold_source_ids,
                        "selected_memory_types": (
                            result.selected_memory_types
                        ),
                        "route_scores": result.route_scores,
                        "raw_retrieved_ids": list(
                            result.raw_retrieved_ids
                        ),
                        "ranked_candidate_ids": list(
                            result.ranked_candidate_ids
                        ),
                        "selected_ids": selected_ids,
                        "selected_source_ids": selected_source_ids,
                        "evidence_precision": evidence["precision"],
                        "evidence_recall": evidence["recall"],
                        "evidence_f1": evidence["f1"],
                        "evidence_hit": evidence["hit"],
                        "selected_count": len(selected_ids),
                        "ranked_count": len(
                            result.ranked_candidate_ids
                        ),
                        "raw_count": len(
                            result.raw_retrieved_ids
                        ),
                        "route_trace": trace,
                    }
                )
    finally:
        for pipeline in pipelines.values():
            pipeline.close()

    variants = {
        variant: aggregate(rows, variant)
        for variant in VARIANTS
    }

    contrasts = {
        "type_access_effect": {
            "left": "base_hybrid",
            "right": "all_types_preserve_scores",
        },
        "score_inflation_effect_given_all_types": {
            "left": "all_types_preserve_scores",
            "right": "all_route_unit_scores",
        },
        "adaptive_floor_effect": {
            "left": "adaptive_preserve_scores",
            "right": "adaptive_threshold_floor",
        },
        "adaptive_total_effect": {
            "left": "base_hybrid",
            "right": "adaptive_threshold_floor",
        },
    }

    causal_contrasts = {}

    for label, comparison in contrasts.items():
        causal_contrasts[label] = {
            **comparison,
            "delta_f1": metric_delta(
                variants,
                left=comparison["left"],
                right=comparison["right"],
                metric="mean_f1",
            ),
            "delta_hit": metric_delta(
                variants,
                left=comparison["left"],
                right=comparison["right"],
                metric="hit_rate",
            ),
            "delta_precision": metric_delta(
                variants,
                left=comparison["left"],
                right=comparison["right"],
                metric="mean_precision",
            ),
            "delta_recall": metric_delta(
                variants,
                left=comparison["left"],
                right=comparison["right"],
                metric="mean_recall",
            ),
        }

    summary = {
        "experiment": (
            "rc8_8b10_route_score_deconfounding_pilot200"
        ),
        "questions": len(question_ids),
        "variants": variants,
        "route_behaviour": {
            variant: {
                "expanded_questions": int(
                    expansion_counts.get(variant, 0)
                ),
                "added_type_counts": dict(
                    added_type_counts.get(variant, Counter())
                ),
            }
            for variant in routers
        },
        "paired_vs_base": {
            variant: paired_hits(rows, variant)
            for variant in VARIANTS[1:]
        },
        "causal_contrasts": causal_contrasts,
        "category_breakdown": {
            variant: category_breakdown(rows, variant)
            for variant in VARIANTS
        },
    }

    write_jsonl(
        args.output_dir / "predictions.jsonl",
        rows,
    )

    (
        args.output_dir / "summary.json"
    ).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )

    (
        args.output_dir / "report.md"
    ).write_text(
        build_report(summary),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print()
    print("ROUTE SCORE DECONFOUNDING PILOT: PASSED")


if __name__ == "__main__":
    main()
