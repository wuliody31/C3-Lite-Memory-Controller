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
from src.schemas import MemoryCandidate, MemoryType, QueryFeatures, QueryState, RouteDecision


VARIANTS = (
    "hard_route",
    "soft_probe_1",
    "soft_probe_3",
    "soft_probe_5",
    "soft_companion_full",
    "all_types_full",
)

PROBE_BUDGETS: dict[str, int | None] = {
    "soft_probe_1": 1,
    "soft_probe_3": 3,
    "soft_probe_5": 5,
    "soft_companion_full": None,
}


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
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
            raise TypeError(f"{path}:{line_number} is not a JSON object")
        rows.append(value)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


class CompanionSoftRoutePlanner:
    """Add one non-procedural companion type while preserving route scores."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.base = RoutePlanner(config)
        self.last_trace: dict[str, Any] = {}

    def plan(self, features: QueryFeatures) -> RouteDecision:
        original = self.base.plan(features)
        original_types = list(original.selected_types)
        original_set = set(original_types)

        added: list[MemoryType] = []
        reason: str | None = None

        if (
            MemoryType.SEMANTIC in original_set
            and MemoryType.EPISODIC not in original_set
        ):
            added = [MemoryType.EPISODIC]
            reason = "semantic_to_episodic_companion"
        elif (
            MemoryType.EPISODIC in original_set
            and MemoryType.SEMANTIC not in original_set
        ):
            added = [MemoryType.SEMANTIC]
            reason = "episodic_to_semantic_companion"
        elif (
            MemoryType.PROCEDURAL in original_set
            and MemoryType.SEMANTIC not in original_set
        ):
            added = [MemoryType.SEMANTIC]
            reason = "procedural_to_semantic_companion"

        expanded_set = original_set | set(added)
        expanded_types = [
            memory_type
            for memory_type in MemoryType
            if memory_type in expanded_set
        ]

        reasons = deepcopy(original.reasons)
        for memory_type in added:
            reasons.setdefault(memory_type.value, [])
            tag = f"soft_route_probe:{reason}"
            if tag not in reasons[memory_type.value]:
                reasons[memory_type.value].append(tag)

        self.last_trace = {
            "triggered": bool(added),
            "original_route_types": [item.value for item in original_types],
            "expanded_route_types": [item.value for item in expanded_types],
            "added_memory_types": [item.value for item in added],
            "reason": reason,
            "route_scores": dict(original.scores),
        }

        return RouteDecision(
            selected_types=expanded_types,
            scores=dict(original.scores),
            reasons=reasons,
        )


class AllTypesPreserveScoresPlanner:
    """Expose all memory types while keeping original route scores."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.base = RoutePlanner(config)
        self.last_trace: dict[str, Any] = {}

    def plan(self, features: QueryFeatures) -> RouteDecision:
        original = self.base.plan(features)
        original_types = list(original.selected_types)
        reasons = deepcopy(original.reasons)

        for memory_type in MemoryType:
            if memory_type not in original_types:
                reasons.setdefault(memory_type.value, [])
                reasons[memory_type.value].append(
                    "soft_route_probe:all_types_full"
                )

        self.last_trace = {
            "triggered": True,
            "original_route_types": [item.value for item in original_types],
            "expanded_route_types": [item.value for item in MemoryType],
            "added_memory_types": [
                item.value for item in MemoryType if item not in original_types
            ],
            "reason": "all_types_full",
            "route_scores": dict(original.scores),
        }

        return RouteDecision(
            selected_types=list(MemoryType),
            scores=dict(original.scores),
            reasons=reasons,
        )


class ProbeBudgetStore:
    """Replay full original-route pools plus a limited companion-type probe."""

    def __init__(
        self,
        *,
        catalog: dict[str, MemoryCandidate],
        pools: dict[str, Any],
        label: str,
        config: dict[str, Any],
        probe_budget: int | None,
    ) -> None:
        if probe_budget is not None and probe_budget <= 0:
            raise ValueError("probe_budget must be positive or None")
        self.catalog = catalog
        self.pools = pools
        self.label = label
        self.probe_budget = probe_budget
        self.base_router = RoutePlanner(config)
        self.retrieval_trace: dict[tuple[str, str], dict[str, Any]] = {}

    def retrieve(
        self,
        *,
        memory_type: MemoryType,
        state: QueryState,
        features: QueryFeatures,
        top_k: int,
        include_archived: bool = False,
    ) -> list[MemoryCandidate]:
        del top_k, include_archived

        question_id = state.session_id
        if not question_id:
            raise ValueError(
                "Probe replay requires QueryState.session_id=question_id"
            )
        if question_id not in self.pools:
            raise KeyError(f"No frozen pool for {question_id}")

        original_route = self.base_router.plan(features)
        is_original_type = memory_type in original_route.selected_types
        limit = None if is_original_type else self.probe_budget

        spec = self.pools[question_id]
        output: list[MemoryCandidate] = []
        available_count = 0

        for memory_id in spec.ordered_ids:
            template = self.catalog.get(memory_id)
            if template is None:
                raise KeyError(
                    f"{self.label}: memory {memory_id!r} is absent from catalog"
                )
            if template.memory_type != memory_type:
                continue
            if template.user_id and template.user_id != state.user_id:
                raise AssertionError(
                    f"{question_id}: user mismatch for {memory_id}"
                )

            available_count += 1
            if limit is not None and len(output) >= limit:
                continue

            candidate = deepcopy(template)
            sources = spec.source_by_id.get(memory_id, [])
            candidate.metadata["candidate_provider"] = self.label
            candidate.metadata["retrieval_sources"] = sources
            candidate.metadata["retrieved_by_c3"] = "c3" in sources
            candidate.metadata["retrieved_by_mem0"] = "mem0" in sources
            candidate.metadata["soft_route_probe"] = not is_original_type
            candidate.metadata["soft_route_probe_budget"] = limit
            candidate.metadata["original_route_type"] = is_original_type
            output.append(candidate)

        self.retrieval_trace[(question_id, memory_type.value)] = {
            "memory_type": memory_type.value,
            "is_original_type": is_original_type,
            "probe_budget": limit,
            "available_count": available_count,
            "returned_count": len(output),
            "returned_ids": [item.memory_id for item in output],
        }
        return output

    def trace_for_question(self, question_id: str) -> dict[str, Any]:
        return {
            memory_type.value: deepcopy(
                self.retrieval_trace[(question_id, memory_type.value)]
            )
            for memory_type in MemoryType
            if (question_id, memory_type.value) in self.retrieval_trace
        }

    def close(self) -> None:
        return None


def aggregate(rows: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    subset = [row for row in rows if row["variant"] == variant]
    if not subset:
        raise AssertionError(f"No rows for {variant}")

    def mean(key: str) -> float:
        return sum(float(row[key]) for row in subset) / len(subset)

    return {
        "questions": len(subset),
        "mean_precision": mean("evidence_precision"),
        "mean_recall": mean("evidence_recall"),
        "mean_f1": mean("evidence_f1"),
        "hit_rate": mean("evidence_hit"),
        "mean_selected": mean("selected_count"),
        "mean_ranked": mean("ranked_count"),
        "mean_raw": mean("raw_count"),
        "mean_probe_raw": mean("probe_raw_count"),
    }


def paired_hits(
    rows: list[dict[str, Any]],
    comparison_variant: str,
) -> dict[str, int]:
    base = {
        row["question_id"]: bool(row["evidence_hit"])
        for row in rows
        if row["variant"] == "hard_route"
    }
    comparison = {
        row["question_id"]: bool(row["evidence_hit"])
        for row in rows
        if row["variant"] == comparison_variant
    }

    both = base_only = comparison_only = neither = 0
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
            "mean_f1": sum(float(item["evidence_f1"]) for item in items)
            / len(items),
            "hit_rate": sum(int(bool(item["evidence_hit"])) for item in items)
            / len(items),
            "mean_raw": sum(int(item["raw_count"]) for item in items)
            / len(items),
        }
    return output


def safe_capture(*, base: float, candidate: float, upper: float) -> float:
    denominator = upper - base
    if abs(denominator) < 1e-12:
        return 0.0
    return (candidate - base) / denominator


def rescue_capture(
    rows: list[dict[str, Any]],
    variant: str,
) -> dict[str, Any]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[row["question_id"]][row["variant"]] = row

    upper_rescues = {
        question_id
        for question_id, variants in grouped.items()
        if (
            not bool(variants["hard_route"]["evidence_hit"])
            and bool(variants["all_types_full"]["evidence_hit"])
        )
    }
    captured = {
        question_id
        for question_id in upper_rescues
        if bool(grouped[question_id][variant]["evidence_hit"])
    }

    return {
        "upper_bound_rescue_questions": len(upper_rescues),
        "captured_rescue_questions": len(captured),
        "capture_rate": (
            len(captured) / len(upper_rescues) if upper_rescues else 0.0
        ),
        "captured_question_ids": sorted(captured),
        "missed_question_ids": sorted(upper_rescues - captured),
    }


def build_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Soft Route Admission with Asymmetric Probe Budget — Pilot-200",
        "",
        "## Protocol",
        "",
        "- `hard_route`: unchanged C3 type routing.",
        "- `soft_probe_1/3/5`: full original-route pool plus 1, 3, or 5 "
        "candidates from one complementary type.",
        "- `soft_companion_full`: complete complementary-type pool.",
        "- `all_types_full`: all memory types with original route scores.",
        "- No gold evidence is used by the runtime policy.",
        "- Ranker, candidate budget, selector and 1200-token evidence budget "
        "remain unchanged.",
        "",
        "## Results",
        "",
        "| Variant | Precision | Recall | F1 | Hit | Selected | Ranked | Raw | Probe raw |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for variant in VARIANTS:
        row = summary["variants"][variant]
        lines.append(
            f"| {variant} | {row['mean_precision']:.4f} | "
            f"{row['mean_recall']:.4f} | {row['mean_f1']:.4f} | "
            f"{row['hit_rate']:.4f} | {row['mean_selected']:.2f} | "
            f"{row['mean_ranked']:.2f} | {row['mean_raw']:.2f} | "
            f"{row['mean_probe_raw']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Gain and rescue capture",
            "",
            "| Variant | F1 capture | Hit capture | Rescue capture | Net rescues |",
            "|---|---:|---:|---:|---:|",
        ]
    )

    for variant in VARIANTS[1:-1]:
        gain = summary["upper_bound_gain_capture"][variant]
        rescue = summary["upper_bound_rescue_capture"][variant]
        paired = summary["paired_vs_hard"][variant]
        lines.append(
            f"| {variant} | {gain['evidence_f1_capture']:.4f} | "
            f"{gain['hit_rate_capture']:.4f} | "
            f"{rescue['capture_rate']:.4f} | {paired['net_rescues']} |"
        )

    lines.extend(
        [
            "",
            "## Decision rule",
            "",
            (
                "Prefer the smallest probe budget that captures most of the "
                "all-types Evidence-F1 gain, yields positive net rescues, "
                "keeps final selected evidence stable, and reduces raw/ranked "
                "candidates relative to full soft routing."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate asymmetric soft-route probe budgets on the frozen "
            "C3+Mem0 LoCoMo Pilot-200 pool."
        )
    )
    parser.add_argument("--runner-module", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prompt-template", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--memories", type=Path, required=True)
    parser.add_argument("--c3-predictions", type=Path, required=True)
    parser.add_argument("--mem0-top20", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    runner = load_module(args.runner_module, "hybrid_runner_module")
    base_config = load_config(args.config)

    records = runner.load_memory_records(args.memories)
    catalog, source_map = runner.build_catalog(records)

    questions = runner.read_csv(args.questions)
    questions_by_id = {
        str(row["question_id"]): row for row in questions
    }

    c3_rows = [
        row
        for row in read_jsonl(args.c3_predictions)
        if str(row.get("method", "")).lower()
        in {"c3", "c3_lite_controller"}
    ]
    c3_by_id = {
        str(row["question_id"]): row for row in c3_rows
    }

    mem0_rows = read_jsonl(args.mem0_top20)
    mem0_by_id = {
        str(row["question_id"]): row for row in mem0_rows
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

    pools = {}
    for question_id in question_ids:
        c3_ids = runner.split_ids(
            c3_by_id[question_id].get("raw_retrieved_ids")
        )
        mem0_ids = runner.split_ids(
            mem0_by_id[question_id].get("retrieved_memory_ids")
        )
        pools[question_id] = runner.make_pool(c3_ids, mem0_ids)

    pipelines: dict[str, C3Pipeline] = {
        "hard_route": C3Pipeline(
            config=deepcopy(base_config),
            memory_store=runner.FrozenCandidateStore(
                catalog=catalog,
                pools=pools,
                label="hard_route",
            ),
            procedure_store=None,
            backbone=MockBackbone(),
            prompt_template=args.prompt_template,
        )
    }
    routers: dict[str, Any] = {}
    stores: dict[str, ProbeBudgetStore] = {}

    for variant, probe_budget in PROBE_BUDGETS.items():
        router = CompanionSoftRoutePlanner(base_config)
        store = ProbeBudgetStore(
            catalog=catalog,
            pools=pools,
            label=variant,
            config=base_config,
            probe_budget=probe_budget,
        )
        pipeline = C3Pipeline(
            config=deepcopy(base_config),
            memory_store=store,
            procedure_store=None,
            backbone=MockBackbone(),
            prompt_template=args.prompt_template,
        )
        pipeline.router = router
        pipelines[variant] = pipeline
        routers[variant] = router
        stores[variant] = store

    all_router = AllTypesPreserveScoresPlanner(base_config)
    all_pipeline = C3Pipeline(
        config=deepcopy(base_config),
        memory_store=runner.FrozenCandidateStore(
            catalog=catalog,
            pools=pools,
            label="all_types_full",
        ),
        procedure_store=None,
        backbone=MockBackbone(),
        prompt_template=args.prompt_template,
    )
    all_pipeline.router = all_router
    pipelines["all_types_full"] = all_pipeline
    routers["all_types_full"] = all_router

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

                route_trace: dict[str, Any] = {}
                if variant in routers:
                    route_trace = deepcopy(
                        routers[variant].last_trace
                    )
                    if route_trace.get("triggered"):
                        expansion_counts[variant] += 1
                    added_type_counts[variant].update(
                        route_trace.get("added_memory_types", [])
                    )

                retrieval_trace = (
                    stores[variant].trace_for_question(question_id)
                    if variant in stores
                    else {}
                )
                probe_raw_count = sum(
                    int(trace.get("returned_count", 0))
                    for trace in retrieval_trace.values()
                    if not bool(trace.get("is_original_type", True))
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
                        "route_trace": route_trace,
                        "retrieval_trace": retrieval_trace,
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
                        "probe_raw_count": probe_raw_count,
                    }
                )
    finally:
        for pipeline in pipelines.values():
            pipeline.close()

    variants = {
        variant: aggregate(rows, variant)
        for variant in VARIANTS
    }
    hard = variants["hard_route"]
    upper = variants["all_types_full"]

    gain_capture = {}
    rescue_captures = {}
    for variant in VARIANTS[1:-1]:
        candidate = variants[variant]
        gain_capture[variant] = {
            "evidence_f1_capture": safe_capture(
                base=hard["mean_f1"],
                candidate=candidate["mean_f1"],
                upper=upper["mean_f1"],
            ),
            "hit_rate_capture": safe_capture(
                base=hard["hit_rate"],
                candidate=candidate["hit_rate"],
                upper=upper["hit_rate"],
            ),
        }
        rescue_captures[variant] = rescue_capture(rows, variant)

    summary = {
        "experiment": "rc8_8b11_soft_route_probe_budget_pilot200",
        "questions": len(question_ids),
        "variants": variants,
        "route_behaviour": {
            variant: {
                "expanded_questions": int(
                    expansion_counts.get(variant, 0)
                ),
                "expansion_rate": (
                    expansion_counts.get(variant, 0)
                    / len(question_ids)
                ),
                "added_type_counts": dict(
                    added_type_counts.get(variant, Counter())
                ),
            }
            for variant in routers
        },
        "paired_vs_hard": {
            variant: paired_hits(rows, variant)
            for variant in VARIANTS[1:]
        },
        "upper_bound_gain_capture": gain_capture,
        "upper_bound_rescue_capture": rescue_captures,
        "category_breakdown": {
            variant: category_breakdown(rows, variant)
            for variant in VARIANTS
        },
    }

    write_jsonl(args.output_dir / "predictions.jsonl", rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "report.md").write_text(
        build_report(summary),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print()
    print("SOFT ROUTE PROBE-BUDGET PILOT: PASSED")


if __name__ == "__main__":
    main()
