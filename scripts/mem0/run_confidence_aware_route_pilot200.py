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
from src.schemas import MemoryType, QueryFeatures, QueryMode, QueryState, RouteDecision

VARIANTS = ("base_hybrid", "adaptive_route_hybrid", "all_route_hybrid")


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
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


def all_route_config(base: dict[str, Any]) -> dict[str, Any]:
    config = deepcopy(base)
    config.setdefault("ablation", {}).update(
        {
            "variant": "no_route_planner",
            "disable_route_planner": True,
            "disable_conflict_handling": False,
            "disable_coverage_confidence_gate": False,
            "disable_evidence_selector": False,
        }
    )
    return config


class ConfidenceAwareRoutePlanner:
    """Query-only one-type route expansion; never reads gold evidence."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.base = RoutePlanner(config)
        self.thresholds = {
            key: float(value)
            for key, value in config["routing"]["thresholds"].items()
        }
        raw = config.get("routing", {}).get("confidence_aware_expansion", {}) or {}
        self.min_excluded_score = float(raw.get("min_excluded_score", 0.30))
        self.max_route_margin = float(raw.get("max_route_margin", 0.15))
        self.max_threshold_gap = float(raw.get("max_threshold_gap", 0.10))
        self.allow_procedural_without_intent = bool(
            raw.get("allow_procedural_without_intent", False)
        )
        self.last_trace: dict[str, Any] = {}

    def plan(self, features: QueryFeatures) -> RouteDecision:
        original = self.base.plan(features)
        original_types = list(original.selected_types)
        selected_set = set(original_types)
        scores = {
            memory_type: float(original.scores.get(memory_type.value, 0.0))
            for memory_type in MemoryType
        }
        excluded = [item for item in MemoryType if item not in selected_set]
        best_selected_score = max((scores[item] for item in original_types), default=0.0)
        best_excluded = max(excluded, key=lambda item: scores[item], default=None)
        best_excluded_score = scores[best_excluded] if best_excluded is not None else 0.0
        route_margin = (
            best_selected_score - best_excluded_score
            if best_excluded is not None
            else 1.0
        )

        preferred: list[MemoryType] = []
        if MemoryType.SEMANTIC in selected_set and MemoryType.EPISODIC in excluded:
            preferred.append(MemoryType.EPISODIC)
        if MemoryType.EPISODIC in selected_set and MemoryType.SEMANTIC in excluded:
            preferred.append(MemoryType.SEMANTIC)
        if MemoryType.PROCEDURAL in selected_set and MemoryType.SEMANTIC in excluded:
            preferred.append(MemoryType.SEMANTIC)
        if (
            MemoryType.PROCEDURAL in excluded
            and (features.asks_procedure or self.allow_procedural_without_intent)
        ):
            preferred.append(MemoryType.PROCEDURAL)

        remaining = sorted(
            [
                item
                for item in excluded
                if item not in preferred
                and (
                    item != MemoryType.PROCEDURAL
                    or features.asks_procedure
                    or self.allow_procedural_without_intent
                )
            ],
            key=lambda item: scores[item],
            reverse=True,
        )
        candidates = list(dict.fromkeys([*preferred, *remaining]))

        chosen: MemoryType | None = None
        chosen_reasons: list[str] = []
        chosen_gap: float | None = None
        multi_need = len(features.information_needs) >= 2
        atemporal = features.query_mode == QueryMode.ATEMPORAL

        for candidate in candidates:
            score = scores[candidate]
            gap = self.thresholds[candidate.value] - score
            reasons: list[str] = []

            if score < self.min_excluded_score:
                continue
            if route_margin <= self.max_route_margin:
                reasons.append("low_route_margin")
            if atemporal and gap <= self.max_threshold_gap:
                reasons.append("atemporal_near_threshold")
            if multi_need and gap <= self.max_threshold_gap:
                reasons.append("multi_need_near_threshold")
            if (
                candidate == MemoryType.PROCEDURAL
                and features.asks_procedure
                and gap <= self.max_threshold_gap
            ):
                reasons.append("explicit_procedural_intent")

            if reasons:
                chosen = candidate
                chosen_reasons = reasons
                chosen_gap = gap
                break

        added = [chosen] if chosen is not None else []
        expanded_set = selected_set | set(added)
        expanded_types = [item for item in MemoryType if item in expanded_set]
        reasons = deepcopy(original.reasons)
        for item in added:
            reasons.setdefault(item.value, [])
            for reason in chosen_reasons:
                tag = f"confidence_aware_expansion:{reason}"
                if tag not in reasons[item.value]:
                    reasons[item.value].append(tag)

        self.last_trace = {
            "triggered": bool(added),
            "original_route_types": [item.value for item in original_types],
            "expanded_route_types": [item.value for item in expanded_types],
            "added_memory_types": [item.value for item in added],
            "query_mode": features.query_mode.value,
            "information_need_count": len(features.information_needs),
            "best_selected_score": best_selected_score,
            "best_excluded_type": best_excluded.value if best_excluded else None,
            "best_excluded_score": best_excluded_score,
            "route_margin": route_margin,
            "chosen_threshold_gap": chosen_gap,
            "trigger_reasons": chosen_reasons,
        }

        return RouteDecision(
            selected_types=expanded_types,
            scores=dict(original.scores),
            reasons=reasons,
        )


def aggregate(rows: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    selected = [row for row in rows if row["variant"] == variant]
    if not selected:
        raise AssertionError(f"No rows for {variant}")

    def mean(key: str) -> float:
        return sum(float(row[key]) for row in selected) / len(selected)

    return {
        "questions": len(selected),
        "mean_precision": mean("evidence_precision"),
        "mean_recall": mean("evidence_recall"),
        "mean_f1": mean("evidence_f1"),
        "hit_rate": mean("evidence_hit"),
        "mean_selected": mean("selected_count"),
        "mean_ranked": mean("ranked_count"),
        "mean_raw": mean("raw_count"),
    }


def paired_hits(rows: list[dict[str, Any]], variant: str) -> dict[str, int]:
    base = {
        row["question_id"]: bool(row["evidence_hit"])
        for row in rows
        if row["variant"] == "base_hybrid"
    }
    other = {
        row["question_id"]: bool(row["evidence_hit"])
        for row in rows
        if row["variant"] == variant
    }
    both = base_only = other_only = neither = 0
    for question_id in sorted(base):
        if base[question_id] and other[question_id]:
            both += 1
        elif base[question_id]:
            base_only += 1
        elif other[question_id]:
            other_only += 1
        else:
            neither += 1
    return {
        "both_hit": both,
        "base_only_hit": base_only,
        "comparison_only_hit": other_only,
        "neither_hit": neither,
        "net_rescues": other_only - base_only,
    }


def category_breakdown(rows: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["variant"] == variant:
            grouped[str(row["question_type"])].append(row)
    output: dict[str, Any] = {}
    for category, items in sorted(grouped.items()):
        output[category] = {
            "questions": len(items),
            "evidence_f1": sum(float(item["evidence_f1"]) for item in items)
            / len(items),
            "hit_rate": sum(int(bool(item["evidence_hit"])) for item in items)
            / len(items),
        }
    return output


def capture(base: float, adaptive: float, upper: float) -> float:
    denominator = upper - base
    return 0.0 if abs(denominator) < 1e-12 else (adaptive - base) / denominator


def build_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Confidence-Aware Route Expansion — LoCoMo Pilot-200",
        "",
        "## Protocol",
        "",
        "- Candidate pool: frozen C3 raw candidates union Mem0 top-20.",
        "- Adaptive policy uses only query features and route scores.",
        "- At most one excluded memory type may be added.",
        "- Gold evidence is used only for evaluation.",
        "- Ranker, candidate budget, selector and 1200-token budget are unchanged.",
        "",
        "## Results",
        "",
        "| Variant | Precision | Recall | Evidence F1 | Hit | Mean selected | Mean ranked |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        row = summary["variants"][variant]
        lines.append(
            f"| {variant} | {row['mean_precision']:.4f} | "
            f"{row['mean_recall']:.4f} | {row['mean_f1']:.4f} | "
            f"{row['hit_rate']:.4f} | {row['mean_selected']:.2f} | "
            f"{row['mean_ranked']:.2f} |"
        )
    adaptive = summary["adaptive_route"]
    gain = summary["upper_bound_gain_capture"]
    lines.extend(
        [
            "",
            "## Adaptive behaviour",
            "",
            f"- Expanded: {adaptive['expanded_questions']}/{summary['questions']} "
            f"({adaptive['expansion_rate']:.4f}).",
            f"- Added types: {adaptive['added_type_counts']}.",
            f"- Evidence-F1 upper-bound gain captured: {gain['evidence_f1_capture']:.4f}.",
            f"- Hit-rate upper-bound gain captured: {gain['hit_rate_capture']:.4f}.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
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
    config = load_config(args.config)
    records = runner.load_memory_records(args.memories)
    catalog, source_map = runner.build_catalog(records)
    questions = runner.read_csv(args.questions)
    questions_by_id = {str(row["question_id"]): row for row in questions}
    c3_rows = [
        row
        for row in read_jsonl(args.c3_predictions)
        if str(row.get("method", "")).lower() in {"c3", "c3_lite_controller"}
    ]
    c3_by_id = {str(row["question_id"]): row for row in c3_rows}
    mem0_rows = read_jsonl(args.mem0_top20)
    mem0_by_id = {str(row["question_id"]): row for row in mem0_rows}
    question_ids = sorted(questions_by_id)

    if len(question_ids) != 200:
        raise AssertionError(f"Expected 200 questions, found {len(question_ids)}")
    if set(c3_by_id) != set(question_ids) or set(mem0_by_id) != set(question_ids):
        raise AssertionError("Question IDs are not aligned")

    pools = {}
    for question_id in question_ids:
        c3_ids = runner.split_ids(c3_by_id[question_id].get("raw_retrieved_ids"))
        mem0_ids = runner.split_ids(mem0_by_id[question_id].get("retrieved_memory_ids"))
        pools[question_id] = runner.make_pool(c3_ids, mem0_ids)

    base_pipeline = C3Pipeline(
        config=deepcopy(config),
        memory_store=runner.FrozenCandidateStore(
            catalog=catalog, pools=pools, label="base_hybrid"
        ),
        procedure_store=None,
        backbone=MockBackbone(),
        prompt_template=args.prompt_template,
    )
    adaptive_pipeline = C3Pipeline(
        config=deepcopy(config),
        memory_store=runner.FrozenCandidateStore(
            catalog=catalog, pools=pools, label="adaptive_route_hybrid"
        ),
        procedure_store=None,
        backbone=MockBackbone(),
        prompt_template=args.prompt_template,
    )
    adaptive_router = ConfidenceAwareRoutePlanner(config)
    adaptive_pipeline.router = adaptive_router
    upper_pipeline = C3Pipeline(
        config=all_route_config(config),
        memory_store=runner.FrozenCandidateStore(
            catalog=catalog, pools=pools, label="all_route_hybrid"
        ),
        procedure_store=None,
        backbone=MockBackbone(),
        prompt_template=args.prompt_template,
    )
    pipelines = {
        "base_hybrid": base_pipeline,
        "adaptive_route_hybrid": adaptive_pipeline,
        "all_route_hybrid": upper_pipeline,
    }

    rows: list[dict[str, Any]] = []
    expanded_questions = 0
    added_type_counts = Counter()
    trigger_reason_counts = Counter()

    try:
        for question_id in question_ids:
            c3 = c3_by_id[question_id]
            canonical = questions_by_id[question_id]
            gold_memory_ids = runner.split_ids(
                canonical.get("supporting_memory_ids", "")
            )
            gold_source_ids = runner.project_ids(gold_memory_ids, source_map)
            state = QueryState(
                query=str(c3["query"]),
                user_id=str(c3["user_id"]),
                session_id=question_id,
            )

            for variant in VARIANTS:
                result = pipelines[variant].answer(deepcopy(state))
                selected_ids = list(result.selected_ids)
                selected_source_ids = runner.project_ids(selected_ids, source_map)
                metrics = runner.evidence_metrics(selected_source_ids, gold_source_ids)
                trace: dict[str, Any] = {}
                if variant == "adaptive_route_hybrid":
                    trace = deepcopy(adaptive_router.last_trace)
                    if trace.get("triggered"):
                        expanded_questions += 1
                        added_type_counts.update(trace.get("added_memory_types", []))
                        trigger_reason_counts.update(trace.get("trigger_reasons", []))

                rows.append(
                    {
                        "question_id": question_id,
                        "question_type": str(
                            c3.get(
                                "question_type",
                                canonical.get("question_type", "unknown"),
                            )
                        ),
                        "query": c3["query"],
                        "variant": variant,
                        "gold_memory_ids": gold_memory_ids,
                        "gold_source_ids": gold_source_ids,
                        "selected_memory_types": result.selected_memory_types,
                        "route_scores": result.route_scores,
                        "raw_retrieved_ids": list(result.raw_retrieved_ids),
                        "ranked_candidate_ids": list(result.ranked_candidate_ids),
                        "selected_ids": selected_ids,
                        "selected_source_ids": selected_source_ids,
                        "evidence_precision": metrics["precision"],
                        "evidence_recall": metrics["recall"],
                        "evidence_f1": metrics["f1"],
                        "evidence_hit": metrics["hit"],
                        "selected_count": len(selected_ids),
                        "ranked_count": len(result.ranked_candidate_ids),
                        "raw_count": len(result.raw_retrieved_ids),
                        "route_expansion": trace,
                    }
                )
    finally:
        for pipeline in pipelines.values():
            pipeline.close()

    variants = {variant: aggregate(rows, variant) for variant in VARIANTS}
    base = variants["base_hybrid"]
    adaptive = variants["adaptive_route_hybrid"]
    upper = variants["all_route_hybrid"]
    summary = {
        "experiment": "rc8_8b9_confidence_aware_route_expansion_pilot200",
        "questions": len(question_ids),
        "policy": {
            "min_excluded_score": adaptive_router.min_excluded_score,
            "max_route_margin": adaptive_router.max_route_margin,
            "max_threshold_gap": adaptive_router.max_threshold_gap,
            "max_added_types": 1,
            "allow_procedural_without_intent": (
                adaptive_router.allow_procedural_without_intent
            ),
        },
        "variants": variants,
        "adaptive_route": {
            "expanded_questions": expanded_questions,
            "expansion_rate": expanded_questions / len(question_ids),
            "added_type_counts": dict(added_type_counts),
            "trigger_reason_counts": dict(trigger_reason_counts),
        },
        "paired_vs_base": {
            variant: paired_hits(rows, variant)
            for variant in ("adaptive_route_hybrid", "all_route_hybrid")
        },
        "upper_bound_gain_capture": {
            "evidence_f1_capture": capture(
                base["mean_f1"], adaptive["mean_f1"], upper["mean_f1"]
            ),
            "hit_rate_capture": capture(
                base["hit_rate"], adaptive["hit_rate"], upper["hit_rate"]
            ),
        },
        "category_breakdown": {
            variant: category_breakdown(rows, variant)
            for variant in VARIANTS
        },
        "decision_inputs": {
            "adaptive_delta_precision": adaptive["mean_precision"]
            - base["mean_precision"],
            "adaptive_delta_recall": adaptive["mean_recall"]
            - base["mean_recall"],
            "adaptive_delta_evidence_f1": adaptive["mean_f1"]
            - base["mean_f1"],
            "adaptive_delta_hit_rate": adaptive["hit_rate"] - base["hit_rate"],
            "adaptive_delta_mean_selected": adaptive["mean_selected"]
            - base["mean_selected"],
        },
    }

    write_jsonl(args.output_dir / "predictions.jsonl", rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "report.md").write_text(
        build_report(summary), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print()
    print("CONFIDENCE-AWARE ROUTE EXPANSION PILOT: PASSED")


if __name__ == "__main__":
    main()
