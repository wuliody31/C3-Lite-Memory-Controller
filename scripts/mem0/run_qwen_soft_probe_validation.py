from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import load_config
from src.pipeline import C3Pipeline
from src.schemas import QueryState


VARIANTS = (
    "hard_route",
    "soft_probe_1",
    "soft_probe_3",
)

COMPARISONS = {
    "soft_probe_1_minus_hard": ("hard_route", "soft_probe_1"),
    "soft_probe_3_minus_hard": ("hard_route", "soft_probe_3"),
    "soft_probe_3_minus_soft_probe_1": ("soft_probe_1", "soft_probe_3"),
}


def load_module(path: Path, module_name: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def average(rows: list[dict[str, Any]], field: str) -> float:
    if not rows:
        return 0.0
    return sum(float(row.get(field, 0.0) or 0.0) for row in rows) / len(rows)


def aggregate_rows(qwen: Any, rows: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    summary = qwen.aggregate_rows(rows, variant)
    selected = [row for row in rows if row["variant"] == variant]
    summary.update(
        {
            "mean_raw": average(selected, "raw_count"),
            "mean_ranked": average(selected, "ranked_count"),
            "mean_probe_raw": average(selected, "probe_raw_count"),
        }
    )
    return summary


def paired_deltas(
    rows: list[dict[str, Any]],
    question_ids: list[str],
    *,
    left_variant: str,
    right_variant: str,
    field: str,
) -> list[float]:
    by_variant = {
        variant: {
            str(row["question_id"]): row
            for row in rows
            if row["variant"] == variant
        }
        for variant in (left_variant, right_variant)
    }
    return [
        float(by_variant[right_variant][question_id][field])
        - float(by_variant[left_variant][question_id][field])
        for question_id in question_ids
    ]


def paired_outcomes(deltas: list[float]) -> dict[str, int]:
    better = sum(delta > 0.0 for delta in deltas)
    worse = sum(delta < 0.0 for delta in deltas)
    return {
        "right_better": better,
        "right_worse": worse,
        "tied": len(deltas) - better - worse,
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
            "answer_f1": average(items, "answer_f1"),
            "evidence_f1": average(items, "evidence_f1"),
            "evidence_hit_rate": average(items, "evidence_hit"),
            "mean_input_tokens": average(items, "input_tokens_numeric"),
            "mean_latency_ms": average(items, "latency_ms"),
        }
    return output


def select_variant(paired_statistics: dict[str, Any]) -> dict[str, Any]:
    """Efficiency-first selection rule fixed before reading B12 results."""
    p1_vs_hard = paired_statistics["soft_probe_1_minus_hard"]["answer_f1"]
    p3_vs_hard = paired_statistics["soft_probe_3_minus_hard"]["answer_f1"]
    p3_vs_p1 = paired_statistics["soft_probe_3_minus_soft_probe_1"]["answer_f1"]

    p1_rejected = float(p1_vs_hard["ci_high"]) < 0.0
    p3_rejected = float(p3_vs_hard["ci_high"]) < 0.0
    p3_significantly_better = float(p3_vs_p1["ci_low"]) > 0.0

    if p3_significantly_better and not p3_rejected:
        chosen = "soft_probe_3"
        reason = (
            "Probe-3 is significantly better than Probe-1 for Answer F1 "
            "and is not rejected versus hard route."
        )
    elif not p1_rejected:
        chosen = "soft_probe_1"
        reason = (
            "Probe-1 is not significantly worse than hard route, and Probe-3 "
            "is not significantly better than Probe-1; the efficiency-first "
            "rule selects Probe-1."
        )
    elif not p3_rejected:
        chosen = "soft_probe_3"
        reason = (
            "Probe-1 is rejected versus hard route, while Probe-3 is not rejected."
        )
    else:
        chosen = "hard_route"
        reason = (
            "Both soft-probe variants are significantly worse than hard route "
            "for Answer F1."
        )

    return {
        "chosen_variant": chosen,
        "reason": reason,
        "probe_1_rejected": p1_rejected,
        "probe_3_rejected": p3_rejected,
        "probe_3_significantly_better_than_probe_1": p3_significantly_better,
    }


def build_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Qwen3 Soft-Route Probe Validation — LoCoMo Pilot-200",
        "",
        "## Role of this experiment",
        "",
        (
            "This experiment tests whether the evidence gains from asymmetric "
            "soft memory admission translate into end-to-end answer quality "
            "under a frozen Qwen3-8B backbone."
        ),
        "",
        "## Protocol",
        "",
        "- Backbone: Qwen/Qwen3-8B, frozen 4-bit NF4.",
        "- Temperature: 0.0; thinking disabled; no LoRA adapter.",
        "- Candidate pool: frozen C3 raw candidates union Mem0 top-20.",
        "- `hard_route`: unchanged C3 hard memory-type routing.",
        "- `soft_probe_1`: one candidate from a complementary memory type.",
        "- `soft_probe_3`: three candidates from a complementary memory type.",
        "- Ranker, candidate budget, selector, confidence gate, prompt and "
        "generation settings are otherwise unchanged.",
        "- Pilot-200 is treated as a development/model-selection set.",
        "",
        "## Main results",
        "",
        "| Variant | Answer F1 | Answerable F1 | Evidence P | Evidence R | "
        "Evidence F1 | Hit | Selected | Raw | Probe raw | Input tokens |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for variant in VARIANTS:
        row = summary["variants"][variant]
        lines.append(
            f"| {variant} | {row['answer_f1_all']:.4f} | "
            f"{row['answer_f1_answerable']:.4f} | "
            f"{row['evidence_precision']:.4f} | {row['evidence_recall']:.4f} | "
            f"{row['evidence_f1']:.4f} | {row['evidence_hit_rate']:.4f} | "
            f"{row['mean_selected']:.2f} | {row['mean_raw']:.2f} | "
            f"{row['mean_probe_raw']:.2f} | {row['mean_input_tokens']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Paired bootstrap",
            "",
            "| Comparison | Metric | Delta | 95% CI | p-value |",
            "|---|---|---:|---:|---:|",
        ]
    )

    for comparison_name, metrics in summary["paired_statistics"].items():
        for metric_name in ("answer_f1", "evidence_f1"):
            row = metrics[metric_name]
            lines.append(
                f"| {comparison_name} | {metric_name} | "
                f"{row['mean_delta']:+.4f} | "
                f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] | "
                f"{row['p_two_sided']:.6f} |"
            )

    selection = summary["selection"]
    lines.extend(
        [
            "",
            "## Development-set selection",
            "",
            f"- Chosen variant: `{selection['chosen_variant']}`.",
            f"- Reason: {selection['reason']}",
            "",
            "## Interpretation boundary",
            "",
            (
                "The selected probe budget must be frozen after this development-set "
                "comparison. Generalisation claims require an untouched held-out "
                "LoCoMo evaluation."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run end-to-end Qwen3-8B validation for hard route, soft Probe-1 "
            "and soft Probe-3 on LoCoMo Pilot-200."
        )
    )
    parser.add_argument("--qwen-runner-module", type=Path, required=True)
    parser.add_argument("--probe-module", type=Path, required=True)
    parser.add_argument("--runner-module", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prompt-template", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--memories", type=Path, required=True)
    parser.add_argument("--c3-predictions", type=Path, required=True)
    parser.add_argument("--mem0-top20", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=50000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260803)
    parser.add_argument(
        "--evaluation-time",
        type=str,
        default="2026-08-03T00:00:00+00:00",
        help="Frozen ISO-8601 evaluation time for deterministic temporal scoring.",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=0,
        help="Zero means all questions. A positive value is for smoke tests.",
    )
    parser.add_argument("--skip-backbone-smoke", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        evaluation_time = datetime.fromisoformat(args.evaluation_time)
    except ValueError as exc:
        raise ValueError(
            "--evaluation-time must be a valid ISO-8601 timestamp"
        ) from exc
    if evaluation_time.tzinfo is None:
        evaluation_time = evaluation_time.replace(tzinfo=timezone.utc)

    qwen = load_module(args.qwen_runner_module, "qwen_route_runner_module")
    probe = load_module(args.probe_module, "soft_probe_module")
    runner = load_module(args.runner_module, "hybrid_runner_module")

    base_config = load_config(args.config)
    generation_config = base_config["generation"]

    records = runner.load_memory_records(args.memories)
    catalog, source_map = runner.build_catalog(records)

    questions = qwen.read_csv(args.questions)
    questions_by_id = {str(row["question_id"]): row for row in questions}

    c3_rows = [
        row
        for row in qwen.read_jsonl(args.c3_predictions)
        if str(row.get("method", "")).lower()
        in {"c3", "c3_lite_controller"}
    ]
    c3_by_id = {str(row["question_id"]): row for row in c3_rows}

    mem0_rows = qwen.read_jsonl(args.mem0_top20)
    mem0_by_id = {str(row["question_id"]): row for row in mem0_rows}

    question_ids = sorted(questions_by_id)
    if len(question_ids) != 200:
        raise AssertionError("Expected 200 canonical Pilot questions.")
    if set(c3_by_id) != set(question_ids):
        raise AssertionError("C3 IDs do not match canonical questions.")
    if set(mem0_by_id) != set(question_ids):
        raise AssertionError("Mem0 IDs do not match canonical questions.")
    if args.max_questions > 0:
        question_ids = question_ids[: args.max_questions]

    hybrid_pools = {}
    missing_catalog: set[str] = set()
    for question_id in question_ids:
        c3_ids = qwen.split_ids(c3_by_id[question_id].get("raw_retrieved_ids"))
        mem0_ids = qwen.split_ids(mem0_by_id[question_id].get("retrieved_memory_ids"))
        for memory_id in [*c3_ids, *mem0_ids]:
            if memory_id not in catalog:
                missing_catalog.add(memory_id)
        hybrid_pools[question_id] = runner.make_pool(c3_ids, mem0_ids)

    if missing_catalog:
        raise AssertionError(
            f"{len(missing_catalog)} candidate IDs are absent from the catalog. "
            f"Examples: {sorted(missing_catalog)[:20]}"
        )

    print("Loading Qwen3 backbone...", flush=True)
    backbone_start = time.perf_counter()
    backbone, backbone_manifest = qwen.build_backbone(generation_config)
    backbone_manifest["construction_seconds"] = time.perf_counter() - backbone_start

    if not args.skip_backbone_smoke:
        smoke_start = time.perf_counter()
        smoke_result = backbone.generate(
            "Reply with exactly: BACKBONE_READY",
            temperature=0.0,
            max_new_tokens=16,
        )
        backbone_manifest["smoke"] = {
            "text": smoke_result.text,
            "input_tokens": smoke_result.input_tokens,
            "output_tokens": smoke_result.output_tokens,
            "latency_seconds": time.perf_counter() - smoke_start,
        }
        print("Backbone smoke response:", smoke_result.text, flush=True)

    backbone_manifest["experiment_variants"] = list(VARIANTS)
    backbone_manifest["evaluation_time"] = evaluation_time.isoformat()
    (args.output_dir / "backbone_manifest.json").write_text(
        json.dumps(backbone_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    hard_pipeline = C3Pipeline(
        config=deepcopy(base_config),
        memory_store=runner.FrozenCandidateStore(
            catalog=catalog,
            pools=hybrid_pools,
            label="hard_route",
        ),
        procedure_store=None,
        backbone=backbone,
        prompt_template=args.prompt_template,
    )

    pipelines: dict[str, C3Pipeline] = {"hard_route": hard_pipeline}
    routers: dict[str, Any] = {}
    stores: dict[str, Any] = {}

    for variant, probe_budget in (("soft_probe_1", 1), ("soft_probe_3", 3)):
        router = probe.CompanionSoftRoutePlanner(base_config)
        store = probe.ProbeBudgetStore(
            catalog=catalog,
            pools=hybrid_pools,
            label=variant,
            config=base_config,
            probe_budget=probe_budget,
        )
        pipeline = C3Pipeline(
            config=deepcopy(base_config),
            memory_store=store,
            procedure_store=None,
            backbone=backbone,
            prompt_template=args.prompt_template,
        )
        pipeline.router = router
        pipelines[variant] = pipeline
        routers[variant] = router
        stores[variant] = store

    predictions_path = args.output_dir / "predictions.jsonl"
    existing_rows = qwen.read_jsonl(predictions_path)
    completed = {
        (str(row["question_id"]), str(row["variant"]))
        for row in existing_rows
    }
    expected_keys = {
        (question_id, variant)
        for question_id in question_ids
        for variant in VARIANTS
    }
    unknown_keys = completed - expected_keys
    if unknown_keys:
        raise AssertionError(
            "Predictions file contains unexpected rows: "
            f"{sorted(unknown_keys)[:10]}"
        )

    print(
        f"Resume state: {len(completed)}/{len(expected_keys)} generations complete.",
        flush=True,
    )

    try:
        with predictions_path.open("a", encoding="utf-8") as output_handle:
            total = len(expected_keys)
            for question_index, question_id in enumerate(question_ids, start=1):
                c3 = c3_by_id[question_id]
                canonical = questions_by_id[question_id]
                query = str(c3["query"])
                user_id = str(c3["user_id"])
                question_type = str(
                    c3.get(
                        "question_type",
                        canonical.get("question_type", "unknown"),
                    )
                )
                gold_memory_ids = qwen.split_ids(
                    canonical.get("supporting_memory_ids", "")
                )
                gold_source_ids = runner.project_ids(gold_memory_ids, source_map)
                references = qwen.answer_references(
                    canonical.get("gold_answer", c3.get("gold_answer"))
                )
                if not references:
                    references = qwen.answer_references(c3.get("gold_answer"))
                should_abstain = qwen.parse_bool(
                    canonical.get(
                        "should_abstain",
                        c3.get("should_abstain", False),
                    )
                )
                shared_time = evaluation_time

                for variant in VARIANTS:
                    key = (question_id, variant)
                    if key in completed:
                        continue

                    print(
                        f"[{len(completed) + 1}/{total}] "
                        f"question={question_index}/{len(question_ids)} "
                        f"id={question_id} variant={variant}",
                        flush=True,
                    )

                    result = pipelines[variant].answer(
                        QueryState(
                            query=query,
                            user_id=user_id,
                            session_id=question_id,
                            current_time=shared_time,
                        )
                    )

                    selected_ids = list(result.selected_ids)
                    selected_source_ids = runner.project_ids(selected_ids, source_map)
                    evidence = runner.evidence_metrics(
                        selected_source_ids,
                        gold_source_ids,
                    )
                    answer_scores = qwen.score_answer(result.answer, references)
                    predicted_abstain = result.decision.value == "abstain"

                    route_trace = (
                        deepcopy(routers[variant].last_trace)
                        if variant in routers
                        else {}
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

                    row = {
                        "question_id": question_id,
                        "variant": variant,
                        "question_type": question_type,
                        "query": query,
                        "user_id": user_id,
                        "gold_answer": references,
                        "should_abstain": should_abstain,
                        "answer": result.answer,
                        "decision": result.decision.value,
                        "predicted_abstain": predicted_abstain,
                        "predicted_response": not predicted_abstain,
                        "abstention_correct": predicted_abstain == should_abstain,
                        **answer_scores,
                        "gold_memory_ids": gold_memory_ids,
                        "gold_source_ids": gold_source_ids,
                        "raw_retrieved_ids": list(result.raw_retrieved_ids),
                        "ranked_candidate_ids": list(result.ranked_candidate_ids),
                        "selected_ids": selected_ids,
                        "selected_source_ids": selected_source_ids,
                        "selected_evidence": [
                            candidate.to_dict()
                            for candidate in result.selected_evidence
                        ],
                        "selected_memory_types": result.selected_memory_types,
                        "route_scores": result.route_scores,
                        "route_trace": route_trace,
                        "retrieval_trace": retrieval_trace,
                        "evidence_precision": evidence["precision"],
                        "evidence_recall": evidence["recall"],
                        "evidence_f1": evidence["f1"],
                        "evidence_hit": evidence["hit"],
                        "evidence_overlap_source_ids": evidence[
                            "overlap_source_ids"
                        ],
                        "selected_count": len(selected_ids),
                        "raw_count": len(result.raw_retrieved_ids),
                        "ranked_count": len(result.ranked_candidate_ids),
                        "probe_raw_count": probe_raw_count,
                        "latency_ms": result.latency_ms,
                        "input_tokens": result.input_tokens,
                        "output_tokens": result.output_tokens,
                        "input_tokens_numeric": result.input_tokens or 0,
                        "output_tokens_numeric": result.output_tokens or 0,
                    }
                    qwen.append_jsonl(output_handle, row)
                    completed.add(key)
    finally:
        for pipeline in pipelines.values():
            pipeline.close()

    final_rows = qwen.read_jsonl(predictions_path)
    by_key = {
        (str(row["question_id"]), str(row["variant"])): row
        for row in final_rows
    }
    if set(by_key) != expected_keys:
        missing = sorted(expected_keys - set(by_key))
        raise AssertionError(f"Incomplete generation. Missing: {missing[:20]}")

    final_rows = [
        by_key[(question_id, variant)]
        for question_id in question_ids
        for variant in VARIANTS
    ]

    variant_summaries = {
        variant: aggregate_rows(qwen, final_rows, variant)
        for variant in VARIANTS
    }

    paired_statistics = {}
    paired_outcome_summary = {}
    for comparison_index, (
        comparison_name,
        (left_variant, right_variant),
    ) in enumerate(COMPARISONS.items()):
        answer_deltas = paired_deltas(
            final_rows,
            question_ids,
            left_variant=left_variant,
            right_variant=right_variant,
            field="answer_f1",
        )
        evidence_deltas = paired_deltas(
            final_rows,
            question_ids,
            left_variant=left_variant,
            right_variant=right_variant,
            field="evidence_f1",
        )
        paired_statistics[comparison_name] = {
            "left_variant": left_variant,
            "right_variant": right_variant,
            "answer_f1": qwen.bootstrap_mean_delta(
                answer_deltas,
                repetitions=args.bootstrap_repetitions,
                seed=args.bootstrap_seed + comparison_index * 10,
            ),
            "evidence_f1": qwen.bootstrap_mean_delta(
                evidence_deltas,
                repetitions=args.bootstrap_repetitions,
                seed=args.bootstrap_seed + comparison_index * 10 + 1,
            ),
        }
        paired_outcome_summary[comparison_name] = {
            "answer_f1": paired_outcomes(answer_deltas),
            "evidence_f1": paired_outcomes(evidence_deltas),
        }

    selection = select_variant(paired_statistics)
    summary = {
        "experiment": "rc8_8b12_qwen3_soft_probe_validation",
        "dataset_role": "development/model-selection",
        "questions": len(question_ids),
        "generations": len(final_rows),
        "backbone": backbone_manifest,
        "variants": variant_summaries,
        "paired_statistics": paired_statistics,
        "paired_outcomes": paired_outcome_summary,
        "category_breakdown": {
            variant: category_breakdown(final_rows, variant)
            for variant in VARIANTS
        },
        "selection": selection,
    }

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
    print("QWEN3 SOFT-PROBE VALIDATION: PASSED")


if __name__ == "__main__":
    main()
