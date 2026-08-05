from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

from src.backbones import MockBackbone
from src.config import load_config
from src.pipeline import C3Pipeline
from src.schemas import QueryState

VARIANTS = ("hard_route", "soft_probe_1")


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def append_jsonl(handle: Any, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    handle.flush()


def avg(rows: list[dict[str, Any]], field: str) -> float:
    return sum(float(row.get(field, 0.0) or 0.0) for row in rows) / len(rows)


def aggregate(rows: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    subset = [row for row in rows if row["variant"] == variant]
    return {
        "questions": len(subset),
        "evidence_precision": avg(subset, "evidence_precision"),
        "evidence_recall": avg(subset, "evidence_recall"),
        "evidence_f1": avg(subset, "evidence_f1"),
        "evidence_hit_rate": avg(subset, "evidence_hit"),
        "mean_selected": avg(subset, "selected_count"),
        "mean_ranked": avg(subset, "ranked_count"),
        "mean_raw": avg(subset, "raw_count"),
        "mean_probe_raw": avg(subset, "probe_raw_count"),
    }


def paired_deltas(
    rows: list[dict[str, Any]],
    question_ids: list[str],
    field: str,
) -> list[float]:
    lookup = {
        variant: {
            str(row["question_id"]): row
            for row in rows
            if row["variant"] == variant
        }
        for variant in VARIANTS
    }
    return [
        float(lookup["soft_probe_1"][qid][field])
        - float(lookup["hard_route"][qid][field])
        for qid in question_ids
    ]


def paired_hits(rows: list[dict[str, Any]]) -> dict[str, int]:
    lookup = {
        variant: {
            str(row["question_id"]): bool(row["evidence_hit"])
            for row in rows
            if row["variant"] == variant
        }
        for variant in VARIANTS
    }
    both = hard_only = soft_only = neither = 0
    for qid in sorted(lookup["hard_route"]):
        hard = lookup["hard_route"][qid]
        soft = lookup["soft_probe_1"][qid]
        if hard and soft:
            both += 1
        elif hard:
            hard_only += 1
        elif soft:
            soft_only += 1
        else:
            neither += 1
    return {
        "both_hit": both,
        "hard_only_hit": hard_only,
        "soft_only_hit": soft_only,
        "neither_hit": neither,
        "net_rescues": soft_only - hard_only,
    }


def category_breakdown(rows: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row["variant"] == variant:
            groups.setdefault(str(row["question_type"]), []).append(row)
    return {
        category: {
            "questions": len(items),
            "evidence_precision": avg(items, "evidence_precision"),
            "evidence_recall": avg(items, "evidence_recall"),
            "evidence_f1": avg(items, "evidence_f1"),
            "evidence_hit_rate": avg(items, "evidence_hit"),
            "mean_selected": avg(items, "selected_count"),
            "mean_raw": avg(items, "raw_count"),
        }
        for category, items in sorted(groups.items())
    }


def build_report(summary: dict[str, Any]) -> str:
    hard = summary["variants"]["hard_route"]
    soft = summary["variants"]["soft_probe_1"]
    paired = summary["paired_statistics"]
    gate = summary["qwen_gate"]
    lines = [
        "# Frozen Soft Probe-1 Controller Replay — LoCoMo Held-out-from-Pilot",
        "",
        "- Split: 1,786 questions excluding Pilot-200.",
        "- Candidate pool: frozen C3 raw union Mem0 top-20.",
        "- Runtime policy: frozen Soft Probe-1; no gold used.",
        "- Mock backbone: controller evidence only.",
        "",
        "| Variant | Evidence P | Evidence R | Evidence F1 | Hit | Selected | Ranked | Raw | Probe raw |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| hard_route | {hard['evidence_precision']:.4f} | {hard['evidence_recall']:.4f} | {hard['evidence_f1']:.4f} | {hard['evidence_hit_rate']:.4f} | {hard['mean_selected']:.2f} | {hard['mean_ranked']:.2f} | {hard['mean_raw']:.2f} | {hard['mean_probe_raw']:.2f} |",
        f"| soft_probe_1 | {soft['evidence_precision']:.4f} | {soft['evidence_recall']:.4f} | {soft['evidence_f1']:.4f} | {soft['evidence_hit_rate']:.4f} | {soft['mean_selected']:.2f} | {soft['mean_ranked']:.2f} | {soft['mean_raw']:.2f} | {soft['mean_probe_raw']:.2f} |",
        "",
        "| Metric | Delta | 95% CI | p-value |",
        "|---|---:|---:|---:|",
    ]
    for metric in ("evidence_precision", "evidence_recall", "evidence_f1", "evidence_hit"):
        row = paired[metric]
        lines.append(
            f"| {metric} | {row['mean_delta']:+.4f} | "
            f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] | "
            f"{row['p_two_sided']:.6f} |"
        )
    lines += [
        "",
        f"Strict Qwen progression gate: `{gate['passed']}`.",
        "",
        "This split excludes Pilot-200 but is described as post-development held-out-from-pilot, not a pristine benchmark test set.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner-module", type=Path, required=True)
    parser.add_argument("--probe-module", type=Path, required=True)
    parser.add_argument("--qwen-runner-module", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prompt-template", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--memories", type=Path, required=True)
    parser.add_argument("--c3-predictions", type=Path, required=True)
    parser.add_argument("--mem0-top20", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-questions", type=int, default=1786)
    parser.add_argument("--max-questions", type=int, default=0)
    parser.add_argument("--bootstrap-repetitions", type=int, default=50000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260803)
    parser.add_argument("--evaluation-time", default="2026-08-03T00:00:00+00:00")
    args = parser.parse_args()

    runner = load_module(args.runner_module, "heldout_hybrid_runner")
    probe = load_module(args.probe_module, "heldout_probe_module")
    qwen = load_module(args.qwen_runner_module, "heldout_qwen_runner")

    config = load_config(args.config)
    questions = read_csv(args.questions)
    questions_by_id = {str(row["question_id"]): row for row in questions}
    if len(questions_by_id) != len(questions):
        raise AssertionError("Question IDs are not unique.")
    if args.expected_questions > 0 and len(questions) != args.expected_questions:
        raise AssertionError(
            f"Expected {args.expected_questions} questions, found {len(questions)}."
        )

    c3_rows = [
        row
        for row in read_jsonl(args.c3_predictions)
        if str(row.get("method", "")).lower() in {"c3", "c3_lite_controller"}
    ]
    c3_by_id = {str(row["question_id"]): row for row in c3_rows}
    mem0_by_id = {
        str(row["question_id"]): row
        for row in read_jsonl(args.mem0_top20)
    }

    all_ids = sorted(questions_by_id)
    if set(c3_by_id) != set(all_ids):
        raise AssertionError("C3 IDs do not match held-out questions.")
    if set(mem0_by_id) != set(all_ids):
        raise AssertionError("Mem0 IDs do not match held-out questions.")

    question_ids = all_ids[: args.max_questions] if args.max_questions > 0 else all_ids

    records = runner.load_memory_records(args.memories)
    catalog, source_map = runner.build_catalog(records)
    hybrid_pools = {}
    missing_catalog: set[str] = set()

    for qid in question_ids:
        c3_ids = qwen.split_ids(c3_by_id[qid].get("raw_retrieved_ids"))
        mem0_ids = qwen.split_ids(mem0_by_id[qid].get("retrieved_memory_ids"))
        for memory_id in [*c3_ids, *mem0_ids]:
            if memory_id not in catalog:
                missing_catalog.add(memory_id)
        hybrid_pools[qid] = runner.make_pool(c3_ids, mem0_ids)

    if missing_catalog:
        raise AssertionError(
            f"{len(missing_catalog)} IDs absent from catalog. "
            f"Examples: {sorted(missing_catalog)[:20]}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    hard_pipeline = C3Pipeline(
        config=deepcopy(config),
        memory_store=runner.FrozenCandidateStore(
            catalog=catalog,
            pools=hybrid_pools,
            label="hard_route",
        ),
        procedure_store=None,
        backbone=MockBackbone(),
        prompt_template=args.prompt_template,
    )

    soft_router = probe.CompanionSoftRoutePlanner(config)
    soft_store = probe.ProbeBudgetStore(
        catalog=catalog,
        pools=hybrid_pools,
        label="soft_probe_1",
        config=config,
        probe_budget=1,
    )
    soft_pipeline = C3Pipeline(
        config=deepcopy(config),
        memory_store=soft_store,
        procedure_store=None,
        backbone=MockBackbone(),
        prompt_template=args.prompt_template,
    )
    soft_pipeline.router = soft_router
    pipelines = {"hard_route": hard_pipeline, "soft_probe_1": soft_pipeline}

    predictions_path = args.output_dir / "predictions.jsonl"
    existing = read_jsonl(predictions_path)
    expected_keys = {
        (qid, variant)
        for qid in question_ids
        for variant in VARIANTS
    }
    completed = {
        (str(row["question_id"]), str(row["variant"]))
        for row in existing
    }
    if completed - expected_keys:
        raise AssertionError("Predictions contain unexpected rows.")

    current_time = datetime.fromisoformat(args.evaluation_time)
    print(
        f"Resume state: {len(completed)}/{len(expected_keys)} controller runs complete.",
        flush=True,
    )

    try:
        with predictions_path.open("a", encoding="utf-8") as handle:
            for index, qid in enumerate(question_ids, start=1):
                canonical = questions_by_id[qid]
                c3 = c3_by_id[qid]
                query = str(c3.get("query", canonical["question"]))
                user_id = str(c3.get("user_id", canonical["user_id"]))
                question_type = str(
                    c3.get("question_type", canonical.get("question_type", "unknown"))
                )
                gold_memory_ids = qwen.split_ids(
                    canonical.get("supporting_memory_ids", "")
                )
                gold_source_ids = runner.project_ids(gold_memory_ids, source_map)

                for variant in VARIANTS:
                    key = (qid, variant)
                    if key in completed:
                        continue
                    result = pipelines[variant].answer(
                        QueryState(
                            query=query,
                            user_id=user_id,
                            session_id=qid,
                            current_time=current_time,
                        )
                    )
                    selected_ids = list(result.selected_ids)
                    selected_source_ids = runner.project_ids(selected_ids, source_map)
                    evidence = runner.evidence_metrics(
                        selected_source_ids,
                        gold_source_ids,
                    )
                    route_trace = (
                        deepcopy(soft_router.last_trace)
                        if variant == "soft_probe_1"
                        else {}
                    )
                    retrieval_trace = (
                        soft_store.trace_for_question(qid)
                        if variant == "soft_probe_1"
                        else {}
                    )
                    probe_raw = sum(
                        int(trace.get("returned_count", 0))
                        for trace in retrieval_trace.values()
                        if not bool(trace.get("is_original_type", True))
                    )
                    row = {
                        "question_id": qid,
                        "variant": variant,
                        "question_type": question_type,
                        "query": query,
                        "user_id": user_id,
                        "gold_memory_ids": gold_memory_ids,
                        "gold_source_ids": gold_source_ids,
                        "raw_retrieved_ids": list(result.raw_retrieved_ids),
                        "ranked_candidate_ids": list(result.ranked_candidate_ids),
                        "selected_ids": selected_ids,
                        "selected_source_ids": selected_source_ids,
                        "route_scores": result.route_scores,
                        "route_trace": route_trace,
                        "retrieval_trace": retrieval_trace,
                        "evidence_precision": evidence["precision"],
                        "evidence_recall": evidence["recall"],
                        "evidence_f1": evidence["f1"],
                        "evidence_hit": evidence["hit"],
                        "evidence_overlap_source_ids": evidence["overlap_source_ids"],
                        "selected_count": len(selected_ids),
                        "raw_count": len(result.raw_retrieved_ids),
                        "ranked_count": len(result.ranked_candidate_ids),
                        "probe_raw_count": probe_raw,
                        "decision": result.decision.value,
                        "latency_ms": result.latency_ms,
                    }
                    append_jsonl(handle, row)
                    completed.add(key)

                if index % 50 == 0 or index == len(question_ids):
                    print(
                        f"questions: {index}/{len(question_ids)}; "
                        f"runs: {len(completed)}/{len(expected_keys)}",
                        flush=True,
                    )
    finally:
        for pipeline in pipelines.values():
            pipeline.close()

    rows = read_jsonl(predictions_path)
    by_key = {
        (str(row["question_id"]), str(row["variant"])): row
        for row in rows
    }
    if set(by_key) != expected_keys:
        raise AssertionError("Controller replay is incomplete.")
    rows = [
        by_key[(qid, variant)]
        for qid in question_ids
        for variant in VARIANTS
    ]

    variants = {variant: aggregate(rows, variant) for variant in VARIANTS}
    metrics = (
        "evidence_precision",
        "evidence_recall",
        "evidence_f1",
        "evidence_hit",
        "selected_count",
        "raw_count",
    )
    paired = {
        metric: qwen.bootstrap_mean_delta(
            paired_deltas(rows, question_ids, metric),
            repetitions=args.bootstrap_repetitions,
            seed=args.bootstrap_seed + index,
        )
        for index, metric in enumerate(metrics)
    }

    expansion_rows = [
        row
        for row in rows
        if row["variant"] == "soft_probe_1"
        and bool(row.get("route_trace", {}).get("triggered"))
    ]
    added_counts: dict[str, int] = {}
    for row in expansion_rows:
        for memory_type in row["route_trace"].get("added_memory_types", []):
            added_counts[memory_type] = added_counts.get(memory_type, 0) + 1

    selected_delta = paired["selected_count"]["mean_delta"]
    hit_delta = paired["evidence_hit"]["mean_delta"]
    f1_boot = paired["evidence_f1"]
    qwen_gate = {
        "evidence_f1_confirmed": (
            f1_boot["mean_delta"] > 0.0 and f1_boot["ci_low"] >= 0.0
        ),
        "hit_non_regressive": hit_delta >= 0.0,
        "selected_context_stable": abs(selected_delta) <= 0.05,
    }
    qwen_gate["passed"] = all(qwen_gate.values())

    summary = {
        "experiment": "rc8_8b13_frozen_soft_probe1_heldout_controller_replay",
        "dataset_role": "post-development held-out-from-pilot",
        "questions": len(question_ids),
        "controller_runs": len(rows),
        "frozen_policy": {
            "variant": "soft_probe_1",
            "probe_budget": 1,
            "route_scores_preserved": True,
            "runtime_uses_gold": False,
        },
        "variants": variants,
        "paired_statistics": paired,
        "paired_hits": paired_hits(rows),
        "route_behaviour": {
            "expanded_questions": len(expansion_rows),
            "expansion_rate": len(expansion_rows) / len(question_ids),
            "added_type_counts": added_counts,
        },
        "category_breakdown": {
            variant: category_breakdown(rows, variant)
            for variant in VARIANTS
        },
        "qwen_gate": qwen_gate,
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
    print("HELD-OUT SOFT-PROBE CONTROLLER REPLAY: PASSED")


if __name__ == "__main__":
    main()
