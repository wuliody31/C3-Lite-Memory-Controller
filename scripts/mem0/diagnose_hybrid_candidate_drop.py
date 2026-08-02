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
from src.schemas import QueryState


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


def trace_by_id(
    result: Any,
) -> dict[str, dict[str, Any]]:
    trace = result.debug.get(
        "full_candidate_score_trace",
        [],
    )

    return {
        str(item["memory_id"]): item
        for item in trace
        if isinstance(item, dict)
        and item.get("memory_id")
    }


def candidate_stage(
    *,
    memory_id: str,
    union_pool_ids: list[str],
    actual_raw_ids: list[str],
    trace: dict[str, dict[str, Any]],
    selected_ids: list[str],
) -> str:
    if memory_id not in union_pool_ids:
        return "not_in_union_pool"

    if memory_id not in actual_raw_ids:
        return "route_filtered"

    item = trace.get(memory_id)

    if item is None:
        return "missing_trace"

    if memory_id in selected_ids:
        return "selected"

    if not bool(
        item.get(
            "passed_ranker_gate",
            False,
        )
    ):
        return "ranker_gate"

    if not bool(
        item.get(
            "kept_after_type_top_k",
            False,
        )
    ):
        return "type_top_k"

    if not bool(
        item.get(
            "kept_after_candidate_budget",
            False,
        )
    ):
        return "candidate_budget"

    if not bool(
        item.get(
            "survived_conflict_resolution",
            False,
        )
    ):
        return "conflict_resolution"

    if not bool(
        item.get(
            "selected_final",
            False,
        )
    ):
        return "evidence_selector"

    return "unclassified"


def furthest_stage(
    stages: list[str],
) -> str:
    order = {
        "not_in_union_pool": 0,
        "route_filtered": 1,
        "missing_trace": 2,
        "ranker_gate": 3,
        "type_top_k": 4,
        "candidate_budget": 5,
        "conflict_resolution": 6,
        "evidence_selector": 7,
        "selected": 8,
        "unclassified": 9,
    }

    return max(
        stages,
        key=lambda stage: order.get(
            stage,
            -1,
        ),
    )


def mean(
    values: list[float],
) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def build_report(
    summary: dict[str, Any],
) -> str:
    lines = [
        "# C3 + Mem0 Hybrid Candidate-Drop Diagnosis",
        "",
        "## Scope",
        "",
        (
            "This diagnosis examines the Pilot-200 questions for which "
            "Mem0 top-20 retrieved at least one gold-equivalent candidate "
            "that was absent from the frozen C3 raw pool."
        ),
        "",
        "## Main counts",
        "",
        (
            f"- Backend-rescue questions: "
            f"{summary['backend_rescue_questions']}"
        ),
        (
            f"- Questions with a Mem0-only gold candidate selected: "
            f"{summary['questions_with_mem0_gold_selected']}"
        ),
        (
            f"- Questions with no Mem0-only gold candidate selected: "
            f"{summary['questions_without_mem0_gold_selected']}"
        ),
        "",
        "## Candidate-level loss stages",
        "",
        "| Stage | Gold candidates |",
        "|---|---:|",
    ]

    for stage, count in summary[
        "candidate_stage_counts"
    ].items():
        lines.append(
            f"| {stage} | {count} |"
        )

    lines.extend(
        [
            "",
            "## Question-level furthest stage",
            "",
            "| Furthest stage reached | Questions |",
            "|---|---:|",
        ]
    )

    for stage, count in summary[
        "question_furthest_stage_counts"
    ].items():
        lines.append(
            f"| {stage} | {count} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation gate",
            "",
            (
                "- A high `route_filtered` count indicates that Mem0 "
                "retrieves useful evidence from memory types excluded by "
                "the current route planner."
            ),
            (
                "- A high `ranker_gate` or `type_top_k` count indicates "
                "that the shared ranker or fixed per-type budget cannot "
                "exploit complementary Mem0 candidates."
            ),
            (
                "- A high `evidence_selector` count indicates that useful "
                "candidates survive ranking but are removed by final "
                "evidence selection."
            ),
            (
                "- Only after identifying the dominant stage should the "
                "utility estimator or dynamic budget be modified."
            ),
            "",
        ]
    )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose where Mem0-only gold candidates are removed "
            "inside the unchanged C3 controller."
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
        "--output-dir",
        type=Path,
        required=True,
    )

    args = parser.parse_args()
    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    runner = load_runner(
        args.runner_module
    )

    config = load_config(
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
            row.get(
                "method",
                "",
            )
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
            "Expected 200 canonical questions."
        )

    if set(c3_by_id) != question_ids:
        raise AssertionError(
            "C3 question IDs do not match."
        )

    if set(mem0_by_id) != question_ids:
        raise AssertionError(
            "Mem0 question IDs do not match."
        )

    hybrid_pools = {}

    backend_rescue_ids: list[str] = []

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

        canonical = (
            questions_by_id[
                question_id
            ]
        )
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
            backend_rescue_ids.append(
                question_id
            )

    pipeline = C3Pipeline(
        config=deepcopy(config),
        memory_store=(
            runner.FrozenCandidateStore(
                catalog=catalog,
                pools=hybrid_pools,
                label="c3_mem0_union_diagnostic",
            )
        ),
        procedure_store=None,
        backbone=MockBackbone(),
        prompt_template=(
            args.prompt_template
        ),
    )

    rows: list[dict[str, Any]] = []
    candidate_stage_counts = Counter()
    question_stage_counts = Counter()

    try:
        for question_id in backend_rescue_ids:
            c3 = c3_by_id[question_id]
            mem0 = mem0_by_id[question_id]
            canonical = (
                questions_by_id[
                    question_id
                ]
            )

            c3_ids = runner.split_ids(
                c3.get(
                    "raw_retrieved_ids"
                )
            )
            mem0_ids = runner.split_ids(
                mem0.get(
                    "retrieved_memory_ids"
                )
            )

            c3_set = set(c3_ids)

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

            mem0_only_gold_ids = []

            for memory_id in mem0_ids:
                if memory_id in c3_set:
                    continue

                projected = set(
                    source_map.get(
                        memory_id,
                        [memory_id],
                    )
                )

                if projected & gold_source_ids:
                    mem0_only_gold_ids.append(
                        memory_id
                    )

            if not mem0_only_gold_ids:
                raise AssertionError(
                    f"{question_id}: no Mem0-only gold candidate "
                    "despite backend-rescue classification."
                )

            result = pipeline.answer(
                QueryState(
                    query=str(c3["query"]),
                    user_id=str(
                        c3["user_id"]
                    ),
                    session_id=(
                        question_id
                    ),
                )
            )

            actual_raw_ids = list(
                result.raw_retrieved_ids
            )
            selected_ids = list(
                result.selected_ids
            )
            trace = trace_by_id(
                result
            )
            union_pool_ids = (
                hybrid_pools[
                    question_id
                ].ordered_ids
            )

            candidate_details = []
            stages = []

            for memory_id in (
                mem0_only_gold_ids
            ):
                stage = candidate_stage(
                    memory_id=memory_id,
                    union_pool_ids=(
                        union_pool_ids
                    ),
                    actual_raw_ids=(
                        actual_raw_ids
                    ),
                    trace=trace,
                    selected_ids=(
                        selected_ids
                    ),
                )

                stages.append(stage)
                candidate_stage_counts[
                    stage
                ] += 1

                item = trace.get(
                    memory_id,
                    {},
                )

                candidate_details.append(
                    {
                        "memory_id": memory_id,
                        "projected_source_ids": (
                            source_map.get(
                                memory_id,
                                [memory_id],
                            )
                        ),
                        "stage": stage,
                        "memory_type": (
                            catalog[
                                memory_id
                            ].memory_type.value
                        ),
                        "final_score": (
                            item.get(
                                "final_score"
                            )
                        ),
                        "rank_global_all_scored": (
                            item.get(
                                "rank_global_all_scored"
                            )
                        ),
                        "rank_global_after_gate": (
                            item.get(
                                "rank_global_after_gate"
                            )
                        ),
                        "passed_ranker_gate": (
                            item.get(
                                "passed_ranker_gate"
                            )
                        ),
                        "kept_after_type_top_k": (
                            item.get(
                                "kept_after_type_top_k"
                            )
                        ),
                        "kept_after_candidate_budget": (
                            item.get(
                                "kept_after_candidate_budget"
                            )
                        ),
                        "survived_conflict_resolution": (
                            item.get(
                                "survived_conflict_resolution"
                            )
                        ),
                        "selected_final": (
                            item.get(
                                "selected_final"
                            )
                        ),
                    }
                )

            question_stage = (
                furthest_stage(stages)
            )
            question_stage_counts[
                question_stage
            ] += 1

            rows.append(
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
                    "gold_source_ids": sorted(
                        gold_source_ids
                    ),
                    "c3_raw_ids": c3_ids,
                    "mem0_top20_ids": (
                        mem0_ids
                    ),
                    "union_pool_ids": (
                        union_pool_ids
                    ),
                    "actual_route_filtered_raw_ids": (
                        actual_raw_ids
                    ),
                    "hybrid_ranked_ids": (
                        list(
                            result.ranked_candidate_ids
                        )
                    ),
                    "hybrid_selected_ids": (
                        selected_ids
                    ),
                    "selected_source_ids": (
                        runner.project_ids(
                            selected_ids,
                            source_map,
                        )
                    ),
                    "mem0_only_gold_candidate_ids": (
                        mem0_only_gold_ids
                    ),
                    "candidate_details": (
                        candidate_details
                    ),
                    "question_furthest_stage": (
                        question_stage
                    ),
                    "any_mem0_gold_selected": (
                        "selected" in stages
                    ),
                    "selected_hits_gold": bool(
                        set(
                            runner.project_ids(
                                selected_ids,
                                source_map,
                            )
                        )
                        & gold_source_ids
                    ),
                    "route_types": (
                        result.selected_memory_types
                    ),
                }
            )
    finally:
        pipeline.close()

    questions_with_selected = sum(
        bool(
            row[
                "any_mem0_gold_selected"
            ]
        )
        for row in rows
    )

    stage_order = [
        "route_filtered",
        "ranker_gate",
        "type_top_k",
        "candidate_budget",
        "conflict_resolution",
        "evidence_selector",
        "selected",
        "missing_trace",
        "unclassified",
        "not_in_union_pool",
    ]

    summary = {
        "experiment": (
            "rc8_8b5_hybrid_candidate_drop_diagnosis"
        ),
        "backend_rescue_questions": (
            len(
                backend_rescue_ids
            )
        ),
        "questions_with_mem0_gold_selected": (
            questions_with_selected
        ),
        "questions_without_mem0_gold_selected": (
            len(rows)
            - questions_with_selected
        ),
        "candidate_stage_counts": {
            stage: (
                candidate_stage_counts[
                    stage
                ]
            )
            for stage in stage_order
            if candidate_stage_counts[
                stage
            ]
        },
        "question_furthest_stage_counts": {
            stage: (
                question_stage_counts[
                    stage
                ]
            )
            for stage in stage_order
            if question_stage_counts[
                stage
            ]
        },
        "mean_mem0_only_gold_candidates_per_rescue_question": (
            mean(
                [
                    float(
                        len(
                            row[
                                "mem0_only_gold_candidate_ids"
                            ]
                        )
                    )
                    for row in rows
                ]
            )
        ),
        "selected_gold_hit_questions": sum(
            bool(
                row[
                    "selected_hits_gold"
                ]
            )
            for row in rows
        ),
        "category_breakdown": {},
    }

    grouped = defaultdict(list)

    for row in rows:
        grouped[
            str(
                row[
                    "question_type"
                ]
            )
        ].append(row)

    for category, items in sorted(
        grouped.items()
    ):
        category_stage_counts = Counter(
            row[
                "question_furthest_stage"
            ]
            for row in items
        )

        summary[
            "category_breakdown"
        ][category] = {
            "questions": len(items),
            "mem0_gold_selected": sum(
                bool(
                    row[
                        "any_mem0_gold_selected"
                    ]
                )
                for row in items
            ),
            "selected_gold_hit": sum(
                bool(
                    row[
                        "selected_hits_gold"
                    ]
                )
                for row in items
            ),
            "furthest_stage_counts": (
                dict(
                    category_stage_counts
                )
            ),
        }

    write_jsonl(
        args.output_dir
        / "hybrid_candidate_drop_details.jsonl",
        rows,
    )

    (
        args.output_dir
        / "hybrid_candidate_drop_summary.json"
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
        / "hybrid_candidate_drop_report.md"
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
        "HYBRID CANDIDATE-DROP "
        "DIAGNOSIS: PASSED"
    )


if __name__ == "__main__":
    main()
