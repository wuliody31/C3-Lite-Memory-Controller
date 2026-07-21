from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

from src.config import get_config
from src.io_utils import (
    read_jsonl,
    write_jsonl,
    write_json,
    find_eval_questions_path,
    find_procedural_rules_path,
)
from src.neo4j_adapter import Neo4jMemoryAdapter
from src.procedural_matcher import ProceduralRuleMatcher
from src.retrieval_engine import MemoryRetrievalEngine
from src.controller import C3LiteController
from src.baselines import (
    no_memory_baseline,
    simple_retrieval_baseline,
    all_memory_baseline,
)
from src.evaluation_utils import score_prediction


METHODS = [
    "no_memory",
    "simple_retrieval",
    "all_memory",
    "c3_lite_controller",
]


def select_questions(
    eval_questions: list[dict[str, Any]],
    limit: int | None = None,
    balanced: bool = False,
) -> list[dict[str, Any]]:
    if (
        limit is None
        or limit >= len(eval_questions)
    ):
        return eval_questions

    if not balanced:
        return eval_questions[:limit]

    groups: dict[
        str,
        list[dict[str, Any]],
    ] = {}

    for question in eval_questions:
        groups.setdefault(
            question["question_type"],
            [],
        ).append(question)

    selected: list[dict[str, Any]] = []

    while len(selected) < limit:
        added = False

        for question_type in sorted(groups):
            if (
                groups[question_type]
                and len(selected) < limit
            ):
                selected.append(
                    groups[question_type].pop(0)
                )
                added = True

        if not added:
            break

    return selected


def _summaries(
    output_dir: Path,
    scores: list[dict[str, Any]],
) -> None:
    import pandas as pd

    dataframe = pd.DataFrame(scores)

    dataframe.to_csv(
        output_dir / "scores.csv",
        index=False,
    )

    metrics = {
        "route_exact_match": "mean",
        "route_precision": "mean",
        "route_recall": "mean",
        "route_f1": "mean",
        "evidence_recall": "mean",
        "evidence_precision": "mean",
        "evidence_density": "mean",
        "include_score": "mean",
        "not_include_violations": "mean",
        "abstention_correctness": "mean",
        "confidence_score": "mean",
        "query_coverage": "mean",
        "latency_seconds": "mean",
        "num_retrieved": "mean",
        "num_used": "mean",
        "num_outdated": "mean",
        "conflict_action": "mean",
    }

    summary = (
        dataframe
        .groupby("method")
        .agg(metrics)
        .reset_index()
    )

    summary.to_csv(
        output_dir / "summary.csv",
        index=False,
    )

    (
        dataframe
        .groupby([
            "question_type",
            "method",
        ])
        .agg(metrics)
        .reset_index()
        .to_csv(
            output_dir
            / "summary_by_question_type.csv",
            index=False,
        )
    )

    summary.sort_values(
        [
            "route_f1",
            "evidence_density",
            "abstention_correctness",
        ],
        ascending=False,
    ).to_csv(
        output_dir / "method_ranking.csv",
        index=False,
    )


def run_experiment(
    output_dir: Path,
    limit: int | None = None,
    balanced: bool = False,
    dataset_path_override: str | None = None,
    methods: list[str] | None = None,
) -> dict[str, Any]:
    config = get_config(
        dataset_path_override
    )

    eval_path = find_eval_questions_path(
        config.dataset_path
    )
    rules_path = (
        find_procedural_rules_path(
            config.dataset_path
        )
    )

    questions = select_questions(
        read_jsonl(eval_path),
        limit,
        balanced,
    )

    run_methods = methods or METHODS

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    neo4j = Neo4jMemoryAdapter(
        config.neo4j_uri,
        config.neo4j_user,
        config.neo4j_password,
    )

    procedural = ProceduralRuleMatcher(
        rules_path
    )

    retrieval = MemoryRetrievalEngine(
        neo4j,
        procedural,
    )

    controller = C3LiteController(
        retrieval,
        neo4j,
        config.answer_mode,
        config.openai_model,
        config.openai_api_key,
    )

    predictions: list[
        dict[str, Any]
    ] = []
    scores: list[
        dict[str, Any]
    ] = []

    try:
        counts = neo4j.smoke_counts()

        for question_row in tqdm(
            questions,
            desc="Running questions",
        ):
            for method in run_methods:
                start = time.time()

                user_id = question_row["user_id"]
                question = question_row["question"]

                if method == "no_memory":
                    prediction = no_memory_baseline(
                        question
                    )

                elif method == "simple_retrieval":
                    prediction = (
                        simple_retrieval_baseline(
                            retrieval,
                            user_id,
                            question,
                            config.answer_mode,
                            config.openai_model,
                            config.openai_api_key,
                        )
                    )

                elif method == "all_memory":
                    prediction = all_memory_baseline(
                        retrieval,
                        user_id,
                        question,
                        config.answer_mode,
                        config.openai_model,
                        config.openai_api_key,
                    )

                elif method == "c3_lite_controller":
                    prediction = controller.answer(
                        user_id,
                        question,
                    )

                else:
                    raise ValueError(method)

                expected_abstention = (
                    question_row.get(
                        "should_abstain",
                        not bool(
                            question_row.get(
                                "supporting_memory_ids",
                                [],
                            )
                        ),
                    )
                )

                record = {
                    "question_id": question_row[
                        "question_id"
                    ],
                    "user_id": user_id,
                    "method": method,
                    "question_type": question_row[
                        "question_type"
                    ],
                    "question": question,
                    "gold_answer": question_row.get(
                        "gold_answer"
                    ),
                    "expected_route": question_row.get(
                        "expected_route",
                        [],
                    ),
                    "required_memory": question_row.get(
                        "required_memory",
                        [],
                    ),
                    "supporting_memory_ids": (
                        question_row.get(
                            "supporting_memory_ids",
                            [],
                        )
                    ),
                    "expected_abstention": (
                        expected_abstention
                    ),
                    "answer_should_include": (
                        question_row.get(
                            "answer_should_include",
                            [],
                        )
                    ),
                    "answer_should_not_include": (
                        question_row.get(
                            "answer_should_not_include",
                            [],
                        )
                    ),
                    "explanation_should_mention": (
                        question_row.get(
                            "explanation_should_mention",
                            [],
                        )
                    ),
                    "rubric_score": question_row.get(
                        "rubric_score"
                    ),
                    "predicted_answer": prediction[
                        "answer"
                    ],
                    "predicted_query_type": prediction[
                        "query_type"
                    ],
                    "predicted_route": prediction[
                        "predicted_route"
                    ],
                    "query_profile": prediction.get(
                        "query_profile"
                    ),
                    "route_plan": prediction.get(
                        "route_plan"
                    ),
                    "retrieved_memory_ids": prediction[
                        "retrieved_memory_ids"
                    ],
                    "used_memory_ids": prediction[
                        "used_memory_ids"
                    ],
                    "outdated_memory_ids": prediction[
                        "outdated_memory_ids"
                    ],
                    "historical_memory_ids": (
                        prediction.get(
                            "historical_memory_ids",
                            [],
                        )
                    ),
                    "newer_memory_ids": prediction.get(
                        "newer_memory_ids",
                        [],
                    ),
                    "conflict_notes": prediction[
                        "conflict_notes"
                    ],
                    "supportive_notes": prediction.get(
                        "supportive_notes",
                        [],
                    ),
                    "unresolved_conflicts": (
                        prediction.get(
                            "unresolved_conflicts",
                            [],
                        )
                    ),
                    "confidence_score": prediction.get(
                        "confidence_score"
                    ),
                    "confidence_state": prediction.get(
                        "confidence_state"
                    ),
                    "confidence_reasons": prediction.get(
                        "confidence_reasons",
                        [],
                    ),
                    "query_coverage": prediction.get(
                        "query_coverage"
                    ),
                    "answerability": prediction.get(
                        "answerability"
                    ),
                    "latency_seconds": round(
                        time.time() - start,
                        4,
                    ),
                }

                predictions.append(record)
                scores.append(
                    score_prediction(record)
                )

        write_jsonl(
            output_dir / "predictions.jsonl",
            predictions,
        )
        write_jsonl(
            output_dir / "scores.jsonl",
            scores,
        )

        _summaries(
            output_dir,
            scores,
        )

        run_info = {
            "algorithm_version": (
                "C3-Lite Algorithm v2.1"
            ),
            "dataset_path": str(
                config.dataset_path
            ),
            "eval_questions_path": str(
                eval_path
            ),
            "procedural_rules_path": str(
                rules_path
            ),
            "answer_mode": config.answer_mode,
            "methods": run_methods,
            "num_questions": len(questions),
            "num_predictions": len(
                predictions
            ),
            "neo4j_counts": counts,
            "abstention_label_policy": (
                "Use should_abstain when present; "
                "otherwise derive from empty "
                "supporting_memory_ids for Dataset A v0.1."
            ),
            "output_dir": str(output_dir),
        }

        write_json(
            output_dir / "run_info.json",
            run_info,
        )

        return run_info

    finally:
        neo4j.close()
