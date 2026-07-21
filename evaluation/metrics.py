from __future__ import annotations

from statistics import mean
from typing import Iterable


def safe_divide(a: float, b: float) -> float:
    return a / b if b else 0.0


def set_metrics(
    predicted: Iterable[str],
    gold: Iterable[str],
) -> dict[str, float]:
    predicted_set = set(predicted)
    gold_set = set(gold)
    overlap = predicted_set & gold_set

    precision = safe_divide(
        len(overlap),
        len(predicted_set),
    )
    recall = safe_divide(
        len(overlap),
        len(gold_set),
    )

    return {
        "exact": float(predicted_set == gold_set),
        "precision": precision,
        "recall": recall,
        "f1": safe_divide(
            2 * precision * recall,
            precision + recall,
        ),
    }


def per_question_metrics(
    *,
    selected_memory_types: list[str],
    expected_memory_types: list[str],
    route_metric_applicable: bool,
    used_ids: list[str],
    gold_ids: list[str],
    decision: str,
    should_abstain: bool,
    expected_outdated_ids: list[str],

    # Legacy post-ranking list retained for compatibility.
    retrieved_ids: list[str],

    # Explicit stages. When omitted, legacy retrieved_ids is used.
    raw_retrieved_ids: list[str] | None = None,
    ranked_candidate_ids: list[str] | None = None,
) -> dict[str, float]:
    raw_ids = (
        raw_retrieved_ids
        if raw_retrieved_ids is not None
        else retrieved_ids
    )
    ranked_ids = (
        ranked_candidate_ids
        if ranked_candidate_ids is not None
        else retrieved_ids
    )

    evidence = set_metrics(used_ids, gold_ids)

    # Outdated-memory retrieval is a retriever-stage metric, so it
    # must use the raw retrieved pool rather than selected evidence.
    outdated = set_metrics(
        raw_ids,
        expected_outdated_ids,
    )

    abstention_correct = (
        decision == "abstain"
        if should_abstain
        else decision != "abstain"
    )

    route_scored = bool(
        route_metric_applicable
        and expected_memory_types
    )

    metrics: dict[str, float] = {
        "route_scored": float(route_scored),
        "evidence_exact": evidence["exact"],
        "evidence_precision": evidence["precision"],
        "evidence_recall": evidence["recall"],
        "evidence_f1": evidence["f1"],
        "evidence_density": evidence["precision"],
        "abstention_correctness": float(
            abstention_correct
        ),
        "expected_outdated_recall": outdated["recall"],

        # Legacy metric: post-ranking candidate count.
        "num_retrieved": float(len(ranked_ids)),

        "num_raw_retrieved": float(len(raw_ids)),
        "num_ranked_candidates": float(
            len(ranked_ids)
        ),
        "num_used": float(len(used_ids)),
    }

    if route_scored:
        route = set_metrics(
            selected_memory_types,
            expected_memory_types,
        )
        metrics.update(
            {
                "route_exact": route["exact"],
                "route_precision": route["precision"],
                "route_recall": route["recall"],
                "route_f1": route["f1"],
            }
        )

    return metrics


def aggregate(
    rows: list[dict[str, float]],
) -> dict[str, float]:
    if not rows:
        return {}

    keys = sorted(
        {
            key
            for row in rows
            for key in row
        }
    )

    return {
        key: mean(
            row[key]
            for row in rows
            if key in row
        )
        for key in keys
    }
