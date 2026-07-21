from __future__ import annotations

from typing import Any


def route_accuracy(
    predicted_route: list[str],
    expected_route: list[str],
) -> float:
    return (
        1.0
        if set(predicted_route)
        == set(expected_route)
        else 0.0
    )


def route_precision_recall_f1(
    predicted_route: list[str],
    expected_route: list[str],
) -> dict[str, float]:
    predicted = set(predicted_route)
    gold = set(expected_route)

    if not predicted and not gold:
        return {
            "route_precision": 1.0,
            "route_recall": 1.0,
            "route_f1": 1.0,
        }

    if not predicted:
        return {
            "route_precision": 0.0,
            "route_recall": 0.0,
            "route_f1": 0.0,
        }

    true_positive = len(predicted & gold)

    precision = (
        true_positive / len(predicted)
    )
    recall = (
        true_positive / len(gold)
        if gold
        else 0.0
    )

    f1 = (
        2.0
        * precision
        * recall
        / (precision + recall)
        if precision + recall
        else 0.0
    )

    return {
        "route_precision": precision,
        "route_recall": recall,
        "route_f1": f1,
    }


def evidence_recall(
    used_memory_ids: list[str],
    supporting_memory_ids: list[str],
) -> float | None:
    if not supporting_memory_ids:
        return None

    used = set(used_memory_ids)
    gold = set(supporting_memory_ids)

    return len(used & gold) / len(gold)


def evidence_precision(
    used_memory_ids: list[str],
    supporting_memory_ids: list[str],
) -> float | None:
    if not used_memory_ids:
        return None

    used = set(used_memory_ids)
    gold = set(supporting_memory_ids)

    return len(used & gold) / len(used)


def evidence_density(
    used_memory_ids: list[str],
    supporting_memory_ids: list[str],
) -> float:
    if not used_memory_ids:
        return 0.0

    return (
        len(
            set(used_memory_ids)
            & set(supporting_memory_ids)
        )
        / len(set(used_memory_ids))
    )


def conflict_precision_recall_f1(
    predicted_outdated_ids: list[str],
    expected_outdated_ids: list[str],
) -> dict[str, float | None]:
    """
    Evaluate conflict-aware outdated-memory detection.

    predicted_outdated_ids:
        Memory IDs marked by the system as outdated / historical.

    expected_outdated_ids:
        Gold outdated-memory IDs annotated in Dataset A.

    If neither predicted nor gold outdated memories exist, the
    question is treated as not conflict-relevant for this metric,
    so None is returned for precision/recall/F1. Pandas will ignore
    these values when taking the mean.
    """
    predicted = set(predicted_outdated_ids or [])
    gold = set(expected_outdated_ids or [])

    if not predicted and not gold:
        return {
            "conflict_precision": None,
            "conflict_recall": None,
            "conflict_f1": None,
        }

    true_positive = len(predicted & gold)

    precision = (
        true_positive / len(predicted)
        if predicted
        else 0.0
    )

    recall = (
        true_positive / len(gold)
        if gold
        else 0.0
    )

    f1 = (
        2.0
        * precision
        * recall
        / (precision + recall)
        if precision + recall
        else 0.0
    )

    return {
        "conflict_precision": precision,
        "conflict_recall": recall,
        "conflict_f1": f1,
    }


def should_include_score(
    answer: str,
    should_include: list[str],
) -> float | None:
    if not should_include:
        return None

    answer_lower = answer.lower()

    hits = sum(
        1
        for item in should_include
        if item.lower() in answer_lower
    )

    return hits / len(should_include)


def should_not_include_violations(
    answer: str,
    should_not_include: list[str],
) -> int:
    answer_lower = answer.lower()

    return sum(
        1
        for item in should_not_include
        if item.lower() in answer_lower
    )


def abstention_decision(
    answer: str,
    confidence_state: str | None,
) -> float:
    """
    Decide whether the system abstained.

    Structured controller state is authoritative. Text matching is
    only used as a fallback for methods without confidence_state.
    """
    if confidence_state is not None:
        return (
            1.0
            if confidence_state == "abstain"
            else 0.0
        )

    answer_lower = answer.lower()

    phrases = [
        "not enough",
        "insufficient",
        "cannot confirm",
        "do not have access",
        "unknown",
        "needs to be checked",
        "没有足够",
        "不能确认",
        "不知道",
    ]

    return (
        1.0
        if any(
            phrase in answer_lower
            for phrase in phrases
        )
        else 0.0
    )


def score_prediction(
    record: dict[str, Any],
) -> dict[str, Any]:
    abstained = abstention_decision(
        record["predicted_answer"],
        record.get("confidence_state"),
    )

    # question_type == "abstention" is a capability category,
    # not necessarily the gold action. Some abstention-category
    # questions have explicit negative evidence and should be answered.
    expected_abstention = float(
        record.get(
            "expected_abstention",
            not bool(
                record.get(
                    "supporting_memory_ids",
                    [],
                )
            ),
        )
    )

    route_scores = (
        route_precision_recall_f1(
            record["predicted_route"],
            record["expected_route"],
        )
    )

    conflict_scores = (
        conflict_precision_recall_f1(
            record.get("outdated_memory_ids", []),
            record.get("expected_outdated_memory_ids", []),
        )
    )

    return {
        "question_id": record["question_id"],
        "method": record["method"],
        "question_type": record[
            "question_type"
        ],
        "conflict_type": record.get(
            "conflict_type",
            "none",
        ),
        "route_exact_match": route_accuracy(
            record["predicted_route"],
            record["expected_route"],
        ),
        **route_scores,
        "evidence_recall": evidence_recall(
            record["used_memory_ids"],
            record["supporting_memory_ids"],
        ),
        "evidence_precision": evidence_precision(
            record["used_memory_ids"],
            record["supporting_memory_ids"],
        ),
        "evidence_density": evidence_density(
            record["used_memory_ids"],
            record["supporting_memory_ids"],
        ),
        "include_score": should_include_score(
            record["predicted_answer"],
            record.get(
                "answer_should_include",
                [],
            ),
        ),
        "not_include_violations": (
            should_not_include_violations(
                record["predicted_answer"],
                record.get(
                    "answer_should_not_include",
                    [],
                ),
            )
        ),
        "abstained": abstained,
        "expected_abstention": (
            expected_abstention
        ),
        "abstention_correctness": (
            1.0
            if abstained
            == expected_abstention
            else 0.0
        ),
        "confidence_score": record.get(
            "confidence_score"
        ),
        "query_coverage": record.get(
            "query_coverage"
        ),
        "latency_seconds": record[
            "latency_seconds"
        ],
        "num_retrieved": len(
            record["retrieved_memory_ids"]
        ),
        "num_used": len(
            record["used_memory_ids"]
        ),
        "num_outdated": len(
            record["outdated_memory_ids"]
        ),
        "expected_num_outdated": len(
            record.get(
                "expected_outdated_memory_ids",
                [],
            )
        ),
        "conflict_action": (
            1.0
            if record.get(
                "outdated_memory_ids"
            )
            else 0.0
        ),
        "expected_conflict_action": (
            1.0
            if record.get(
                "expected_outdated_memory_ids"
            )
            else 0.0
        ),
        **conflict_scores,
    }