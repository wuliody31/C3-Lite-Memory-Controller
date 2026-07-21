from __future__ import annotations
from typing import Any


def route_accuracy(predicted_route: list[str], expected_route: list[str]) -> float:
    return 1.0 if set(predicted_route) == set(expected_route) else 0.0


def route_precision_recall_f1(predicted_route: list[str], expected_route: list[str]) -> dict[str, float]:
    pred, gold = set(predicted_route), set(expected_route)
    if not pred and not gold:
        return {'route_precision': 1.0, 'route_recall': 1.0, 'route_f1': 1.0}
    if not pred:
        return {'route_precision': 0.0, 'route_recall': 0.0, 'route_f1': 0.0}
    tp = len(pred & gold)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {'route_precision': precision, 'route_recall': recall, 'route_f1': f1}


def evidence_recall(used_memory_ids: list[str], supporting_memory_ids: list[str]) -> float | None:
    if not supporting_memory_ids:
        return None
    return len(set(used_memory_ids) & set(supporting_memory_ids)) / len(set(supporting_memory_ids))


def evidence_precision(used_memory_ids: list[str], supporting_memory_ids: list[str]) -> float | None:
    if not used_memory_ids:
        return None
    return len(set(used_memory_ids) & set(supporting_memory_ids)) / len(set(used_memory_ids))


def should_include_score(answer: str, should_include: list[str]) -> float | None:
    if not should_include:
        return None
    a = answer.lower()
    return sum(1 for item in should_include if item.lower() in a) / len(should_include)


def should_not_include_violations(answer: str, should_not_include: list[str]) -> int:
    a = answer.lower()
    return sum(1 for item in should_not_include if item.lower() in a)


def abstention_heuristic(answer: str) -> float:
    a = answer.lower()
    phrases = ['not enough','insufficient','cannot confirm','do not have','unknown','needs to be checked','没有足够','不能确认','不知道']
    return 1.0 if any(p in a for p in phrases) else 0.0


def score_prediction(record: dict[str, Any]) -> dict[str, Any]:
    answer = record['predicted_answer']
    qtype = record['question_type']
    route_scores = route_precision_recall_f1(record['predicted_route'], record['expected_route'])
    return {'question_id': record['question_id'], 'method': record['method'], 'question_type': qtype, 'route_exact_match': route_accuracy(record['predicted_route'], record['expected_route']), **route_scores, 'evidence_recall': evidence_recall(record['used_memory_ids'], record['supporting_memory_ids']), 'evidence_precision': evidence_precision(record['used_memory_ids'], record['supporting_memory_ids']), 'include_score': should_include_score(answer, record.get('answer_should_include', [])), 'not_include_violations': should_not_include_violations(answer, record.get('answer_should_not_include', [])), 'abstention_heuristic': abstention_heuristic(answer) if qtype == 'abstention' else None, 'latency_seconds': record['latency_seconds'], 'num_retrieved': len(record['retrieved_memory_ids']), 'num_used': len(record['used_memory_ids']), 'num_outdated': len(record['outdated_memory_ids'])}
