from __future__ import annotations
from typing import Any

def route_accuracy(predicted_route,expected_route): return 1.0 if set(predicted_route)==set(expected_route) else 0.0

def route_precision_recall_f1(predicted_route,expected_route):
    pred,gold=set(predicted_route),set(expected_route)
    if not pred and not gold: return {'route_precision':1.0,'route_recall':1.0,'route_f1':1.0}
    if not pred: return {'route_precision':0.0,'route_recall':0.0,'route_f1':0.0}
    tp=len(pred&gold); p=tp/len(pred); r=tp/len(gold) if gold else 0.0; f=2*p*r/(p+r) if p+r else 0.0
    return {'route_precision':p,'route_recall':r,'route_f1':f}

def evidence_recall(used,gold): return None if not gold else len(set(used)&set(gold))/len(set(gold))
def evidence_precision(used,gold): return None if not used else len(set(used)&set(gold))/len(set(used))
def evidence_density(used,gold): return 0.0 if not used else len(set(used)&set(gold))/len(set(used))
def should_include_score(answer,items): return None if not items else sum(1 for x in items if x.lower() in answer.lower())/len(items)
def should_not_include_violations(answer,items): return sum(1 for x in items if x.lower() in answer.lower())
def abstention_decision(
    answer: str,
    confidence_state: str | None,
) -> float:
    if confidence_state is not None:
        return 1.0 if confidence_state == "abstain" else 0.0

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

    return 1.0 if any(
        phrase in answer_lower
        for phrase in phrases
    ) else 0.0

def score_prediction(record:dict[str,Any]):
    abstained=abstention_decision(record['predicted_answer'],record.get('confidence_state')); expected=1.0 if record['question_type']=='abstention' else 0.0; rs=route_precision_recall_f1(record['predicted_route'],record['expected_route'])
    return {'question_id':record['question_id'],'method':record['method'],'question_type':record['question_type'],'route_exact_match':route_accuracy(record['predicted_route'],record['expected_route']),**rs,'evidence_recall':evidence_recall(record['used_memory_ids'],record['supporting_memory_ids']),'evidence_precision':evidence_precision(record['used_memory_ids'],record['supporting_memory_ids']),'evidence_density':evidence_density(record['used_memory_ids'],record['supporting_memory_ids']),'include_score':should_include_score(record['predicted_answer'],record.get('answer_should_include',[])),'not_include_violations':should_not_include_violations(record['predicted_answer'],record.get('answer_should_not_include',[])),'abstained':abstained,'expected_abstention':expected,'abstention_correctness':1.0 if abstained==expected else 0.0,'confidence_score':record.get('confidence_score'),'latency_seconds':record['latency_seconds'],'num_retrieved':len(record['retrieved_memory_ids']),'num_used':len(record['used_memory_ids']),'num_outdated':len(record['outdated_memory_ids']),'conflict_action':1.0 if record.get('outdated_memory_ids') else 0.0}
