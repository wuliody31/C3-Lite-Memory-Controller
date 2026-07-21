from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from src.query_analyzer import QueryProfile

E = 'neo4j_episodic_graph'
S = 'neo4j_semantic_graph'
P = 'json_procedural_rules'

@dataclass(frozen=True)
class RoutePlan:
    routes: tuple[str, ...]
    route_scores: dict[str, float]
    confidence: float
    include_archived_semantic: bool
    reasons: tuple[str, ...]
    def to_dict(self) -> dict[str, Any]:
        return {'routes': list(self.routes), 'route_scores': self.route_scores, 'confidence': self.confidence, 'include_archived_semantic': self.include_archived_semantic, 'reasons': list(self.reasons)}

def _scores(profile: QueryProfile) -> dict[str, float]:
    s = profile.intent_scores
    return {
        E: round(min(1.0, 0.90*s['episodic_recall'] + 0.85*s['temporal_update'] + 0.72*s['conflict_resolution'] + 0.38*s['explainability'] + 0.42*s['abstention']), 6),
        S: round(min(1.0, 0.90*s['current_fact'] + 0.88*s['conflict_resolution'] + 0.80*s['temporal_update'] + 0.62*s['procedural_following'] + 0.62*s['explainability'] + 0.72*s['abstention']), 6),
        P: round(min(1.0, 0.96*s['procedural_following'] + 0.42*s['conflict_resolution'] + 0.24*s['explainability']), 6),
    }

def plan_routes(question: str, profile: QueryProfile) -> RoutePlan:
    q = question.lower().replace('‑','-').replace('–','-')
    f, s, rs = set(profile.flags), profile.intent_scores, _scores(profile)
    reasons: list[str] = []

    if 'pure_meta_policy' in f:
        routes, conf = (P,), 0.95; reasons.append('general memory-handling policy')
    elif 'plan_history' in f:
        routes, conf = (E,S), 0.90; reasons.append('planned/detail recall needs event history and current facts')
    elif 'temporal_comparison' in f and s['procedural_following'] < 0.45:
        routes, conf = (E,S), 0.95; reasons.append('temporal comparison needs old and current state')
    elif 'past_only' in f:
        routes, conf = (E,), 0.95; reasons.append('past-only recall')
    elif 'domain_policy' in f and s['procedural_following'] >= 0.35:
        if s['explainability'] >= 0.38 or s['conflict_resolution'] >= 0.38:
            routes, conf = (E,S,P), 0.90; reasons.append('domain policy plus historical/explanation context')
        else:
            routes, conf = (P,S), 0.90; reasons.append('domain policy plus current domain-state evidence')
    elif 'decision_conflict' in f:
        routes, conf = (E,S,P), 0.92; reasons.append('decision conflict needs old/new evidence and policy')
    elif s['explainability'] >= 0.50:
        if q.startswith('why ') and not any(x in q for x in ['first-day','first day']):
            routes, conf = (S,), 0.85; reasons.append('design rationale is primarily semantic')
        else:
            routes, conf = (E,S), 0.85; reasons.append('evidence attribution needs event and current-fact evidence')
    elif s['procedural_following'] >= 0.45:
        routes, conf = (P,S), 0.88; reasons.append('normative task needs procedural policy and current facts')
    elif s['conflict_resolution'] >= 0.43:
        routes, conf = (E,S), 0.88; reasons.append('state conflict needs historical and current evidence')
    elif s['abstention'] >= 0.40:
        routes, conf = (E,S), 0.85; reasons.append('evidence-existence check searches both graph memory types')
    elif s['episodic_recall'] >= 0.47 and s['temporal_update'] < 0.38:
        routes, conf = (E,), 0.88; reasons.append('past-event signal dominates')
    else:
        routes, conf = (S,), 0.80; reasons.append('default current-fact route')

    include_archived = E in routes and S in routes and (s['conflict_resolution'] >= 0.35 or s['temporal_update'] >= 0.35 or s['explainability'] >= 0.45)
    return RoutePlan(routes, rs, conf, include_archived, tuple(reasons))
