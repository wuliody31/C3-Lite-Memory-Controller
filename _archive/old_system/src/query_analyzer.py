from __future__ import annotations
from dataclasses import dataclass
from typing import Any

INTENTS = ('episodic_recall','current_fact','temporal_update','conflict_resolution','procedural_following','abstention','explainability','general')

@dataclass(frozen=True)
class QueryProfile:
    primary_intent: str
    intent_scores: dict[str, float]
    flags: tuple[str, ...]
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {'primary_intent': self.primary_intent, 'intent_scores': self.intent_scores, 'flags': list(self.flags), 'reasons': list(self.reasons)}

def _any(text: str, patterns: list[str]) -> bool:
    return any(p in text for p in patterns)

def analyse_query(question: str) -> QueryProfile:
    q = question.lower().replace('‑','-').replace('–','-')
    raw = {k: 0.05 for k in INTENTS}
    flags: set[str] = set()
    reasons: list[str] = []

    def add(intent: str, weight: float, reason: str) -> None:
        raw[intent] += weight
        reasons.append(f'{intent}: {reason}')

    cue_groups = {
        'abstention': (0.52, ['not enough','no stored evidence','without evidence','not stored','exact current','exact price','exact opening','explicitly','definitely','cannot confirm','do you know','invent exact','unknown']),
        'explainability': (0.42, ['explain','why','evidence','support','cite','which memory','memory types used','and why']),
        'conflict_resolution': (0.42, ['which is more current','conflict','supersede','outdated','rather than',' or ','mainly about proving','completely removed']),
        'temporal_update': (0.45, ['over time','changed','change over time','shifted','became','more current','always','earlier','later','updated']),
        'procedural_following': (0.52, ['how should','what should you do','should the system','how should the system','what should i say','answer in ','style','format','answer structure','rewrite','should you','should the itinerary','how should my cv wording']),
        'episodic_recall': (0.45, ['originally','previously','before','past','did i ask','what happened','at york']),
        'current_fact': (0.28, ['current','currently','main','focus','role','types','baselines','abilities','selected','include','preference','location']),
    }
    for intent, (weight, cues) in cue_groups.items():
        for cue in cues:
            if cue in q: add(intent, weight, f"matched cue '{cue}'")

    if _any(q, ['how did my','how did the','how did i','over time','which is more current','always']):
        add('temporal_update', 0.55, 'explicit state-change or temporal-comparison question')
        flags.add('temporal_comparison')
    if 'originally' in q and not _any(q, ['how did','over time','changed','change','more current']):
        add('episodic_recall', 0.50, 'past-only recall')
        raw['temporal_update'] = max(0.0, raw['temporal_update'] - 0.30)
        flags.add('past_only')
    if _any(q, ['what evidence supports','which memories support','cite the memory','which memory supports','memory types used']):
        add('explainability', 0.45, 'explicit evidence-attribution request')
        flags.add('evidence_attribution')
    if _any(q, ['if an older memory conflicts','if a cv claim is not stored','if there is no stored evidence']):
        add('procedural_following', 0.65, 'general memory-handling policy')
        flags.add('pure_meta_policy')
    if _any(q, ['what technical details should i add','what should i mention','what msc project difficulty themes','did i ask for walking routes']):
        flags.add('plan_history')
    if _any(q, ['current travel prices are not stored','invent exact prices','booking information','booking screen','ticket instruction','bus or train','restaurant recommendations','itinerary']):
        flags.add('domain_policy')
    if _any(q, ['should my evaluation mainly','should my cv now prioritise','station area or the current','large multi-agent system or a smaller']):
        flags.add('decision_conflict')

    scores = {k: round(max(0.0,v)/(1.0+max(0.0,v)), 6) for k,v in raw.items()}
    primary = max(scores, key=scores.get)
    return QueryProfile(primary, scores, tuple(sorted(flags)), tuple(reasons))
