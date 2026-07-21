from __future__ import annotations
from dataclasses import dataclass
from statistics import mean
from typing import Any

@dataclass(frozen=True)
class ConfidenceDecision:
    score: float
    state: str
    reasons: tuple[str,...]
    def to_dict(self)->dict[str,Any]: return {'score':self.score,'state':self.state,'reasons':list(self.reasons)}

def calibrate_confidence(selected_memories:list[dict[str,Any]],route_confidence:float,unresolved_conflicts:list[dict[str,Any]],query_type:str)->ConfidenceDecision:
    if not selected_memories: return ConfidenceDecision(0.0,'abstain',('no relevant memory evidence selected',))
    vals=sorted([float(x.get('evidence_score') or 0) for x in selected_memories],reverse=True); top=vals[0]; top3=mean(vals[:3]); agreement=max(0.0,1.0-min(.8,.35*len(unresolved_conflicts)))
    adequacy=.45*top+.25*top3+.15*route_confidence+.15*agreement
    reasons=(f'top_evidence={top:.3f}',f'mean_top3={top3:.3f}',f'route_confidence={route_confidence:.3f}',f'agreement={agreement:.3f}')
    if top<.22: state='abstain'
    elif query_type=='abstention' and adequacy<.58: state='abstain'
    elif adequacy>=.64: state='direct'
    elif adequacy>=.46: state='caveat'
    else: state='abstain'
    return ConfidenceDecision(round(adequacy,6),state,reasons)
