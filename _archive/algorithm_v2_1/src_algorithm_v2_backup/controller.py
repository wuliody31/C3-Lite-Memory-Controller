from __future__ import annotations
from typing import Any
from src.query_analyzer import analyse_query
from src.route_planner import plan_routes
from src.conflict_resolver import ConflictResolver
from src.confidence_calibrator import calibrate_confidence
from src.answer_generator import generate_answer

class C3LiteController:
    """C3-Lite v2: context-adaptive routing, conflict resolution, confidence calibration."""
    def __init__(self,retrieval_engine,neo4j_adapter,answer_mode:str='mock',openai_model:str='gpt-4o-mini',openai_api_key:str|None=None):
        self.retrieval=retrieval_engine; self.resolver=ConflictResolver(neo4j_adapter); self.answer_mode=answer_mode; self.openai_model=openai_model; self.openai_api_key=openai_api_key
    @staticmethod
    def _select(resolved:dict[str,Any],routes:list[str],budget:int):
        usable=resolved['usable_memories']; selected={}; route_type={'neo4j_episodic_graph':'episodic','neo4j_semantic_graph':'semantic','json_procedural_rules':'procedural'}
        for route in routes:
            c=[x for x in usable if x.get('memory_type')==route_type[route]]
            if c:
                best=max(c,key=lambda x:x.get('evidence_score',0)); selected[best['id']]=best
        for x in sorted(usable,key=lambda x:x.get('evidence_score',0),reverse=True):
            if len(selected)>=budget: break
            selected[x['id']]=x
        return list(selected.values())[:budget]
    def answer(self,user_id:str,question:str):
        profile=analyse_query(question); plan=plan_routes(question,profile)
        bundle=self.retrieval.retrieve_routes(user_id,question,plan.routes,plan.include_archived_semantic,8); retrieved=self.retrieval.flatten(bundle)
        resolution=self.resolver.resolve(retrieved,profile.primary_intent); budget=12 if profile.primary_intent in {'temporal_update','conflict_resolution','explainability'} else 10; selected=self._select(resolution,list(plan.routes),budget)
        conf=calibrate_confidence(selected,plan.confidence,resolution['unresolved_conflicts'],profile.primary_intent)
        if conf.state=='abstain': answer=f'There is not enough stored evidence to answer this question confidently. Confidence score: {conf.score:.3f}.'
        else:
            answer=generate_answer(self.answer_mode,question,selected,resolution, [x for x in selected if x.get('memory_type')=='procedural'],profile.primary_intent,self.openai_model,self.openai_api_key)
            if conf.state=='caveat': answer='Based on limited stored evidence, '+answer
        return {'question':question,'query_type':profile.primary_intent,'predicted_route':list(plan.routes),'query_profile':profile.to_dict(),'route_plan':plan.to_dict(),'retrieved_memory_ids':[x['id'] for x in retrieved],'used_memory_ids':[x['id'] for x in selected],'outdated_memory_ids':resolution['outdated_memory_ids'],'historical_memory_ids':[x['id'] for x in resolution['historical_memories']],'newer_memory_ids':resolution['newer_memory_ids'],'conflict_notes':resolution['conflict_notes'],'supportive_notes':resolution['supportive_notes'],'unresolved_conflicts':resolution['unresolved_conflicts'],'confidence_score':conf.score,'confidence_state':conf.state,'confidence_reasons':list(conf.reasons),'answer':answer}
