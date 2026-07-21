from __future__ import annotations
from typing import Any
from src.answer_generator import generate_answer

def _resolution(memories): return {'resolved_memories':memories,'usable_memories':memories,'current_memories':memories,'historical_memories':[],'outdated_memory_ids':[],'newer_memory_ids':[],'conflict_notes':[],'supportive_notes':[],'unresolved_conflicts':[]}

def no_memory_baseline(question:str):
    return {'question':question,'query_type':'no_memory','predicted_route':[],'retrieved_memory_ids':[],'used_memory_ids':[],'outdated_memory_ids':[],'historical_memory_ids':[],'newer_memory_ids':[],'conflict_notes':[],'supportive_notes':[],'unresolved_conflicts':[],'confidence_score':0.0,'confidence_state':'abstain','confidence_reasons':['no memory available'],'answer':'I do not have access to stored memory for this question.'}

def simple_retrieval_baseline(retrieval_engine,user_id:str,question:str,answer_mode:str='mock',openai_model:str='gpt-4o-mini',openai_api_key:str|None=None):
    routes=['neo4j_episodic_graph','neo4j_semantic_graph']; bundle=retrieval_engine.retrieve_routes(user_id,question,routes,True,8); memories=retrieval_engine.flatten(bundle)[:10]; res=_resolution(memories); answer=generate_answer(answer_mode,question,memories,res,[],'simple_retrieval',openai_model,openai_api_key)
    return {'question':question,'query_type':'simple_retrieval','predicted_route':routes,'retrieved_memory_ids':[x['id'] for x in memories],'used_memory_ids':[x['id'] for x in memories],'outdated_memory_ids':[],'historical_memory_ids':[],'newer_memory_ids':[],'conflict_notes':[],'supportive_notes':[],'unresolved_conflicts':[],'confidence_score':1.0,'confidence_state':'direct','confidence_reasons':['baseline fixed state'],'answer':answer}

def all_memory_baseline(retrieval_engine,user_id:str,question:str,answer_mode:str='mock',openai_model:str='gpt-4o-mini',openai_api_key:str|None=None):
    routes=['neo4j_episodic_graph','neo4j_semantic_graph','json_procedural_rules']; bundle=retrieval_engine.retrieve_routes(user_id,question,routes,True,10); memories=retrieval_engine.flatten(bundle)[:14]; res=_resolution(memories); rules=[x for x in memories if x.get('memory_type')=='procedural']; answer=generate_answer(answer_mode,question,memories,res,rules,'all_memory',openai_model,openai_api_key)
    return {'question':question,'query_type':'all_memory','predicted_route':routes,'retrieved_memory_ids':[x['id'] for x in memories],'used_memory_ids':[x['id'] for x in memories],'outdated_memory_ids':[],'historical_memory_ids':[],'newer_memory_ids':[],'conflict_notes':[],'supportive_notes':[],'unresolved_conflicts':[],'confidence_score':1.0,'confidence_state':'direct','confidence_reasons':['baseline fixed state'],'answer':answer}
