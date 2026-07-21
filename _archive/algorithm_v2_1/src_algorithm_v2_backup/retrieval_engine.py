from __future__ import annotations
from typing import Any
from src.ranking import rank_candidates
from src.route_planner import E,S,P

class MemoryRetrievalEngine:
    """Shared BM25+metadata retrieval primitive for C3-Lite and baselines."""
    def __init__(self, neo4j_adapter, procedural_matcher):
        self.neo4j=neo4j_adapter; self.procedural=procedural_matcher
    def retrieve_episodic(self,user_id:str,question:str,limit:int=8):
        return rank_candidates(question,self.neo4j.get_episodic_candidates(user_id),'episodic',limit)
    def retrieve_semantic(self,user_id:str,question:str,limit:int=8,include_archived:bool=False):
        return rank_candidates(question,self.neo4j.get_semantic_candidates(user_id,include_archived),'semantic',limit)
    def retrieve_procedural(self,user_id:str,question:str,limit:int=4):
        return rank_candidates(question,self.procedural.get_candidates(user_id),'procedural',limit)
    def retrieve_routes(self,user_id:str,question:str,routes,include_archived_semantic:bool=False,per_route_limit:int=8):
        out={}
        if E in routes: out[E]=self.retrieve_episodic(user_id,question,per_route_limit)
        if S in routes: out[S]=self.retrieve_semantic(user_id,question,per_route_limit,include_archived_semantic)
        if P in routes: out[P]=self.retrieve_procedural(user_id,question,min(5,per_route_limit))
        return out
    @staticmethod
    def flatten(bundle: dict[str,list[dict[str,Any]]]):
        merged={}
        for items in bundle.values():
            for x in items:
                if x['id'] not in merged or x.get('evidence_score',0)>merged[x['id']].get('evidence_score',0): merged[x['id']]=x
        return sorted(merged.values(),key=lambda x:x.get('evidence_score',0),reverse=True)
