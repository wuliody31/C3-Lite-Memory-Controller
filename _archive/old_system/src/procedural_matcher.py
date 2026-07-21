from __future__ import annotations
from pathlib import Path
from typing import Any
from src.io_utils import read_json

class ProceduralRuleMatcher:
    def __init__(self,rules_path:Path|str): self.rules_path=Path(rules_path); self.rules:list[dict[str,Any]]=read_json(self.rules_path)
    def get_candidates(self,user_id:str):
        return [{'id':r['rule_id'],'memory_type':'procedural','text':f"{r['condition']} -> {r['action']}",'name':r.get('name'),'condition':r.get('condition'),'action':r.get('action'),'priority':int(r.get('priority',0))} for r in self.rules if r.get('user_id')==user_id]
    def match(self,user_id:str,question:str,limit:int=5):
        from src.ranking import rank_candidates
        return rank_candidates(question,self.get_candidates(user_id),'procedural',limit)
