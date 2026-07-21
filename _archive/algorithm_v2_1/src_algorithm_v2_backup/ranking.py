from __future__ import annotations
import math, re
from collections import Counter
from datetime import date
from typing import Any

ALIASES = {
    'multi-agent':['multi','agent','memory system'], 'all-memory':['all','memory','prompt'],
    'user study':['user','study','evaluation'], 'data analyst':['data','analyst','career target'],
    'llm agent':['llm','agent','engineer','applied ai'], 'city-centre':['city','centre','hotel location'],
    'city center':['city','centre','hotel location'], 'train station':['train','station','hotel location'],
    'fine-tuning':['fine','tuning','training'], 'production deployment':['production','deployment','claim'],
    'random forest':['random','forest','mlp']}

def tokenize(text: str) -> list[str]:
    return [t for t in re.split(r'[^a-z0-9]+', str(text).lower().replace('_',' ').replace('‑','-').replace('–','-')) if len(t)>=2]

def expand_query(q: str) -> str:
    low=q.lower(); extras=[]
    for phrase, vals in ALIASES.items():
        if phrase in low: extras.extend(vals)
    return q+' '+' '.join(extras)

def _bm25(query: str, docs: list[str], k1: float=1.5, b: float=0.75) -> list[float]:
    qt=tokenize(expand_query(query)); ds=[tokenize(d) for d in docs]; n=len(ds)
    if not n: return []
    avg=sum(map(len,ds))/max(n,1); df=Counter()
    for d in ds:
        for t in set(d): df[t]+=1
    scores=[]
    for d in ds:
        tf=Counter(d); score=0.0
        for t in qt:
            if not df.get(t): continue
            idf=math.log(1+(n-df[t]+0.5)/(df[t]+0.5))
            den=tf[t]+k1*(1-b+b*len(d)/max(avg,1e-9))
            score += idf*(tf[t]*(k1+1))/max(den,1e-9)
        scores.append(score)
    return scores

def _overlap(query: str, doc: str) -> float:
    q,d=set(tokenize(expand_query(query))),set(tokenize(doc))
    return len(q&d)/math.sqrt(len(q)*len(d)) if q and d else 0.0

def _parse(v: Any):
    try: return date.fromisoformat(str(v)[:10])
    except (ValueError,TypeError): return None

def _recency(candidates: list[dict[str,Any]]) -> dict[str,float]:
    dated=[(x['id'],_parse(x.get('date') or x.get('last_updated'))) for x in candidates]
    dated=[x for x in dated if x[1] is not None]
    if not dated: return {x['id']:0.5 for x in candidates}
    lo=min(d for _,d in dated); hi=max(d for _,d in dated); span=max((hi-lo).days,1)
    out={x['id']:0.5 for x in candidates}
    for mid,d in dated: out[mid]=(d-lo).days/span
    return out

def rank_candidates(question: str, candidates: list[dict[str,Any]], memory_type: str, limit: int) -> list[dict[str,Any]]:
    if not candidates: return []
    docs=[str(x.get('text','')) for x in candidates]; raw=_bm25(question,docs); mx=max(raw) if raw else 0.0; rec=_recency(candidates)
    ranked=[]
    for item,bm in zip(candidates,raw):
        bm_n=bm/mx if mx>0 else 0.0; ov=_overlap(question,str(item.get('text',''))); lex=.75*bm_n+.25*ov
        if lex<.03: continue
        if memory_type=='episodic': score=.68*lex+.17*(float(item.get('importance') or 0)/5)+.15*rec[item['id']]
        elif memory_type=='semantic': score=.68*lex+.14*float(item.get('confidence') or 0)+.10*(1.0 if item.get('status')=='active' else .25)+.08*rec[item['id']]
        elif memory_type=='procedural': score=.82*lex+.18*(float(item.get('priority') or 0)/5)
        else: score=lex
        x=dict(item); x.update({'bm25_score':round(bm_n,6),'lexical_overlap':round(ov,6),'relevance_score':round(lex,6),'evidence_score':round(min(1.0,score),6)})
        ranked.append(x)
    ranked.sort(key=lambda x:(x['evidence_score'],x['relevance_score']),reverse=True)
    return ranked[:limit]
