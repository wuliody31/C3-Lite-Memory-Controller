from __future__ import annotations
from collections import defaultdict
from typing import Any

INVALIDATING={'SUPERSEDES','UPDATES','OVERRIDES','REPLACES'}
SUPPORTIVE={'CLARIFIES','ALIGNS_WITH','DERIVED_FROM'}

def _date_key(x): return str(x.get('last_updated') or x.get('date') or '')

class ConflictResolver:
    """Relation-aware conflict resolution; supportive relations do not invalidate memories."""
    def __init__(self,neo4j_adapter): self.neo4j=neo4j_adapter
    def resolve(self,memories:list[dict[str,Any]],query_type:str):
        mmap={x['id']:dict(x) for x in memories}; rels=self.neo4j.find_memory_relations(list(mmap))
        outdated=set(); newer=set(); conflict_notes=[]; supportive_notes=[]
        for r in rels:
            rel=str(r['relation']).upper(); note={'older_memory':r['older_memory'],'newer_memory':r['newer_memory'],'relation':rel,'reason':r.get('reason')}
            if rel in INVALIDATING: outdated.add(r['older_memory']); newer.add(r['newer_memory']); conflict_notes.append(note)
            elif rel in SUPPORTIVE: supportive_notes.append(note)
        for mid in sorted(newer):
            if mid not in mmap:
                m=self.neo4j.get_memory_by_id(mid)
                if m: m['evidence_score']=max(.72,float(m.get('evidence_score') or 0)); m['relevance_score']=float(m.get('relevance_score') or 0); m['graph_expanded']=True; mmap[mid]=m

        groups=defaultdict(list)
        for x in mmap.values():
            if x.get('memory_type')=='semantic': groups[(str(x.get('subject')),str(x.get('relation')))].append(x)
        unresolved=[]
        for key,group in groups.items():
            if len({str(x.get('object')) for x in group})<=1: continue
            ordered=sorted(group,key=lambda x:(1 if x.get('status')=='active' else 0,_date_key(x),float(x.get('confidence') or 0)),reverse=True); preferred=ordered[0]
            for old in ordered[1:]:
                if old['id'] in outdated: continue
                if old.get('status')=='archived' or _date_key(old)<_date_key(preferred):
                    outdated.add(old['id']); conflict_notes.append({'older_memory':old['id'],'newer_memory':preferred['id'],'relation':'IMPLICIT_VERSION_CONFLICT','reason':f'Same subject/relation {key} has different objects; active/newer fact preferred.'})
                else: unresolved.append({'memory_ids':[preferred['id'],old['id']],'reason':f'Multiple active semantic facts disagree for {key}.'})

        allow_history=query_type in {'temporal_update','conflict_resolution','explainability'}
        current=[]; historical=[]
        for x in mmap.values():
            y=dict(x)
            if x['id'] in outdated: y['evidence_role']='historical'; historical.append(y)
            else: y['evidence_role']='current'; current.append(y)
        current.sort(key=lambda x:x.get('evidence_score',0),reverse=True); historical.sort(key=lambda x:x.get('evidence_score',0),reverse=True)
        return {'usable_memories':current+historical if allow_history else current,'current_memories':current,'historical_memories':historical,'outdated_memory_ids':sorted(outdated),'newer_memory_ids':sorted(newer),'conflict_notes':conflict_notes,'supportive_notes':supportive_notes,'unresolved_conflicts':unresolved}
