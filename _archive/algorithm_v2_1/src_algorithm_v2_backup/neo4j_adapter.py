from __future__ import annotations
from typing import Any
from neo4j import GraphDatabase

class Neo4jMemoryAdapter:
    def __init__(self,uri:str,user:str,password:str|None):
        if not password: raise ValueError('NEO4J_PASSWORD is missing. Set it in .env.')
        self.driver=GraphDatabase.driver(uri,auth=(user,password))
    def close(self): self.driver.close()
    def ping(self):
        with self.driver.session() as s: return s.run('RETURN 1 AS ok').single()['ok']==1
    def smoke_counts(self):
        q='''MATCH (u:User) WITH count(u) AS users MATCH (s:Session) WITH users,count(s) AS sessions MATCH (e:EpisodicEvent) WITH users,sessions,count(e) AS episodic MATCH (f:SemanticFact) WITH users,sessions,episodic,count(f) AS semantic MATCH (en:Entity) WITH users,sessions,episodic,semantic,count(en) AS entities RETURN users,sessions,episodic,semantic,entities'''
        with self.driver.session() as s:
            row=s.run(q).single(); return dict(row) if row else {}
    def get_episodic_candidates(self,user_id:str,limit:int=200):
        q='''MATCH (:User {user_id:$user_id})-[:HAS_EPISODIC_MEMORY]->(e:EpisodicEvent) RETURN e.memory_id AS id,'episodic' AS memory_type,toString(e.date) AS date,e.event AS text,e.importance AS importance,e.source_session AS source,e.entities AS entities ORDER BY e.date DESC,e.importance DESC LIMIT $limit'''
        with self.driver.session() as s: return [dict(r) for r in s.run(q,user_id=user_id,limit=limit)]
    def get_semantic_candidates(self,user_id:str,include_archived:bool=False,limit:int=300):
        filt='' if include_archived else "WHERE f.status='active'"
        q=f'''MATCH (:User {{user_id:$user_id}})-[:HAS_SEMANTIC_FACT]->(f:SemanticFact) {filt} RETURN f.triple_id AS id,'semantic' AS memory_type,f.subject+' --'+f.relation+'--> '+f.object AS text,f.subject AS subject,f.relation AS relation,f.object AS object,f.status AS status,f.confidence AS confidence,toString(f.last_updated) AS last_updated,f.source AS source ORDER BY f.last_updated DESC,f.confidence DESC LIMIT $limit'''
        with self.driver.session() as s: return [dict(r) for r in s.run(q,user_id=user_id,limit=limit)]
    def search_episodic(self,user_id:str,keyword:str,limit:int=8):
        q='''MATCH (:User {user_id:$user_id})-[:HAS_EPISODIC_MEMORY]->(e:EpisodicEvent) WHERE toLower(e.event) CONTAINS toLower($keyword) OR toLower(coalesce(e.entities,'')) CONTAINS toLower($keyword) RETURN e.memory_id AS id,'episodic' AS memory_type,toString(e.date) AS date,e.event AS text,e.importance AS importance,e.source_session AS source ORDER BY e.date DESC,e.importance DESC LIMIT $limit'''
        with self.driver.session() as s: return [dict(r) for r in s.run(q,user_id=user_id,keyword=keyword,limit=limit)]
    def search_semantic(self,user_id:str,keyword:str,limit:int=8,active_only:bool=True):
        filt="AND f.status='active'" if active_only else ''
        q=f'''MATCH (:User {{user_id:$user_id}})-[:HAS_SEMANTIC_FACT]->(f:SemanticFact) WHERE (toLower(f.subject) CONTAINS toLower($keyword) OR toLower(f.relation) CONTAINS toLower($keyword) OR toLower(f.object) CONTAINS toLower($keyword)) {filt} RETURN f.triple_id AS id,'semantic' AS memory_type,f.subject+' --'+f.relation+'--> '+f.object AS text,f.subject AS subject,f.relation AS relation,f.object AS object,f.status AS status,f.confidence AS confidence,toString(f.last_updated) AS last_updated,f.source AS source ORDER BY f.confidence DESC,f.last_updated DESC LIMIT $limit'''
        with self.driver.session() as s: return [dict(r) for r in s.run(q,user_id=user_id,keyword=keyword,limit=limit)]
    def get_memory_by_id(self,memory_id:str):
        q='''OPTIONAL MATCH (e:EpisodicEvent {memory_id:$memory_id}) OPTIONAL MATCH (f:SemanticFact {triple_id:$memory_id}) RETURN CASE WHEN e IS NOT NULL THEN e.memory_id ELSE f.triple_id END AS id,CASE WHEN e IS NOT NULL THEN 'episodic' ELSE 'semantic' END AS memory_type,CASE WHEN e IS NOT NULL THEN e.event ELSE f.subject+' --'+f.relation+'--> '+f.object END AS text,CASE WHEN e IS NOT NULL THEN toString(e.date) ELSE toString(f.last_updated) END AS date,CASE WHEN f IS NOT NULL THEN f.status ELSE null END AS status,CASE WHEN e IS NOT NULL THEN e.importance ELSE null END AS importance,CASE WHEN f IS NOT NULL THEN f.confidence ELSE null END AS confidence,CASE WHEN f IS NOT NULL THEN f.subject ELSE null END AS subject,CASE WHEN f IS NOT NULL THEN f.relation ELSE null END AS relation,CASE WHEN f IS NOT NULL THEN f.object ELSE null END AS object,CASE WHEN f IS NOT NULL THEN toString(f.last_updated) ELSE null END AS last_updated'''
        with self.driver.session() as s:
            row=s.run(q,memory_id=memory_id).single(); return None if not row or row['id'] is None else dict(row)
    def find_memory_relations(self,memory_ids:list[str]):
        if not memory_ids: return []
        q='''MATCH (new)-[r:MEMORY_RELATION]->(old) WHERE coalesce(old.memory_id,old.triple_id) IN $memory_ids RETURN coalesce(new.memory_id,new.triple_id) AS newer_memory,r.relation AS relation,coalesce(old.memory_id,old.triple_id) AS older_memory,r.reason AS reason'''
        with self.driver.session() as s: return [dict(r) for r in s.run(q,memory_ids=memory_ids)]
    def find_superseding_memory(self,memory_id:str): return self.find_memory_relations([memory_id])
    def get_timeline(self,user_id:str,limit:int=20): return self.get_episodic_candidates(user_id,limit)
