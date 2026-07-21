from __future__ import annotations
from typing import Any
from src.query_classifier import classify_query
from src.memory_router import select_routes
from src.keyword_extractor import extract_keywords
from src.conflict_checker import check_conflicts, enrich_with_newer_memories
from src.answer_generator import generate_answer


class C3LiteController:
    def __init__(self, neo4j_adapter, procedural_matcher, answer_mode: str = 'mock', openai_model: str = 'gpt-4o-mini', openai_api_key: str | None = None):
        self.neo4j = neo4j_adapter
        self.procedural = procedural_matcher
        self.answer_mode = answer_mode
        self.openai_model = openai_model
        self.openai_api_key = openai_api_key

    def answer(self, user_id: str, question: str) -> dict[str, Any]:
        query_type = classify_query(question)
        routes = select_routes(query_type)
        keywords = extract_keywords(question)
        retrieved, procedural_rules = [], []
        for keyword in keywords:
            if 'neo4j_episodic_graph' in routes:
                retrieved.extend(self.neo4j.search_episodic(user_id, keyword, limit=6))
            if 'neo4j_semantic_graph' in routes:
                retrieved.extend(self.neo4j.search_semantic(user_id, keyword, limit=6, active_only=True))
        if 'json_procedural_rules' in routes:
            procedural_rules = self.procedural.match(user_id, question, limit=5)
        unique = {}
        for m in retrieved:
            unique[m['id']] = m
        retrieved = list(unique.values())
        conflict_result = check_conflicts(self.neo4j, retrieved)
        newer_memories = enrich_with_newer_memories(self.neo4j, conflict_result)
        selected_map = {}
        for m in conflict_result['resolved_memories']:
            selected_map[m['id']] = m
        for m in newer_memories:
            selected_map[m['id']] = m
        selected_memories = list(selected_map.values())[:10]
        answer = generate_answer(self.answer_mode, question, selected_memories, conflict_result, procedural_rules, query_type, self.openai_model, self.openai_api_key)
        return {'question': question, 'query_type': query_type, 'predicted_route': routes, 'keywords': keywords, 'retrieved_memory_ids': [m['id'] for m in retrieved], 'used_memory_ids': [m['id'] for m in selected_memories] + [r['id'] for r in procedural_rules], 'outdated_memory_ids': conflict_result['outdated_memory_ids'], 'newer_memory_ids': conflict_result['newer_memory_ids'], 'conflict_notes': conflict_result['conflict_notes'], 'answer': answer}
