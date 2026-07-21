from __future__ import annotations
from typing import Any
from src.keyword_extractor import extract_keywords
from src.answer_generator import generate_answer


def no_memory_baseline(question: str) -> dict[str, Any]:
    return {'question': question, 'query_type': 'no_memory', 'predicted_route': [], 'keywords': [], 'retrieved_memory_ids': [], 'used_memory_ids': [], 'outdated_memory_ids': [], 'newer_memory_ids': [], 'conflict_notes': [], 'answer': 'I do not have access to stored memory for this question.'}


def simple_retrieval_baseline(neo4j_adapter, user_id: str, question: str, answer_mode: str = 'mock', openai_model: str = 'gpt-4o-mini', openai_api_key: str | None = None) -> dict[str, Any]:
    keywords = extract_keywords(question, max_keywords=4)
    memories = []
    for kw in keywords:
        memories.extend(neo4j_adapter.search_episodic(user_id, kw, limit=4))
        memories.extend(neo4j_adapter.search_semantic(user_id, kw, limit=4, active_only=False))
    unique = {}
    for m in memories:
        unique[m['id']] = m
    memories = list(unique.values())[:10]
    conflict_result = {'resolved_memories': memories, 'outdated_memory_ids': [], 'newer_memory_ids': [], 'conflict_notes': []}
    answer = generate_answer(answer_mode, question, memories, conflict_result, [], 'simple_retrieval', openai_model, openai_api_key)
    return {'question': question, 'query_type': 'simple_retrieval', 'predicted_route': ['neo4j_episodic_graph','neo4j_semantic_graph'], 'keywords': keywords, 'retrieved_memory_ids': [m['id'] for m in memories], 'used_memory_ids': [m['id'] for m in memories], 'outdated_memory_ids': [], 'newer_memory_ids': [], 'conflict_notes': [], 'answer': answer}


def all_memory_baseline(neo4j_adapter, procedural_matcher, user_id: str, question: str, answer_mode: str = 'mock', openai_model: str = 'gpt-4o-mini', openai_api_key: str | None = None) -> dict[str, Any]:
    keywords = extract_keywords(question, max_keywords=6)
    memories = []
    for kw in keywords:
        memories.extend(neo4j_adapter.search_episodic(user_id, kw, limit=5))
        memories.extend(neo4j_adapter.search_semantic(user_id, kw, limit=5, active_only=False))
    unique = {}
    for m in memories:
        unique[m['id']] = m
    memories = list(unique.values())[:14]
    rules = procedural_matcher.match(user_id, question, limit=5)
    conflict_result = {'resolved_memories': memories, 'outdated_memory_ids': [], 'newer_memory_ids': [], 'conflict_notes': []}
    answer = generate_answer(answer_mode, question, memories, conflict_result, rules, 'all_memory', openai_model, openai_api_key)
    return {'question': question, 'query_type': 'all_memory', 'predicted_route': ['neo4j_episodic_graph','neo4j_semantic_graph','json_procedural_rules'], 'keywords': keywords, 'retrieved_memory_ids': [m['id'] for m in memories] + [r['id'] for r in rules], 'used_memory_ids': [m['id'] for m in memories] + [r['id'] for r in rules], 'outdated_memory_ids': [], 'newer_memory_ids': [], 'conflict_notes': [], 'answer': answer}
