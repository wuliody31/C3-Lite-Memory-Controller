from __future__ import annotations

ROUTE_MAP = {
    'current_fact': ['neo4j_semantic_graph'],
    'episodic_recall': ['neo4j_episodic_graph'],
    'temporal_update': ['neo4j_episodic_graph', 'neo4j_semantic_graph'],
    'conflict_resolution': ['neo4j_episodic_graph', 'neo4j_semantic_graph', 'json_procedural_rules'],
    'procedural_following': ['json_procedural_rules', 'neo4j_semantic_graph'],
    'abstention': ['neo4j_episodic_graph', 'neo4j_semantic_graph'],
    'explainability': ['neo4j_episodic_graph', 'neo4j_semantic_graph', 'json_procedural_rules'],
    'general': ['neo4j_semantic_graph'],
}


def select_routes(query_type: str) -> list[str]:
    return ROUTE_MAP.get(query_type, ['neo4j_semantic_graph'])
