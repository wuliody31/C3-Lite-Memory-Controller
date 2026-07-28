from __future__ import annotations

import pytest

from src.query_analyzer import QueryAnalyzer
from src.route_planner import RoutePlanner


@pytest.fixture()
def components() -> tuple[QueryAnalyzer, RoutePlanner]:
    config = {
        "query_analysis": {
            "stopwords": [],
            "aliases": {},
        },
        "routing": {
            "thresholds": {
                "episodic": 0.50,
                "semantic": 0.50,
                "procedural": 0.50,
            },
            "utility_weights": {
                "intent": 0.35,
                "lexical": 0.20,
                "entity": 0.15,
                "temporal_task": 0.20,
                "prior": 0.10,
            },
            "priors": {
                "episodic": 0.35,
                "semantic": 0.65,
                "procedural": 0.25,
            },
            "structural_gates": {
                "force_timeline_route": True,
                "force_conflict_route": True,
                "force_historical_semantic_companion": True,
                "force_procedural_semantic_companion": True,
                "force_explanation_policy_route": True,
                "force_factual_selection_route": True,
                "fallback_top_route": True,
            },
        },
    }
    return QueryAnalyzer(config), RoutePlanner(config)


@pytest.mark.parametrize(
    ("query", "expected_types", "expected_procedure"),
    [
        (
            "Should I claim production deployment in my CV if there is no evidence?",
            {"semantic", "procedural"},
            True,
        ),
        (
            "When discussing my dissertation with my supervisor, how should the answer style be?",
            {"semantic", "procedural"},
            True,
        ),
        (
            "Did I decide to train my own model for this project?",
            {"episodic", "semantic"},
            False,
        ),
        (
            "Is the project mainly about proving Neo4j is useful?",
            {"episodic", "semantic"},
            False,
        ),
        (
            "If an older memory conflicts with a later project decision, how should the system answer?",
            {"procedural"},
            True,
        ),
        (
            "Should you invent exact prices if the price is not stored?",
            {"semantic", "procedural"},
            True,
        ),
        (
            "What should I do if booking information is unclear?",
            {"semantic", "procedural"},
            True,
        ),
    ],
)
def test_rc8_3a_keeps_rc4_and_rc8_3_routes(
    components: tuple[QueryAnalyzer, RoutePlanner],
    query: str,
    expected_types: set[str],
    expected_procedure: bool,
) -> None:
    analyzer, planner = components
    features = analyzer.analyse(query)
    decision = planner.plan(features)

    actual_types = {
        memory_type.value
        for memory_type in decision.selected_types
    }

    assert features.asks_procedure is expected_procedure
    assert actual_types == expected_types
