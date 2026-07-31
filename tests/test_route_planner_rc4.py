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
                "fallback_top_route": True,
            },
        },
    }
    return QueryAnalyzer(config), RoutePlanner(config)


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        (
            "Is my project currently a large multi-agent system or a smaller C3-Lite controller prototype?",
            {"episodic", "semantic"},
        ),
        (
            "When discussing my dissertation with my supervisor, how should the answer style be?",
            {"semantic", "procedural"},
        ),
        (
            "Please explain why my dataset should not be centred on Neo4j.",
            {"episodic", "semantic", "procedural"},
        ),
        (
            "Should my evaluation mainly be a user study or a controlled fictional dataset with baseline comparison?",
            {"episodic", "semantic", "procedural"},
        ),
        (
            "What should I say if there is no stored evidence for a project claim?",
            {"procedural"},
        ),
        (
            "Did I decide to train my own model for this project?",
            {"episodic", "semantic"},
        ),
        (
            "What are the three memory types in my project and what does each record?",
            {"semantic"},
        ),
        (
            "Is the project mainly about proving Neo4j is useful?",
            {"episodic", "semantic"},
        ),
        (
            "If an older memory conflicts with a later project decision, how should the system answer?",
            {"procedural"},
        ),
        (
            "Should I claim production deployment in my CV if there is no evidence?",
            {"semantic", "procedural"},
        ),
        (
            "What evidence supports using memory explanations as an evaluation dimension?",
            {"episodic", "semantic", "procedural"},
        ),
        (
            "What baselines have I selected for the project evaluation?",
            {"episodic", "semantic"},
        ),
        (
            "Should my CV now prioritise Data Analyst or LLM Agent Engineer?",
            {"episodic", "semantic", "procedural"},
        ),
    ],
)
def test_rc4_route_planner(
    components: tuple[QueryAnalyzer, RoutePlanner],
    query: str,
    expected: set[str],
) -> None:
    analyzer, planner = components
    features = analyzer.analyse(query)
    decision = planner.plan(features)

    selected = decision.selected_types
    actual = {memory_type.value for memory_type in selected}
    scores = decision.scores
    reasons = decision.reasons

    assert actual == expected, (
        f"Query: {query}\n"
        f"Expected: {sorted(expected)}\n"
        f"Actual: {sorted(actual)}\n"
        f"Scores: {scores}\n"
        f"Reasons: {reasons}"
    )
