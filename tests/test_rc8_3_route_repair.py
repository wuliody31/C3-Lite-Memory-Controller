from __future__ import annotations

from pathlib import Path

import pytest

from src.config import load_config
from src.query_analyzer import QueryAnalyzer
from src.route_planner import RoutePlanner


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def analyzer_and_router():
    config = load_config(
        ROOT / "configs" / "c3_lite_v2_2_final.yaml"
    )
    return QueryAnalyzer(config), RoutePlanner(config)


def route_for(analyzer, router, query: str):
    features = analyzer.analyse(query)
    route = router.plan(features)
    return features, [item.value for item in route.selected_types]


@pytest.mark.parametrize(
    "query",
    [
        "What technical details should I add to the RAG project description?",
        "What should I mention for the recommendation system project?",
        "What MSc project difficulty themes can I discuss in interviews?",
        "What is the role of the memory controller in the project?",
    ],
)
def test_factual_selection_routes_event_and_fact_without_procedure(
    analyzer_and_router,
    query,
):
    analyzer, router = analyzer_and_router
    features, selected = route_for(analyzer, router, query)

    assert features.asks_procedure is False
    assert selected == ["episodic", "semantic"]


def test_removed_option_is_a_conflict_requiring_event_and_fact(
    analyzer_and_router,
):
    analyzer, router = analyzer_and_router
    features, selected = route_for(
        analyzer,
        router,
        "Was Data Analyst completely removed from my career options?",
    )

    assert features.asks_conflict is True
    assert selected == ["episodic", "semantic"]


def test_always_question_is_a_temporal_comparison(
    analyzer_and_router,
):
    analyzer, router = analyzer_and_router
    features, selected = route_for(
        analyzer,
        router,
        "Was my travel preference always low-cost but time-aware?",
    )

    assert features.asks_timeline is True
    assert selected == ["episodic", "semantic"]


@pytest.mark.parametrize(
    "query",
    [
        "Where did I originally think I would stay?",
        "Did I ask for walking routes?",
        "What happened at York and how did it affect the plan?",
    ],
)
def test_historical_state_queries_keep_semantic_endpoint(
    analyzer_and_router,
    query,
):
    analyzer, router = analyzer_and_router
    _, selected = route_for(analyzer, router, query)

    assert selected == ["episodic", "semantic"]


@pytest.mark.parametrize(
    "query",
    [
        "Explain my current CV positioning and cite the memory types used.",
        "Explain which memory supports using city centre for food recommendations.",
        "If I ask for restaurant recommendations, what location should be used and why?",
    ],
)
def test_policy_explanations_route_all_memory_types(
    analyzer_and_router,
    query,
):
    analyzer, router = analyzer_and_router
    _, selected = route_for(analyzer, router, query)

    assert selected == ["episodic", "semantic", "procedural"]


@pytest.mark.parametrize(
    "query",
    [
        "Should you invent exact prices if the price is not stored?",
        "What should I do if booking information is unclear?",
    ],
)
def test_travel_policy_routes_semantic_and_procedural(
    analyzer_and_router,
    query,
):
    analyzer, router = analyzer_and_router
    features, selected = route_for(analyzer, router, query)

    assert features.asks_procedure is True
    assert selected == ["semantic", "procedural"]


def test_memory_type_definition_remains_semantic_only(
    analyzer_and_router,
):
    analyzer, router = analyzer_and_router
    features, selected = route_for(
        analyzer,
        router,
        "What are the three memory types in my project and what does each record?",
    )

    assert features.asks_procedure is False
    assert selected == ["semantic"]


def test_scope_explanation_does_not_add_unrelated_procedural_route(
    analyzer_and_router,
):
    analyzer, router = analyzer_and_router
    _, selected = route_for(
        analyzer,
        router,
        "Answer my current project scope and explain which memories support it.",
    )

    assert selected == ["episodic", "semantic"]


def test_normative_evaluation_choice_still_routes_all_memories(
    analyzer_and_router,
):
    analyzer, router = analyzer_and_router
    features, selected = route_for(
        analyzer,
        router,
        (
            "Should my evaluation mainly be a user study or a controlled "
            "fictional dataset with baseline comparison?"
        ),
    )

    assert features.asks_procedure is True
    assert features.asks_conflict is True
    assert selected == ["episodic", "semantic", "procedural"]
