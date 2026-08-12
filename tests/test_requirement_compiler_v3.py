from __future__ import annotations

from src.config import load_config
from src.evidence_requirements import (
    EvidenceRequirementPlanner,
)
from src.query_analyzer import QueryAnalyzer
from src.requirement_compiler import (
    RequirementCompiler,
)
from src.route_planner import RoutePlanner
from src.schemas import QueryMode


CONFIG_PATH = "configs/c3_lite_v2_2_final.yaml"


def config() -> dict:
    return load_config(CONFIG_PATH)


def test_compiler_preserves_query_analysis() -> None:
    cfg = config()

    query = "What is my current MSc project scope?"

    legacy_features = QueryAnalyzer(
        cfg
    ).analyse(query)

    compiled = RequirementCompiler(
        cfg
    ).compile(query)

    assert (
        compiled.features
        == legacy_features
    )

    assert (
        compiled.spec.temporal_mode
        == legacy_features.query_mode
    )

    assert (
        compiled.spec.entities
        == legacy_features.entities
    )

    assert (
        compiled.spec.information_needs
        == legacy_features.information_needs
    )


def test_compiler_preserves_route() -> None:
    cfg = config()

    query = (
        "How did my project scope "
        "change over time?"
    )

    features = QueryAnalyzer(
        cfg
    ).analyse(query)

    legacy_route = RoutePlanner(
        cfg
    ).plan(features)

    compiled = RequirementCompiler(
        cfg
    ).compile(query)

    assert (
        compiled.route
        == legacy_route
    )

    assert (
        compiled.spec.memory_types
        == legacy_route.selected_types
    )

    assert (
        compiled.spec.route_scores
        == legacy_route.scores
    )


def test_compiler_preserves_evidence_plan() -> None:
    cfg = config()

    query = (
        "What are the three memory types "
        "in my project and what does each record?"
    )

    features = QueryAnalyzer(
        cfg
    ).analyse(query)

    route = RoutePlanner(
        cfg
    ).plan(features)

    legacy_plan = (
        EvidenceRequirementPlanner(
            cfg
        ).plan(
            features=features,
            route=route,
            conflicts=[],
        )
    )

    compiled = RequirementCompiler(
        cfg
    ).compile(query)

    assert (
        compiled.spec.requirements
        == legacy_plan.requirements
    )

    assert (
        compiled.spec.token_budget
        == legacy_plan.token_budget
    )

    assert (
        compiled.spec.max_evidence
        == legacy_plan.max_evidence
    )

    assert (
        compiled.spec.explicit_cardinality
        == legacy_plan.explicit_cardinality
    )


def test_timeline_becomes_temporal_requirement() -> None:
    cfg = config()

    compiled = RequirementCompiler(
        cfg
    ).compile(
        "How did my project scope change over time?"
    )

    assert (
        compiled.spec.temporal_mode
        == QueryMode.TIMELINE
    )

    assert compiled.spec.asks_timeline

    needs = " ".join(
        compiled.spec.information_needs
    ).lower()

    assert "earlier" in needs
    assert "current" in needs

def test_previous_query_compiles_historical_requirement() -> None:
    cfg = config()

    compiled = RequirementCompiler(
        cfg
    ).compile(
        "What was my previous MSc project scope?"
    )

    assert (
        compiled.spec.temporal_mode
        == QueryMode.HISTORICAL
    )

    assert (
        compiled.spec.asks_historical_state
    )

    roles = {
        requirement.role
        for requirement
        in compiled.spec.requirements
    }

    assert "answer_target" in roles
    assert "historical_state" in roles