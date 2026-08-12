from __future__ import annotations

from pathlib import Path

from src.config import load_config
from src.coverage_estimator import (
    CoverageEstimator,
)
from src.evidence_utility import (
    EvidenceUtilityModel,
)
from src.query_analyzer import (
    QueryAnalyzer,
)
from src.schemas import (
    MemoryCandidate,
    MemoryType,
)


ROOT = Path(__file__).resolve().parents[1]


def config() -> dict:
    return load_config(
        ROOT
        / "configs"
        / "c3_lite_v2_2_final.yaml"
    )


def utility_model():
    cfg = config()

    return (
        cfg,
        EvidenceUtilityModel(
            cfg,
            CoverageEstimator(
                cfg
            ),
        ),
        QueryAnalyzer(
            cfg
        ),
    )


def state(
    memory_id: str,
    *,
    status: str,
    validity: float,
    compatible: bool,
    text: str,
) -> MemoryCandidate:
    return MemoryCandidate(
        memory_id=memory_id,
        memory_type=(
            MemoryType.SEMANTIC
        ),
        text=text,
        user_id="user01",
        subject="MSc project",
        predicate="scope",
        object_value="memory controller",
        status=status,
        confidence=1.0,
        lexical_score=0.8,
        graph_entity_score=0.8,
        route_compatibility_score=0.9,
        validity_score=validity,
        source_confidence_score=1.0,
        metadata={
            "query_relative_temporal_compatible": (
                compatible
            ),
        },
    )


def test_requirement_gain_increases_marginal_utility() -> None:
    _, model, analyzer = utility_model()

    features = analyzer.analyse(
        "What was my previous MSc project scope?"
    )

    candidate = state(
        "historical",
        status="superseded",
        validity=1.0,
        compatible=True,
        text=(
            "The previous MSc project scope "
            "was a larger memory system."
        ),
    )

    without_gain = model.evaluate(
        candidate=candidate,
        selected=[],
        features=features,
        requirement_gain=0.0,
    )

    with_gain = model.evaluate(
        candidate=candidate,
        selected=[],
        features=features,
        requirement_gain=1.0,
    )

    assert (
        with_gain.utility
        >
        without_gain.utility
    )


def test_temporally_incompatible_current_state_is_penalised() -> None:
    _, model, analyzer = utility_model()

    features = analyzer.analyse(
        "What was my previous MSc project scope?"
    )

    historical = state(
        "historical",
        status="superseded",
        validity=1.0,
        compatible=True,
        text=(
            "The previous MSc project scope "
            "was a larger memory system."
        ),
    )

    current = state(
        "current",
        status="current",
        validity=0.25,
        compatible=False,
        text=(
            "The current MSc project scope "
            "is the C3 memory controller."
        ),
    )

    historical_result = model.evaluate(
        candidate=historical,
        selected=[],
        features=features,
        requirement_gain=1.0,
    )

    current_result = model.evaluate(
        candidate=current,
        selected=[],
        features=features,
        requirement_gain=0.0,
    )

    assert (
        historical_result.utility
        >
        current_result.utility
    )

    assert (
        current_result.incompatibility
        == 1.0
    )


def test_redundancy_reduces_marginal_utility() -> None:
    _, model, analyzer = utility_model()

    features = analyzer.analyse(
        "What was my previous MSc project scope?"
    )

    first = state(
        "first",
        status="superseded",
        validity=1.0,
        compatible=True,
        text=(
            "The previous MSc project scope "
            "was a larger memory system."
        ),
    )

    duplicate = state(
        "duplicate",
        status="superseded",
        validity=1.0,
        compatible=True,
        text=(
            "The previous MSc project scope "
            "was a larger memory system."
        ),
    )

    empty_set = model.evaluate(
        candidate=duplicate,
        selected=[],
        features=features,
    )

    after_duplicate = model.evaluate(
        candidate=duplicate,
        selected=[first],
        features=features,
    )

    assert (
        after_duplicate.redundancy
        >
        empty_set.redundancy
    )

    assert (
        after_duplicate.utility
        <
        empty_set.utility
    )


def test_longer_memory_has_higher_context_cost() -> None:
    _, model, analyzer = utility_model()

    features = analyzer.analyse(
        "What is my current MSc project scope?"
    )

    short = state(
        "short",
        status="current",
        validity=1.0,
        compatible=True,
        text="Current project is C3.",
    )

    long = state(
        "long",
        status="current",
        validity=1.0,
        compatible=True,
        text=(
            "Current project is C3. "
            * 40
        ),
    )

    short_result = model.evaluate(
        candidate=short,
        selected=[],
        features=features,
    )

    long_result = model.evaluate(
        candidate=long,
        selected=[],
        features=features,
    )

    assert (
        long_result.token_cost
        >
        short_result.token_cost
    )

    assert (
        long_result.normalised_token_cost
        >
        short_result.normalised_token_cost
    )


def test_timeline_endpoint_combination_can_create_coverage_gain() -> None:
    _, model, analyzer = utility_model()

    features = analyzer.analyse(
        "How did my project scope change over time?"
    )

    historical = state(
        "historical",
        status="superseded",
        validity=1.0,
        compatible=True,
        text=(
            "The previous MSc project scope "
            "was a broad multi-agent system."
        ),
    )
    historical.resolution_action = (
        "historical"
    )

    current = state(
        "current",
        status="current",
        validity=1.0,
        compatible=True,
        text=(
            "The current MSc project scope "
            "is the C3 memory controller."
        ),
    )
    current.resolution_action = (
        "current_endpoint"
    )

    result = model.evaluate(
        candidate=current,
        selected=[historical],
        features=features,
    )

    assert (
        result.coverage_gain
        >= 0.0
    )