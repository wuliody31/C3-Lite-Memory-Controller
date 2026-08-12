from __future__ import annotations

from pathlib import Path

from src.config import load_config
from src.query_analyzer import QueryAnalyzer
from src.schemas import (
    MemoryCandidate,
    MemoryType,
)
from src.temporal_validity import (
    QueryRelativeTemporalValidity,
)


ROOT = Path(__file__).resolve().parents[1]


def config() -> dict:
    return load_config(
        ROOT
        / "configs"
        / "c3_lite_v2_2_final.yaml"
    )


def current_candidate() -> MemoryCandidate:
    return MemoryCandidate(
        memory_id="current_fact",
        memory_type=MemoryType.SEMANTIC,
        text="The current project scope is C3.",
        user_id="user01",
        status="current",
    )


def historical_candidate() -> MemoryCandidate:
    return MemoryCandidate(
        memory_id="historical_fact",
        memory_type=MemoryType.SEMANTIC,
        text=(
            "The previous project scope was "
            "a larger multi-agent memory system."
        ),
        user_id="user01",
        status="outdated",
    )


def test_current_query_prefers_current_version() -> None:
    cfg = config()

    analyzer = QueryAnalyzer(cfg)
    evaluator = QueryRelativeTemporalValidity(
        cfg
    )

    features = analyzer.analyse(
        "What is my current MSc project scope?"
    )

    current = evaluator.evaluate(
        candidate=current_candidate(),
        features=features,
    )

    historical = evaluator.evaluate(
        candidate=historical_candidate(),
        features=features,
    )

    assert current.score > historical.score

    assert (
        current.temporal_role
        == "current_state"
    )

    assert (
        historical.temporal_role
        == "historical_state"
    )


def test_historical_query_prefers_historical_version() -> None:
    cfg = config()

    analyzer = QueryAnalyzer(cfg)
    evaluator = QueryRelativeTemporalValidity(
        cfg
    )

    features = analyzer.analyse(
        "What was my previous MSc project scope?"
    )

    current = evaluator.evaluate(
        candidate=current_candidate(),
        features=features,
    )

    historical = evaluator.evaluate(
        candidate=historical_candidate(),
        features=features,
    )

    assert historical.score > current.score

    assert (
        historical.temporal_role
        == "historical_state"
    )

    assert historical.compatible

    assert not current.compatible


def test_currentness_is_not_recency() -> None:
    cfg = config()

    analyzer = QueryAnalyzer(cfg)
    evaluator = QueryRelativeTemporalValidity(
        cfg
    )

    historical_features = analyzer.analyse(
        "What was my previous MSc project scope?"
    )

    newer_current = current_candidate()
    older_historical = historical_candidate()

    current_score = evaluator.evaluate(
        candidate=newer_current,
        features=historical_features,
    ).score

    historical_score = evaluator.evaluate(
        candidate=older_historical,
        features=historical_features,
    ).score

    # Even though the current fact represents the newer
    # version, it must not dominate a historical query.
    assert historical_score > current_score


def test_timeline_accepts_both_endpoints() -> None:
    cfg = config()

    analyzer = QueryAnalyzer(cfg)
    evaluator = QueryRelativeTemporalValidity(
        cfg
    )

    features = analyzer.analyse(
        "How did my project scope change over time?"
    )

    current = evaluator.evaluate(
        candidate=current_candidate(),
        features=features,
    )

    historical = evaluator.evaluate(
        candidate=historical_candidate(),
        features=features,
    )

    assert current.compatible
    assert historical.compatible

    assert current.score >= 0.90
    assert historical.score >= 0.90

    assert (
        current.temporal_role
        == "current_endpoint"
    )

    assert (
        historical.temporal_role
        == "historical_state"
    )


def test_resolution_action_overrides_raw_status() -> None:
    cfg = config()

    analyzer = QueryAnalyzer(cfg)
    evaluator = QueryRelativeTemporalValidity(
        cfg
    )

    features = analyzer.analyse(
        "What was my previous MSc project scope?"
    )

    candidate = current_candidate()

    # Conflict resolution can explicitly identify a
    # candidate as historical even if its raw status is
    # otherwise ambiguous/current.
    candidate.resolution_action = "historical"

    result = evaluator.evaluate(
        candidate=candidate,
        features=features,
    )

    assert result.score == 1.0

    assert (
        result.temporal_role
        == "historical_state"
    )