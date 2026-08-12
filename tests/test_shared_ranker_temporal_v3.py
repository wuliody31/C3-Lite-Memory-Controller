from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from src.config import load_config
from src.query_analyzer import QueryAnalyzer
from src.route_planner import RoutePlanner
from src.schemas import (
    MemoryCandidate,
    MemoryType,
)
from src.shared_ranker import SharedRanker


ROOT = Path(__file__).resolve().parents[1]


def config() -> dict:
    return load_config(
        ROOT
        / "configs"
        / "c3_lite_v2_2_final.yaml"
    )


def current_memory() -> MemoryCandidate:
    """Current semantic version of the same factual slot."""

    return MemoryCandidate(
        memory_id="current_scope",
        memory_type=MemoryType.SEMANTIC,
        text=(
            "The MSc project scope is "
            "a memory controller."
        ),
        user_id="user01",
        subject="MSc project",
        predicate="scope",
        object_value="memory controller",
        status="current",
        confidence=1.0,
        importance=0.5,
        authority="unknown",
    )


def historical_memory() -> MemoryCandidate:
    """Historical semantic version of the same factual slot."""

    return MemoryCandidate(
        memory_id="previous_scope",
        memory_type=MemoryType.SEMANTIC,
        text=(
            "The MSc project scope is "
            "a memory controller."
        ),
        user_id="user01",
        subject="MSc project",
        predicate="scope",
        object_value="memory controller",
        status="superseded",
        confidence=1.0,
        importance=0.5,
        authority="unknown",
    )


def rank_for_query(
    query: str,
) -> list[MemoryCandidate]:
    """Rank two otherwise matched temporal versions."""

    cfg = config()

    analyzer = QueryAnalyzer(
        cfg
    )

    router = RoutePlanner(
        cfg
    )

    ranker = SharedRanker(
        cfg
    )

    features = analyzer.analyse(
        query
    )

    route = router.plan(
        features
    )

    candidates = [
        deepcopy(
            current_memory()
        ),
        deepcopy(
            historical_memory()
        ),
    ]

    ranker.rank(
        candidates=candidates,
        features=features,
        route=route,
        current_time=datetime.now(
            timezone.utc
        ),
    )

    return candidates


def by_id(
    candidates: list[MemoryCandidate],
) -> dict[str, MemoryCandidate]:
    return {
        candidate.memory_id: candidate
        for candidate in candidates
    }


def test_historical_query_reverses_legacy_validity_bias() -> None:
    candidates = by_id(
        rank_for_query(
            "What was my previous MSc project scope?"
        )
    )

    current = candidates[
        "current_scope"
    ]

    historical = candidates[
        "previous_scope"
    ]

    # RC8 legacy status validity favours
    # the current semantic version.
    assert (
        current.metadata[
            "legacy_validity_score"
        ]
        >
        historical.metadata[
            "legacy_validity_score"
        ]
    )

    assert (
        current.metadata[
            "legacy_validity_score"
        ]
        == 1.0
    )

    assert (
        historical.metadata[
            "legacy_validity_score"
        ]
        == 0.85
    )

    # C3-v3 query-relative validity reverses
    # the preference for a historical query.
    assert (
        historical.validity_score
        >
        current.validity_score
    )

    assert (
        historical.validity_score
        == 1.0
    )

    assert (
        current.validity_score
        == 0.25
    )


def test_historical_query_improves_historical_final_score() -> None:
    candidates = by_id(
        rank_for_query(
            "What was my previous MSc project scope?"
        )
    )

    current = candidates[
        "current_scope"
    ]

    historical = candidates[
        "previous_scope"
    ]

    # The two memories deliberately use the same
    # text, confidence, authority and memory type.
    # Query-relative temporal validity should
    # therefore make the historical version rank
    # above the current version.
    assert (
        historical.final_score
        >
        current.final_score
    )


def test_current_query_still_prefers_current_validity() -> None:
    candidates = by_id(
        rank_for_query(
            "What is my current MSc project scope?"
        )
    )

    current = candidates[
        "current_scope"
    ]

    historical = candidates[
        "previous_scope"
    ]

    assert (
        current.validity_score
        >
        historical.validity_score
    )

    assert (
        current.validity_score
        == 1.0
    )

    assert (
        historical.validity_score
        == 0.10
    )

    assert (
        current.metadata[
            "query_relative_temporal_role"
        ]
        == "current_state"
    )

    assert (
        historical.metadata[
            "query_relative_temporal_role"
        ]
        == "historical_state"
    )


def test_timeline_keeps_both_state_versions_valid() -> None:
    candidates = by_id(
        rank_for_query(
            "How did my project scope change over time?"
        )
    )

    current = candidates[
        "current_scope"
    ]

    historical = candidates[
        "previous_scope"
    ]

    assert (
        current.validity_score
        == 1.0
    )

    assert (
        historical.validity_score
        == 1.0
    )

    assert (
        current.metadata[
            "query_relative_temporal_role"
        ]
        == "current_endpoint"
    )

    assert (
        historical.metadata[
            "query_relative_temporal_role"
        ]
        == "historical_state"
    )

    assert (
        current.metadata[
            "query_relative_temporal_compatible"
        ]
        is True
    )

    assert (
        historical.metadata[
            "query_relative_temporal_compatible"
        ]
        is True
    )


def test_legacy_score_remains_available_for_ablation() -> None:
    candidates = by_id(
        rank_for_query(
            "What was my previous MSc project scope?"
        )
    )

    for candidate in candidates.values():
        assert (
            "legacy_validity_score"
            in candidate.metadata
        )

        assert (
            "query_relative_validity_score"
            in candidate.metadata
        )

        assert (
            "query_relative_temporal_role"
            in candidate.metadata
        )

        assert (
            "query_relative_temporal_compatible"
            in candidate.metadata
        )

        assert (
            "query_relative_temporal_reasons"
            in candidate.metadata
        )

        assert (
            candidate.validity_score
            == candidate.metadata[
                "query_relative_validity_score"
            ]
        )