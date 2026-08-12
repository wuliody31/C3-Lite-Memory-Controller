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


def current_state() -> MemoryCandidate:
    return MemoryCandidate(
        memory_id="current_state",
        memory_type=MemoryType.SEMANTIC,
        text="The MSc project scope is a memory controller.",
        user_id="user01",
        subject="MSc project",
        predicate="scope",
        object_value="memory controller",
        status="current",
        confidence=1.0,
        importance=0.5,
        authority="unknown",
    )


def historical_state() -> MemoryCandidate:
    return MemoryCandidate(
        memory_id="historical_state",
        memory_type=MemoryType.SEMANTIC,
        text="The MSc project scope is a memory controller.",
        user_id="user01",
        subject="MSc project",
        predicate="scope",
        object_value="memory controller",
        status="superseded",
        confidence=1.0,
        importance=0.5,
        authority="unknown",
    )


def legacy_final_score(
    candidate: MemoryCandidate,
    weights: dict,
) -> float:
    """Reconstruct the RC8 score using legacy validity."""

    score = (
        float(weights["lexical_weight"])
        * candidate.lexical_score

        + float(weights["graph_entity_weight"])
        * candidate.graph_entity_score

        + float(weights["temporal_task_weight"])
        * candidate.temporal_task_score

        + float(weights["validity_weight"])
        * float(
            candidate.metadata[
                "legacy_validity_score"
            ]
        )

        + float(weights["source_confidence_weight"])
        * candidate.source_confidence_score

        + float(weights["route_compatibility_weight"])
        * candidate.route_compatibility_score
    )

    return max(
        0.0,
        min(
            1.0,
            score,
        ),
    )


def run_historical_case():
    cfg = config()

    analyzer = QueryAnalyzer(cfg)
    router = RoutePlanner(cfg)
    ranker = SharedRanker(cfg)

    features = analyzer.analyse(
        "What was my previous MSc project scope?"
    )

    route = router.plan(
        features
    )

    candidates = [
        deepcopy(current_state()),
        deepcopy(historical_state()),
    ]

    ranker.rank(
        candidates=candidates,
        features=features,
        route=route,
        current_time=datetime.now(
            timezone.utc
        ),
    )

    by_id = {
        candidate.memory_id: candidate
        for candidate in candidates
    }

    return (
        cfg,
        by_id["current_state"],
        by_id["historical_state"],
    )


def test_query_relative_validity_causes_rank_inversion() -> None:
    cfg, current, historical = (
        run_historical_case()
    )

    weights = cfg["ranking"]

    legacy_current = (
        legacy_final_score(
            current,
            weights,
        )
    )

    legacy_historical = (
        legacy_final_score(
            historical,
            weights,
        )
    )

    active_current = (
        current.final_score
    )

    active_historical = (
        historical.final_score
    )

    # -----------------------------------------------------
    # RC8 legacy ranking:
    # current state is incorrectly preferred.
    # -----------------------------------------------------

    assert (
        legacy_current
        >
        legacy_historical
    )

    # -----------------------------------------------------
    # C3-v3 ranking:
    # historical state becomes preferred for a
    # historical query.
    # -----------------------------------------------------

    assert (
        active_historical
        >
        active_current
    )


def test_only_validity_term_is_required_for_controlled_inversion() -> None:
    cfg, current, historical = (
        run_historical_case()
    )

    # Controlled candidates use the same content,
    # memory type, confidence, authority and route.
    # Their non-validity ranking features should match.

    assert (
        current.lexical_score
        == historical.lexical_score
    )

    assert (
        current.graph_entity_score
        == historical.graph_entity_score
    )

    assert (
        current.temporal_task_score
        == historical.temporal_task_score
    )

    assert (
        current.source_confidence_score
        == historical.source_confidence_score
    )

    assert (
        current.route_compatibility_score
        == historical.route_compatibility_score
    )

    # Legacy validity prefers currentness.
    assert (
        current.metadata[
            "legacy_validity_score"
        ]
        >
        historical.metadata[
            "legacy_validity_score"
        ]
    )

    # Query-relative validity reverses that relation.
    assert (
        historical.validity_score
        >
        current.validity_score
    )