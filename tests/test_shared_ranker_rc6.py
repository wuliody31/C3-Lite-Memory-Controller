from __future__ import annotations

from types import SimpleNamespace

from src.config import load_config
from src.schemas import MemoryType, QueryMode
from src.shared_ranker import SharedRanker


def build_ranker() -> SharedRanker:
    config = load_config(
        "configs/c3_lite_v2_2_final.yaml"
    )
    return SharedRanker(config)


def candidate(
    *,
    memory_type: MemoryType = MemoryType.EPISODIC,
    lexical: float = 0.0,
    entity: float = 0.0,
    utility: float = 0.0,
    temporal: float = 0.0,
    validity: float = 0.0,
    route: float = 0.0,
):
    return SimpleNamespace(
        memory_type=memory_type,
        lexical_score=lexical,
        graph_entity_score=entity,
        final_score=utility,
        temporal_task_score=temporal,
        validity_score=validity,
        route_compatibility_score=route,
    )


def features(
    *,
    mode: QueryMode,
    asks_explanation: bool = False,
):
    return SimpleNamespace(
        query_mode=mode,
        asks_explanation=asks_explanation,
    )


def test_rc6_utility_gate_requires_weak_content_support() -> None:
    ranker = build_ranker()

    item = candidate(
        utility=0.60,
        lexical=0.0,
        entity=0.0,
    )

    reasons = ranker._gate_reasons(
        item,
        features(mode=QueryMode.CURRENT),
    )

    assert "utility_gate" not in reasons


def test_rc6_utility_gate_keeps_supported_candidate() -> None:
    ranker = build_ranker()

    item = candidate(
        utility=0.41,
        lexical=0.0,
        entity=0.10,
    )

    reasons = ranker._gate_reasons(
        item,
        features(mode=QueryMode.CURRENT),
    )

    assert "utility_gate" in reasons


def test_rc6_temporal_route_rescue_keeps_structural_evidence() -> None:
    ranker = build_ranker()

    item = candidate(
        utility=0.39,
        lexical=0.0,
        entity=0.0,
        temporal=0.98,
        validity=1.0,
        route=0.60,
    )

    reasons = ranker._gate_reasons(
        item,
        features(mode=QueryMode.TIMELINE),
    )

    assert "utility_gate" not in reasons
    assert "temporal_route_rescue" in reasons


def test_rc6_temporal_rescue_does_not_pollute_current_queries() -> None:
    ranker = build_ranker()

    item = candidate(
        utility=0.60,
        lexical=0.0,
        entity=0.0,
        temporal=1.0,
        validity=1.0,
        route=0.90,
    )

    reasons = ranker._gate_reasons(
        item,
        features(mode=QueryMode.CURRENT),
    )

    assert "utility_gate" not in reasons
    assert "temporal_route_rescue" not in reasons


def test_rc6_explanation_rescue_keeps_supporting_evidence() -> None:
    ranker = build_ranker()

    item = candidate(
        utility=0.39,
        lexical=0.0,
        entity=0.0,
        temporal=0.80,
        validity=1.0,
        route=0.60,
    )

    reasons = ranker._gate_reasons(
        item,
        features(
            mode=QueryMode.CURRENT,
            asks_explanation=True,
        ),
    )

    assert "explanation_rescue" in reasons


def test_rc6_low_signal_candidate_is_rejected() -> None:
    ranker = build_ranker()

    item = candidate(
        utility=0.25,
        lexical=0.0,
        entity=0.0,
        temporal=0.40,
        validity=0.55,
        route=0.20,
    )

    reasons = ranker._gate_reasons(
        item,
        features(mode=QueryMode.CURRENT),
    )

    assert reasons == []


def test_rc6_procedural_safety_gate_is_preserved() -> None:
    ranker = build_ranker()

    item = candidate(
        memory_type=MemoryType.PROCEDURAL,
        utility=0.10,
    )

    reasons = ranker._gate_reasons(
        item,
        features(mode=QueryMode.PROCEDURAL),
    )

    assert "procedural_safety_gate" in reasons
