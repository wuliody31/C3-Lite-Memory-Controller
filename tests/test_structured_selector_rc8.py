from __future__ import annotations

from datetime import datetime, timezone

from src.coverage_estimator import CoverageEstimator
from src.evidence_requirements import EvidenceRequirementPlanner
from src.evidence_selector import EvidenceSelector
from src.schemas import (
    MemoryCandidate,
    MemoryType,
    QueryFeatures,
    QueryMode,
    RouteDecision,
)


def config() -> dict:
    return {
        "query_analysis": {
            "stopwords": [
                "the",
                "a",
                "an",
                "is",
                "are",
                "was",
                "were",
                "do",
                "does",
                "did",
                "my",
                "me",
                "i",
                "to",
                "of",
                "and",
                "or",
                "in",
                "on",
                "for",
                "what",
                "which",
                "how",
                "why",
                "when",
            ]
        },
        "selection": {
            "method": "structured_requirement_mmr",
            "max_evidence": 5,
            "max_per_memory_type": 4,
            "relevance_weight": 0.45,
            "requirement_weight": 0.30,
            "coverage_gain_weight": 0.15,
            "redundancy_weight": 0.10,
            "evidence_token_budget": 1200,
            "minimum_evidence": 2,
            "min_marginal_score": 0.18,
            "min_topical_score": 0.12,
            "guarantee_procedural_if_routed": True,
        },
        "evidence_requirements": {
            "enabled": True,
            "default_max_evidence": 5,
            "hard_max_evidence": 8,
            "max_per_memory_type": 4,
            "evidence_token_budget": 1200,
            "timeline_min_historical_states": 2,
            "timeline_min_current_states": 1,
            "timeline_min_transition_evidence": 1,
            "timeline_preferred_evidence": 6,
            "procedural_min_rules": 1,
            "procedural_preferred_evidence": 6,
            "conflict_min_alternatives": 1,
            "conflict_min_resolutions": 1,
            "conflict_preferred_evidence": 6,
            "explanation_min_supporting_evidence": 2,
            "explanation_preferred_evidence": 6,
            "cardinality_extra_buffer": 1,
        },
    }


def features(
    *,
    query: str,
    mode: QueryMode,
    needs: list[str],
    asks_explanation: bool = False,
    asks_conflict: bool = False,
    asks_procedure: bool = False,
) -> QueryFeatures:
    return QueryFeatures(
        normalised_query=query.lower(),
        tokens=query.lower().split(),
        entities=[],
        temporal_expressions=[],
        query_mode=mode,
        task_type=None,
        asks_current_state=(mode == QueryMode.CURRENT),
        asks_historical_state=(mode == QueryMode.HISTORICAL),
        asks_timeline=(mode == QueryMode.TIMELINE),
        asks_procedure=asks_procedure,
        asks_explanation=asks_explanation,
        asks_conflict=asks_conflict,
        information_needs=needs,
    )


def route(*types: MemoryType) -> RouteDecision:
    return RouteDecision(
        selected_types=list(types),
        scores={item.value: 0.9 for item in types},
        reasons={item.value: ["test"] for item in types},
    )


def candidate(
    memory_id: str,
    text: str,
    *,
    memory_type: MemoryType = MemoryType.SEMANTIC,
    score: float = 0.5,
    status: str = "current",
    timestamp: datetime | None = None,
    action: str | None = None,
    metadata: dict | None = None,
) -> MemoryCandidate:
    return MemoryCandidate(
        memory_id=memory_id,
        memory_type=memory_type,
        text=text,
        user_id="user01",
        timestamp=timestamp,
        status=status,
        confidence=1.0,
        final_score=score,
        lexical_score=0.0,
        graph_entity_score=0.0,
        resolution_action=action,
        metadata=metadata or {},
    )


def selector() -> EvidenceSelector:
    cfg = config()
    return EvidenceSelector(
        cfg,
        CoverageEstimator(cfg),
    )


def test_planner_detects_three_distinct_items() -> None:
    cfg = config()
    planner = EvidenceRequirementPlanner(cfg)
    plan = planner.plan(
        features=features(
            query=(
                "What are the three memory types in my project "
                "and what does each record?"
            ),
            mode=QueryMode.ATEMPORAL,
            needs=[
                "three memory types project",
                "each record",
            ],
        ),
        route=route(MemoryType.SEMANTIC),
        conflicts=[],
    )

    assert plan.explicit_cardinality == 3
    requirement = next(
        item
        for item in plan.requirements
        if item.role == "distinct_item"
    )
    assert requirement.min_count == 3
    assert requirement.distinct is True


def test_selector_preserves_three_memory_type_facts() -> None:
    evidence_selector = selector()
    query_features = features(
        query=(
            "What are the three memory types in my project "
            "and what does each record?"
        ),
        mode=QueryMode.ATEMPORAL,
        needs=[
            "three memory types project",
            "each record",
        ],
    )
    candidates = [
        candidate(
            "s_irrelevant",
            "baseline set includes no memory",
            score=0.90,
        ),
        candidate(
            "s_ep",
            "episodic memory records what happened and when",
            score=0.58,
        ),
        candidate(
            "s_sem",
            "semantic memory records currently believed facts",
            score=0.57,
        ),
        candidate(
            "s_proc",
            "procedural memory records rules and response policies",
            score=0.56,
        ),
    ]

    selected = evidence_selector.select(
        candidates=candidates,
        features=query_features,
        route=route(MemoryType.SEMANTIC),
        conflicts=[],
    )
    selected_ids = {
        item.memory_id
        for item in selected
    }

    assert {"s_ep", "s_sem", "s_proc"}.issubset(selected_ids)


def test_current_scope_beats_generic_answer_style() -> None:
    evidence_selector = selector()
    query_features = features(
        query=(
            "Answer my current project scope and explain which "
            "memories support it."
        ),
        mode=QueryMode.CURRENT,
        needs=[
            "answer current project scope",
            "explain memories support it",
            "current valid state",
            "supporting reason or evidence",
        ],
        asks_explanation=True,
    )
    current_scope = candidate(
        "s_scope",
        "project current scope is small MSc level memory controller prototype",
        score=0.76,
    )
    generic_style = candidate(
        "s_style",
        "project answer style current preference is structured academic",
        score=0.82,
    )
    transition = candidate(
        "e_transition",
        "The user narrowed the project from a broad chatbot platform to controlled memory selection",
        memory_type=MemoryType.EPISODIC,
        score=0.48,
        timestamp=datetime(2026, 1, 2, tzinfo=timezone.utc),
    )

    selected = evidence_selector.select(
        candidates=[generic_style, current_scope, transition],
        features=query_features,
        route=route(
            MemoryType.SEMANTIC,
            MemoryType.EPISODIC,
        ),
        conflicts=[],
    )

    assert "s_scope" in [item.memory_id for item in selected]


def test_timeline_selects_historical_current_and_transition() -> None:
    evidence_selector = selector()
    query_features = features(
        query="How did my project scope change over time?",
        mode=QueryMode.TIMELINE,
        needs=[
            "project scope change over time",
            "earlier state",
            "current state",
        ],
    )
    candidates = [
        candidate(
            "s_old_1",
            "project earlier scope was broad long term memory chatbot",
            score=0.70,
            status="outdated",
            action="historical",
        ),
        candidate(
            "s_old_2",
            "project earlier scope was large multi agent memory system",
            score=0.69,
            status="outdated",
            action="historical",
        ),
        candidate(
            "s_current",
            "project current scope is small MSc memory controller prototype",
            score=0.80,
            action="current_endpoint",
        ),
        candidate(
            "e_change",
            "The user narrowed the project from a broad chatbot to controlled memory selection",
            memory_type=MemoryType.EPISODIC,
            score=0.60,
            timestamp=datetime(2026, 1, 2, tzinfo=timezone.utc),
        ),
        candidate(
            "s_noise",
            "Dataset A tests conflict resolution",
            score=0.90,
        ),
    ]

    selected = evidence_selector.select(
        candidates=candidates,
        features=query_features,
        route=route(
            MemoryType.SEMANTIC,
            MemoryType.EPISODIC,
        ),
        conflicts=[],
    )
    selected_ids = {
        item.memory_id
        for item in selected
    }

    assert {
        "s_old_1",
        "s_old_2",
        "s_current",
        "e_change",
    }.issubset(selected_ids)


def test_procedural_comparison_keeps_rule_and_resolution() -> None:
    evidence_selector = selector()
    query_features = features(
        query=(
            "Should my evaluation mainly be a user study or a "
            "controlled fictional dataset with baseline comparison?"
        ),
        mode=QueryMode.PROCEDURAL,
        needs=[
            "should evaluation mainly be user study controlled fictional dataset with baseline comparison",
            "applicable procedure or rule",
            "alternative or conflicting states",
            "preferred or current resolution",
        ],
        asks_conflict=True,
        asks_procedure=True,
    )
    candidates = [
        candidate(
            "p_rule",
            "Use controlled benchmark evidence before optional user study",
            memory_type=MemoryType.PROCEDURAL,
            score=0.50,
            metadata={
                "direct_match": True,
                "task_match": 1.0,
            },
        ),
        candidate(
            "s_core",
            "project evaluation core method is controlled fictional dataset plus baseline comparison",
            score=0.88,
        ),
        candidate(
            "s_alt",
            "user study status is stretch goal",
            score=0.52,
            status="outdated",
        ),
        candidate(
            "e_decision",
            "The user made the user study a stretch goal rather than the core evaluation",
            memory_type=MemoryType.EPISODIC,
            score=0.60,
        ),
        candidate(
            "s_noise",
            "baseline set includes no memory",
            score=0.80,
        ),
    ]

    selected = evidence_selector.select(
        candidates=candidates,
        features=query_features,
        route=route(
            MemoryType.PROCEDURAL,
            MemoryType.SEMANTIC,
            MemoryType.EPISODIC,
        ),
        conflicts=[],
    )
    selected_ids = {
        item.memory_id
        for item in selected
    }

    assert "p_rule" in selected_ids
    assert "s_core" in selected_ids
    assert "s_alt" in selected_ids or "e_decision" in selected_ids
