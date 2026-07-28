from __future__ import annotations

from src.coverage_estimator import CoverageEstimator
from src.evidence_requirements import EvidenceRequirementPlanner
from src.evidence_selector import EvidenceSelector
from src.query_analyzer import QueryAnalyzer
from src.schemas import (
    ConflictGroup,
    MemoryCandidate,
    MemoryType,
    QueryFeatures,
    QueryMode,
    RouteDecision,
)


def config() -> dict:
    return {
        "query_analysis": {
            "aliases": {},
            "stopwords": [
                "the", "a", "an", "is", "are", "was", "were",
                "do", "does", "did", "my", "me", "i", "to", "of",
                "and", "or", "in", "on", "for", "what", "which",
                "how", "why", "when",
            ],
        },
        "selection": {
            "max_evidence": 5,
            "max_per_memory_type": 4,
            "evidence_token_budget": 1200,
            "minimum_evidence": 2,
            "relevance_weight": 0.45,
            "requirement_weight": 0.30,
            "coverage_gain_weight": 0.15,
            "redundancy_weight": 0.10,
            "min_marginal_score": 0.18,
            "min_topical_score": 0.12,
            "guarantee_procedural_if_routed": True,
        },
        "evidence_requirements": {
            "default_max_evidence": 5,
            "hard_max_evidence": 8,
            "max_per_memory_type": 4,
            "evidence_token_budget": 1200,
            "timeline_min_historical_states": 1,
            "timeline_min_current_states": 1,
            "timeline_min_transition_evidence": 1,
            "timeline_transition_hard": False,
            "timeline_preferred_evidence": 5,
            "procedural_min_rules": 1,
            "procedural_preferred_evidence": 5,
            "conflict_min_alternatives": 1,
            "conflict_min_resolutions": 1,
            "conflict_preferred_evidence": 5,
            "explanation_min_supporting_evidence": 1,
            "explanation_preferred_evidence": 5,
            "answer_anchor_enabled": True,
            "conflict_preservation_min_topical": 0.18,
            "role_topical_thresholds": {
                "answer_target": 0.20,
                "current_state": 0.18,
                "historical_state": 0.15,
                "transition": 0.15,
                "procedural_rule": 0.05,
                "alternative_state": 0.16,
                "preferred_resolution": 0.18,
                "supporting_evidence": 0.14,
                "distinct_item": 0.18,
            },
        },
    }


def features(
    query: str,
    mode: QueryMode,
    *,
    asks_procedure: bool = False,
    asks_conflict: bool = False,
    asks_explanation: bool = False,
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
        information_needs=[query.lower()],
    )


def route(*types: MemoryType) -> RouteDecision:
    return RouteDecision(
        selected_types=list(types),
        scores={item.value: 0.9 for item in types},
        reasons={item.value: ["test"] for item in types},
    )


def conflict() -> ConflictGroup:
    return ConflictGroup(
        conflict_id="c1",
        key="unrelated",
        candidate_ids=["s1", "s2"],
        conflict_type="implicit",
        explicit=False,
        unresolved=True,
    )


def candidate(
    memory_id: str,
    text: str,
    memory_type: MemoryType,
    *,
    score: float = 0.7,
    status: str = "current",
    metadata: dict | None = None,
) -> MemoryCandidate:
    return MemoryCandidate(
        memory_id=memory_id,
        memory_type=memory_type,
        text=text,
        user_id="user01",
        status=status,
        confidence=1.0,
        final_score=score,
        lexical_score=0.2,
        graph_entity_score=0.1,
        metadata=metadata or {},
    )


def test_incidental_conflict_does_not_create_state_requirements() -> None:
    planner = EvidenceRequirementPlanner(config())
    plan = planner.plan(
        features=features(
            "What are the three memory types and what does each record?",
            QueryMode.ATEMPORAL,
        ),
        route=route(MemoryType.SEMANTIC),
        conflicts=[conflict()],
    )
    roles = {requirement.role for requirement in plan.requirements}
    assert "distinct_item" in roles
    assert "alternative_state" not in roles
    assert "preferred_resolution" not in roles


def test_timeline_uses_one_historical_hard_and_soft_transition() -> None:
    planner = EvidenceRequirementPlanner(config())
    plan = planner.plan(
        features=features(
            "How did my career target change over time?",
            QueryMode.TIMELINE,
        ),
        route=route(MemoryType.EPISODIC, MemoryType.SEMANTIC),
        conflicts=[],
    )
    requirements = {item.role: item for item in plan.requirements}
    assert requirements["historical_state"].min_count == 1
    assert requirements["historical_state"].hard is True
    assert requirements["current_state"].hard is True
    assert requirements["transition"].hard is False


def test_route_only_procedure_is_soft() -> None:
    planner = EvidenceRequirementPlanner(config())
    plan = planner.plan(
        features=features(
            "What are the three memory types?",
            QueryMode.ATEMPORAL,
        ),
        route=route(MemoryType.SEMANTIC, MemoryType.PROCEDURAL),
        conflicts=[],
    )
    procedural = next(
        item for item in plan.requirements
        if item.role == "procedural_rule"
    )
    assert procedural.hard is False


def test_hypothetical_conflict_policy_is_procedural_mode() -> None:
    analyzer = QueryAnalyzer(config())
    result = analyzer.analyse(
        "If an older memory conflicts with a later project decision, "
        "how should the system answer?"
    )
    assert result.asks_procedure is True
    assert result.query_mode == QueryMode.PROCEDURAL


def test_should_you_is_detected_as_procedural() -> None:
    analyzer = QueryAnalyzer(config())
    result = analyzer.analyse(
        "Should you invent exact prices if the price is not stored?"
    )
    assert result.asks_procedure is True
    assert result.query_mode == QueryMode.PROCEDURAL


def test_procedural_rule_can_be_answer_target() -> None:
    cfg = config()
    selector = EvidenceSelector(cfg, CoverageEstimator(cfg))
    procedure = candidate(
        "p1",
        "Do not invent unsupported facts; state that evidence is missing.",
        MemoryType.PROCEDURAL,
        metadata={
            "direct_match": True,
            "task_match": 1.0,
            "instruction": "Do not invent unsupported facts.",
        },
    )
    selected = selector.select(
        candidates=[procedure],
        features=features(
            "What should I say if there is no stored evidence?",
            QueryMode.PROCEDURAL,
            asks_procedure=True,
        ),
        route=route(MemoryType.PROCEDURAL),
        conflicts=[],
    )
    assert [item.memory_id for item in selected] == ["p1"]
    assert selector.last_requirement_status["answer_target"]["complete"]
    assert selector.last_requirement_status["procedural_rule"]["complete"]


def test_no_candidates_records_infeasible_status() -> None:
    cfg = config()
    selector = EvidenceSelector(cfg, CoverageEstimator(cfg))
    selected = selector.select(
        candidates=[],
        features=features(
            "What should I do if booking information is unclear?",
            QueryMode.PROCEDURAL,
            asks_procedure=True,
        ),
        route=route(MemoryType.PROCEDURAL),
        conflicts=[],
    )
    assert selected == []
    assert selector.last_requirement_status
    assert all(
        status["feasible"] is False
        for status in selector.last_requirement_status.values()
    )


def test_selector_annotates_eligible_roles_and_feasibility() -> None:
    cfg = config()
    selector = EvidenceSelector(cfg, CoverageEstimator(cfg))
    current = candidate(
        "s1",
        "project current scope is a small memory controller prototype",
        MemoryType.SEMANTIC,
    )
    selector.select(
        candidates=[current],
        features=features(
            "What is my current project scope?",
            QueryMode.CURRENT,
        ),
        route=route(MemoryType.SEMANTIC),
        conflicts=[],
    )
    assert "answer_target" in current.metadata["eligible_evidence_roles"]
    assert selector.last_requirement_status["answer_target"]["feasible"]
