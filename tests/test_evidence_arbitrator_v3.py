from __future__ import annotations

from pathlib import Path

from src.config import load_config
from src.coverage_estimator import (
    CoverageEstimator,
)
from src.evidence_arbitrator_v3 import (
    EvidenceArbitratorV3,
)
from src.evidence_requirements import (
    EvidenceRequirement,
)
from src.evidence_utility import (
    EvidenceUtilityModel,
)
from src.query_analyzer import (
    QueryAnalyzer,
)
from src.requirement_gain_v3 import (
    QueryConsistentRequirementGain,
)
from src.schemas import (
    MemoryCandidate,
    MemoryType,
)


ROOT = Path(__file__).resolve().parents[1]


def build():
    cfg = load_config(
        ROOT
        / "configs"
        / "c3_lite_v2_2_final.yaml"
    )

    utility = EvidenceUtilityModel(
        cfg,
        CoverageEstimator(
            cfg
        ),
    )

    gain = (
        QueryConsistentRequirementGain()
    )

    arbitrator = EvidenceArbitratorV3(
        cfg,
        utility,
        gain,
    )

    analyzer = QueryAnalyzer(
        cfg
    )

    return (
        arbitrator,
        analyzer,
    )


def memory(
    memory_id: str,
    *,
    temporal_role: str,
    compatible: bool,
    evidence_roles: list[str],
    validity: float = 1.0,
) -> MemoryCandidate:
    return MemoryCandidate(
        memory_id=memory_id,
        memory_type=(
            MemoryType.SEMANTIC
        ),
        text=(
            "The MSc project scope "
            "is a memory controller."
        ),
        user_id="user01",
        status="current",
        confidence=1.0,
        lexical_score=0.8,
        graph_entity_score=0.8,
        route_compatibility_score=0.9,
        validity_score=validity,
        source_confidence_score=1.0,
        final_score=0.8,
        metadata={
            "evidence_roles": list(
                evidence_roles
            ),
            "query_relative_temporal_role": (
                temporal_role
            ),
            "query_relative_temporal_compatible": (
                compatible
            ),
        },
    )


def test_historical_query_rejects_incompatible_current_filler() -> None:
    arbitrator, analyzer = build()

    features = analyzer.analyse(
        "What was my previous MSc project scope?"
    )

    historical = memory(
        "historical",
        temporal_role="historical_state",
        compatible=True,
        evidence_roles=[
            "answer_target",
            "historical_state",
        ],
    )

    current = memory(
        "current",
        temporal_role="current_state",
        compatible=False,
        evidence_roles=[
            "historical_state",
        ],
        validity=0.25,
    )

    requirements = [
        EvidenceRequirement(
            role="answer_target",
            hard=True,
        ),
        EvidenceRequirement(
            role="historical_state",
            hard=True,
        ),
    ]

    result = arbitrator.arbitrate(
        candidates=[
            current,
            historical,
        ],
        features=features,
        requirements=requirements,
    )

    assert result.selected_ids == [
        "historical"
    ]

    assert result.hard_complete

    assert (
        "current"
        in result.rejected_incompatible
    )


def test_current_query_selects_current_answer() -> None:
    arbitrator, analyzer = build()

    features = analyzer.analyse(
        "What is my current MSc project scope?"
    )

    current = memory(
        "current",
        temporal_role="current_state",
        compatible=True,
        evidence_roles=[
            "answer_target",
            "current_state",
        ],
    )

    requirements = [
        EvidenceRequirement(
            role="answer_target",
            hard=True,
        ),
        EvidenceRequirement(
            role="current_state",
            hard=True,
        ),
    ]

    result = arbitrator.arbitrate(
        candidates=[current],
        features=features,
        requirements=requirements,
    )

    assert result.selected_ids == [
        "current"
    ]

    assert result.hard_complete


def test_timeline_selects_both_required_endpoints() -> None:
    arbitrator, analyzer = build()

    features = analyzer.analyse(
        "How did my project scope change over time?"
    )

    current = memory(
        "current",
        temporal_role="current_endpoint",
        compatible=True,
        evidence_roles=[
            "answer_target",
        ],
    )

    historical = memory(
        "historical",
        temporal_role="historical_state",
        compatible=True,
        evidence_roles=[],
    )

    requirements = [
        EvidenceRequirement(
            role="answer_target",
            hard=True,
        ),
        EvidenceRequirement(
            role="historical_state",
            hard=True,
        ),
        EvidenceRequirement(
            role="current_state",
            hard=True,
        ),
    ]

    result = arbitrator.arbitrate(
        candidates=[
            current,
            historical,
        ],
        features=features,
        requirements=requirements,
    )

    assert set(
        result.selected_ids
    ) == {
        "current",
        "historical",
    }

    assert result.hard_complete


def test_soft_transition_is_selected_after_hard_requirements() -> None:
    arbitrator, analyzer = build()

    features = analyzer.analyse(
        "How did my project scope change over time?"
    )

    current = memory(
        "current",
        temporal_role="current_endpoint",
        compatible=True,
        evidence_roles=[
            "answer_target",
        ],
    )

    historical = memory(
        "historical",
        temporal_role="historical_state",
        compatible=True,
        evidence_roles=[],
    )

    transition = memory(
        "transition",
        temporal_role="transition_event",
        compatible=True,
        evidence_roles=[],
        validity=0.95,
    )

    requirements = [
        EvidenceRequirement(
            role="answer_target",
            hard=True,
        ),
        EvidenceRequirement(
            role="historical_state",
            hard=True,
        ),
        EvidenceRequirement(
            role="current_state",
            hard=True,
        ),
        EvidenceRequirement(
            role="transition",
            hard=False,
        ),
    ]

    result = arbitrator.arbitrate(
        candidates=[
            transition,
            historical,
            current,
        ],
        features=features,
        requirements=requirements,
    )

    assert set(
        result.selected_ids
    ) == {
        "current",
        "historical",
        "transition",
    }

    assert result.hard_complete
    assert result.soft_complete

    assert result.steps[-1].phase == (
        "soft"
    )


def test_zero_gain_procedural_filler_is_not_added_by_default() -> None:
    arbitrator, analyzer = build()

    features = analyzer.analyse(
        (
            "When discussing my dissertation "
            "with my supervisor, how should "
            "the answer be written?"
        )
    )

    primary = MemoryCandidate(
        memory_id="primary",
        memory_type=MemoryType.PROCEDURAL,
        text="Use concise academic language.",
        user_id="user01",
        status="current",
        confidence=1.0,
        lexical_score=1.0,
        graph_entity_score=1.0,
        route_compatibility_score=1.0,
        validity_score=1.0,
        source_confidence_score=1.0,
        final_score=1.0,
        metadata={
            "evidence_roles": [
                "answer_target",
                "procedural_rule",
            ],
            "query_relative_temporal_role": (
                "procedural_rule"
            ),
            "query_relative_temporal_compatible": (
                True
            ),
        },
    )

    filler = MemoryCandidate(
        memory_id="filler",
        memory_type=MemoryType.PROCEDURAL,
        text=(
            "Another generally relevant "
            "writing instruction."
        ),
        user_id="user01",
        status="current",
        confidence=1.0,
        lexical_score=0.9,
        graph_entity_score=0.9,
        route_compatibility_score=1.0,
        validity_score=1.0,
        source_confidence_score=1.0,
        final_score=0.9,
        metadata={
            "evidence_roles": [],
            "query_relative_temporal_role": (
                "procedural_rule"
            ),
            "query_relative_temporal_compatible": (
                True
            ),
        },
    )

    requirements = [
        EvidenceRequirement(
            role="answer_target",
            hard=True,
        ),
        EvidenceRequirement(
            role="procedural_rule",
            hard=True,
        ),
    ]

    result = arbitrator.arbitrate(
        candidates=[
            filler,
            primary,
        ],
        features=features,
        requirements=requirements,
    )

    assert result.selected_ids == [
        "primary"
    ]

    assert result.hard_complete