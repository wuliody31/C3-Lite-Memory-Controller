from __future__ import annotations

from src.config import load_config
from src.evidence_requirements import EvidenceRequirement
from src.requirement_spec import RequirementSpec
from src.requirement_sufficiency import (
    RequirementSufficiencyEvaluator,
)
from src.schemas import (
    MemoryCandidate,
    MemoryType,
    QueryMode,
)


CONFIG_PATH = "configs/c3_lite_v2_2_final.yaml"


def config() -> dict:
    return load_config(CONFIG_PATH)


def candidate(
    memory_id: str,
    *,
    roles: list[str],
) -> MemoryCandidate:
    return MemoryCandidate(
        memory_id=memory_id,
        memory_type=MemoryType.SEMANTIC,
        text="test evidence",
        user_id="user01",
        status="current",
        confidence=1.0,
        final_score=1.0,
        lexical_score=0.0,
        graph_entity_score=0.0,
        metadata={
            "evidence_roles": roles,
        },
    )


def spec(
    requirements: list[EvidenceRequirement],
) -> RequirementSpec:
    return RequirementSpec(
        query="test query",
        memory_types=[
            MemoryType.SEMANTIC
        ],
        temporal_mode=QueryMode.ATEMPORAL,
        information_needs=[
            "test evidence"
        ],
        requirements=requirements,
        token_budget=1200,
        max_evidence=5,
        max_per_memory_type=3,
    )


def test_all_hard_requirements_are_sufficient() -> None:
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

    requirement_spec = spec(requirements)

    selected = [
        candidate(
            "s1",
            roles=[
                "answer_target",
                "current_state",
            ],
        )
    ]

    selector_status = {
        "answer_target": {
            "required": 1,
            "satisfied": 1,
            "hard": True,
            "distinct": False,
            "complete": True,
        },
        "current_state": {
            "required": 1,
            "satisfied": 1,
            "hard": True,
            "distinct": False,
            "complete": True,
        },
    }

    result = RequirementSufficiencyEvaluator(
        config()
    ).evaluate(
        spec=requirement_spec,
        selected=selected,
        selector_status=selector_status,
    )

    assert result.sufficient
    assert result.hard_requirement_coverage == 1.0
    assert result.missing_hard_requirements == []


def test_missing_hard_requirement_is_insufficient() -> None:
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

    requirement_spec = spec(requirements)

    selected = [
        candidate(
            "s1",
            roles=[
                "answer_target",
            ],
        )
    ]

    selector_status = {
        "answer_target": {
            "required": 1,
            "satisfied": 1,
            "hard": True,
            "distinct": False,
            "complete": True,
        },
        "current_state": {
            "required": 1,
            "satisfied": 0,
            "hard": True,
            "distinct": False,
            "complete": False,
        },
    }

    result = RequirementSufficiencyEvaluator(
        config()
    ).evaluate(
        spec=requirement_spec,
        selected=selected,
        selector_status=selector_status,
    )

    assert not result.sufficient

    assert [
        requirement.role
        for requirement
        in result.missing_hard_requirements
    ] == ["current_state"]

    assert result.hard_requirement_coverage == 0.5


def test_soft_requirement_does_not_block_answer() -> None:
    requirements = [
        EvidenceRequirement(
            role="answer_target",
            hard=True,
        ),
        EvidenceRequirement(
            role="transition",
            hard=False,
        ),
    ]

    requirement_spec = spec(requirements)

    selected = [
        candidate(
            "s1",
            roles=[
                "answer_target",
            ],
        )
    ]

    selector_status = {
        "answer_target": {
            "required": 1,
            "satisfied": 1,
            "hard": True,
            "distinct": False,
            "complete": True,
        },
        "transition": {
            "required": 1,
            "satisfied": 0,
            "hard": False,
            "distinct": False,
            "complete": False,
        },
    }

    result = RequirementSufficiencyEvaluator(
        config()
    ).evaluate(
        spec=requirement_spec,
        selected=selected,
        selector_status=selector_status,
    )

    assert result.sufficient

    assert [
        requirement.role
        for requirement
        in result.missing_requirements
    ] == ["transition"]

    assert result.missing_hard_requirements == []


def test_min_count_produces_fractional_coverage() -> None:
    requirements = [
        EvidenceRequirement(
            role="distinct_item",
            min_count=3,
            hard=True,
            distinct=True,
        )
    ]

    requirement_spec = spec(requirements)

    selected = [
        candidate(
            "s1",
            roles=["distinct_item"],
        ),
        candidate(
            "s2",
            roles=["distinct_item"],
        ),
    ]

    selector_status = {
        "distinct_item": {
            "required": 3,
            "satisfied": 2,
            "hard": True,
            "distinct": True,
            "complete": False,
        }
    }

    result = RequirementSufficiencyEvaluator(
        config()
    ).evaluate(
        spec=requirement_spec,
        selected=selected,
        selector_status=selector_status,
    )

    assert not result.sufficient

    assert result.hard_requirement_coverage == 0.666667

    assert (
        result.requirement_statuses[0].required_count
        == 3
    )

    assert (
        result.requirement_statuses[0].satisfied_count
        == 2
    )


def test_supporting_memory_ids_are_traced() -> None:
    requirements = [
        EvidenceRequirement(
            role="current_state",
            hard=True,
        )
    ]

    requirement_spec = spec(requirements)

    selected = [
        candidate(
            "s_current",
            roles=["current_state"],
        ),
        candidate(
            "s_other",
            roles=["answer_target"],
        ),
    ]

    selector_status = {
        "current_state": {
            "required": 1,
            "satisfied": 1,
            "hard": True,
            "distinct": False,
            "complete": True,
        }
    }

    result = RequirementSufficiencyEvaluator(
        config()
    ).evaluate(
        spec=requirement_spec,
        selected=selected,
        selector_status=selector_status,
    )

    assert (
        result.requirement_statuses[
            0
        ].supporting_memory_ids
        == ["s_current"]
    )