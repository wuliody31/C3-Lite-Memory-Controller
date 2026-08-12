from __future__ import annotations

from src.evidence_requirements import (
    EvidenceRequirement,
)
from src.repair_planner import (
    TargetedRepairPlanner,
)
from src.requirement_spec import (
    RequirementSpec,
)
from src.schemas import (
    MemoryType,
    QueryMode,
)


def make_spec(
    memory_types: list[MemoryType],
) -> RequirementSpec:
    return RequirementSpec(
        query="test query",
        memory_types=memory_types,
        temporal_mode=QueryMode.ATEMPORAL,
        information_needs=[
            "test need"
        ],
        requirements=[],
        token_budget=1200,
        max_evidence=5,
        max_per_memory_type=3,
    )


def test_no_missing_requirement_needs_no_repair():
    planner = TargetedRepairPlanner()

    result = planner.plan(
        spec=make_spec(
            [MemoryType.SEMANTIC]
        ),
        missing_requirements=[],
    )

    assert not result.needed

    assert (
        result.target_memory_types
        == []
    )


def test_current_state_targets_semantic_memory():
    planner = TargetedRepairPlanner()

    missing = [
        EvidenceRequirement(
            role="current_state",
            hard=True,
        )
    ]

    result = planner.plan(
        spec=make_spec(
            [MemoryType.SEMANTIC]
        ),
        missing_requirements=missing,
    )

    assert result.needed

    assert result.target_memory_types == [
        MemoryType.SEMANTIC
    ]

    assert not (
        result.expands_beyond_initial_route
    )


def test_historical_state_can_target_two_memory_types():
    planner = TargetedRepairPlanner()

    missing = [
        EvidenceRequirement(
            role="historical_state",
            hard=True,
        )
    ]

    result = planner.plan(
        spec=make_spec(
            [MemoryType.SEMANTIC]
        ),
        missing_requirements=missing,
    )

    assert set(
        result.target_memory_types
    ) == {
        MemoryType.EPISODIC,
        MemoryType.SEMANTIC,
    }

    assert (
        result.expands_beyond_initial_route
    )


def test_procedural_requirement_can_expand_route():
    planner = TargetedRepairPlanner()

    missing = [
        EvidenceRequirement(
            role="procedural_rule",
            hard=True,
        )
    ]

    result = planner.plan(
        spec=make_spec(
            [MemoryType.SEMANTIC]
        ),
        missing_requirements=missing,
    )

    assert result.target_memory_types == [
        MemoryType.PROCEDURAL
    ]

    assert (
        result.expands_beyond_initial_route
    )

    assert (
        "repair_requires_route_expansion"
        in result.reasons
    )


def test_generic_answer_target_uses_compiled_route():
    planner = TargetedRepairPlanner()

    missing = [
        EvidenceRequirement(
            role="answer_target",
            hard=True,
        )
    ]

    result = planner.plan(
        spec=make_spec(
            [
                MemoryType.EPISODIC,
                MemoryType.SEMANTIC,
            ]
        ),
        missing_requirements=missing,
    )

    assert result.target_memory_types == [
        MemoryType.EPISODIC,
        MemoryType.SEMANTIC,
    ]

    assert not (
        result.expands_beyond_initial_route
    )