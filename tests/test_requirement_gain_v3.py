from __future__ import annotations

from src.evidence_requirements import (
    EvidenceRequirement,
)
from src.requirement_gain_v3 import (
    QueryConsistentRequirementGain,
)
from src.schemas import (
    MemoryCandidate,
    MemoryType,
)


def memory(
    memory_id: str,
    *,
    temporal_role: str,
    compatible: bool,
    legacy_roles: list[str],
) -> MemoryCandidate:
    return MemoryCandidate(
        memory_id=memory_id,
        memory_type=MemoryType.SEMANTIC,
        text="Project scope memory.",
        user_id="user01",
        status="current",
        metadata={
            "evidence_roles": (
                legacy_roles
            ),
            "query_relative_temporal_role": (
                temporal_role
            ),
            "query_relative_temporal_compatible": (
                compatible
            ),
        },
    )


def test_incompatible_current_memory_cannot_satisfy_historical_requirement() -> None:
    estimator = (
        QueryConsistentRequirementGain()
    )

    requirements = [
        EvidenceRequirement(
            role="historical_state",
            hard=True,
        )
    ]

    candidate = memory(
        "current",
        temporal_role="current_state",
        compatible=False,
        legacy_roles=[
            "historical_state",
        ],
    )

    result = estimator.evaluate(
        candidate=candidate,
        selected=[],
        requirements=requirements,
    )

    assert (
        "historical_state"
        not in result.effective_roles
    )

    assert result.gain == 0.0


def test_valid_historical_memory_satisfies_historical_requirement() -> None:
    estimator = (
        QueryConsistentRequirementGain()
    )

    requirements = [
        EvidenceRequirement(
            role="historical_state",
            hard=True,
        )
    ]

    candidate = memory(
        "historical",
        temporal_role="historical_state",
        compatible=True,
        legacy_roles=[
            "historical_state",
        ],
    )

    result = estimator.evaluate(
        candidate=candidate,
        selected=[],
        requirements=requirements,
    )

    assert (
        "historical_state"
        in result.effective_roles
    )

    assert result.gain == 1.0

    assert result.hard_gain == 1.0


def test_current_endpoint_satisfies_timeline_current_requirement() -> None:
    estimator = (
        QueryConsistentRequirementGain()
    )

    requirements = [
        EvidenceRequirement(
            role="current_state",
            hard=True,
        )
    ]

    candidate = memory(
        "current",
        temporal_role="current_endpoint",
        compatible=True,
        legacy_roles=[],
    )

    result = estimator.evaluate(
        candidate=candidate,
        selected=[],
        requirements=requirements,
    )

    assert (
        "current_state"
        in result.effective_roles
    )

    assert result.gain == 1.0


def test_transition_event_satisfies_transition_requirement() -> None:
    estimator = (
        QueryConsistentRequirementGain()
    )

    requirements = [
        EvidenceRequirement(
            role="transition",
            hard=False,
        )
    ]

    candidate = memory(
        "transition",
        temporal_role="transition_event",
        compatible=True,
        legacy_roles=[],
    )

    result = estimator.evaluate(
        candidate=candidate,
        selected=[],
        requirements=requirements,
    )

    assert (
        "transition"
        in result.effective_roles
    )

    assert result.gain == 1.0

    assert result.soft_gain == 1.0


def test_already_satisfied_requirement_has_zero_marginal_gain() -> None:
    estimator = (
        QueryConsistentRequirementGain()
    )

    requirements = [
        EvidenceRequirement(
            role="historical_state",
            hard=True,
        )
    ]

    first = memory(
        "historical_1",
        temporal_role="historical_state",
        compatible=True,
        legacy_roles=[
            "historical_state",
        ],
    )

    second = memory(
        "historical_2",
        temporal_role="historical_state",
        compatible=True,
        legacy_roles=[
            "historical_state",
        ],
    )

    result = estimator.evaluate(
        candidate=second,
        selected=[first],
        requirements=requirements,
    )

    assert result.gain == 0.0