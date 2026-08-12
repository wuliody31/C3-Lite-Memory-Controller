from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .coverage_estimator import CoverageEstimator
from .evidence_requirements import EvidenceRequirement
from .requirement_spec import RequirementSpec
from .schemas import MemoryCandidate


@dataclass(slots=True)
class RequirementStatus:
    """Satisfaction state of one query-conditioned evidence requirement."""

    requirement: EvidenceRequirement
    required_count: int
    satisfied_count: int
    complete: bool
    supporting_memory_ids: list[str] = field(default_factory=list)

    @property
    def coverage(self) -> float:
        if self.required_count <= 0:
            return 1.0

        return min(
            1.0,
            self.satisfied_count / self.required_count,
        )


@dataclass(slots=True)
class SufficiencyResult:
    """Paper-level evidence sufficiency result for C3-v3."""

    sufficient: bool

    hard_requirement_coverage: float
    total_requirement_coverage: float
    information_coverage: float

    requirement_statuses: list[RequirementStatus] = field(
        default_factory=list
    )

    missing_requirements: list[EvidenceRequirement] = field(
        default_factory=list
    )

    missing_hard_requirements: list[EvidenceRequirement] = field(
        default_factory=list
    )

    satisfied_requirements: list[EvidenceRequirement] = field(
        default_factory=list
    )

    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sufficient": self.sufficient,
            "hard_requirement_coverage": (
                self.hard_requirement_coverage
            ),
            "total_requirement_coverage": (
                self.total_requirement_coverage
            ),
            "information_coverage": self.information_coverage,
            "requirement_statuses": [
                {
                    "role": status.requirement.role,
                    "hard": status.requirement.hard,
                    "distinct": status.requirement.distinct,
                    "required_count": status.required_count,
                    "satisfied_count": status.satisfied_count,
                    "coverage": status.coverage,
                    "complete": status.complete,
                    "supporting_memory_ids": list(
                        status.supporting_memory_ids
                    ),
                }
                for status in self.requirement_statuses
            ],
            "missing_requirements": [
                requirement.role
                for requirement in self.missing_requirements
            ],
            "missing_hard_requirements": [
                requirement.role
                for requirement in self.missing_hard_requirements
            ],
            "satisfied_requirements": [
                requirement.role
                for requirement in self.satisfied_requirements
            ],
            "reasons": list(self.reasons),
        }


class RequirementSufficiencyEvaluator:
    """Unify RC8.3 requirement status into C3-v3 sufficiency.

    In this behaviour-preserving version, role satisfaction comes directly
    from EvidenceSelector.last_requirement_status. The evaluator does not
    duplicate or replace the validated RC8.3 role-matching logic.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.coverage = CoverageEstimator(config)

    def evaluate(
        self,
        *,
        spec: RequirementSpec,
        selected: list[MemoryCandidate],
        selector_status: dict[str, dict[str, Any]],
    ) -> SufficiencyResult:
        statuses: list[RequirementStatus] = []

        for requirement in spec.requirements:
            raw_status = selector_status.get(
                requirement.role,
                {},
            )

            required_count = int(
                raw_status.get(
                    "required",
                    requirement.min_count,
                )
            )

            satisfied_count = int(
                raw_status.get(
                    "satisfied",
                    0,
                )
            )

            complete = bool(
                raw_status.get(
                    "complete",
                    satisfied_count >= required_count,
                )
            )

            supporting_memory_ids = [
                candidate.memory_id
                for candidate in selected
                if requirement.role
                in candidate.metadata.get(
                    "evidence_roles",
                    [],
                )
            ]

            statuses.append(
                RequirementStatus(
                    requirement=requirement,
                    required_count=required_count,
                    satisfied_count=satisfied_count,
                    complete=complete,
                    supporting_memory_ids=supporting_memory_ids,
                )
            )

        hard_statuses = [
            status
            for status in statuses
            if status.requirement.hard
        ]

        if hard_statuses:
            hard_requirement_coverage = (
                sum(
                    status.coverage
                    for status in hard_statuses
                )
                / len(hard_statuses)
            )
        else:
            hard_requirement_coverage = 1.0

        if statuses:
            total_requirement_coverage = (
                sum(
                    status.coverage
                    for status in statuses
                )
                / len(statuses)
            )
        else:
            total_requirement_coverage = (
                1.0 if selected else 0.0
            )

        missing_requirements = [
            status.requirement
            for status in statuses
            if not status.complete
        ]

        missing_hard_requirements = [
            status.requirement
            for status in statuses
            if (
                status.requirement.hard
                and not status.complete
            )
        ]

        satisfied_requirements = [
            status.requirement
            for status in statuses
            if status.complete
        ]

        sufficient = (
            len(missing_hard_requirements) == 0
        )

        information_coverage = self.coverage.compute(
            spec.information_needs,
            selected,
        )

        reasons: list[str] = []

        if sufficient:
            reasons.append(
                "all hard evidence requirements satisfied"
            )
        else:
            reasons.append(
                "one or more hard evidence requirements missing"
            )

        if missing_hard_requirements:
            reasons.append(
                "missing_hard_roles="
                + ",".join(
                    requirement.role
                    for requirement in missing_hard_requirements
                )
            )

        return SufficiencyResult(
            sufficient=sufficient,
            hard_requirement_coverage=round(
                hard_requirement_coverage,
                6,
            ),
            total_requirement_coverage=round(
                total_requirement_coverage,
                6,
            ),
            information_coverage=round(
                information_coverage,
                6,
            ),
            requirement_statuses=statuses,
            missing_requirements=missing_requirements,
            missing_hard_requirements=(
                missing_hard_requirements
            ),
            satisfied_requirements=satisfied_requirements,
            reasons=reasons,
        )