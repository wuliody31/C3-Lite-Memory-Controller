from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .evidence_requirements import (
    EvidenceRequirement,
)
from .schemas import MemoryCandidate


TEMPORAL_REQUIREMENT_ROLES = {
    "current_state",
    "historical_state",
    "transition",
}


@dataclass(slots=True)
class RequirementGainResult:
    memory_id: str

    effective_roles: list[str]
    gained_roles: list[str]

    before_coverage: float
    after_coverage: float
    gain: float

    hard_gain: float
    soft_gain: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class QueryConsistentRequirementGain:
    """Validity-constrained requirement gain for C3-v3.

    Legacy selector roles are retained for non-temporal requirements,
    but temporal roles are reconstructed from query-relative temporal
    semantics.

    This prevents a temporally incompatible memory from satisfying a
    historical/current/timeline requirement merely because a legacy
    conflict-resolution label assigned it that role.
    """

    HARD_WEIGHT = 1.0
    SOFT_WEIGHT = 0.5

    def evaluate(
        self,
        *,
        candidate: MemoryCandidate,
        selected: list[MemoryCandidate],
        requirements: list[EvidenceRequirement],
    ) -> RequirementGainResult:
        candidate_roles = (
            self.effective_roles(
                candidate
            )
        )

        before_counts = (
            self._role_counts(
                selected
            )
        )

        after_counts = dict(
            before_counts
        )

        for role in candidate_roles:
            after_counts[role] = (
                after_counts.get(
                    role,
                    0,
                )
                + 1
            )

        total_weight = 0.0

        before_value = 0.0
        after_value = 0.0

        hard_before = 0.0
        hard_after = 0.0
        hard_total = 0.0

        soft_before = 0.0
        soft_after = 0.0
        soft_total = 0.0

        gained_roles: list[str] = []

        for requirement in requirements:
            required = max(
                1,
                int(
                    requirement.min_count
                ),
            )

            weight = (
                self.HARD_WEIGHT
                if requirement.hard
                else self.SOFT_WEIGHT
            )

            weighted_required = (
                weight
                * required
            )

            total_weight += (
                weighted_required
            )

            before_count = min(
                before_counts.get(
                    requirement.role,
                    0,
                ),
                required,
            )

            after_count = min(
                after_counts.get(
                    requirement.role,
                    0,
                ),
                required,
            )

            before_component = (
                weight
                * before_count
            )

            after_component = (
                weight
                * after_count
            )

            before_value += (
                before_component
            )

            after_value += (
                after_component
            )

            if (
                after_count
                > before_count
            ):
                gained_roles.append(
                    requirement.role
                )

            if requirement.hard:
                hard_total += (
                    weighted_required
                )
                hard_before += (
                    before_component
                )
                hard_after += (
                    after_component
                )

            else:
                soft_total += (
                    weighted_required
                )
                soft_before += (
                    before_component
                )
                soft_after += (
                    after_component
                )

        before_coverage = (
            before_value
            / total_weight
            if total_weight
            else 1.0
        )

        after_coverage = (
            after_value
            / total_weight
            if total_weight
            else 1.0
        )

        gain = max(
            0.0,
            after_coverage
            - before_coverage,
        )

        hard_gain = (
            max(
                0.0,
                (
                    hard_after
                    - hard_before
                )
                / hard_total,
            )
            if hard_total
            else 0.0
        )

        soft_gain = (
            max(
                0.0,
                (
                    soft_after
                    - soft_before
                )
                / soft_total,
            )
            if soft_total
            else 0.0
        )

        return RequirementGainResult(
            memory_id=(
                candidate.memory_id
            ),
            effective_roles=sorted(
                candidate_roles
            ),
            gained_roles=sorted(
                set(
                    gained_roles
                )
            ),
            before_coverage=round(
                before_coverage,
                6,
            ),
            after_coverage=round(
                after_coverage,
                6,
            ),
            gain=round(
                gain,
                6,
            ),
            hard_gain=round(
                hard_gain,
                6,
            ),
            soft_gain=round(
                soft_gain,
                6,
            ),
        )

    def effective_roles(
        self,
        candidate: MemoryCandidate,
    ) -> set[str]:
        """Return requirement roles consistent with C3-v3 validity."""

        legacy_roles = {
            str(role)
            for role
            in candidate.metadata.get(
                "evidence_roles",
                [],
            )
        }

        # Remove legacy temporal assignments.
        roles = (
            legacy_roles
            - TEMPORAL_REQUIREMENT_ROLES
        )

        temporal_role = str(
            candidate.metadata.get(
                "query_relative_temporal_role",
                "",
            )
            or ""
        )

        compatible = (
            candidate.metadata.get(
                "query_relative_temporal_compatible"
            )
        )

        # A temporally incompatible factual memory must
        # not satisfy the answer target solely because
        # the legacy selector chose it as an anchor.
        if compatible is False:
            roles.discard(
                "answer_target"
            )

        # Reconstruct temporal requirement roles from
        # query-relative semantics.
        if compatible is not False:
            if temporal_role in {
                "current_state",
                "current_endpoint",
            }:
                roles.add(
                    "current_state"
                )

            elif (
                temporal_role
                == "historical_state"
            ):
                roles.add(
                    "historical_state"
                )

            elif (
                temporal_role
                == "transition_event"
            ):
                roles.add(
                    "transition"
                )

        return roles

    def _role_counts(
        self,
        candidates: list[
            MemoryCandidate
        ],
    ) -> dict[str, int]:
        counts: dict[
            str,
            int
        ] = {}

        for candidate in candidates:
            for role in (
                self.effective_roles(
                    candidate
                )
            ):
                counts[role] = (
                    counts.get(
                        role,
                        0,
                    )
                    + 1
                )

        return counts