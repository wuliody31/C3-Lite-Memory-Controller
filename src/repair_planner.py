from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .evidence_requirements import EvidenceRequirement
from .requirement_spec import RequirementSpec
from .schemas import MemoryType


@dataclass(slots=True)
class RepairPlan:
    """Targeted repair request for missing evidence requirements."""

    needed: bool

    missing_roles: list[str] = field(
        default_factory=list
    )

    target_memory_types: list[MemoryType] = field(
        default_factory=list
    )

    role_targets: dict[str, list[MemoryType]] = field(
        default_factory=dict
    )

    expands_beyond_initial_route: bool = False

    reasons: list[str] = field(
        default_factory=list
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "needed": self.needed,
            "missing_roles": list(
                self.missing_roles
            ),
            "target_memory_types": [
                memory_type.value
                for memory_type
                in self.target_memory_types
            ],
            "role_targets": {
                role: [
                    memory_type.value
                    for memory_type in targets
                ]
                for role, targets
                in self.role_targets.items()
            },
            "expands_beyond_initial_route": (
                self.expands_beyond_initial_route
            ),
            "reasons": list(self.reasons),
        }


class TargetedRepairPlanner:
    """Map missing evidence roles to targeted memory sources.

    C3-v3 interpretation:

        MissingRequirements(S, z_q)
            -> TargetedRepairPlan

    This stage only plans repair. It does not perform retrieval.
    """

    ROLE_TARGETS: dict[
        str,
        tuple[MemoryType, ...],
    ] = {
        "current_state": (
            MemoryType.SEMANTIC,
        ),

        "historical_state": (
            MemoryType.EPISODIC,
            MemoryType.SEMANTIC,
        ),

        "transition": (
            MemoryType.EPISODIC,
            MemoryType.SEMANTIC,
        ),

        "procedural_rule": (
            MemoryType.PROCEDURAL,
        ),

        "alternative_state": (
            MemoryType.EPISODIC,
            MemoryType.SEMANTIC,
        ),

        "preferred_resolution": (
            MemoryType.SEMANTIC,
        ),
    }

    def plan(
        self,
        *,
        spec: RequirementSpec,
        missing_requirements: list[
            EvidenceRequirement
        ],
    ) -> RepairPlan:
        if not missing_requirements:
            return RepairPlan(
                needed=False,
                reasons=[
                    "no missing evidence requirements"
                ],
            )

        role_targets: dict[
            str,
            list[MemoryType],
        ] = {}

        combined_targets: list[
            MemoryType
        ] = []

        missing_roles: list[str] = []

        for requirement in missing_requirements:
            role = requirement.role

            missing_roles.append(role)

            targets = self._targets_for_role(
                role=role,
                spec=spec,
            )

            role_targets[role] = targets

            for memory_type in targets:
                if (
                    memory_type
                    not in combined_targets
                ):
                    combined_targets.append(
                        memory_type
                    )

        ordered_targets = [
            memory_type
            for memory_type in MemoryType
            if memory_type in combined_targets
        ]

        initial_route = set(
            spec.memory_types
        )

        expands_beyond_initial_route = any(
            memory_type not in initial_route
            for memory_type
            in ordered_targets
        )

        reasons = [
            "missing_roles="
            + ",".join(missing_roles)
        ]

        if expands_beyond_initial_route:
            reasons.append(
                "repair_requires_route_expansion"
            )
        else:
            reasons.append(
                "repair_within_initial_route"
            )

        return RepairPlan(
            needed=True,
            missing_roles=missing_roles,
            target_memory_types=(
                ordered_targets
            ),
            role_targets=role_targets,
            expands_beyond_initial_route=(
                expands_beyond_initial_route
            ),
            reasons=reasons,
        )

    def _targets_for_role(
        self,
        *,
        role: str,
        spec: RequirementSpec,
    ) -> list[MemoryType]:
        explicit_targets = (
            self.ROLE_TARGETS.get(role)
        )

        if explicit_targets is not None:
            return list(explicit_targets)

        # Generic requirements such as answer_target,
        # supporting_evidence and distinct_item remain
        # grounded in the query's compiled route.
        if spec.memory_types:
            return list(spec.memory_types)

        # Defensive fallback only.
        return list(MemoryType)