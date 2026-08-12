from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .evidence_requirements import (
    EvidenceRequirement,
)
from .evidence_utility import (
    EvidenceUtilityModel,
    EvidenceUtilityBreakdown,
)
from .requirement_gain_v3 import (
    QueryConsistentRequirementGain,
    RequirementGainResult,
)
from .schemas import (
    MemoryCandidate,
    QueryFeatures,
)
from .text_utils import (
    approximate_token_count,
)


@dataclass(slots=True)
class ArbitrationStep:
    step: int
    phase: str
    memory_id: str

    hard_gain: float
    soft_gain: float
    requirement_gain: float
    utility: float

    token_cost: int
    tokens_after: int

    effective_roles: list[str]
    gained_roles: list[str]

    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ArbitrationResult:
    selected: list[MemoryCandidate]
    selected_ids: list[str]

    steps: list[ArbitrationStep]

    hard_complete: bool
    soft_complete: bool

    used_tokens: int

    rejected_incompatible: list[str]
    rejected_budget: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_ids": list(
                self.selected_ids
            ),
            "steps": [
                step.to_dict()
                for step in self.steps
            ],
            "hard_complete": (
                self.hard_complete
            ),
            "soft_complete": (
                self.soft_complete
            ),
            "used_tokens": (
                self.used_tokens
            ),
            "rejected_incompatible": list(
                self.rejected_incompatible
            ),
            "rejected_budget": list(
                self.rejected_budget
            ),
        }


class EvidenceArbitratorV3:
    """Greedy validity-constrained evidence-set arbitration.

    Selection is lexicographic:

        Phase A:
            satisfy hard requirements.

        Phase B:
            satisfy soft requirements.

        Phase C:
            optional enrichment.

    In the initial C3-v3 shadow implementation Phase C is disabled by
    default. This produces the smallest evidence set justified by the
    compiled requirements, while allowing us to compare it against the
    legacy selector before changing production behaviour.
    """

    def __init__(
        self,
        config: dict[str, Any],
        utility_model: EvidenceUtilityModel,
        requirement_gain_model: (
            QueryConsistentRequirementGain
        ),
    ) -> None:
        self.config = config

        self.utility_model = (
            utility_model
        )

        self.requirement_gain_model = (
            requirement_gain_model
        )

        settings = config.get(
            "evidence_arbitration_v3",
            {},
        )

        selection = config.get(
            "selection",
            {},
        )

        self.default_max_evidence = int(
            settings.get(
                "max_evidence",
                selection.get(
                    "max_evidence",
                    5,
                ),
            )
        )

        self.default_token_budget = int(
            settings.get(
                "token_budget",
                selection.get(
                    "evidence_token_budget",
                    1200,
                ),
            )
        )

        self.default_max_per_memory_type = int(
            settings.get(
                "max_per_memory_type",
                selection.get(
                    "max_per_memory_type",
                    3,
                ),
            )
        )

        # Deliberately disabled for M2-C2A.
        # We first validate requirement-complete minimal selection.
        self.allow_optional_enrichment = bool(
            settings.get(
                "allow_optional_enrichment",
                False,
            )
        )

    def arbitrate(
        self,
        *,
        candidates: list[MemoryCandidate],
        features: QueryFeatures,
        requirements: list[EvidenceRequirement],
        max_evidence: int | None = None,
        token_budget: int | None = None,
        max_per_memory_type: int | None = None,
    ) -> ArbitrationResult:
        max_evidence = (
            self.default_max_evidence
            if max_evidence is None
            else int(max_evidence)
        )

        token_budget = (
            self.default_token_budget
            if token_budget is None
            else int(token_budget)
        )

        max_per_memory_type = (
            self.default_max_per_memory_type
            if max_per_memory_type is None
            else int(max_per_memory_type)
        )

        selected: list[
            MemoryCandidate
        ] = []

        remaining = list(
            candidates
        )

        steps: list[
            ArbitrationStep
        ] = []

        rejected_incompatible: list[
            str
        ] = []

        rejected_budget: list[
            str
        ] = []

        # ---------------------------------------------
        # Phase A: hard requirements
        # ---------------------------------------------

        while (
            len(selected)
            < max_evidence
            and not self._requirements_complete(
                selected=selected,
                requirements=requirements,
                hard_only=True,
            )
        ):
            choice = self._best_candidate(
                candidates=remaining,
                selected=selected,
                features=features,
                requirements=requirements,
                phase="hard",
                token_budget=token_budget,
                max_per_memory_type=(
                    max_per_memory_type
                ),
                rejected_incompatible=(
                    rejected_incompatible
                ),
                rejected_budget=(
                    rejected_budget
                ),
            )

            if choice is None:
                break

            candidate, gain, utility = (
                choice
            )

            self._append_choice(
                selected=selected,
                remaining=remaining,
                candidate=candidate,
                gain=gain,
                utility=utility,
                steps=steps,
                phase="hard",
                reason=(
                    "hard_requirement_gain"
                ),
            )

        # ---------------------------------------------
        # Phase B: soft requirements
        # ---------------------------------------------

        while (
            len(selected)
            < max_evidence
            and not self._requirements_complete(
                selected=selected,
                requirements=requirements,
                soft_only=True,
            )
        ):
            choice = self._best_candidate(
                candidates=remaining,
                selected=selected,
                features=features,
                requirements=requirements,
                phase="soft",
                token_budget=token_budget,
                max_per_memory_type=(
                    max_per_memory_type
                ),
                rejected_incompatible=(
                    rejected_incompatible
                ),
                rejected_budget=(
                    rejected_budget
                ),
            )

            if choice is None:
                break

            candidate, gain, utility = (
                choice
            )

            self._append_choice(
                selected=selected,
                remaining=remaining,
                candidate=candidate,
                gain=gain,
                utility=utility,
                steps=steps,
                phase="soft",
                reason=(
                    "soft_requirement_gain"
                ),
            )

        # ---------------------------------------------
        # Phase C: optional enrichment
        #
        # Disabled by default for shadow validation.
        # ---------------------------------------------

        if self.allow_optional_enrichment:
            while (
                remaining
                and len(selected)
                < max_evidence
            ):
                choice = (
                    self._best_candidate(
                        candidates=remaining,
                        selected=selected,
                        features=features,
                        requirements=requirements,
                        phase="optional",
                        token_budget=(
                            token_budget
                        ),
                        max_per_memory_type=(
                            max_per_memory_type
                        ),
                        rejected_incompatible=(
                            rejected_incompatible
                        ),
                        rejected_budget=(
                            rejected_budget
                        ),
                    )
                )

                if choice is None:
                    break

                candidate, gain, utility = (
                    choice
                )

                # Optional evidence must still have
                # positive net marginal utility.
                if utility.utility <= 0.0:
                    break

                self._append_choice(
                    selected=selected,
                    remaining=remaining,
                    candidate=candidate,
                    gain=gain,
                    utility=utility,
                    steps=steps,
                    phase="optional",
                    reason=(
                        "positive_optional_utility"
                    ),
                )

        return ArbitrationResult(
            selected=selected,
            selected_ids=[
                candidate.memory_id
                for candidate in selected
            ],
            steps=steps,
            hard_complete=(
                self._requirements_complete(
                    selected=selected,
                    requirements=requirements,
                    hard_only=True,
                )
            ),
            soft_complete=(
                self._requirements_complete(
                    selected=selected,
                    requirements=requirements,
                    soft_only=True,
                )
            ),
            used_tokens=(
                self._used_tokens(
                    selected
                )
            ),
            rejected_incompatible=sorted(
                set(
                    rejected_incompatible
                )
            ),
            rejected_budget=sorted(
                set(
                    rejected_budget
                )
            ),
        )

    def _best_candidate(
        self,
        *,
        candidates: list[MemoryCandidate],
        selected: list[MemoryCandidate],
        features: QueryFeatures,
        requirements: list[EvidenceRequirement],
        phase: str,
        token_budget: int,
        max_per_memory_type: int,
        rejected_incompatible: list[str],
        rejected_budget: list[str],
    ) -> tuple[
        MemoryCandidate,
        RequirementGainResult,
        EvidenceUtilityBreakdown,
    ] | None:
        scored: list[
            tuple[
                tuple[
                    float,
                    float,
                    float,
                    float,
                    str,
                ],
                MemoryCandidate,
                RequirementGainResult,
                EvidenceUtilityBreakdown,
            ]
        ] = []

        for candidate in candidates:
            compatible = (
                candidate.metadata.get(
                    "query_relative_temporal_compatible"
                )
            )

            # Query-relative incompatibility is a hard
            # feasibility constraint in C3-v3.
            if compatible is False:
                rejected_incompatible.append(
                    candidate.memory_id
                )
                continue

            if (
                self._memory_type_count(
                    selected,
                    candidate,
                )
                >= max_per_memory_type
            ):
                continue

            candidate_cost = (
                approximate_token_count(
                    self.utility_model
                    ._candidate_text(
                        candidate
                    )
                )
            )

            if (
                self._used_tokens(
                    selected
                )
                + candidate_cost
                > token_budget
            ):
                rejected_budget.append(
                    candidate.memory_id
                )
                continue

            gain = (
                self.requirement_gain_model
                .evaluate(
                    candidate=candidate,
                    selected=selected,
                    requirements=requirements,
                )
            )

            if (
                phase == "hard"
                and gain.hard_gain <= 0.0
            ):
                continue

            if (
                phase == "soft"
                and gain.soft_gain <= 0.0
            ):
                continue

            utility = (
                self.utility_model.evaluate(
                    candidate=candidate,
                    selected=selected,
                    features=features,
                    requirement_gain=(
                        gain.gain
                    ),
                    token_budget=(
                        token_budget
                    ),
                )
            )

            if phase == "hard":
                key = (
                    gain.hard_gain,
                    gain.gain,
                    utility.utility,
                    candidate.final_score,
                    candidate.memory_id,
                )

            elif phase == "soft":
                key = (
                    gain.soft_gain,
                    gain.gain,
                    utility.utility,
                    candidate.final_score,
                    candidate.memory_id,
                )

            else:
                key = (
                    utility.utility,
                    gain.gain,
                    candidate.final_score,
                    candidate.validity_score,
                    candidate.memory_id,
                )

            scored.append(
                (
                    key,
                    candidate,
                    gain,
                    utility,
                )
            )

        if not scored:
            return None

        scored.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        _, candidate, gain, utility = (
            scored[0]
        )

        return (
            candidate,
            gain,
            utility,
        )

    def _append_choice(
        self,
        *,
        selected: list[MemoryCandidate],
        remaining: list[MemoryCandidate],
        candidate: MemoryCandidate,
        gain: RequirementGainResult,
        utility: EvidenceUtilityBreakdown,
        steps: list[ArbitrationStep],
        phase: str,
        reason: str,
    ) -> None:
        selected.append(
            candidate
        )

        remaining.remove(
            candidate
        )

        steps.append(
            ArbitrationStep(
                step=len(
                    selected
                ),
                phase=phase,
                memory_id=(
                    candidate.memory_id
                ),
                hard_gain=(
                    gain.hard_gain
                ),
                soft_gain=(
                    gain.soft_gain
                ),
                requirement_gain=(
                    gain.gain
                ),
                utility=(
                    utility.utility
                ),
                token_cost=(
                    utility.token_cost
                ),
                tokens_after=(
                    self._used_tokens(
                        selected
                    )
                ),
                effective_roles=list(
                    gain.effective_roles
                ),
                gained_roles=list(
                    gain.gained_roles
                ),
                reason=reason,
            )
        )

    def _requirements_complete(
        self,
        *,
        selected: list[MemoryCandidate],
        requirements: list[
            EvidenceRequirement
        ],
        hard_only: bool = False,
        soft_only: bool = False,
    ) -> bool:
        relevant = [
            requirement
            for requirement
            in requirements
            if (
                (
                    not hard_only
                    and not soft_only
                )
                or (
                    hard_only
                    and requirement.hard
                )
                or (
                    soft_only
                    and not requirement.hard
                )
            )
        ]

        if not relevant:
            return True

        role_counts: dict[
            str,
            int
        ] = {}

        for candidate in selected:
            roles = (
                self.requirement_gain_model
                .effective_roles(
                    candidate
                )
            )

            for role in roles:
                role_counts[role] = (
                    role_counts.get(
                        role,
                        0,
                    )
                    + 1
                )

        for requirement in relevant:
            if (
                role_counts.get(
                    requirement.role,
                    0,
                )
                < max(
                    1,
                    requirement.min_count,
                )
            ):
                return False

        return True

    @staticmethod
    def _memory_type_count(
        selected: list[MemoryCandidate],
        candidate: MemoryCandidate,
    ) -> int:
        return sum(
            1
            for item in selected
            if (
                item.memory_type
                == candidate.memory_type
            )
        )

    def _used_tokens(
        self,
        selected: list[
            MemoryCandidate
        ],
    ) -> int:
        return sum(
            approximate_token_count(
                self.utility_model
                ._candidate_text(
                    candidate
                )
            )
            for candidate
            in selected
        )