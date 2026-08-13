from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .coverage_estimator import CoverageEstimator
from .evidence_requirements import EvidenceRequirement
from .information_need_gain_v3 import (
    InformationNeedMatcherV3,
)
from .requirement_spec import (
    RequirementSlot,
    RequirementSpec,
)
from .schemas import (
    MemoryCandidate,
    MemoryType,
)


@dataclass(slots=True)
class RequirementStatus:
    """Satisfaction state of one coarse evidence requirement."""

    requirement: EvidenceRequirement
    required_count: int
    satisfied_count: int
    complete: bool
    supporting_memory_ids: list[str] = field(
        default_factory=list
    )

    @property
    def coverage(self) -> float:
        if self.required_count <= 0:
            return 1.0

        return min(
            1.0,
            self.satisfied_count
            / self.required_count,
        )


@dataclass(slots=True)
class SlotStatus:
    """Shadow satisfaction state of one semantic RequirementSlot."""

    slot: RequirementSlot

    required_count: int
    satisfied_count: int
    complete: bool

    supporting_memory_ids: list[str] = field(
        default_factory=list
    )

    semantic_matching_memory_ids: list[str] = field(
        default_factory=list
    )

    structurally_rejected_memory_ids: list[str] = field(
        default_factory=list
    )

    reasons: list[str] = field(
        default_factory=list
    )

    @property
    def coverage(self) -> float:
        if self.required_count <= 0:
            return 1.0

        return min(
            1.0,
            self.satisfied_count
            / self.required_count,
        )


@dataclass(slots=True)
class SufficiencyResult:
    """Role sufficiency plus C3-v3 semantic-slot shadow sufficiency.

    IMPORTANT:
    ``sufficient`` preserves the validated role-level behaviour.

    ``slot_sufficient`` and ``combined_sufficient_shadow`` remain
    descriptive only in M2-C3.3B.
    """

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

    reasons: list[str] = field(
        default_factory=list
    )

    slot_sufficient: bool = True

    hard_slot_coverage: float = 1.0
    total_slot_coverage: float = 1.0

    combined_sufficient_shadow: bool = True

    slot_statuses: list[SlotStatus] = field(
        default_factory=list
    )

    missing_slots: list[RequirementSlot] = field(
        default_factory=list
    )

    missing_hard_slots: list[RequirementSlot] = field(
        default_factory=list
    )

    satisfied_slots: list[RequirementSlot] = field(
        default_factory=list
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sufficient": self.sufficient,

            "hard_requirement_coverage": (
                self.hard_requirement_coverage
            ),

            "total_requirement_coverage": (
                self.total_requirement_coverage
            ),

            "information_coverage": (
                self.information_coverage
            ),

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
                for status
                in self.requirement_statuses
            ],

            "missing_requirements": [
                requirement.role
                for requirement
                in self.missing_requirements
            ],

            "missing_hard_requirements": [
                requirement.role
                for requirement
                in self.missing_hard_requirements
            ],

            "satisfied_requirements": [
                requirement.role
                for requirement
                in self.satisfied_requirements
            ],

            "slot_sufficient": (
                self.slot_sufficient
            ),

            "hard_slot_coverage": (
                self.hard_slot_coverage
            ),

            "total_slot_coverage": (
                self.total_slot_coverage
            ),

            "combined_sufficient_shadow": (
                self.combined_sufficient_shadow
            ),

            "slot_statuses": [
                {
                    "slot_id": status.slot.slot_id,
                    "kind": status.slot.kind,
                    "target": status.slot.target,
                    "hard": status.slot.hard,
                    "distinct": status.slot.distinct,
                    "temporal_role": (
                        status.slot.temporal_role
                    ),
                    "required_count": (
                        status.required_count
                    ),
                    "satisfied_count": (
                        status.satisfied_count
                    ),
                    "coverage": status.coverage,
                    "complete": status.complete,
                    "supporting_memory_ids": list(
                        status.supporting_memory_ids
                    ),
                    "semantic_matching_memory_ids": list(
                        status.semantic_matching_memory_ids
                    ),
                    "structurally_rejected_memory_ids": list(
                        status.structurally_rejected_memory_ids
                    ),
                    "reasons": list(
                        status.reasons
                    ),
                }
                for status
                in self.slot_statuses
            ],

            "missing_slots": [
                slot.slot_id
                for slot
                in self.missing_slots
            ],

            "missing_hard_slots": [
                slot.slot_id
                for slot
                in self.missing_hard_slots
            ],

            "satisfied_slots": [
                slot.slot_id
                for slot
                in self.satisfied_slots
            ],

            "reasons": list(
                self.reasons
            ),
        }


class RequirementSufficiencyEvaluator:
    """Evaluate legacy roles plus semantic-slot sufficiency in shadow.

    M2-C3.3B separates two kinds of slot matching:

    1. semantic + structural slots:
       CONTENT, CURRENT_ENDPOINT, HISTORICAL_ENDPOINT,
       CONTRAST, RESOLUTION;

    2. set/role/temporal-structural slots:
       MULTI_FACET, PROCEDURE, SUPPORT, PROVENANCE,
       TEMPORAL_TRANSITION.

    This avoids forcing every structural obligation through one lexical
    semantic matcher.
    """

    SLOT_TARGET_FRAMING_TOKENS = {
        "explain",
        "explains",
        "explained",
        "explanation",
        "cite",
        "cites",
        "cited",
        "citation",
        "memory",
        "memories",
        "supports",
        "supported",
        "using",
        "used",
        "answer",
        "answers",
        "why",
        "how",
        "what",
        "which",
        "who",
        "when",
        "where",
        "should",
        "would",
        "could",
        "does",
        "do",
        "did",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "my",
        "your",
        "our",
        "their",
        "mainly",
        "msc",
    }

    SUPPORTED_SLOT_KINDS = {
        "CONTENT",
        "CURRENT_ENDPOINT",
        "HISTORICAL_ENDPOINT",
        "TEMPORAL_TRANSITION",
        "PROCEDURE",
        "CONTRAST",
        "RESOLUTION",
        "SUPPORT",
        "PROVENANCE",
        "MULTI_FACET",
    }

    CURRENT_TEMPORAL_ROLES = {
        "current_state",
        "current_endpoint",
    }

    HISTORICAL_TEMPORAL_ROLES = {
        "historical_state",
        "historical_endpoint",
    }

    TRANSITION_TEMPORAL_ROLES = {
        "transition_event",
        "transition",
    }

    RESOLUTION_EVIDENCE_ROLES = {
        "preferred_resolution",
        "answer_target",
        "current_state",
    }

    ANSWER_BEARING_ROLES = {
        "answer_target",
        "current_state",
        "historical_state",
        "preferred_resolution",
        "distinct_item",
    }

    PROVENANCE_RELEVANT_ROLES = {
        "answer_target",
        "current_state",
        "historical_state",
        "preferred_resolution",
        "supporting_evidence",
        "procedural_rule",
        "transition",
    }

    def __init__(
        self,
        config: dict[str, Any],
    ) -> None:
        self.coverage = CoverageEstimator(
            config
        )

        self.slot_matcher = (
            InformationNeedMatcherV3(
                self.coverage
            )
        )

    # =====================================================
    # Public evaluation
    # =====================================================

    def evaluate(
        self,
        *,
        spec: RequirementSpec,
        selected: list[MemoryCandidate],
        selector_status: dict[str, dict[str, Any]],
    ) -> SufficiencyResult:
        """Evaluate legacy roles plus shadow semantic slots."""

        statuses: list[
            RequirementStatus
        ] = []

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
                    (
                        satisfied_count
                        >= required_count
                    ),
                )
            )

            supporting_memory_ids = [
                candidate.memory_id
                for candidate
                in selected
                if requirement.role
                in self._candidate_evidence_roles(
                    candidate
                )
            ]

            statuses.append(
                RequirementStatus(
                    requirement=requirement,
                    required_count=required_count,
                    satisfied_count=satisfied_count,
                    complete=complete,
                    supporting_memory_ids=(
                        supporting_memory_ids
                    ),
                )
            )

        hard_statuses = [
            status
            for status
            in statuses
            if status.requirement.hard
        ]

        if hard_statuses:
            hard_requirement_coverage = (
                sum(
                    status.coverage
                    for status
                    in hard_statuses
                )
                / len(
                    hard_statuses
                )
            )
        else:
            hard_requirement_coverage = (
                1.0
            )

        if statuses:
            total_requirement_coverage = (
                sum(
                    status.coverage
                    for status
                    in statuses
                )
                / len(
                    statuses
                )
            )
        else:
            total_requirement_coverage = (
                1.0
                if selected
                else 0.0
            )

        missing_requirements = [
            status.requirement
            for status
            in statuses
            if not status.complete
        ]

        missing_hard_requirements = [
            status.requirement
            for status
            in statuses
            if (
                status.requirement.hard
                and not status.complete
            )
        ]

        satisfied_requirements = [
            status.requirement
            for status
            in statuses
            if status.complete
        ]

        # Preserve legacy behaviour exactly.
        sufficient = (
            len(
                missing_hard_requirements
            )
            == 0
        )

        information_coverage = (
            self.coverage.compute(
                spec.information_needs,
                selected,
            )
        )

        # -------------------------------------------------
        # Semantic-slot shadow evaluation
        # -------------------------------------------------

        slot_statuses = (
            self._evaluate_slots(
                slots=list(
                    getattr(
                        spec,
                        "slots",
                        [],
                    )
                ),
                selected=selected,
            )
        )

        hard_slot_statuses = [
            status
            for status
            in slot_statuses
            if status.slot.hard
        ]

        if hard_slot_statuses:
            hard_slot_coverage = (
                sum(
                    status.coverage
                    for status
                    in hard_slot_statuses
                )
                / len(
                    hard_slot_statuses
                )
            )
        else:
            hard_slot_coverage = (
                1.0
            )

        if slot_statuses:
            total_slot_coverage = (
                sum(
                    status.coverage
                    for status
                    in slot_statuses
                )
                / len(
                    slot_statuses
                )
            )
        else:
            total_slot_coverage = (
                1.0
            )

        missing_slots = [
            status.slot
            for status
            in slot_statuses
            if not status.complete
        ]

        missing_hard_slots = [
            status.slot
            for status
            in slot_statuses
            if (
                status.slot.hard
                and not status.complete
            )
        ]

        satisfied_slots = [
            status.slot
            for status
            in slot_statuses
            if status.complete
        ]

        slot_sufficient = (
            len(
                missing_hard_slots
            )
            == 0
        )

        combined_sufficient_shadow = (
            sufficient
            and slot_sufficient
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
                    for requirement
                    in missing_hard_requirements
                )
            )

        if slot_statuses:
            if slot_sufficient:
                reasons.append(
                    "all hard semantic slots satisfied in shadow"
                )
            else:
                reasons.append(
                    "one or more hard semantic slots missing in shadow"
                )

            if missing_hard_slots:
                reasons.append(
                    "missing_hard_slots="
                    + ",".join(
                        slot.slot_id
                        for slot
                        in missing_hard_slots
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

            requirement_statuses=(
                statuses
            ),

            missing_requirements=(
                missing_requirements
            ),

            missing_hard_requirements=(
                missing_hard_requirements
            ),

            satisfied_requirements=(
                satisfied_requirements
            ),

            reasons=reasons,

            slot_sufficient=(
                slot_sufficient
            ),

            hard_slot_coverage=round(
                hard_slot_coverage,
                6,
            ),

            total_slot_coverage=round(
                total_slot_coverage,
                6,
            ),

            combined_sufficient_shadow=(
                combined_sufficient_shadow
            ),

            slot_statuses=(
                slot_statuses
            ),

            missing_slots=(
                missing_slots
            ),

            missing_hard_slots=(
                missing_hard_slots
            ),

            satisfied_slots=(
                satisfied_slots
            ),
        )

    # =====================================================
    # Slot evaluation
    # =====================================================

    def _evaluate_slots(
        self,
        *,
        slots: list[RequirementSlot],
        selected: list[MemoryCandidate],
    ) -> list[SlotStatus]:
        return [
            self._evaluate_slot(
                slot=slot,
                selected=selected,
            )
            for slot in slots
        ]

    def _evaluate_slot(
        self,
        *,
        slot: RequirementSlot,
        selected: list[MemoryCandidate],
    ) -> SlotStatus:
        kind = str(
            slot.kind
        ).upper()

        required_count = max(
            1,
            int(
                slot.min_count
            ),
        )

        if (
            kind
            not in self.SUPPORTED_SLOT_KINDS
        ):
            return SlotStatus(
                slot=slot,
                required_count=required_count,
                satisfied_count=0,
                complete=False,
                reasons=[
                    (
                        "unsupported_slot_kind="
                        + kind
                    )
                ],
            )

        # -------------------------------------------------
        # Set-/role-/temporal-structural slots
        # -------------------------------------------------

        if kind == "MULTI_FACET":
            return self._evaluate_multi_facet_slot(
                slot=slot,
                selected=selected,
            )

        if kind == "PROCEDURE":
            return self._evaluate_procedure_slot(
                slot=slot,
                selected=selected,
            )

        if kind == "SUPPORT":
            return self._evaluate_support_slot(
                slot=slot,
                selected=selected,
            )

        if kind == "PROVENANCE":
            return self._evaluate_provenance_slot(
                slot=slot,
                selected=selected,
            )

        if kind == "TEMPORAL_TRANSITION":
            return self._evaluate_transition_slot(
                slot=slot,
                selected=selected,
            )

        # -------------------------------------------------
        # Semantic + structural slots
        # -------------------------------------------------

        semantic_target = (
            self._normalise_slot_target(
                slot.target
            )
        )

        semantic_matching: list[
            MemoryCandidate
        ] = []

        structurally_rejected: list[
            MemoryCandidate
        ] = []

        matched: list[
            MemoryCandidate
        ] = []

        for candidate in selected:
            semantic_match = (
                self._semantic_match(
                    candidate=candidate,
                    target=semantic_target,
                )
            )

            if not semantic_match:
                continue

            semantic_matching.append(
                candidate
            )

            if self._structural_match(
                candidate=candidate,
                slot=slot,
                kind=kind,
            ):
                matched.append(
                    candidate
                )
            else:
                structurally_rejected.append(
                    candidate
                )

        supporting_memory_ids = [
            candidate.memory_id
            for candidate
            in matched
        ]

        satisfied_count = len(
            supporting_memory_ids
        )

        complete = (
            satisfied_count
            >= required_count
        )

        reasons = [
            (
                "matching_mode="
                "semantic_and_structural"
            ),
            (
                "semantic_matches="
                + str(
                    len(
                        semantic_matching
                    )
                )
            ),
            (
                "structural_matches="
                + str(
                    len(
                        matched
                    )
                )
            ),
            (
                "slot_complete"
                if complete
                else "slot_incomplete"
            ),
        ]

        if not semantic_target:
            reasons.insert(
                1,
                "slot_target_has_no_semantic_content",
            )

        return SlotStatus(
            slot=slot,

            required_count=(
                required_count
            ),

            satisfied_count=(
                satisfied_count
            ),

            complete=complete,

            supporting_memory_ids=(
                supporting_memory_ids
            ),

            semantic_matching_memory_ids=[
                candidate.memory_id
                for candidate
                in semantic_matching
            ],

            structurally_rejected_memory_ids=[
                candidate.memory_id
                for candidate
                in structurally_rejected
            ],

            reasons=reasons,
        )

    # =====================================================
    # Structural slot evaluators
    # =====================================================

    def _evaluate_multi_facet_slot(
        self,
        *,
        slot: RequirementSlot,
        selected: list[MemoryCandidate],
    ) -> SlotStatus:
        """Require multiple distinct answer-bearing semantic units."""

        required_count = max(
            1,
            int(
                slot.min_count
            ),
        )

        answer_bearing = [
            candidate
            for candidate
            in selected
            if (
                self._candidate_evidence_roles(
                    candidate
                )
                & self.ANSWER_BEARING_ROLES
            )
        ]

        supporting_ids = (
            self._distinct_supporting_ids(
                answer_bearing
            )
        )

        satisfied_count = len(
            supporting_ids
        )

        complete = (
            satisfied_count
            >= required_count
        )

        return SlotStatus(
            slot=slot,

            required_count=(
                required_count
            ),

            satisfied_count=(
                satisfied_count
            ),

            complete=complete,

            supporting_memory_ids=(
                supporting_ids
            ),

            semantic_matching_memory_ids=[
                candidate.memory_id
                for candidate
                in answer_bearing
            ],

            reasons=[
                "matching_mode=set_structural",
                (
                    "answer_bearing_candidates="
                    + str(
                        len(
                            answer_bearing
                        )
                    )
                ),
                (
                    "distinct_answer_units="
                    + str(
                        satisfied_count
                    )
                ),
                (
                    "slot_complete"
                    if complete
                    else "slot_incomplete"
                ),
            ],
        )

    def _evaluate_procedure_slot(
        self,
        *,
        slot: RequirementSlot,
        selected: list[MemoryCandidate],
    ) -> SlotStatus:
        """Require at least one applicable procedural/rule candidate."""

        matched = [
            candidate
            for candidate
            in selected
            if (
                candidate.memory_type
                == MemoryType.PROCEDURAL
                or (
                    "procedural_rule"
                    in self._candidate_evidence_roles(
                        candidate
                    )
                )
                or (
                    self._candidate_temporal_role(
                        candidate
                    )
                    == "procedural_rule"
                )
            )
        ]

        required_count = max(
            1,
            int(
                slot.min_count
            ),
        )

        satisfied_count = len(
            matched
        )

        complete = (
            satisfied_count
            >= required_count
        )

        return SlotStatus(
            slot=slot,

            required_count=(
                required_count
            ),

            satisfied_count=(
                satisfied_count
            ),

            complete=complete,

            supporting_memory_ids=[
                candidate.memory_id
                for candidate
                in matched
            ],

            semantic_matching_memory_ids=[
                candidate.memory_id
                for candidate
                in matched
            ],

            reasons=[
                "matching_mode=role_structural",
                (
                    "procedural_candidates="
                    + str(
                        len(
                            matched
                        )
                    )
                ),
                (
                    "slot_complete"
                    if complete
                    else "slot_incomplete"
                ),
            ],
        )

    def _evaluate_support_slot(
        self,
        *,
        slot: RequirementSlot,
        selected: list[MemoryCandidate],
    ) -> SlotStatus:
        """Require explicit supporting evidence selected for the query."""

        matched = [
            candidate
            for candidate
            in selected
            if (
                "supporting_evidence"
                in self._candidate_evidence_roles(
                    candidate
                )
            )
        ]

        required_count = max(
            1,
            int(
                slot.min_count
            ),
        )

        satisfied_count = len(
            matched
        )

        complete = (
            satisfied_count
            >= required_count
        )

        return SlotStatus(
            slot=slot,

            required_count=(
                required_count
            ),

            satisfied_count=(
                satisfied_count
            ),

            complete=complete,

            supporting_memory_ids=[
                candidate.memory_id
                for candidate
                in matched
            ],

            semantic_matching_memory_ids=[
                candidate.memory_id
                for candidate
                in matched
            ],

            reasons=[
                "matching_mode=role_structural",
                (
                    "supporting_candidates="
                    + str(
                        len(
                            matched
                        )
                    )
                ),
                (
                    "slot_complete"
                    if complete
                    else "slot_incomplete"
                ),
            ],
        )

    def _evaluate_provenance_slot(
        self,
        *,
        slot: RequirementSlot,
        selected: list[MemoryCandidate],
    ) -> SlotStatus:
        """Require typed provenance for relevant selected evidence."""

        target = self._normalise_slot_target(
            slot.target
        )

        relevant: list[
            MemoryCandidate
        ] = []

        for candidate in selected:
            roles = (
                self._candidate_evidence_roles(
                    candidate
                )
            )

            role_relevant = bool(
                roles
                & self.PROVENANCE_RELEVANT_ROLES
            )

            semantic_relevant = (
                self._semantic_match(
                    candidate=candidate,
                    target=target,
                )
                if target
                else False
            )

            if (
                role_relevant
                or semantic_relevant
            ):
                relevant.append(
                    candidate
                )

        matched = [
            candidate
            for candidate
            in relevant
            if isinstance(
                candidate.memory_type,
                MemoryType,
            )
        ]

        required_count = max(
            1,
            int(
                slot.min_count
            ),
        )

        satisfied_count = len(
            matched
        )

        complete = (
            satisfied_count
            >= required_count
        )

        return SlotStatus(
            slot=slot,

            required_count=(
                required_count
            ),

            satisfied_count=(
                satisfied_count
            ),

            complete=complete,

            supporting_memory_ids=[
                candidate.memory_id
                for candidate
                in matched
            ],

            semantic_matching_memory_ids=[
                candidate.memory_id
                for candidate
                in relevant
            ],

            reasons=[
                "matching_mode=provenance_structural",
                (
                    "relevant_candidates="
                    + str(
                        len(
                            relevant
                        )
                    )
                ),
                (
                    "typed_provenance_candidates="
                    + str(
                        len(
                            matched
                        )
                    )
                ),
                (
                    "slot_complete"
                    if complete
                    else "slot_incomplete"
                ),
            ],
        )

    def _evaluate_transition_slot(
        self,
        *,
        slot: RequirementSlot,
        selected: list[MemoryCandidate],
    ) -> SlotStatus:
        """Require explicit query-conditioned transition evidence.

        A transition is primarily a temporal structural relation. A valid
        transition candidate should not be rejected merely because its text
        does not repeat the lexical name of the queried state variable.

        The candidate pool is already query-conditioned by the retrieval /
        ranking / conflict-resolution pipeline, so explicit transition
        metadata is the primary criterion here.
        """

        matched = [
            candidate
            for candidate
            in selected
            if (
                self._candidate_temporal_role(
                    candidate
                )
                in self.TRANSITION_TEMPORAL_ROLES
                or (
                    "transition"
                    in self._candidate_evidence_roles(
                        candidate
                    )
                )
            )
        ]

        required_count = max(
            1,
            int(
                slot.min_count
            ),
        )

        satisfied_count = len(
            matched
        )

        complete = (
            satisfied_count
            >= required_count
        )

        return SlotStatus(
            slot=slot,

            required_count=(
                required_count
            ),

            satisfied_count=(
                satisfied_count
            ),

            complete=complete,

            supporting_memory_ids=[
                candidate.memory_id
                for candidate
                in matched
            ],

            semantic_matching_memory_ids=[
                candidate.memory_id
                for candidate
                in matched
            ],

            reasons=[
                "matching_mode=temporal_structural",
                (
                    "transition_candidates="
                    + str(
                        len(
                            matched
                        )
                    )
                ),
                (
                    "slot_complete"
                    if complete
                    else "slot_incomplete"
                ),
            ],
        )

    # =====================================================
    # Semantic + structural matching primitives
    # =====================================================

    def _semantic_match(
        self,
        *,
        candidate: MemoryCandidate,
        target: str,
    ) -> bool:
        if not target:
            return False

        covered = (
            self.slot_matcher
            .candidate_coverage(
                candidate,
                [target],
            )
        )

        return 0 in covered

    def _structural_match(
        self,
        *,
        candidate: MemoryCandidate,
        slot: RequirementSlot,
        kind: str,
    ) -> bool:
        temporal_role = (
            self._candidate_temporal_role(
                candidate
            )
        )

        evidence_roles = (
            self._candidate_evidence_roles(
                candidate
            )
        )

        if kind == "CONTENT":
            return True

        if kind == "CONTRAST":
            return True

        if kind == "CURRENT_ENDPOINT":
            return (
                temporal_role
                in self.CURRENT_TEMPORAL_ROLES
                or (
                    "current_state"
                    in evidence_roles
                )
            )

        if kind == "HISTORICAL_ENDPOINT":
            return (
                temporal_role
                in self.HISTORICAL_TEMPORAL_ROLES
                or (
                    "historical_state"
                    in evidence_roles
                )
            )

        if kind == "RESOLUTION":
            return bool(
                evidence_roles
                & self.RESOLUTION_EVIDENCE_ROLES
            )

        return False

    # =====================================================
    # Candidate helpers
    # =====================================================

    @staticmethod
    def _candidate_evidence_roles(
        candidate: MemoryCandidate,
    ) -> set[str]:
        raw = candidate.metadata.get(
            "evidence_roles",
            [],
        )

        if isinstance(
            raw,
            str,
        ):
            return {
                raw
            }

        if isinstance(
            raw,
            (
                list,
                tuple,
                set,
            ),
        ):
            return {
                str(item)
                for item
                in raw
            }

        return set()

    @staticmethod
    def _candidate_temporal_role(
        candidate: MemoryCandidate,
    ) -> str:
        return str(
            candidate.metadata.get(
                "query_relative_temporal_role",
                "",
            )
            or ""
        )

    @classmethod
    def _normalise_slot_target(
        cls,
        target: str,
    ) -> str:
        """Remove residual query framing while preserving domain semantics."""

        text = " ".join(
            str(
                target
            )
            .lower()
            .split()
        )

        tokens = re.findall(
            r"[\w'-]+",
            text,
            flags=re.UNICODE,
        )

        filtered = [
            token
            for token
            in tokens
            if token
            not in cls.SLOT_TARGET_FRAMING_TOKENS
        ]

        return " ".join(
            filtered
        )

    @staticmethod
    def _candidate_semantic_fingerprint(
        candidate: MemoryCandidate,
    ) -> str:
        """Deterministic fingerprint for set-level distinctness."""

        values = [
            candidate.subject or "",
            candidate.predicate or "",
            candidate.object_value or "",
            candidate.text or "",
        ]

        text = " ".join(
            value
            for value
            in values
            if value
        ).lower()

        tokens = re.findall(
            r"[\w'-]+",
            text,
            flags=re.UNICODE,
        )

        return " ".join(
            tokens
        )

    def _distinct_supporting_ids(
        self,
        candidates: list[MemoryCandidate],
    ) -> list[str]:
        seen: set[str] = set()

        output: list[str] = []

        for candidate in candidates:
            fingerprint = (
                self._candidate_semantic_fingerprint(
                    candidate
                )
            )

            key = (
                fingerprint
                or candidate.memory_id
            )

            if key in seen:
                continue

            seen.add(
                key
            )

            output.append(
                candidate.memory_id
            )

        return output