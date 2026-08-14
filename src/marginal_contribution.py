from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .requirement_spec import RequirementSpec
from .requirement_sufficiency import (
    RequirementSufficiencyEvaluator,
    SufficiencyResult,
)
from .schemas import MemoryCandidate
from .slot_fidelity import (
    TemporalPathState,
    TransitionSlotFidelityEvaluator,
)


@dataclass(frozen=True, slots=True)
class MarginalEvidenceContribution:
    """Leave-one-out structural contribution of one selected memory."""

    memory_id: str

    criticality_class: str

    # Direct slot loss observed after leave-one-out.
    hard_slots_lost: tuple[str, ...] = ()

    # Slot losses that remain unresolved after considering
    # valid alternative information paths.
    effective_hard_slots_lost: tuple[str, ...] = ()

    # Direct hard-slot losses whose obligation remains
    # recoverable through an alternative evidence path.
    path_substituted_hard_slots: tuple[str, ...] = ()

    hard_slots_gained: tuple[str, ...] = ()

    support_count_drop_slots: tuple[str, ...] = ()

    hard_slot_coverage_before: float = 1.0
    hard_slot_coverage_after: float = 1.0
    hard_slot_coverage_delta: float = 0.0

    slot_sufficient_before: bool = True
    slot_sufficient_after: bool = True

    temporal_path_applicable: bool = False
    endpoint_path_before: bool = False
    endpoint_path_after: bool = False
    endpoint_path_lost: bool = False

    transition_support_before: bool = False
    transition_support_after: bool = False
    transition_support_lost: bool = False

    support_count_before: int = 0
    support_count_after: int = 0

    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "criticality_class": self.criticality_class,

            "hard_slots_lost": list(
                self.hard_slots_lost
            ),
            "effective_hard_slots_lost": list(
                self.effective_hard_slots_lost
            ),
            "path_substituted_hard_slots": list(
                self.path_substituted_hard_slots
            ),
            "hard_slots_gained": list(
                self.hard_slots_gained
            ),
            "support_count_drop_slots": list(
                self.support_count_drop_slots
            ),

            "hard_slot_coverage_before": (
                self.hard_slot_coverage_before
            ),
            "hard_slot_coverage_after": (
                self.hard_slot_coverage_after
            ),
            "hard_slot_coverage_delta": (
                self.hard_slot_coverage_delta
            ),

            "slot_sufficient_before": (
                self.slot_sufficient_before
            ),
            "slot_sufficient_after": (
                self.slot_sufficient_after
            ),

            "temporal_path_applicable": (
                self.temporal_path_applicable
            ),
            "endpoint_path_before": (
                self.endpoint_path_before
            ),
            "endpoint_path_after": (
                self.endpoint_path_after
            ),
            "endpoint_path_lost": (
                self.endpoint_path_lost
            ),

            "transition_support_before": (
                self.transition_support_before
            ),
            "transition_support_after": (
                self.transition_support_after
            ),
            "transition_support_lost": (
                self.transition_support_lost
            ),

            "support_count_before": (
                self.support_count_before
            ),
            "support_count_after": (
                self.support_count_after
            ),

            "reasons": list(
                self.reasons
            ),
        }


@dataclass(frozen=True, slots=True)
class MarginalContributionAudit:
    """Leave-one-out audit for one selected evidence set."""

    available: bool
    reason: str

    selected_ids: tuple[str, ...] = ()

    contributions: tuple[
        MarginalEvidenceContribution,
        ...
    ] = ()

    critical_ids: tuple[str, ...] = ()
    contributory_ids: tuple[str, ...] = ()
    redundant_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "active_for_generation": False,
            "reason": self.reason,

            "selected_ids": list(
                self.selected_ids
            ),

            "contributions": [
                item.to_dict()
                for item
                in self.contributions
            ],

            "critical_ids": list(
                self.critical_ids
            ),
            "contributory_ids": list(
                self.contributory_ids
            ),
            "redundant_ids": list(
                self.redundant_ids
            ),
        }


@dataclass(frozen=True, slots=True)
class AddBackRestorationContribution:
    """Structural gain from adding one pruned memory back."""

    memory_id: str
    restoration_class: str

    hard_slots_restored: tuple[str, ...] = ()
    effective_hard_slots_restored: tuple[str, ...] = ()

    # Direct TEMPORAL_TRANSITION restoration that is unnecessary
    # because endpoint reconstruction was already available.
    path_redundant_restored_slots: tuple[str, ...] = ()

    support_count_gain_slots: tuple[str, ...] = ()

    hard_slot_coverage_before: float = 1.0
    hard_slot_coverage_after: float = 1.0
    hard_slot_coverage_delta: float = 0.0

    slot_sufficient_before: bool = True
    slot_sufficient_after: bool = True

    temporal_path_applicable: bool = False

    endpoint_path_before: bool = False
    endpoint_path_after: bool = False
    endpoint_path_gained: bool = False

    transition_support_before: bool = False
    transition_support_after: bool = False
    transition_support_gained: bool = False

    support_count_before: int = 0
    support_count_after: int = 0

    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "restoration_class": self.restoration_class,

            "hard_slots_restored": list(
                self.hard_slots_restored
            ),
            "effective_hard_slots_restored": list(
                self.effective_hard_slots_restored
            ),
            "path_redundant_restored_slots": list(
                self.path_redundant_restored_slots
            ),
            "support_count_gain_slots": list(
                self.support_count_gain_slots
            ),

            "hard_slot_coverage_before": (
                self.hard_slot_coverage_before
            ),
            "hard_slot_coverage_after": (
                self.hard_slot_coverage_after
            ),
            "hard_slot_coverage_delta": (
                self.hard_slot_coverage_delta
            ),

            "slot_sufficient_before": (
                self.slot_sufficient_before
            ),
            "slot_sufficient_after": (
                self.slot_sufficient_after
            ),

            "temporal_path_applicable": (
                self.temporal_path_applicable
            ),

            "endpoint_path_before": (
                self.endpoint_path_before
            ),
            "endpoint_path_after": (
                self.endpoint_path_after
            ),
            "endpoint_path_gained": (
                self.endpoint_path_gained
            ),

            "transition_support_before": (
                self.transition_support_before
            ),
            "transition_support_after": (
                self.transition_support_after
            ),
            "transition_support_gained": (
                self.transition_support_gained
            ),

            "support_count_before": (
                self.support_count_before
            ),
            "support_count_after": (
                self.support_count_after
            ),

            "reasons": list(
                self.reasons
            ),
        }


@dataclass(frozen=True, slots=True)
class AddBackRestorationAudit:
    """Paired add-back audit for memories pruned relative to Legacy."""

    available: bool
    reason: str

    base_selected_ids: tuple[str, ...] = ()
    candidate_ids: tuple[str, ...] = ()

    contributions: tuple[
        AddBackRestorationContribution,
        ...
    ] = ()

    restorative_ids: tuple[str, ...] = ()
    supportive_ids: tuple[str, ...] = ()
    no_restoration_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "active_for_generation": False,
            "reason": self.reason,

            "base_selected_ids": list(
                self.base_selected_ids
            ),
            "candidate_ids": list(
                self.candidate_ids
            ),

            "contributions": [
                item.to_dict()
                for item
                in self.contributions
            ],

            "restorative_ids": list(
                self.restorative_ids
            ),
            "supportive_ids": list(
                self.supportive_ids
            ),
            "no_restoration_ids": list(
                self.no_restoration_ids
            ),
        }


class MarginalContributionEvaluator:
    """Threshold-free leave-one-out requirement contribution audit.

    This component is descriptive only. It does not modify arbitration,
    repair, confidence, prompt construction or generation.

    A memory is:

      critical:
        removing it destroys at least one previously complete hard slot,
        or destroys an available endpoint reconstruction path;

      contributory:
        no hard requirement/path is destroyed, but removal reduces
        supporting evidence for at least one slot or explicit transition;

      redundant:
        removal changes neither required slot state nor structural support.

    No learned weights or fitted thresholds are used.
    """

    def __init__(
        self,
        *,
        sufficiency: RequirementSufficiencyEvaluator,
        transition_fidelity: TransitionSlotFidelityEvaluator,
    ) -> None:
        self.sufficiency = sufficiency
        self.transition_fidelity = transition_fidelity

    def evaluate(
        self,
        *,
        spec: RequirementSpec,
        selected: list[MemoryCandidate],
    ) -> MarginalContributionAudit:

        if not selected:
            return MarginalContributionAudit(
                available=False,
                reason="selected_evidence_empty",
            )

        full_result = self.sufficiency.evaluate(
            spec=spec,
            selected=selected,
            selector_status={},
        )

        # RequirementSpec is the authoritative source of slot hardness.
        # Do not depend on optional serialization fields in SlotStatus.
        hard_slot_ids = {
            slot.slot_id
            for slot in spec.slots
            if slot.hard
        }

        full_dict = full_result.to_dict()

        full_path = self._temporal_path(
            selected=selected,
            sufficiency=full_result,
        )

        contributions: list[
            MarginalEvidenceContribution
        ] = []

        for removed in selected:
            reduced = [
                candidate
                for candidate in selected
                if (
                    candidate.memory_id
                    != removed.memory_id
                )
            ]

            reduced_result = (
                self.sufficiency.evaluate(
                    spec=spec,
                    selected=reduced,
                    selector_status={},
                )
            )

            reduced_dict = (
                reduced_result.to_dict()
            )

            reduced_path = self._temporal_path(
                selected=reduced,
                sufficiency=reduced_result,
            )

            contribution = (
                self._compare(
                    memory_id=removed.memory_id,
                    before=full_result,
                    before_dict=full_dict,
                    before_path=full_path,
                    after=reduced_result,
                    after_dict=reduced_dict,
                    after_path=reduced_path,
                    hard_slot_ids=hard_slot_ids,
                )
            )

            contributions.append(
                contribution
            )

        critical_ids = tuple(
            item.memory_id
            for item in contributions
            if (
                item.criticality_class
                == "critical"
            )
        )

        contributory_ids = tuple(
            item.memory_id
            for item in contributions
            if (
                item.criticality_class
                == "contributory"
            )
        )

        redundant_ids = tuple(
            item.memory_id
            for item in contributions
            if (
                item.criticality_class
                == "redundant"
            )
        )

        return MarginalContributionAudit(
            available=True,
            reason="leave_one_out_complete",

            selected_ids=tuple(
                candidate.memory_id
                for candidate in selected
            ),

            contributions=tuple(
                contributions
            ),

            critical_ids=critical_ids,
            contributory_ids=contributory_ids,
            redundant_ids=redundant_ids,
        )

    def evaluate_add_back(
        self,
        *,
        spec: RequirementSpec,
        selected: list[MemoryCandidate],
        candidates: list[MemoryCandidate],
    ) -> AddBackRestorationAudit:
        """Evaluate structural restoration from adding pruned evidence back."""

        selected_ids = {
            candidate.memory_id
            for candidate in selected
        }

        unique_candidates: list[MemoryCandidate] = []
        seen: set[str] = set()

        for candidate in candidates:
            if candidate.memory_id in selected_ids:
                continue

            if candidate.memory_id in seen:
                continue

            seen.add(
                candidate.memory_id
            )

            unique_candidates.append(
                candidate
            )

        if not unique_candidates:
            return AddBackRestorationAudit(
                available=True,
                reason="no_add_back_candidates",

                base_selected_ids=tuple(
                    candidate.memory_id
                    for candidate in selected
                ),
            )

        before = self.sufficiency.evaluate(
            spec=spec,
            selected=selected,
            selector_status={},
        )

        before_dict = before.to_dict()

        before_path = self._temporal_path(
            selected=selected,
            sufficiency=before,
        )

        hard_slot_ids = {
            slot.slot_id
            for slot in spec.slots
            if slot.hard
        }

        contributions: list[
            AddBackRestorationContribution
        ] = []

        for candidate in unique_candidates:
            augmented = [
                *selected,
                candidate,
            ]

            after = self.sufficiency.evaluate(
                spec=spec,
                selected=augmented,
                selector_status={},
            )

            after_dict = after.to_dict()

            after_path = self._temporal_path(
                selected=augmented,
                sufficiency=after,
            )

            contributions.append(
                self._compare_add_back(
                    memory_id=candidate.memory_id,

                    before=before,
                    before_dict=before_dict,
                    before_path=before_path,

                    after=after,
                    after_dict=after_dict,
                    after_path=after_path,

                    hard_slot_ids=hard_slot_ids,
                )
            )

        restorative_ids = tuple(
            item.memory_id
            for item in contributions
            if (
                item.restoration_class
                == "restorative"
            )
        )

        supportive_ids = tuple(
            item.memory_id
            for item in contributions
            if (
                item.restoration_class
                == "supportive"
            )
        )

        no_restoration_ids = tuple(
            item.memory_id
            for item in contributions
            if (
                item.restoration_class
                == "no_restoration"
            )
        )

        return AddBackRestorationAudit(
            available=True,
            reason="add_back_complete",

            base_selected_ids=tuple(
                candidate.memory_id
                for candidate in selected
            ),

            candidate_ids=tuple(
                candidate.memory_id
                for candidate
                in unique_candidates
            ),

            contributions=tuple(
                contributions
            ),

            restorative_ids=restorative_ids,
            supportive_ids=supportive_ids,
            no_restoration_ids=no_restoration_ids,
        )

    def _compare_add_back(
        self,
        *,
        memory_id: str,

        before: SufficiencyResult,
        before_dict: dict[str, Any],
        before_path: TemporalPathState,

        after: SufficiencyResult,
        after_dict: dict[str, Any],
        after_path: TemporalPathState,

        hard_slot_ids: set[str],
    ) -> AddBackRestorationContribution:

        before_hard = self._hard_slot_map(
            before_dict,
            hard_slot_ids,
        )

        after_hard = self._hard_slot_map(
            after_dict,
            hard_slot_ids,
        )

        all_hard_ids = sorted(
            set(before_hard)
            | set(after_hard)
        )

        hard_slots_restored: list[str] = []

        for slot_id in all_hard_ids:
            before_status = before_hard.get(
                slot_id,
                {},
            )

            after_status = after_hard.get(
                slot_id,
                {},
            )

            if (
                not bool(
                    before_status.get(
                        "complete",
                        False,
                    )
                )
                and bool(
                    after_status.get(
                        "complete",
                        False,
                    )
                )
            ):
                hard_slots_restored.append(
                    slot_id
                )

        # -------------------------------------------------
        # Path-aware restoration semantics
        #
        # Restoring an explicit transition slot is not an
        # effective requirement restoration if the endpoint
        # reconstruction path was already available.
        # -------------------------------------------------

        path_redundant_restored_slots: list[str] = []

        for slot_id in hard_slots_restored:
            after_status = after_hard.get(
                slot_id,
                {},
            )

            kind = str(
                after_status.get(
                    "kind",
                    "",
                )
            ).upper()

            if (
                kind == "TEMPORAL_TRANSITION"
                and (
                    before_path
                    .endpoint_reconstruction_available
                )
                and (
                    after_path
                    .endpoint_reconstruction_available
                )
            ):
                path_redundant_restored_slots.append(
                    slot_id
                )

        redundant_restoration_set = set(
            path_redundant_restored_slots
        )

        effective_hard_slots_restored = [
            slot_id
            for slot_id in hard_slots_restored
            if (
                slot_id
                not in redundant_restoration_set
            )
        ]

        before_all = {
            str(status.get("slot_id", "")): status
            for status
            in before_dict.get(
                "slot_statuses",
                [],
            )
        }

        after_all = {
            str(status.get("slot_id", "")): status
            for status
            in after_dict.get(
                "slot_statuses",
                [],
            )
        }

        support_count_gain_slots: list[str] = []

        for slot_id in sorted(
            set(before_all)
            | set(after_all)
        ):
            before_count = int(
                before_all.get(
                    slot_id,
                    {},
                ).get(
                    "satisfied_count",
                    0,
                )
            )

            after_count = int(
                after_all.get(
                    slot_id,
                    {},
                ).get(
                    "satisfied_count",
                    0,
                )
            )

            if after_count > before_count:
                support_count_gain_slots.append(
                    slot_id
                )

        endpoint_path_gained = (
            after_path.applicable
            and not (
                before_path
                .endpoint_reconstruction_available
            )
            and (
                after_path
                .endpoint_reconstruction_available
            )
        )

        transition_support_gained = (
            after_path.applicable
            and not (
                before_path
                .explicit_transition_support_available
            )
            and (
                after_path
                .explicit_transition_support_available
            )
        )

        support_count_before = (
            self._total_support_count(
                before_dict
            )
        )

        support_count_after = (
            self._total_support_count(
                after_dict
            )
        )

        reasons: list[str] = []

        if effective_hard_slots_restored:
            reasons.append(
                "hard_slot_restored"
            )

        if path_redundant_restored_slots:
            reasons.append(
                "direct_transition_restoration_path_redundant"
            )

        if endpoint_path_gained:
            reasons.append(
                "endpoint_reconstruction_restored"
            )

        if support_count_gain_slots:
            reasons.append(
                "slot_support_increased"
            )

        if transition_support_gained:
            reasons.append(
                "explicit_transition_support_added"
            )

        if (
            effective_hard_slots_restored
            or endpoint_path_gained
        ):
            restoration_class = (
                "restorative"
            )

        elif (
            support_count_gain_slots
            or transition_support_gained
            or path_redundant_restored_slots
        ):
            restoration_class = (
                "supportive"
            )

        else:
            restoration_class = (
                "no_restoration"
            )

            reasons.append(
                "no_requirement_or_path_gain"
            )

        return AddBackRestorationContribution(
            memory_id=memory_id,

            restoration_class=(
                restoration_class
            ),

            hard_slots_restored=tuple(
                hard_slots_restored
            ),

            effective_hard_slots_restored=tuple(
                effective_hard_slots_restored
            ),

            path_redundant_restored_slots=tuple(
                path_redundant_restored_slots
            ),

            support_count_gain_slots=tuple(
                support_count_gain_slots
            ),

            hard_slot_coverage_before=(
                before.hard_slot_coverage
            ),

            hard_slot_coverage_after=(
                after.hard_slot_coverage
            ),

            hard_slot_coverage_delta=round(
                (
                    after.hard_slot_coverage
                    - before.hard_slot_coverage
                ),
                6,
            ),

            slot_sufficient_before=(
                before.slot_sufficient
            ),

            slot_sufficient_after=(
                after.slot_sufficient
            ),

            temporal_path_applicable=(
                after_path.applicable
            ),

            endpoint_path_before=(
                before_path
                .endpoint_reconstruction_available
            ),

            endpoint_path_after=(
                after_path
                .endpoint_reconstruction_available
            ),

            endpoint_path_gained=(
                endpoint_path_gained
            ),

            transition_support_before=(
                before_path
                .explicit_transition_support_available
            ),

            transition_support_after=(
                after_path
                .explicit_transition_support_available
            ),

            transition_support_gained=(
                transition_support_gained
            ),

            support_count_before=(
                support_count_before
            ),

            support_count_after=(
                support_count_after
            ),

            reasons=tuple(
                reasons
            ),
        )

    def _temporal_path(
        self,
        *,
        selected: list[MemoryCandidate],
        sufficiency: SufficiencyResult,
    ) -> TemporalPathState:

        slot_dict = sufficiency.to_dict()

        fidelity = (
            self.transition_fidelity.evaluate_set(
                selected=selected,
                slot_statuses=list(
                    slot_dict[
                        "slot_statuses"
                    ]
                ),
            )
        )

        return (
            self.transition_fidelity
            ._temporal_path_state(
                fidelity
            )
        )

    @staticmethod
    def _hard_slot_map(
        result_dict: dict[str, Any],
        hard_slot_ids: set[str],
    ) -> dict[str, dict[str, Any]]:

        return {
            str(status["slot_id"]): status
            for status
            in result_dict.get(
                "slot_statuses",
                [],
            )
            if str(
                status.get(
                    "slot_id",
                    "",
                )
            ) in hard_slot_ids
        }

    @staticmethod
    def _total_support_count(
        result_dict: dict[str, Any],
    ) -> int:

        support_ids: set[str] = set()

        for status in result_dict.get(
            "slot_statuses",
            [],
        ):
            support_ids.update(
                str(memory_id)
                for memory_id
                in status.get(
                    "supporting_memory_ids",
                    [],
                )
            )

        return len(
            support_ids
        )

    def _compare(
        self,
        *,
        memory_id: str,

        before: SufficiencyResult,
        before_dict: dict[str, Any],
        before_path: TemporalPathState,

        after: SufficiencyResult,
        after_dict: dict[str, Any],
        after_path: TemporalPathState,
        hard_slot_ids: set[str],
    ) -> MarginalEvidenceContribution:

        before_slots = self._hard_slot_map(
            before_dict,
            hard_slot_ids,
        )

        after_slots = self._hard_slot_map(
            after_dict,
            hard_slot_ids,
        )

        all_slot_ids = sorted(
            set(
                before_slots
            )
            | set(
                after_slots
            )
        )

        hard_slots_lost: list[str] = []
        hard_slots_gained: list[str] = []
        support_count_drop_slots: list[str] = []

        for slot_id in all_slot_ids:
            before_status = (
                before_slots.get(
                    slot_id,
                    {},
                )
            )

            after_status = (
                after_slots.get(
                    slot_id,
                    {},
                )
            )

            before_complete = bool(
                before_status.get(
                    "complete",
                    False,
                )
            )

            after_complete = bool(
                after_status.get(
                    "complete",
                    False,
                )
            )

            if (
                before_complete
                and not after_complete
            ):
                hard_slots_lost.append(
                    slot_id
                )

            if (
                not before_complete
                and after_complete
            ):
                hard_slots_gained.append(
                    slot_id
                )

            before_count = int(
                before_status.get(
                    "satisfied_count",
                    0,
                )
            )

            after_count = int(
                after_status.get(
                    "satisfied_count",
                    0,
                )
            )

            if after_count < before_count:
                support_count_drop_slots.append(
                    slot_id
                )

        # -------------------------------------------------
        # Requirement-path substitution
        #
        # Losing direct TEMPORAL_TRANSITION support does not
        # imply effective temporal information loss when the
        # historical/current endpoint reconstruction path
        # remains available.
        # -------------------------------------------------

        path_substituted_hard_slots: list[str] = []

        for slot_id in hard_slots_lost:
            before_status = before_slots.get(
                slot_id,
                {},
            )

            slot_kind = str(
                before_status.get(
                    "kind",
                    "",
                )
            ).upper()

            if (
                slot_kind == "TEMPORAL_TRANSITION"
                and (
                    before_path
                    .endpoint_reconstruction_available
                )
                and (
                    after_path
                    .endpoint_reconstruction_available
                )
            ):
                path_substituted_hard_slots.append(
                    slot_id
                )

        substituted_set = set(
            path_substituted_hard_slots
        )

        effective_hard_slots_lost = [
            slot_id
            for slot_id in hard_slots_lost
            if slot_id not in substituted_set
        ]

        endpoint_path_lost = (
            before_path.applicable
            and (
                before_path
                .endpoint_reconstruction_available
            )
            and not (
                after_path
                .endpoint_reconstruction_available
            )
        )

        transition_support_lost = (
            before_path.applicable
            and (
                before_path
                .explicit_transition_support_available
            )
            and not (
                after_path
                .explicit_transition_support_available
            )
        )

        support_count_before = (
            self._total_support_count(
                before_dict
            )
        )

        support_count_after = (
            self._total_support_count(
                after_dict
            )
        )

        reasons: list[str] = []

        if effective_hard_slots_lost:
            reasons.append(
                "hard_slot_destroyed"
            )

        if path_substituted_hard_slots:
            reasons.append(
                "hard_slot_loss_substituted_by_endpoint_path"
            )

        if endpoint_path_lost:
            reasons.append(
                "endpoint_reconstruction_destroyed"
            )

        if support_count_drop_slots:
            reasons.append(
                "slot_support_reduced"
            )

        if transition_support_lost:
            reasons.append(
                "explicit_transition_support_removed"
            )

        if (
            effective_hard_slots_lost
            or endpoint_path_lost
        ):
            criticality_class = (
                "critical"
            )

        elif (
            support_count_drop_slots
            or transition_support_lost
            or path_substituted_hard_slots
        ):
            criticality_class = (
                "contributory"
            )

        else:
            criticality_class = (
                "redundant"
            )

            reasons.append(
                "no_requirement_or_path_state_changed"
            )

        return MarginalEvidenceContribution(
            memory_id=memory_id,

            criticality_class=(
                criticality_class
            ),

            hard_slots_lost=tuple(
                hard_slots_lost
            ),
            effective_hard_slots_lost=tuple(
                effective_hard_slots_lost
            ),
            path_substituted_hard_slots=tuple(
                path_substituted_hard_slots
            ),
            hard_slots_gained=tuple(
                hard_slots_gained
            ),

            support_count_drop_slots=tuple(
                support_count_drop_slots
            ),

            hard_slot_coverage_before=(
                before.hard_slot_coverage
            ),
            hard_slot_coverage_after=(
                after.hard_slot_coverage
            ),
            hard_slot_coverage_delta=round(
                (
                    after.hard_slot_coverage
                    - before.hard_slot_coverage
                ),
                6,
            ),

            slot_sufficient_before=(
                before.slot_sufficient
            ),
            slot_sufficient_after=(
                after.slot_sufficient
            ),

            temporal_path_applicable=(
                before_path.applicable
            ),
            endpoint_path_before=(
                before_path
                .endpoint_reconstruction_available
            ),
            endpoint_path_after=(
                after_path
                .endpoint_reconstruction_available
            ),
            endpoint_path_lost=(
                endpoint_path_lost
            ),

            transition_support_before=(
                before_path
                .explicit_transition_support_available
            ),
            transition_support_after=(
                after_path
                .explicit_transition_support_available
            ),
            transition_support_lost=(
                transition_support_lost
            ),

            support_count_before=(
                support_count_before
            ),
            support_count_after=(
                support_count_after
            ),

            reasons=tuple(
                reasons
            ),
        )
