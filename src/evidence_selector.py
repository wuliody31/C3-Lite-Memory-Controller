from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any

from .coverage_estimator import CoverageEstimator
from .evidence_requirements import (
    EvidencePlan,
    EvidenceRequirement,
    EvidenceRequirementPlanner,
)
from .schemas import (
    ConflictGroup,
    MemoryCandidate,
    MemoryType,
    QueryFeatures,
    QueryMode,
    RouteDecision,
)
from .text_utils import (
    approximate_token_count,
    jaccard_similarity,
    overlap_ratio,
    tokenize,
)


class EvidenceSelector:
    """RC8 structured requirement-aware evidence selector.

    Selection has two phases:

    1. satisfy query-conditioned evidence roles such as current state,
       historical states, transition evidence, procedural rules, explicit
       cardinality and explanation support;
    2. use coverage-aware MMR only for useful remaining capacity.

    The selector is deterministic and writes its decisions into candidate
    metadata for auditability.
    """

    HISTORICAL_STATUSES = {
        "outdated",
        "superseded",
        "archived",
        "invalid",
        "historical",
    }
    CURRENT_STATUSES = {
        "current",
        "active",
        "valid",
    }
    HISTORICAL_ACTIONS = {
        "historical",
        "preferred_historical",
    }
    CURRENT_ACTIONS = {
        "current_endpoint",
        "preferred",
    }

    HISTORICAL_SIGNALS = {
        "earlier",
        "previous",
        "previously",
        "initial",
        "initially",
        "original",
        "originally",
        "past",
        "before",
        "backup",
        "stretch goal",
        "considered",
        "曾经",
        "以前",
        "之前",
        "最初",
        "历史",
        "备用",
    }
    CURRENT_SIGNALS = {
        "current",
        "currently",
        "now",
        "latest",
        "present",
        "main target",
        "core method",
        "current focus",
        "should emphasise",
        "should prioritize",
        "should prioritise",
        "当前",
        "现在",
        "目前",
        "核心",
        "主要",
    }
    TRANSITION_SIGNALS = {
        "changed",
        "change",
        "shifted",
        "switched",
        "narrowed",
        "updated",
        "revised",
        "moved from",
        "rather than",
        "made",
        "decided",
        "kept",
        "转变",
        "改为",
        "缩小",
        "从",
        "到",
        "决定",
        "调整",
    }
    SUPPORT_SIGNALS = {
        "evidence",
        "support",
        "supports",
        "supported",
        "because",
        "reason",
        "why",
        "confirmed",
        "decided",
        "wanted",
        "defined",
        "explain",
        "explanation",
        "依据",
        "支持",
        "因为",
        "原因",
        "解释",
        "确认",
        "决定",
    }
    RESOLUTION_SIGNALS = {
        "current",
        "main",
        "core",
        "preferred",
        "priority",
        "prioritise",
        "prioritize",
        "should emphasise",
        "should emphasize",
        "decided",
        "shifted",
        "narrowed",
        "rather than",
        "当前",
        "主要",
        "核心",
        "优先",
        "决定",
    }
    ALTERNATIVE_SIGNALS = HISTORICAL_SIGNALS | {
        "alternative",
        "conflicting",
        "instead",
        "option",
        "possible core",
        "备选",
        "冲突",
        "替代",
    }

    DISTINCT_MEMORY_LABELS = (
        "episodic memory",
        "semantic memory",
        "procedural memory",
        "情景记忆",
        "语义记忆",
        "程序记忆",
    )

    MEMORY_DEFINITION_SIGNALS = {
        "record",
        "records",
        "recorded",
        "store",
        "stores",
        "stored",
        "represents",
        "contains",
        "facts",
        "events",
        "happened",
        "when",
        "rules",
        "policies",
        "currently believed",
        "what happened",
        "response policies",
        "记录",
        "存储",
        "事实",
        "事件",
        "发生",
        "规则",
        "策略",
    }

    DEFAULT_ROLE_TOPICAL_THRESHOLDS = {
        "answer_target": 0.20,
        "current_state": 0.18,
        "historical_state": 0.15,
        "transition": 0.15,
        "procedural_rule": 0.05,
        "alternative_state": 0.16,
        "preferred_resolution": 0.18,
        "supporting_evidence": 0.14,
        "distinct_item": 0.18,
    }

    def __init__(
        self,
        config: dict[str, Any],
        coverage: CoverageEstimator,
    ) -> None:
        self.s = config["selection"]
        self.requirement_settings = config.get(
            "evidence_requirements",
            {},
        )
        self.coverage = coverage
        self.planner = EvidenceRequirementPlanner(
            config
        )
        self.stopwords = {
            str(item).lower()
            for item in config["query_analysis"].get(
                "stopwords",
                [],
            )
        }

        self.relevance_weight = float(
            self.s.get("relevance_weight", 0.45)
        )
        self.requirement_weight = float(
            self.s.get("requirement_weight", 0.30)
        )
        self.coverage_gain_weight = float(
            self.s.get("coverage_gain_weight", 0.15)
        )
        self.redundancy_weight = float(
            self.s.get("redundancy_weight", 0.10)
        )
        self.min_marginal_score = float(
            self.s.get("min_marginal_score", 0.18)
        )
        self.min_topical_score = float(
            self.s.get("min_topical_score", 0.12)
        )
        self.minimum_evidence = int(
            self.s.get("minimum_evidence", 2)
        )

        configured_thresholds = self.requirement_settings.get(
            "role_topical_thresholds",
            {},
        )
        self.role_topical_thresholds = {
            role: float(
                configured_thresholds.get(
                    role,
                    default,
                )
            )
            for role, default
            in self.DEFAULT_ROLE_TOPICAL_THRESHOLDS.items()
        }
        self.conflict_preservation_min_topical = float(
            self.requirement_settings.get(
                "conflict_preservation_min_topical",
                0.18,
            )
        )
        self.answer_anchor_enabled = bool(
            self.requirement_settings.get(
                "answer_anchor_enabled",
                True,
            )
        )

        self.last_plan: EvidencePlan | None = None
        self.last_requirement_status: dict[
            str,
            dict[str, Any],
        ] = {}

    def select(
        self,
        *,
        candidates: list[MemoryCandidate],
        features: QueryFeatures,
        route: RouteDecision,
        conflicts: list[ConflictGroup],
    ) -> list[MemoryCandidate]:
        if not candidates:
            self.last_plan = self.planner.plan(
                features=features,
                route=route,
                conflicts=conflicts,
            )
            self.last_requirement_status = {}
            return []

        plan = self.planner.plan(
            features=features,
            route=route,
            conflicts=conflicts,
        )
        self.last_plan = plan

        selected: list[MemoryCandidate] = []
        selected_ids: set[str] = set()
        covered: set[int] = set()
        counts: Counter[MemoryType] = Counter()
        used_tokens = 0

        satisfied: Counter[str] = Counter()
        distinct_keys: dict[str, set[str]] = defaultdict(set)

        def feasible(item: MemoryCandidate) -> bool:
            token_cost = approximate_token_count(
                self._candidate_text(item)
            )
            return (
                item.memory_id not in selected_ids
                and len(selected) < plan.max_evidence
                and counts[item.memory_type]
                < plan.max_per_memory_type
                and used_tokens + token_cost
                <= plan.token_budget
            )

        def add(
            item: MemoryCandidate,
            *,
            reason: str,
            matched_roles: set[str],
            requirement_gain: float,
        ) -> None:
            nonlocal used_tokens

            selected.append(item)
            selected_ids.add(item.memory_id)
            counts[item.memory_type] += 1
            used_tokens += approximate_token_count(
                self._candidate_text(item)
            )
            covered.update(
                self.coverage.candidate_coverage(
                    item,
                    features.information_needs,
                )
            )

            for requirement in plan.requirements:
                if requirement.role not in matched_roles:
                    continue

                if requirement.distinct:
                    key = self._distinct_item_key(
                        item,
                        features,
                    )
                    if key and key not in distinct_keys[
                        requirement.role
                    ]:
                        distinct_keys[
                            requirement.role
                        ].add(key)
                        satisfied[requirement.role] += 1
                else:
                    satisfied[requirement.role] += 1

            item.metadata["selector_reason"] = reason
            item.metadata["evidence_roles"] = sorted(
                matched_roles
            )
            item.metadata[
                "selector_requirement_gain"
            ] = round(requirement_gain, 6)
            item.metadata[
                "selector_topical_score"
            ] = round(
                self._topical_score(item, features),
                6,
            )

        # Phase A0: select one topically grounded answer anchor before
        # satisfying secondary temporal, conflict or explanation roles.
        # This prevents a generic "current" or "support" memory from
        # satisfying several roles while missing the actual answer topic.
        if self.answer_anchor_enabled:
            anchor_requirement = next(
                (
                    requirement
                    for requirement in plan.requirements
                    if requirement.role == "answer_target"
                ),
                None,
            )

            if anchor_requirement is not None:
                anchor_candidates: list[
                    tuple[
                        tuple[float, float, float, float, str],
                        MemoryCandidate,
                        set[str],
                        float,
                    ]
                ] = []

                for item in candidates:
                    if not feasible(item):
                        continue

                    answer_strength = self._role_strength(
                        role="answer_target",
                        item=item,
                        features=features,
                        conflicts=conflicts,
                    )
                    if answer_strength <= 0.0:
                        continue

                    matched_roles = self._matched_roles(
                        item=item,
                        requirements=plan.requirements,
                        features=features,
                        conflicts=conflicts,
                        distinct_keys=distinct_keys,
                    )
                    matched_roles.add("answer_target")

                    # For explicit-cardinality questions, the answer anchor
                    # must itself be one of the requested distinct items.
                    if (
                        plan.explicit_cardinality is not None
                        and "distinct_item" not in matched_roles
                    ):
                        continue

                    requirement_gain = self._requirement_gain(
                        matched_roles=matched_roles,
                        unsatisfied=plan.requirements,
                        satisfied=satisfied,
                    )
                    primary_topic = self._primary_topic_score(
                        item,
                        features,
                    )
                    anchor_score = (
                        0.55 * primary_topic
                        + 0.35 * float(item.final_score)
                        + 0.10 * answer_strength
                    )

                    if (
                        "current_state" in matched_roles
                        and (
                            features.asks_current_state
                            or features.query_mode
                            in {
                                QueryMode.CURRENT,
                                QueryMode.TIMELINE,
                            }
                        )
                    ):
                        anchor_score += 0.10
                    elif (
                        features.query_mode == QueryMode.HISTORICAL
                        and "historical_state" in matched_roles
                    ):
                        anchor_score += 0.08

                    anchor_candidates.append(
                        (
                            (
                                anchor_score,
                                primary_topic,
                                item.final_score,
                                item.confidence,
                                item.memory_id,
                            ),
                            item,
                            matched_roles,
                            requirement_gain,
                        )
                    )

                if anchor_candidates:
                    _, item, matched_roles, gain = max(
                        anchor_candidates,
                        key=lambda value: value[0],
                    )
                    add(
                        item,
                        reason="answer_anchor",
                        matched_roles=matched_roles,
                        requirement_gain=gain,
                    )

        # Phase A: repeatedly select the candidate that satisfies the most
        # important currently unmet roles.
        while len(selected) < plan.max_evidence:
            unsatisfied = [
                requirement
                for requirement in plan.requirements
                if satisfied[requirement.role]
                < requirement.min_count
            ]
            if not unsatisfied:
                break

            scored: list[
                tuple[
                    tuple[float, float, float, float, str],
                    MemoryCandidate,
                    set[str],
                    float,
                ]
            ] = []

            for item in candidates:
                if not feasible(item):
                    continue

                matched_roles = self._matched_roles(
                    item=item,
                    requirements=unsatisfied,
                    features=features,
                    conflicts=conflicts,
                    distinct_keys=distinct_keys,
                )
                if not matched_roles:
                    continue

                requirement_gain = self._requirement_gain(
                    matched_roles=matched_roles,
                    unsatisfied=unsatisfied,
                    satisfied=satisfied,
                )
                score = self._structured_score(
                    item=item,
                    selected=selected,
                    covered=covered,
                    needs=features.information_needs,
                    features=features,
                    requirement_gain=requirement_gain,
                )

                scored.append(
                    (
                        (
                            requirement_gain,
                            score,
                            item.final_score,
                            item.confidence,
                            item.memory_id,
                        ),
                        item,
                        matched_roles,
                        requirement_gain,
                    )
                )

            if not scored:
                break

            _, item, matched_roles, gain = max(
                scored,
                key=lambda value: value[0],
            )
            add(
                item,
                reason="requirement_satisfaction",
                matched_roles=matched_roles,
                requirement_gain=gain,
            )

        # Safety fallback for a routed procedural query when no procedure was
        # selected because metadata was incomplete.
        if (
            self.s.get(
                "guarantee_procedural_if_routed",
                True,
            )
            and MemoryType.PROCEDURAL in route.selected_types
            and not any(
                item.memory_type == MemoryType.PROCEDURAL
                for item in selected
            )
        ):
            procedures = [
                item
                for item in candidates
                if item.memory_type == MemoryType.PROCEDURAL
                and feasible(item)
            ]
            if procedures:
                item = max(
                    procedures,
                    key=lambda candidate: (
                        candidate.final_score,
                        candidate.confidence,
                        candidate.memory_id,
                    ),
                )
                add(
                    item,
                    reason="procedural_safety_fallback",
                    matched_roles={"procedural_rule"},
                    requirement_gain=1.0,
                )

        # Preserve unresolved conflict alternatives only when they are
        # topically grounded in the current query. A conflict elsewhere in
        # memory must not consume evidence slots for this answer.
        by_id = {
            item.memory_id: item
            for item in candidates
        }
        conflict_roles = {
            "answer_target",
            "current_state",
            "historical_state",
            "alternative_state",
            "preferred_resolution",
            "transition",
        }
        for conflict in conflicts:
            if not conflict.unresolved:
                continue
            for memory_id in conflict.candidate_ids[:2]:
                item = by_id.get(memory_id)
                if item is None or not feasible(item):
                    continue

                topical = self._primary_topic_score(
                    item,
                    features,
                )
                if (
                    topical
                    < self.conflict_preservation_min_topical
                ):
                    continue

                roles = (
                    self._all_matching_roles(
                        item=item,
                        plan=plan,
                        features=features,
                        conflicts=conflicts,
                    )
                    & conflict_roles
                )
                if not roles:
                    continue

                add(
                    item,
                    reason="unresolved_conflict_preservation",
                    matched_roles=roles,
                    requirement_gain=1.0,
                )

        # Phase B: fill only when the candidate is still topically useful,
        # covers a new need, or contributes another recognised role. When an
        # explicit-cardinality request is already fully satisfied, stop at the
        # requested number instead of adding unrelated buffer evidence.
        cardinality_complete = (
            plan.explicit_cardinality is not None
            and satisfied["distinct_item"]
            >= plan.explicit_cardinality
        )
        while (
            len(selected) < plan.max_evidence
            and not cardinality_complete
        ):
            scored_remaining: list[
                tuple[
                    tuple[float, float, float, str],
                    MemoryCandidate,
                    set[str],
                    float,
                    int,
                ]
            ] = []

            for item in candidates:
                if not feasible(item):
                    continue

                all_roles = self._all_matching_roles(
                    item=item,
                    plan=plan,
                    features=features,
                    conflicts=conflicts,
                )
                roles = self._novel_fill_roles(
                    item=item,
                    all_roles=all_roles,
                    plan=plan,
                    features=features,
                    satisfied=satisfied,
                    distinct_keys=distinct_keys,
                )
                new_coverage = len(
                    self.coverage.candidate_coverage(
                        item,
                        features.information_needs,
                    )
                    - covered
                )
                topical = self._topical_score(
                    item,
                    features,
                )
                role_bonus = min(
                    1.0,
                    0.25 * len(roles),
                )
                score = self._structured_score(
                    item=item,
                    selected=selected,
                    covered=covered,
                    needs=features.information_needs,
                    features=features,
                    requirement_gain=role_bonus,
                )

                useful = (
                    len(selected) < self.minimum_evidence
                    or new_coverage > 0
                    or bool(roles)
                )
                if not useful:
                    continue

                # Once the minimum evidence floor is met, do not fill with
                # generic current/support memories that are not grounded in
                # the primary answer topic.
                if (
                    len(selected) >= self.minimum_evidence
                    and topical < self.min_topical_score
                    and new_coverage == 0
                ):
                    continue

                if (
                    len(selected) >= self.minimum_evidence
                    and score < self.min_marginal_score
                    and new_coverage == 0
                    and not roles
                ):
                    continue

                scored_remaining.append(
                    (
                        (
                            score,
                            topical,
                            item.final_score,
                            item.memory_id,
                        ),
                        item,
                        roles,
                        role_bonus,
                        new_coverage,
                    )
                )

            if not scored_remaining:
                break

            _, item, roles, gain, new_coverage = max(
                scored_remaining,
                key=lambda value: value[0],
            )

            add(
                item,
                reason=(
                    "coverage_fill"
                    if new_coverage > 0
                    else "structured_mmr_fill"
                ),
                matched_roles=roles,
                requirement_gain=gain,
            )

        self.last_requirement_status = {
            requirement.role: {
                "required": requirement.min_count,
                "satisfied": min(
                    satisfied[requirement.role],
                    requirement.min_count,
                ),
                "hard": requirement.hard,
                "distinct": requirement.distinct,
                "complete": (
                    satisfied[requirement.role]
                    >= requirement.min_count
                ),
            }
            for requirement in plan.requirements
        }

        return selected

    def _matched_roles(
        self,
        *,
        item: MemoryCandidate,
        requirements: list[EvidenceRequirement],
        features: QueryFeatures,
        conflicts: list[ConflictGroup],
        distinct_keys: dict[str, set[str]],
    ) -> set[str]:
        matched: set[str] = set()

        for requirement in requirements:
            strength = self._role_strength(
                role=requirement.role,
                item=item,
                features=features,
                conflicts=conflicts,
            )
            if strength <= 0.0:
                continue

            if requirement.distinct:
                key = self._distinct_item_key(
                    item,
                    features,
                )
                if (
                    not key
                    or key in distinct_keys[
                        requirement.role
                    ]
                ):
                    continue

            matched.add(requirement.role)

        return matched

    def _all_matching_roles(
        self,
        *,
        item: MemoryCandidate,
        plan: EvidencePlan,
        features: QueryFeatures,
        conflicts: list[ConflictGroup],
    ) -> set[str]:
        return {
            requirement.role
            for requirement in plan.requirements
            if self._role_strength(
                role=requirement.role,
                item=item,
                features=features,
                conflicts=conflicts,
            )
            > 0.0
        }

    def _novel_fill_roles(
        self,
        *,
        item: MemoryCandidate,
        all_roles: set[str],
        plan: EvidencePlan,
        features: QueryFeatures,
        satisfied: Counter[str],
        distinct_keys: dict[str, set[str]],
    ) -> set[str]:
        """Allow limited role redundancy after hard requirements are met."""
        output: set[str] = set()
        extra_roles = {
            "historical_state",
            "transition",
            "supporting_evidence",
            "alternative_state",
        }

        for requirement in plan.requirements:
            if requirement.role not in all_roles:
                continue

            allowance = requirement.min_count + (
                1 if requirement.role in extra_roles else 0
            )
            if satisfied[requirement.role] >= allowance:
                continue

            if requirement.distinct:
                key = self._distinct_item_key(
                    item,
                    features,
                )
                if (
                    not key
                    or key in distinct_keys[requirement.role]
                ):
                    continue

            output.add(requirement.role)

        return output

    def _requirement_gain(
        self,
        *,
        matched_roles: set[str],
        unsatisfied: list[EvidenceRequirement],
        satisfied: Counter[str],
    ) -> float:
        gain = 0.0
        for requirement in unsatisfied:
            if requirement.role not in matched_roles:
                continue
            remaining = max(
                1,
                requirement.min_count
                - satisfied[requirement.role],
            )
            gain += (
                (1.0 if requirement.hard else 0.5)
                / remaining
            )
        return gain

    def _structured_score(
        self,
        *,
        item: MemoryCandidate,
        selected: list[MemoryCandidate],
        covered: set[int],
        needs: list[str],
        features: QueryFeatures,
        requirement_gain: float,
    ) -> float:
        redundancy = max(
            (
                jaccard_similarity(
                    tokenize(
                        self._candidate_text(item),
                        self.stopwords,
                    ),
                    tokenize(
                        self._candidate_text(other),
                        self.stopwords,
                    ),
                )
                for other in selected
            ),
            default=0.0,
        )
        coverage_gain = (
            len(
                self.coverage.candidate_coverage(
                    item,
                    needs,
                )
                - covered
            )
            / len(needs)
            if needs
            else 0.0
        )
        topical = self._topical_score(
            item,
            features,
        )
        relevance = (
            0.65 * float(item.final_score)
            + 0.35 * topical
        )

        return (
            self.relevance_weight * relevance
            + self.requirement_weight
            * min(1.0, requirement_gain)
            + self.coverage_gain_weight
            * coverage_gain
            - self.redundancy_weight
            * redundancy
        )

    def _role_strength(
        self,
        *,
        role: str,
        item: MemoryCandidate,
        features: QueryFeatures,
        conflicts: list[ConflictGroup],
    ) -> float:
        text = self._candidate_text(item).lower()
        primary_topical = self._primary_topic_score(
            item,
            features,
        )
        broad_topical = self._topical_score(
            item,
            features,
        )
        threshold = self.role_topical_thresholds.get(
            role,
            self.min_topical_score,
        )

        # Roles that determine the factual answer must be grounded in the
        # primary topic, not merely in generic intent words such as current,
        # explain, evidence or preferred.
        if role == "answer_target":
            return (
                primary_topical
                if primary_topical >= threshold
                else 0.0
            )

        if role == "current_state":
            if primary_topical < threshold:
                return 0.0

            has_historical_signal = (
                self._contains_historical_signal(text)
            )
            has_current_signal = (
                self._contains_current_signal(text)
            )
            preferred = self._is_preferred(
                item,
                conflicts,
            )

            # A semantic record can still carry status=current while its text
            # explicitly describes an earlier state. Do not let that satisfy
            # the current endpoint unless it also carries an explicit current
            # signal or was selected as the conflict resolution.
            if (
                has_historical_signal
                and not has_current_signal
                and not preferred
            ):
                return 0.0

            if self._is_current(item):
                return min(
                    1.0,
                    0.65 + 0.35 * primary_topical,
                )
            if has_current_signal:
                return min(
                    1.0,
                    0.45 + 0.40 * primary_topical,
                )
            return 0.0

        if role == "historical_state":
            if primary_topical < threshold:
                return 0.0

            if self._is_historical(item):
                return min(
                    1.0,
                    0.65 + 0.35 * primary_topical,
                )
            if self._contains_historical_signal(text):
                return min(
                    1.0,
                    0.50 + 0.40 * primary_topical,
                )
            return 0.0

        if role == "transition":
            if primary_topical < threshold:
                return 0.0
            if any(
                signal in text
                for signal in self.TRANSITION_SIGNALS
            ):
                bonus = (
                    0.15
                    if item.memory_type
                    == MemoryType.EPISODIC
                    else 0.0
                )
                return min(
                    1.0,
                    0.50
                    + bonus
                    + 0.35 * primary_topical,
                )
            return 0.0

        if role == "procedural_rule":
            if item.memory_type != MemoryType.PROCEDURAL:
                return 0.0

            applicability = self._procedure_applicability(
                item
            )
            topical = max(
                primary_topical,
                0.75 * broad_topical,
            )
            if (
                topical < threshold
                and applicability < 0.80
            ):
                return 0.0

            return min(
                1.0,
                0.60 * applicability
                + 0.40 * topical,
            )

        if role == "alternative_state":
            if primary_topical < threshold:
                return 0.0
            if self._is_historical(item):
                return min(
                    1.0,
                    0.65 + 0.30 * primary_topical,
                )
            if any(
                signal in text
                for signal in self.ALTERNATIVE_SIGNALS
            ):
                return min(
                    1.0,
                    0.50 + 0.40 * primary_topical,
                )
            if self._is_conflict_member(
                item,
                conflicts,
            ):
                return min(
                    1.0,
                    0.45 + 0.40 * primary_topical,
                )
            return 0.0

        if role == "preferred_resolution":
            if primary_topical < threshold:
                return 0.0
            if self._is_preferred(
                item,
                conflicts,
            ):
                return min(
                    1.0,
                    0.70 + 0.25 * primary_topical,
                )
            if any(
                signal in text
                for signal in self.RESOLUTION_SIGNALS
            ):
                return min(
                    1.0,
                    0.50 + 0.40 * primary_topical,
                )
            return 0.0

        if role == "supporting_evidence":
            topical = max(
                primary_topical,
                0.80 * broad_topical,
            )
            if topical < threshold:
                return 0.0

            has_support_signal = any(
                signal in text
                for signal in self.SUPPORT_SIGNALS
            )
            if item.memory_type == MemoryType.EPISODIC:
                return min(
                    1.0,
                    0.45 + 0.40 * topical,
                )
            if has_support_signal:
                return min(
                    1.0,
                    0.45 + 0.40 * topical,
                )
            if self._is_historical(item):
                return min(
                    1.0,
                    0.40 + 0.35 * topical,
                )
            return 0.0

        if role == "distinct_item":
            key = self._distinct_item_key(
                item,
                features,
            )
            if not key:
                return 0.0
            if primary_topical < threshold:
                return 0.0
            return min(
                1.0,
                0.70 + 0.25 * primary_topical,
            )

        return 0.0

    def _topical_score(
        self,
        item: MemoryCandidate,
        features: QueryFeatures,
    ) -> float:
        """Broad topical score used by MMR and support-role ranking.

        The primary information need remains dominant so generic query intent
        words such as explain, current or evidence cannot outweigh the actual
        subject being asked about.
        """
        primary = self._primary_topic_score(
            item,
            features,
        )
        candidate_tokens = tokenize(
            self._candidate_text(item),
            self.stopwords,
        )
        query_tokens = tokenize(
            features.normalised_query,
            self.stopwords,
        )
        full_overlap = overlap_ratio(
            query_tokens,
            candidate_tokens,
        )

        lexical_component = (
            0.45 * float(item.lexical_score)
            + 0.55 * primary
        )
        entity_component = (
            0.50 * float(item.graph_entity_score)
            + 0.50 * primary
        )

        return max(
            primary,
            0.35 * full_overlap + 0.65 * primary,
            lexical_component,
            entity_component,
        )

    def _primary_topic_score(
        self,
        item: MemoryCandidate,
        features: QueryFeatures,
    ) -> float:
        """Score overlap with the answer topic, excluding intent scaffolding."""
        candidate_tokens = tokenize(
            self._candidate_text(item),
            self.stopwords,
        )
        primary_need = (
            features.information_needs[0]
            if features.information_needs
            else features.normalised_query
        )
        primary_tokens = tokenize(
            primary_need,
            self.stopwords,
        )
        focus_tokens = self._focus_tokens(
            primary_tokens
        )
        if not focus_tokens:
            focus_tokens = primary_tokens

        slot_text = " ".join(
            value
            for value in (
                item.subject or "",
                item.predicate or "",
                item.object_value or "",
            )
            if value
        )
        slot_tokens = tokenize(
            slot_text,
            self.stopwords,
        )

        text_overlap = overlap_ratio(
            focus_tokens,
            candidate_tokens,
        )
        slot_overlap = overlap_ratio(
            focus_tokens,
            slot_tokens,
        )

        # For "three memory types" queries, the type label itself is a
        # legitimate primary-topic match even when the candidate defines only
        # one of the three types.
        query_lower = features.normalised_query.lower()
        memory_type_bonus = 0.0
        if (
            "memory type" in query_lower
            or "memory types" in query_lower
            or "记忆类型" in query_lower
        ):
            text_lower = self._candidate_text(item).lower()
            if any(
                label in text_lower
                for label in self.DISTINCT_MEMORY_LABELS
            ):
                memory_type_bonus = 0.35

        return max(
            text_overlap,
            slot_overlap,
            memory_type_bonus,
        )

    @staticmethod
    def _focus_tokens(
        tokens: list[str],
    ) -> list[str]:
        generic = {
            "answer",
            "explain",
            "support",
            "supports",
            "supported",
            "evidence",
            "current",
            "currently",
            "earlier",
            "historical",
            "state",
            "states",
            "preferred",
            "alternative",
            "alternatives",
            "conflicting",
            "conflict",
            "should",
            "mainly",
            "more",
            "over",
            "time",
            "change",
            "changed",
            "which",
            "memories",
            "memory",
            "what",
            "does",
            "each",
            "three",
            "originally",
            "before",
            "now",
        }
        return [
            token
            for token in tokens
            if token not in generic
        ]

    def _contains_historical_signal(
        self,
        text: str,
    ) -> bool:
        return any(
            signal in text
            for signal in self.HISTORICAL_SIGNALS
        )

    def _contains_current_signal(
        self,
        text: str,
    ) -> bool:
        return any(
            signal in text
            for signal in self.CURRENT_SIGNALS
        )

    def _distinct_item_key(
        self,
        item: MemoryCandidate,
        features: QueryFeatures,
    ) -> str | None:
        text = self._candidate_text(item).lower()
        query_lower = features.normalised_query.lower()
        memory_type_query = (
            "memory type" in query_lower
            or "memory types" in query_lower
            or "记忆类型" in query_lower
        )

        if memory_type_query:
            has_definition_signal = any(
                signal in text
                for signal in self.MEMORY_DEFINITION_SIGNALS
            )
            if not has_definition_signal:
                return None

            for label in self.DISTINCT_MEMORY_LABELS:
                if label in text:
                    return label.replace(" ", "_")

            return None

        subject = self._normalise_key(item.subject)
        predicate = self._normalise_key(item.predicate)
        object_value = self._normalise_key(
            item.object_value
        )

        if subject and predicate:
            return f"{subject}|{predicate}"
        if subject:
            return subject
        if predicate and object_value:
            return f"{predicate}|{object_value}"

        # For explicit-cardinality queries, fall back to the most informative
        # beginning of the candidate text rather than the memory ID.
        tokens = tokenize(text, self.stopwords)
        if tokens:
            return "_".join(tokens[:4])

        return None

    @staticmethod
    def _normalise_key(value: str | None) -> str:
        if not value:
            return ""
        return re.sub(
            r"\s+",
            "_",
            value.strip().lower(),
        )

    @staticmethod
    def _candidate_text(
        item: MemoryCandidate,
    ) -> str:
        metadata_values: list[str] = []
        for key in (
            "instruction",
            "description",
            "response_policy",
            "task_type",
            "matched_triggers",
            "triggers",
            "steps",
        ):
            value = item.metadata.get(key)
            if isinstance(value, list):
                metadata_values.extend(
                    str(part)
                    for part in value
                )
            elif value not in (None, ""):
                metadata_values.append(str(value))

        values = [
            item.text,
            item.subject or "",
            item.predicate or "",
            item.object_value or "",
            *metadata_values,
        ]
        return " ".join(
            value
            for value in values
            if value
        )

    def _is_historical(
        self,
        item: MemoryCandidate,
    ) -> bool:
        return (
            item.status.lower()
            in self.HISTORICAL_STATUSES
            or (item.resolution_action or "").lower()
            in self.HISTORICAL_ACTIONS
        )

    def _is_current(
        self,
        item: MemoryCandidate,
    ) -> bool:
        action = (
            item.resolution_action or ""
        ).lower()
        if action in self.CURRENT_ACTIONS:
            return True
        return (
            item.memory_type == MemoryType.SEMANTIC
            and item.status.lower()
            in self.CURRENT_STATUSES
            and action
            not in {
                "historical",
                "excluded",
                "unresolved",
            }
        )

    @staticmethod
    def _procedure_applicability(
        item: MemoryCandidate,
    ) -> float:
        task_match = float(
            item.metadata.get("task_match", 0.0)
            or 0.0
        )
        preliminary = float(
            item.metadata.get(
                "preliminary_procedural_score",
                0.0,
            )
            or 0.0
        )
        direct_match = bool(
            item.metadata.get("direct_match", False)
        )
        always_apply = bool(
            item.metadata.get("always_apply", False)
        )

        return min(
            1.0,
            max(
                task_match,
                preliminary,
                1.0 if always_apply else 0.0,
                0.8 if direct_match else 0.0,
            ),
        )

    @staticmethod
    def _is_conflict_member(
        item: MemoryCandidate,
        conflicts: list[ConflictGroup],
    ) -> bool:
        return any(
            item.memory_id in group.candidate_ids
            for group in conflicts
        )

    @staticmethod
    def _is_preferred(
        item: MemoryCandidate,
        conflicts: list[ConflictGroup],
    ) -> bool:
        if (
            item.resolution_action or ""
        ).lower() in {
            "preferred",
            "current_endpoint",
            "preferred_historical",
        }:
            return True

        return any(
            item.memory_id in group.preferred_ids
            for group in conflicts
        )
