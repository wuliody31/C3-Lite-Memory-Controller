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

        # Preserve explicit unresolved conflict alternatives.
        by_id = {
            item.memory_id: item
            for item in candidates
        }
        for conflict in conflicts:
            if not conflict.unresolved:
                continue
            for memory_id in conflict.candidate_ids[:2]:
                item = by_id.get(memory_id)
                if item is None or not feasible(item):
                    continue
                roles = self._all_matching_roles(
                    item=item,
                    plan=plan,
                    features=features,
                    conflicts=conflicts,
                )
                add(
                    item,
                    reason="unresolved_conflict_preservation",
                    matched_roles=roles
                    or {"alternative_state"},
                    requirement_gain=1.0,
                )

        # Phase B: fill only when the candidate is still topically useful,
        # covers a new need, or contributes another recognised role.
        while len(selected) < plan.max_evidence:
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
        topical = self._topical_score(
            item,
            features,
        )

        if role == "answer_target":
            return topical if topical >= 0.12 else 0.0

        if role == "current_state":
            if self._is_current(item):
                return 0.65 + 0.35 * topical
            if any(signal in text for signal in self.CURRENT_SIGNALS):
                return 0.45 + 0.35 * topical
            return 0.0

        if role == "historical_state":
            if self._is_historical(item):
                return 0.65 + 0.35 * topical
            if any(
                signal in text
                for signal in self.HISTORICAL_SIGNALS
            ):
                return 0.50 + 0.35 * topical
            return 0.0

        if role == "transition":
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
                    0.55 + bonus + 0.30 * topical,
                )
            return 0.0

        if role == "procedural_rule":
            if item.memory_type != MemoryType.PROCEDURAL:
                return 0.0
            applicability = self._procedure_applicability(
                item
            )
            return max(
                0.50,
                0.65 * applicability
                + 0.35 * topical,
            )

        if role == "alternative_state":
            if self._is_historical(item):
                return 0.65 + 0.30 * topical
            if any(
                signal in text
                for signal in self.ALTERNATIVE_SIGNALS
            ):
                return 0.50 + 0.35 * topical
            if self._is_conflict_member(item, conflicts):
                return 0.45 + 0.35 * topical
            return 0.0

        if role == "preferred_resolution":
            if self._is_preferred(item, conflicts):
                return 0.70 + 0.25 * topical
            if any(
                signal in text
                for signal in self.RESOLUTION_SIGNALS
            ):
                return 0.50 + 0.35 * topical
            return 0.0

        if role == "supporting_evidence":
            has_support_signal = any(
                signal in text
                for signal in self.SUPPORT_SIGNALS
            )
            if (
                item.memory_type == MemoryType.EPISODIC
                and topical >= 0.10
            ):
                return 0.50 + 0.35 * topical
            if has_support_signal and topical >= 0.08:
                return 0.45 + 0.35 * topical
            if self._is_historical(item) and topical >= 0.15:
                return 0.40 + 0.35 * topical
            return 0.0

        if role == "distinct_item":
            key = self._distinct_item_key(
                item,
                features,
            )
            return (
                0.70 + 0.25 * topical
                if key
                else 0.0
            )

        return 0.0

    def _topical_score(
        self,
        item: MemoryCandidate,
        features: QueryFeatures,
    ) -> float:
        candidate_tokens = tokenize(
            self._candidate_text(item),
            self.stopwords,
        )
        query_tokens = tokenize(
            features.normalised_query,
            self.stopwords,
        )
        primary_need_tokens = tokenize(
            (
                features.information_needs[0]
                if features.information_needs
                else features.normalised_query
            ),
            self.stopwords,
        )

        lexical_proxy = max(
            overlap_ratio(
                query_tokens,
                candidate_tokens,
            ),
            overlap_ratio(
                primary_need_tokens,
                candidate_tokens,
            ),
        )
        focus_tokens = [
            token
            for token in primary_need_tokens
            if token
            not in {
                "answer",
                "explain",
                "support",
                "supports",
                "evidence",
                "current",
                "currently",
                "earlier",
                "historical",
                "state",
                "preferred",
                "alternative",
                "conflicting",
                "should",
                "mainly",
                "more",
                "over",
                "time",
                "change",
                "changed",
            }
        ]
        focus_score = overlap_ratio(
            focus_tokens,
            candidate_tokens,
        )
        blended_proxy = (
            0.60 * lexical_proxy
            + 0.40 * focus_score
        )

        return max(
            float(item.lexical_score),
            float(item.graph_entity_score),
            blended_proxy,
        )

    def _distinct_item_key(
        self,
        item: MemoryCandidate,
        features: QueryFeatures,
    ) -> str | None:
        text = self._candidate_text(item).lower()

        for label in self.DISTINCT_MEMORY_LABELS:
            if label in text:
                return label.replace(" ", "_")

        query_lower = features.normalised_query.lower()
        if (
            "memory type" in query_lower
            or "memory types" in query_lower
            or "记忆类型" in query_lower
        ):
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
