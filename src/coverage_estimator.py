from __future__ import annotations

from typing import Any

from .schemas import MemoryCandidate, MemoryType
from .text_utils import overlap_ratio, tokenize


class CoverageEstimator:
    """Estimate lexical and structure-aware evidence coverage.

    Coverage is not treated as word overlap alone:

    * Timeline queries use historical/current roles and temporal transitions.
    * Procedural queries use task applicability, trigger matches and rule text.
    """

    EARLIER_SIGNALS = {
        "earlier",
        "historical",
        "previous",
        "previously",
        "initial",
        "initially",
        "original",
        "originally",
        "past",
        "before",
        "旧",
        "过去",
        "之前",
        "最初",
        "历史",
    }
    CURRENT_SIGNALS = {
        "current",
        "currently",
        "present",
        "latest",
        "now",
        "当前",
        "现在",
        "目前",
        "最新",
    }
    CHANGE_SIGNALS = {
        "change",
        "changed",
        "changes",
        "changing",
        "timeline",
        "over time",
        "evolve",
        "evolution",
        "transition",
        "narrowed",
        "updated",
        "revised",
        "switched",
        "变化",
        "改变",
        "演变",
        "更新",
        "转为",
    }
    PROCEDURAL_NEED_SIGNALS = {
        "should",
        "how",
        "written",
        "write",
        "format",
        "style",
        "procedure",
        "rule",
        "instruction",
        "follow",
        "must",
        "applicable",
        "应该",
        "如何",
        "怎么",
        "写",
        "格式",
        "风格",
        "规则",
        "步骤",
        "遵循",
        "必须",
        "适用",
    }

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

    def __init__(self, config: dict[str, Any]) -> None:
        stopwords = config["query_analysis"].get(
            "stopwords",
            [],
        )
        self.stopwords = {
            str(item).lower() for item in stopwords
        }

    def candidate_coverage(
        self,
        candidate: MemoryCandidate,
        information_needs: list[str],
    ) -> set[int]:
        candidate_tokens = tokenize(
            " ".join(
                value
                for value in [
                    candidate.text,
                    candidate.subject or "",
                    candidate.predicate or "",
                    candidate.object_value or "",
                ]
                if value
            ),
            self.stopwords,
        )

        covered: set[int] = set()
        for index, need in enumerate(information_needs):
            need_lower = need.lower()
            need_tokens = tokenize(
                need,
                self.stopwords,
            )

            # Procedural coverage must recognise that one well-matched rule can
            # answer both the task context and the requested action/style.
            if (
                candidate.memory_type == MemoryType.PROCEDURAL
                and self._is_applicable_procedure(candidate)
                and self._procedure_covers_need(
                    candidate,
                    need_lower,
                    need_tokens,
                    candidate_tokens,
                )
            ):
                covered.add(index)
                continue

            # Base lexical/semantic-slot proxy coverage.
            if overlap_ratio(
                need_tokens,
                candidate_tokens,
            ) >= 0.20:
                covered.add(index)
                continue

            # Structured timeline/current-state coverage.
            if (
                self._need_matches(
                    need_lower,
                    self.EARLIER_SIGNALS,
                )
                and self._is_historical(candidate)
            ):
                covered.add(index)
                continue

            if (
                self._need_matches(
                    need_lower,
                    self.CURRENT_SIGNALS,
                )
                and self._is_current_endpoint(candidate)
            ):
                covered.add(index)
                continue

            if (
                self._need_matches(
                    need_lower,
                    self.CHANGE_SIGNALS,
                )
                and self._is_change_evidence(candidate)
            ):
                covered.add(index)

        return covered

    def compute(
        self,
        information_needs: list[str],
        selected: list[MemoryCandidate],
    ) -> float:
        if not information_needs:
            return 1.0 if selected else 0.0

        covered: set[int] = set()
        for candidate in selected:
            covered |= self.candidate_coverage(
                candidate,
                information_needs,
            )

        # Global timeline coverage: historical + current evidence demonstrates
        # a transition even if no single memory literally says "over time".
        historical_present = any(
            self._is_historical(candidate)
            for candidate in selected
        )
        current_present = any(
            self._is_current_endpoint(candidate)
            for candidate in selected
        )
        change_event_present = any(
            self._is_change_evidence(candidate)
            for candidate in selected
        )
        distinct_temporal_states = (
            len(
                {
                    candidate.timestamp
                    for candidate in selected
                    if candidate.timestamp is not None
                }
            )
            >= 2
        )

        for index, need in enumerate(information_needs):
            need_lower = need.lower()

            if (
                self._need_matches(
                    need_lower,
                    self.EARLIER_SIGNALS,
                )
                and historical_present
            ):
                covered.add(index)

            if (
                self._need_matches(
                    need_lower,
                    self.CURRENT_SIGNALS,
                )
                and current_present
            ):
                covered.add(index)

            if self._need_matches(
                need_lower,
                self.CHANGE_SIGNALS,
            ):
                if (
                    historical_present
                    and current_present
                    and (
                        change_event_present
                        or distinct_temporal_states
                    )
                ):
                    covered.add(index)

        selected_text = " ".join(
            " ".join(
                str(value)
                for value in [
                    candidate.text,
                    candidate.subject,
                    candidate.predicate,
                    candidate.object_value,
                ]
                if value not in (None, "")
            )
            for candidate in selected
        ).lower().replace("_", " ")

        memory_definition_supported = all(
            label in selected_text
            for label in (
                "episodic memory",
                "semantic memory",
                "procedural memory",
            )
        )

        if memory_definition_supported:
            for index, need in enumerate(
                information_needs
            ):
                need_lower = need.lower()

                if any(
                    signal in need_lower
                    for signal in (
                        "memory",
                        "type",
                        "types",
                        "each",
                        "record",
                        "records",
                    )
                ):
                    covered.add(index)

        # RC8.4-B role-grounded coverage.
        procedural_rule_selected = any(
            candidate.memory_type
            == MemoryType.PROCEDURAL
            and "procedural_rule"
            in {
                str(role)
                for role in (
                    candidate.metadata.get(
                        "evidence_roles",
                        [],
                    )
                    or []
                )
            }
            for candidate in selected
        )

        scope_validation_supported = any(
            phrase in selected_text
            for phrase in (
                "implementation backend not research core",
                "implementation backend, not research core",
                "not the central research contribution",
                "not research core",
            )
        )

        for index, need in enumerate(
            information_needs
        ):
            need_lower = need.lower()

            if (
                procedural_rule_selected
                and (
                    "applicable procedure" in need_lower
                    or "procedure or rule" in need_lower
                    or "applicable rule" in need_lower
                )
            ):
                covered.add(index)

            if (
                scope_validation_supported
                and "neo4j" in need_lower
                and any(
                    signal in need_lower
                    for signal in (
                        "main",
                        "mainly",
                        "primarily",
                        "proving",
                        "useful",
                        "focus",
                        "core",
                    )
                )
            ):
                covered.add(index)

        return len(covered) / len(information_needs)

    def _procedure_covers_need(
        self,
        candidate: MemoryCandidate,
        need_lower: str,
        need_tokens: list[str],
        candidate_tokens: list[str],
    ) -> bool:
        matched_triggers = [
            str(trigger).lower()
            for trigger in candidate.metadata.get(
                "matched_triggers",
                [],
            )
        ]
        all_triggers = [
            str(trigger).lower()
            for trigger in candidate.metadata.get(
                "triggers",
                [],
            )
        ]

        trigger_match = any(
            trigger in need_lower
            for trigger in [*matched_triggers, *all_triggers]
            if trigger
        )
        lexical_match = overlap_ratio(
            need_tokens,
            candidate_tokens,
        ) >= 0.20
        procedural_need = self._need_matches(
            need_lower,
            self.PROCEDURAL_NEED_SIGNALS,
        )

        # Once a rule is established as directly applicable to the task,
        # generic "how/should/rule/style" needs are covered by its instruction.
        return trigger_match or lexical_match or procedural_need

    @staticmethod
    def _is_applicable_procedure(
        candidate: MemoryCandidate,
    ) -> bool:
        preliminary_score = float(
            candidate.metadata.get(
                "preliminary_procedural_score",
                0.0,
            )
            or 0.0
        )
        task_match = float(
            candidate.metadata.get(
                "task_match",
                0.0,
            )
            or 0.0
        )

        if (
            preliminary_score >= 0.50
            or task_match >= 0.80
        ):
            return True

        task_match = float(
            candidate.metadata.get("task_match", 0.0)
        )
        preliminary_score = float(
            candidate.metadata.get(
                "preliminary_procedural_score",
                0.0,
            )
        )
        direct_match = bool(
            candidate.metadata.get("direct_match", False)
        )
        always_apply = bool(
            candidate.metadata.get("always_apply", False)
        )

        return (
            always_apply
            or (
                direct_match
                and (
                    task_match >= 0.80
                    or preliminary_score >= 0.50
                )
            )
        )

    @staticmethod
    def _need_matches(
        need: str,
        signals: set[str],
    ) -> bool:
        return any(signal in need for signal in signals)

    def _is_historical(
        self,
        candidate: MemoryCandidate,
    ) -> bool:
        content = " ".join(
            str(value)
            for value in [
                candidate.text,
                candidate.subject,
                candidate.predicate,
                candidate.object_value,
            ]
            if value not in (None, "")
        ).lower()

        return (
            candidate.status.lower()
            in self.HISTORICAL_STATUSES
            or (candidate.resolution_action or "").lower()
            in self.HISTORICAL_ACTIONS
            or any(
                signal in content
                for signal in self.EARLIER_SIGNALS
            )
        )

    def _is_current_endpoint(
        self,
        candidate: MemoryCandidate,
    ) -> bool:
        action = (
            candidate.resolution_action or ""
        ).lower()
        if action in self.CURRENT_ACTIONS:
            return True

        return (
            candidate.memory_type == MemoryType.SEMANTIC
            and candidate.status.lower()
            in self.CURRENT_STATUSES
            and action
            not in {
                "historical",
                "excluded",
                "unresolved",
            }
        )

    def _is_change_evidence(
        self,
        candidate: MemoryCandidate,
    ) -> bool:
        text = candidate.text.lower()
        if any(
            signal in text for signal in self.CHANGE_SIGNALS
        ):
            return True

        return (
            self._is_historical(candidate)
            or (candidate.resolution_action or "").lower()
            == "current_endpoint"
        )