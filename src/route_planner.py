from __future__ import annotations

import re
from typing import Any

from .schemas import (
    MemoryType,
    QueryFeatures,
    QueryMode,
    RouteDecision,
)


class RoutePlanner:
    TERMS = {
        MemoryType.EPISODIC: {
            "when",
            "happened",
            "event",
            "meeting",
            "originally",
            "previously",
            "before",
            "earlier",
            "experience",
            "did",
            "went",
            "decided",
            "selected",
            "chosen",
            "picked",
            "adopted",
            "发生",
            "最初",
            "之前",
            "经历",
            "决定",
            "选择",
        },
        MemoryType.SEMANTIC: {
            "current",
            "currently",
            "what is",
            "which",
            "uses",
            "preference",
            "focus",
            "role",
            "main",
            "now",
            "目前",
            "现在",
            "是什么",
            "使用",
            "偏好",
        },
        MemoryType.PROCEDURAL: {
            "how should",
            "should i",
            "format",
            "write",
            "follow",
            "must",
            "step",
            "style",
            "procedure",
            "guideline",
            "怎么",
            "应该",
            "格式",
            "写",
            "遵循",
            "必须",
        },
    }

    FACTUAL_SELECTION_PATTERNS = (
        r"\bwhat\s+(?:technical\s+)?details\s+should\s+i\s+"
        r"(?:add|include|mention)\b",
        r"\bwhat\s+should\s+i\s+(?:mention|add|include)\s+"
        r"(?:for|in|about|to)\b",
        r"\bwhat\s+should\s+my\s+(?:current\s+)?cv\s+"
        r"(?:emphasise|emphasize|highlight)\b",
        r"\bwhat\s+.+\s+can\s+i\s+discuss\s+in\s+interviews?\b",
        r"\bwhat\s+should\s+be\s+protected\s+as\s+the\s+core\s+project\b",
        r"\bwhat\s+is\s+the\s+role\s+of\s+the\s+memory\s+controller\b",
    )

    HISTORICAL_SEMANTIC_PATTERNS = (
        r"\bwhere\s+did\s+i\s+originally\s+think\b",
        r"\bdid\s+i\s+ask\s+for\b",
        r"\bhow\s+did\s+it\s+affect\b",
        r"\bwhat\s+happened\b.+\bplan\b",
        r"\b(?:hotel|stay|location|preference|route|plan|role|target|direction)\b",
    )

    EXPLANATION_PROCEDURAL_PATTERNS = (
        r"\bcite\s+the\s+memory\s+types\s+used\b",
        r"\bwhich\s+memory\s+supports\b.+\brecommend",
        r"\bwhat\s+location\s+should\s+be\s+used\b",
        r"\bhow\s+should\s+(?:the\s+answer|you|the\s+system)\b",
    )

    PROCEDURAL_SEMANTIC_PATTERNS = (
        r"\b(?:booking|price|prices|opening\s+hours|restaurant|hotel|travel|"
        r"bus|train|itinerary)\b",
        r"\b(?:unclear|unknown|not\s+stored|not\s+available)\b",
    )

    HISTORICAL_COMMITMENT_PATTERNS = (
        r"\bdid\s+i\s+(?:decide|choose|select|adopt|agree)\b",
        r"\bhave\s+i\s+(?:selected|chosen|adopted|decided)\b",
        r"\bwhat\b.+\bhave\s+i\s+(?:selected|chosen|adopted)\b",
        r"\bwhich\b.+\bhave\s+i\s+(?:selected|chosen|adopted)\b",
    )

    SCOPE_VALIDATION_PATTERNS = (
        r"\bis\s+(?:my|the)\s+.+\s+mainly\s+about\b",
        r"\bis\s+(?:my|the)\s+.+\s+primarily\s+about\b",
        r"\bis\s+.+\s+the\s+main\s+(?:focus|purpose)\b",
    )

    def __init__(
        self,
        config: dict[str, Any],
    ):
        routing = config["routing"]
        self.thresholds = routing["thresholds"]
        self.weights = routing["utility_weights"]
        self.priors = routing["priors"]
        self.gates = routing["structural_gates"]

    def plan(
        self,
        features: QueryFeatures,
    ) -> RouteDecision:
        raw: dict[MemoryType, float] = {}

        reasons = {
            memory_type.value: []
            for memory_type in MemoryType
        }

        for memory_type in MemoryType:
            signals = self._signals(
                memory_type,
                features,
            )

            raw[memory_type] = min(
                1.0,
                max(
                    0.0,
                    sum(
                        float(self.weights[key])
                        * value
                        for key, value
                        in signals.items()
                    ),
                ),
            )

            reasons[memory_type.value].extend(
                key
                for key, value in signals.items()
                if (
                    key != "prior"
                    and value >= 0.6
                )
            )

        selected = [
            memory_type
            for memory_type, score
            in raw.items()
            if score
            >= float(
                self.thresholds[
                    memory_type.value
                ]
            )
        ]

        self._gates(
            features,
            selected,
            reasons,
        )

        if (
            not selected
            and self.gates.get(
                "fallback_top_route",
                True,
            )
        ):
            fallback = max(
                raw,
                key=raw.get,
            )
            selected = [fallback]
            reasons[fallback.value].append(
                "fallback_top_route"
            )

        selected_set = set(selected)

        selected = [
            memory_type
            for memory_type in MemoryType
            if memory_type in selected_set
        ]

        return RouteDecision(
            selected_types=selected,
            scores={
                memory_type.value: round(
                    raw[memory_type],
                    6,
                )
                for memory_type in MemoryType
            },
            reasons=reasons,
        )

    def _signals(
        self,
        memory_type: MemoryType,
        features: QueryFeatures,
    ) -> dict[str, float]:
        query = features.normalised_query
        tokens = set(features.tokens)

        lexical_hits = sum(
            int(
                (
                    term in query
                    if " " in term
                    else term in tokens
                )
            )
            for term in self.TERMS[
                memory_type
            ]
        )

        lexical = min(
            1.0,
            lexical_hits / 2.0,
        )

        if memory_type == MemoryType.EPISODIC:
            return {
                "intent": (
                    1.0
                    if features.query_mode
                    in {
                        QueryMode.HISTORICAL,
                        QueryMode.TIMELINE,
                    }
                    else 0.15
                ),
                "lexical": lexical,
                "entity": (
                    min(
                        1.0,
                        0.2
                        + 0.15
                        * len(features.entities),
                    )
                    if features.entities
                    else 0.15
                ),
                "temporal_task": (
                    1.0
                    if (
                        features.temporal_expressions
                        or features.asks_historical_state
                        or features.asks_timeline
                    )
                    else 0.1
                ),
                "prior": float(
                    self.priors[
                        memory_type.value
                    ]
                ),
            }

        if memory_type == MemoryType.SEMANTIC:
            return {
                "intent": (
                    1.0
                    if features.query_mode
                    in {
                        QueryMode.CURRENT,
                        QueryMode.TIMELINE,
                    }
                    else 0.55
                ),
                "lexical": lexical,
                "entity": (
                    min(
                        1.0,
                        0.35
                        + 0.15
                        * len(features.entities),
                    )
                    if features.entities
                    else 0.3
                ),
                "temporal_task": (
                    1.0
                    if (
                        features.asks_current_state
                        or features.asks_conflict
                        or features.asks_timeline
                    )
                    else 0.45
                ),
                "prior": float(
                    self.priors[
                        memory_type.value
                    ]
                ),
            }

        return {
            "intent": (
                1.0
                if features.asks_procedure
                else (
                    0.15
                    if features.task_type
                    else 0.05
                )
            ),
            "lexical": lexical,
            "entity": (
                0.45
                if features.task_type
                else 0.1
            ),
            "temporal_task": (
                1.0
                if features.asks_procedure
                else (
                    0.25
                    if features.task_type
                    else 0.05
                )
            ),
            "prior": float(
                self.priors[
                    memory_type.value
                ]
            ),
        }

    def _gates(
        self,
        features: QueryFeatures,
        selected: list[MemoryType],
        reasons: dict[str, list[str]],
    ) -> None:
        def add(
            memory_type: MemoryType,
            reason: str,
        ) -> None:
            if memory_type not in selected:
                selected.append(memory_type)

            if reason not in reasons[
                memory_type.value
            ]:
                reasons[
                    memory_type.value
                ].append(reason)

        query = features.normalised_query
        hypothetical_policy = self._is_hypothetical_policy(features)

        if (
            features.asks_timeline
            and self.gates.get(
                "force_timeline_route",
                True,
            )
        ):
            add(
                MemoryType.EPISODIC,
                "timeline_gate",
            )
            add(
                MemoryType.SEMANTIC,
                "timeline_gate",
            )

        if (
            features.asks_conflict
            and self.gates.get(
                "force_conflict_route",
                True,
            )
        ):
            add(
                MemoryType.EPISODIC,
                "conflict_gate",
            )
            add(
                MemoryType.SEMANTIC,
                "conflict_gate",
            )

        if features.asks_historical_state:
            add(
                MemoryType.EPISODIC,
                "historical_gate",
            )
            if (
                self.gates.get(
                    "force_historical_semantic_companion",
                    True,
                )
                and self._matches_any(
                    query,
                    self.HISTORICAL_SEMANTIC_PATTERNS,
                )
            ):
                add(
                    MemoryType.SEMANTIC,
                    "historical_semantic_companion_gate",
                )

        if features.asks_current_state:
            add(
                MemoryType.SEMANTIC,
                "current_state_gate",
            )

        if features.asks_procedure:
            add(
                MemoryType.PROCEDURAL,
                "procedural_gate",
            )
            if (
                self.gates.get(
                    "force_procedural_semantic_companion",
                    True,
                )
                and (
                    features.task_type == "travel_planning"
                    or self._matches_any(
                        query,
                        self.PROCEDURAL_SEMANTIC_PATTERNS,
                    )
                )
            ):
                add(
                    MemoryType.SEMANTIC,
                    "procedural_semantic_companion_gate",
                )

        if (
            features.asks_procedure
            and features.task_type in {"academic_writing", "cv_writing"}
            and not hypothetical_policy
        ):
            add(
                MemoryType.SEMANTIC,
                "procedural_context_gate",
            )

        if self._is_historical_commitment(features):
            add(
                MemoryType.EPISODIC,
                "historical_commitment_gate",
            )
            add(
                MemoryType.SEMANTIC,
                "historical_commitment_gate",
            )

        if self._is_scope_validation(features):
            add(
                MemoryType.EPISODIC,
                "scope_validation_gate",
            )
            add(
                MemoryType.SEMANTIC,
                "scope_validation_gate",
            )

        if features.asks_explanation:
            add(
                MemoryType.EPISODIC,
                "explanation_gate",
            )
            add(
                MemoryType.SEMANTIC,
                "explanation_gate",
            )
            if (
                self.gates.get(
                    "force_explanation_policy_route",
                    True,
                )
                and (
                    features.task_type
                    in {
                        "cv_writing",
                        "travel_planning",
                    }
                    or self._matches_any(
                        query,
                        self.EXPLANATION_PROCEDURAL_PATTERNS,
                    )
                )
            ):
                add(
                    MemoryType.PROCEDURAL,
                    "explanation_policy_gate",
                )

        if (
            self.gates.get(
                "force_factual_selection_route",
                True,
            )
            and self._matches_any(
                query,
                self.FACTUAL_SELECTION_PATTERNS,
            )
        ):
            add(
                MemoryType.EPISODIC,
                "factual_selection_gate",
            )
            add(
                MemoryType.SEMANTIC,
                "factual_selection_gate",
            )

        if (
            features.asks_conflict
            and features.asks_current_state
            and " or "
            in f" {features.normalised_query} "
        ):
            add(
                MemoryType.PROCEDURAL,
                "current_conflict_policy_gate",
            )

        if hypothetical_policy:
            selected[:] = [
                memory_type
                for memory_type in selected
                if memory_type == MemoryType.PROCEDURAL
            ]
            add(
                MemoryType.PROCEDURAL,
                "hypothetical_policy_gate",
            )

    @staticmethod
    def _is_hypothetical_policy(
        features: QueryFeatures,
    ) -> bool:
        query = features.normalised_query
        return (
            features.asks_procedure
            and features.task_type != "travel_planning"
            and (
                query.startswith("if ")
                or " if an " in query
                or " if a " in query
                or " if the " in query
            )
            and re.search(
                r"\b(?:how should|what should|should the system|"
                r"should i say|should the answer)\b",
                query,
            )
            is not None
        )

    @classmethod
    def _is_historical_commitment(
        cls,
        features: QueryFeatures,
    ) -> bool:
        return cls._matches_any(
            features.normalised_query,
            cls.HISTORICAL_COMMITMENT_PATTERNS,
        )

    @classmethod
    def _is_scope_validation(
        cls,
        features: QueryFeatures,
    ) -> bool:
        return cls._matches_any(
            features.normalised_query,
            cls.SCOPE_VALIDATION_PATTERNS,
        )

    @staticmethod
    def _matches_any(
        text: str,
        patterns: tuple[str, ...],
    ) -> bool:
        return any(
            re.search(pattern, text) is not None
            for pattern in patterns
        )
