from __future__ import annotations

from pyexpat import features
import re
from typing import Any

from .schemas import MemoryType, QueryFeatures, QueryMode, RouteDecision


class RoutePlanner:
    """Context-adaptive, training-free multi-memory route planner.

    Important design rules:
    1. ``task_type`` is weak domain context and never forces procedural memory.
    2. Structural gates are driven by query intent.
    3. Hypothetical policy questions should not retrieve historical facts merely
       because the condition mentions an old/new conflict.
    """

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
            "adopted",
            "history",
            "over time",
        },
        MemoryType.SEMANTIC: {
            "current",
            "currently",
            "what is",
            "what are",
            "which",
            "uses",
            "preference",
            "focus",
            "role",
            "main",
            "now",
            "selected",
            "chosen",
            "baseline",
            "baselines",
        },
        MemoryType.PROCEDURAL: {
            "how should",
            "what should",
            "should i",
            "should my",
            "format",
            "write",
            "follow",
            "must",
            "step",
            "style",
            "procedure",
            "prioritise",
            "prioritize",
            "choose between",
        },
    }

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

    def __init__(self, config: dict[str, Any]):
        routing = config["routing"]
        self.thresholds = routing["thresholds"]
        self.weights = routing["utility_weights"]
        self.priors = routing["priors"]
        self.gates = routing.get("structural_gates", {})

    def plan(self, features: QueryFeatures) -> RouteDecision:
        raw_scores: dict[MemoryType, float] = {}
        reasons = {memory_type.value: [] for memory_type in MemoryType}

        for memory_type in MemoryType:
            signals = self._signals(memory_type, features)
            score = sum(
                float(self.weights[key]) * value
                for key, value in signals.items()
            )
            raw_scores[memory_type] = min(1.0, max(0.0, score))

            for key, value in signals.items():
                if key != "prior" and value >= 0.60:
                    reasons[memory_type.value].append(key)

        selected = [
            memory_type
            for memory_type, score in raw_scores.items()
            if score >= float(self.thresholds[memory_type.value])
        ]

        self._apply_structural_gates(
            features,
            selected,
            reasons,
        )

        if (
            not selected
            and self.gates.get("fallback_top_route", True)
        ):
            top_route = max(raw_scores, key=raw_scores.get)
            selected.append(top_route)
            reasons[top_route.value].append("fallback_top_route")

        ordered_selected = [
            memory_type
            for memory_type in MemoryType
            if memory_type in set(selected)
        ]

        cleaned_reasons = {
            key: list(dict.fromkeys(value))
            for key, value in reasons.items()
        }

        return RouteDecision(
            ordered_selected,
            {
                memory_type.value: round(
                    raw_scores[memory_type],
                    6,
                )
                for memory_type in MemoryType
            },
            cleaned_reasons,
        )

    def _signals(
        self,
        memory_type: MemoryType,
        features: QueryFeatures,
    ) -> dict[str, float]:
        lexical = self._lexical_score(
            features.normalised_query,
            set(features.tokens),
            self.TERMS[memory_type],
        )
        entity_signal = (
            min(1.0, 0.20 + 0.15 * len(features.entities))
            if features.entities
            else 0.10
        )
        hypothetical_policy = self._is_hypothetical_policy(features)

        if memory_type == MemoryType.EPISODIC:
            if hypothetical_policy:
                intent = 0.05
                temporal_task = 0.05
                entity = min(entity_signal, 0.30)
            else:
                intent = (
                    1.0
                    if features.query_mode
                    in {QueryMode.HISTORICAL, QueryMode.TIMELINE}
                    else 0.08
                )
                temporal_task = (
                    1.0
                    if (
                        features.temporal_expressions
                        or features.asks_historical_state
                        or features.asks_timeline
                    )
                    else 0.05
                )
                entity = entity_signal

            return {
                "intent": intent,
                "lexical": lexical,
                "entity": entity,
                "temporal_task": temporal_task,
                "prior": float(self.priors[memory_type.value]),
            }

        if memory_type == MemoryType.SEMANTIC:
            if hypothetical_policy:
                intent = 0.25
                temporal_task = 0.20
                entity = min(entity_signal, 0.35)
            else:
                intent = (
                    1.0
                    if features.query_mode
                    in {QueryMode.CURRENT, QueryMode.TIMELINE}
                    else 0.65
                )
                temporal_task = (
                    1.0
                    if (
                        features.asks_current_state
                        or features.asks_conflict
                        or features.asks_timeline
                    )
                    else 0.45
                )
                entity = min(1.0, entity_signal + 0.15)

            return {
                "intent": intent,
                "lexical": lexical,
                "entity": entity,
                "temporal_task": temporal_task,
                "prior": float(self.priors[memory_type.value]),
            }

        # Procedural memory is driven primarily by procedural intent.
        # task_type contributes only a weak domain-context signal.
        return {
            "intent": 1.0 if features.asks_procedure else 0.05,
            "lexical": lexical,
            "entity": 0.25 if features.task_type else 0.05,
            "temporal_task": (
                1.0
                if features.asks_procedure
                else (0.20 if features.task_type else 0.05)
            ),
            "prior": float(self.priors[memory_type.value]),
        }

    def _apply_structural_gates(
        self,
        features: QueryFeatures,
        selected: list[MemoryType],
        reasons: dict[str, list[str]],
    ) -> None:
        def add(memory_type: MemoryType, reason: str) -> None:
            if memory_type not in selected:
                selected.append(memory_type)
            reasons[memory_type.value].append(reason)

        hypothetical_policy = self._is_hypothetical_policy(features)
        normative_alternative = (
            features.asks_conflict
            and features.asks_procedure
            and not hypothetical_policy
        )
        current_conflict = (
            features.asks_conflict
            and features.asks_current_state
            and not hypothetical_policy
        )

        if (
            features.asks_timeline
            and self.gates.get("force_timeline_route", True)
            and not hypothetical_policy
        ):
            add(MemoryType.EPISODIC, "timeline_gate")
            add(MemoryType.SEMANTIC, "timeline_gate")

        if (
            features.asks_conflict
            and self.gates.get("force_conflict_route", True)
            and not hypothetical_policy
        ):
            add(MemoryType.EPISODIC, "conflict_gate")
            add(MemoryType.SEMANTIC, "conflict_gate")

        if (
            features.asks_historical_state
            and not hypothetical_policy
        ):
            add(MemoryType.EPISODIC, "historical_gate")

        if features.asks_current_state:
            add(MemoryType.SEMANTIC, "current_state_gate")

        if features.asks_procedure:
            add(MemoryType.PROCEDURAL, "procedural_gate")

        # A current A-or-B query needs the policy that decides which state wins.
        if current_conflict:
            add(
                MemoryType.PROCEDURAL,
                "current_conflict_policy_gate",
            )

        # A normative A-or-B decision needs history, current facts, and policy.
        if normative_alternative:
            add(
                MemoryType.EPISODIC,
                "normative_alternative_gate",
            )
            add(
                MemoryType.SEMANTIC,
                "normative_alternative_gate",
            )
            add(
                MemoryType.PROCEDURAL,
                "normative_alternative_gate",
            )

        # Historical commitments often require both the original decision and
        # the currently retained semantic state.
        if self._is_historical_commitment(features):
            add(
                MemoryType.EPISODIC,
                "historical_commitment_gate",
            )
            add(
                MemoryType.SEMANTIC,
                "historical_commitment_gate",
            )

        # User-specific CV/academic procedure questions usually require the
        # applicable rule plus the user's current factual context.
        if (
            features.asks_procedure
            and features.task_type
            in {"academic_writing", "cv_writing"}
            and not hypothetical_policy
        ):
            add(
                MemoryType.SEMANTIC,
                "procedural_context_gate",
            )

        # Explanation requests require evidence-bearing episodic and semantic
        # memories. Procedural memory is added independently when requested.
        if features.asks_explanation:
            add(MemoryType.EPISODIC, "explanation_gate")
            add(MemoryType.SEMANTIC, "explanation_gate")

        # Scope/claim validation benefits from the current fact and the
        # decision/history that established it.
        if self._is_scope_validation(features):
            add(
                MemoryType.EPISODIC,
                "scope_validation_gate",
            )
            add(
                MemoryType.SEMANTIC,
                "scope_validation_gate",
            )

        # A hypothetical policy question such as
        # "If an older memory conflicts..., how should the system answer?"
        # asks for the rule itself, not retrieval of a real historical case.
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
    def _lexical_score(
        query: str,
        tokens: set[str],
        terms: set[str],
    ) -> float:
        matches = 0

        for term in terms:
            if " " in term:
                matched = term in query
            else:
                matched = term in tokens

            matches += int(matched)

        return min(1.0, matches / 2.0)

    @staticmethod
    def _is_hypothetical_policy(
        features: QueryFeatures,
    ) -> bool:
        query = features.normalised_query

    # Response-policy question:
    # "What should I say if there is no evidence?"
        
        response_policy = (
            re.search(
                r"^(?:"
                r"what should i say|"
                r"how should i answer|"
                r"what should the system say|"
                r"how should the system answer"
                r")\s+if\b",
                query,
            )
            is not None
        )

    # Condition-first policy:
    # "If an older memory conflicts..., how should the system answer?"
        condition_first_policy = (
            query.startswith("if ")
            and re.search(
                r"\b(?:"
                r"how should|"
                r"what should|"
                r"should the system|"
                r"should i say|"
                r"should the answer"
                r")\b",
                query,
            )
            is not None
        )

        return (
            features.asks_procedure
            and (
                response_policy
                or condition_first_policy
            )
        )

    @classmethod
    def _is_historical_commitment(
        cls,
        features: QueryFeatures,
    ) -> bool:
        return any(
            re.search(
                pattern,
                features.normalised_query,
            )
            is not None
            for pattern in cls.HISTORICAL_COMMITMENT_PATTERNS
        )

    @classmethod
    def _is_scope_validation(
        cls,
        features: QueryFeatures,
    ) -> bool:
        return any(
            re.search(
                pattern,
                features.normalised_query,
            )
            is not None
            for pattern in cls.SCOPE_VALIDATION_PATTERNS
        )


