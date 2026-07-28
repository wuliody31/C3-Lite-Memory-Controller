from __future__ import annotations

import json
from typing import Any, Iterable

from ..schemas import (
    MemoryCandidate,
    MemoryType,
    QueryFeatures,
    QueryState,
)
from ..text_utils import overlap_ratio, tokenize


class ProceduralJsonStore:
    """Retrieve applicable procedural rules from JSON.

    RC8.3-B changes:
    - search rule name, condition, action and instruction rather than only the
      final instruction;
    - infer task families when legacy rules have empty task_type fields;
    - use controlled concept expansion for conflict, academic, CV and travel
      policies;
    - expose field-level applicability metadata for later audit/ranking;
    - keep the existing safety rule that priority/scope alone cannot admit an
      unrelated procedure.
    """

    CONCEPT_ALIASES: dict[str, tuple[str, ...]] = {
        "conflict_policy": (
            "conflict",
            "conflicting",
            "earlier",
            "later",
            "older",
            "newer",
            "previous",
            "current",
            "alternative",
            "prefer",
            "preferred",
            "rather than",
            "instead of",
            "versus",
            "supervisor-confirmed",
            "user-confirmed",
        ),
        "academic_project": (
            "academic",
            "dissertation",
            "thesis",
            "supervisor",
            "project planning",
            "msc project",
            "methodology",
            "dataset",
            "evaluation",
            "neo4j",
            "baseline",
            "benchmark",
        ),
        "cv_hr": (
            "cv",
            "resume",
            "hr",
            "application",
            "interview",
            "hr-friendly",
            "hr facing",
            "project direction",
            "positioning",
            "claim",
            "production deployment",
        ),
        "evidence_uncertainty": (
            "no evidence",
            "not stored",
            "unsupported",
            "not supported",
            "unclear",
            "unknown",
            "uncertain",
            "invent",
            "guess",
            "evidence exists",
            "supported by memory",
        ),
        "travel_transport": (
            "travel",
            "bus",
            "buses",
            "train",
            "trains",
            "ticket",
            "tickets",
            "booking",
            "booking interface",
            "refund",
            "cancellation",
            "step by step",
            "step-by-step",
        ),
        "location_recommendation": (
            "restaurant",
            "food recommendation",
            "food recommendations",
            "route recommendation",
            "route recommendations",
            "recommendation",
            "recommendations",
            "location",
            "hotel",
            "city centre",
            "city center",
        ),
        "answer_style": (
            "answer",
            "response",
            "write",
            "written",
            "wording",
            "style",
            "language",
            "explain",
            "structured",
            "concise",
            "instructions",
        ),
    }

    TASK_BY_CONCEPT: dict[str, str] = {
        "academic_project": "academic_writing",
        "cv_hr": "cv_writing",
        "travel_transport": "travel_planning",
        "location_recommendation": "travel_planning",
    }

    def __init__(
        self,
        rules: list[dict[str, Any]],
        stopwords: set[str] | None = None,
    ) -> None:
        self.rules = rules
        self.stopwords = stopwords or set()

    @classmethod
    def from_file(
        cls,
        path: str,
        stopwords: set[str] | None = None,
    ) -> "ProceduralJsonStore":
        with open(path, "r", encoding="utf-8-sig") as handle:
            data = json.load(handle)

        if isinstance(data, dict):
            data = data.get(
                "procedures",
                data.get("rules", []),
            )

        if not isinstance(data, list):
            raise ValueError(
                "Procedure JSON must be a list or contain "
                "procedures/rules."
            )

        return cls(data, stopwords)

    def retrieve(
        self,
        *,
        memory_type: MemoryType,
        state: QueryState,
        features: QueryFeatures,
        top_k: int,
        include_archived: bool = False,
    ) -> list[MemoryCandidate]:
        del include_archived

        if memory_type != MemoryType.PROCEDURAL:
            return []

        query_text = str(
            features.normalised_query
        ).strip().lower()
        query_tokens = tokenize(
            query_text,
            self.stopwords,
        )
        query_concepts = self._concepts(query_text)

        if bool(getattr(features, "asks_conflict", False)):
            query_concepts.add("conflict_policy")
        if bool(getattr(features, "asks_explanation", False)):
            query_concepts.add("answer_style")
        if (
            "not stored" in query_text
            or "no evidence" in query_text
            or "unclear" in query_text
            or "invent" in query_text
        ):
            query_concepts.add("evidence_uncertainty")

        output: list[tuple[float, MemoryCandidate]] = []

        for rule in self.rules:
            if not isinstance(rule, dict):
                continue
            if not bool(rule.get("enabled", True)):
                continue

            rule_user = str(
                rule.get("user_id", state.user_id)
            )
            if rule_user not in {
                state.user_id,
                "*",
                "global",
            }:
                continue

            rule_id = self._first_text(
                rule,
                ("rule_id", "memory_id", "procedure_id", "id"),
            )
            instruction = self._first_text(
                rule,
                (
                    "instruction",
                    "text",
                    "rule",
                    "content",
                    "description",
                    "action",
                ),
            )
            if not rule_id or not instruction:
                continue

            name = self._first_text(rule, ("name", "title"))
            condition = self._first_text(
                rule,
                ("condition", "when", "applicability"),
            )
            action = self._first_text(
                rule,
                (
                    "action",
                    "recommended_action",
                    "response_rule",
                    "instruction",
                    "text",
                ),
            )

            declared_triggers = self._string_list(
                rule.get("triggers", [])
            )
            declared_tasks = self._string_list(
                rule.get("task_types", [])
            )
            if rule.get("task_type"):
                declared_tasks.append(str(rule["task_type"]))

            rule_search_text = " ".join(
                part
                for part in (
                    name,
                    condition,
                    action,
                    instruction,
                    " ".join(declared_triggers),
                    " ".join(declared_tasks),
                    str(rule.get("scope", "")),
                )
                if part
            ).lower()
            rule_concepts = self._concepts(rule_search_text)

            inferred_tasks = [
                self.TASK_BY_CONCEPT[concept]
                for concept in sorted(rule_concepts)
                if concept in self.TASK_BY_CONCEPT
            ]
            tasks = list(
                dict.fromkeys(
                    [
                        *declared_tasks,
                        *inferred_tasks,
                    ]
                )
            )

            concept_terms = self._concept_expansion_terms(
                rule_concepts
            )
            retrieval_triggers = list(
                dict.fromkeys(
                    [
                        *declared_triggers,
                        *(
                            [name]
                            if name
                            else []
                        ),
                        *(
                            [condition]
                            if condition
                            else []
                        ),
                        *concept_terms,
                    ]
                )
            )

            lexical = overlap_ratio(
                query_tokens,
                tokenize(
                    rule_search_text,
                    self.stopwords,
                ),
            )
            name_match = overlap_ratio(
                query_tokens,
                tokenize(name, self.stopwords),
            )
            condition_match = overlap_ratio(
                query_tokens,
                tokenize(condition, self.stopwords),
            )
            action_match = overlap_ratio(
                query_tokens,
                tokenize(
                    " ".join(
                        part
                        for part in (
                            action,
                            instruction,
                        )
                        if part
                    ),
                    self.stopwords,
                ),
            )

            matched_triggers = [
                trigger
                for trigger in retrieval_triggers
                if trigger
                and trigger.lower() in query_text
            ]
            trigger_hits = min(
                1.0,
                float(len(matched_triggers)),
            )

            task_match = (
                1.0
                if getattr(features, "task_type", None)
                and getattr(features, "task_type") in tasks
                else 0.0
            )

            shared_concepts = (
                query_concepts
                & rule_concepts
            )
            concept_match = (
                len(shared_concepts)
                / max(1, len(query_concepts))
                if query_concepts
                else 0.0
            )

            intent_match = self._intent_match(
                features=features,
                query_concepts=query_concepts,
                rule_concepts=rule_concepts,
            )

            scope_score = self._scope(
                str(rule.get("scope", "global"))
            )
            priority = min(
                1.0,
                max(
                    0.0,
                    float(rule.get("priority", 5))
                    / 10.0,
                ),
            )

            preliminary_score = min(
                1.0,
                (
                    0.18 * lexical
                    + 0.10 * name_match
                    + 0.18 * condition_match
                    + 0.14 * action_match
                    + 0.08 * trigger_hits
                    + 0.12 * task_match
                    + 0.12 * concept_match
                    + 0.05 * intent_match
                    + 0.02 * scope_score
                    + 0.01 * priority
                ),
            )

            # Intent-based rescue should be strong enough to enter the raw
            # pool, but it still requires a matching rule family.
            if intent_match > 0.0:
                preliminary_score = max(
                    preliminary_score,
                    0.52 + 0.18 * concept_match,
                )

            always_apply = bool(
                rule.get("always_apply", False)
            )
            has_direct_match = bool(
                lexical > 0.0
                or name_match > 0.0
                or condition_match > 0.0
                or action_match > 0.0
                or trigger_hits > 0.0
                or task_match > 0.0
                or concept_match > 0.0
                or intent_match > 0.0
            )

            # Scope and priority remain support/tie-break signals only.
            if not always_apply and not has_direct_match:
                continue

            effective_task_type = (
                str(rule.get("task_type"))
                if rule.get("task_type")
                else (
                    tasks[0]
                    if tasks
                    else None
                )
            )

            item = MemoryCandidate(
                memory_id=rule_id,
                memory_type=MemoryType.PROCEDURAL,
                text=instruction,
                user_id=state.user_id,
                status="current",
                confidence=float(
                    rule.get("confidence", 1.0)
                ),
                importance=priority,
                authority=str(
                    rule.get(
                        "authority",
                        "user_confirmed",
                    )
                ),
                metadata={
                    **rule,
                    "task_type": effective_task_type,
                    "task_types": tasks,
                    "declared_triggers": declared_triggers,
                    "triggers": retrieval_triggers,
                    "matched_triggers": matched_triggers,
                    "derived_concepts": sorted(rule_concepts),
                    "matched_concepts": sorted(shared_concepts),
                    "retrieval_search_text": rule_search_text,
                    "preliminary_procedural_score": (
                        preliminary_score
                    ),
                    "scope_score": scope_score,
                    "task_match": task_match,
                    "lexical_match": lexical,
                    "name_match": name_match,
                    "condition_match": condition_match,
                    "action_match": action_match,
                    "concept_match": concept_match,
                    "intent_match": intent_match,
                    "trigger_hits": trigger_hits,
                    "direct_match": has_direct_match,
                    "always_apply": always_apply,
                    "retrieval_version": "rc8.3b",
                },
            )
            output.append(
                (preliminary_score, item)
            )

        output.sort(
            key=lambda entry: (
                entry[0],
                float(
                    entry[1].metadata.get(
                        "intent_match",
                        0.0,
                    )
                ),
                float(
                    entry[1].metadata.get(
                        "concept_match",
                        0.0,
                    )
                ),
                float(
                    entry[1].metadata.get(
                        "priority",
                        0.0,
                    )
                ),
                entry[1].memory_id,
            ),
            reverse=True,
        )
        return [
            item
            for _, item in output[:top_k]
        ]

    def _intent_match(
        self,
        *,
        features: QueryFeatures,
        query_concepts: set[str],
        rule_concepts: set[str],
    ) -> float:
        if (
            bool(getattr(features, "asks_conflict", False))
            and "conflict_policy" in rule_concepts
        ):
            return 1.0

        if (
            bool(getattr(features, "asks_explanation", False))
            and (
                "answer_style" in rule_concepts
                or bool(
                    query_concepts
                    & rule_concepts
                    & {
                        "academic_project",
                        "cv_hr",
                        "travel_transport",
                        "location_recommendation",
                    }
                )
            )
        ):
            return 1.0

        if (
            bool(getattr(features, "asks_procedure", False))
            and bool(query_concepts & rule_concepts)
        ):
            return 0.85

        return 0.0

    @classmethod
    def _concepts(
        cls,
        text: str,
    ) -> set[str]:
        lowered = str(text or "").lower()
        concepts: set[str] = set()

        for concept, aliases in (
            cls.CONCEPT_ALIASES.items()
        ):
            if any(
                alias in lowered
                for alias in aliases
            ):
                concepts.add(concept)

        return concepts

    @classmethod
    def _concept_expansion_terms(
        cls,
        concepts: Iterable[str],
    ) -> list[str]:
        output: list[str] = []
        for concept in concepts:
            output.extend(
                cls.CONCEPT_ALIASES.get(
                    concept,
                    (),
                )
            )
        return list(dict.fromkeys(output))

    @staticmethod
    def _first_text(
        value: dict[str, Any],
        keys: tuple[str, ...],
    ) -> str:
        for key in keys:
            item = value.get(key)
            if item is None:
                continue
            text = str(item).strip()
            if text:
                return text
        return ""

    @staticmethod
    def _string_list(
        value: Any,
    ) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            return [
                str(item)
                for item in value
                if str(item).strip()
            ]
        if isinstance(value, dict):
            output: list[str] = []
            for key, item in value.items():
                if isinstance(item, bool):
                    if item:
                        output.append(str(key))
                elif isinstance(item, list):
                    output.extend(
                        str(part)
                        for part in item
                    )
                elif item not in (None, ""):
                    output.append(str(item))
            return output
        return [str(value)]

    @staticmethod
    def _scope(scope: str) -> float:
        return {
            "task": 1.0,
            "task-specific": 1.0,
            "project": 0.85,
            "project-specific": 0.85,
            "user": 0.70,
            "user-global": 0.70,
            "global": 0.50,
            "system": 0.30,
        }.get(scope.lower(), 0.50)

    def close(self) -> None:
        return None
