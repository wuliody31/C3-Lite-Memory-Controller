from __future__ import annotations

import json
from typing import Any

from ..schemas import (
    MemoryCandidate,
    MemoryType,
    QueryFeatures,
    QueryState,
)
from ..text_utils import overlap_ratio, tokenize


class ProceduralJsonStore:
    """Retrieve task-applicable procedural rules from JSON.

    Rules are filtered before ranking so that high priority or broad project
    scope alone cannot make an unrelated rule enter the evidence set.
    """

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
        with open(path, "r", encoding="utf-8") as handle:
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
        del include_archived  # Procedural rules use enabled/disabled status.

        if memory_type != MemoryType.PROCEDURAL:
            return []

        query_tokens = tokenize(
            features.normalised_query,
            self.stopwords,
        )
        query_text = features.normalised_query.lower()
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

            rule_id = str(
                rule.get("rule_id")
                or rule.get("memory_id")
                or rule.get("id")
                or ""
            ).strip()
            text = str(
                rule.get(
                    "instruction",
                    rule.get("text", ""),
                )
            ).strip()
            if not rule_id or not text:
                continue

            triggers = self._string_list(
                rule.get("triggers", [])
            )
            tasks = self._string_list(
                rule.get("task_types", [])
            )
            if rule.get("task_type"):
                tasks.append(str(rule["task_type"]))
            tasks = list(dict.fromkeys(tasks))

            searchable_text = " ".join(
                [text, *triggers, *tasks]
            )
            lexical = overlap_ratio(
                query_tokens,
                tokenize(searchable_text, self.stopwords),
            )

            matched_triggers = [
                trigger
                for trigger in triggers
                if trigger.lower() in query_text
            ]
            trigger_hits = min(
                1.0,
                float(len(matched_triggers)),
            )

            task_match = (
                1.0
                if features.task_type
                and features.task_type in tasks
                else 0.0
            )
            scope_score = self._scope(
                str(rule.get("scope", "global"))
            )
            priority = min(
                1.0,
                max(
                    0.0,
                    float(rule.get("priority", 5)) / 10.0,
                ),
            )

            preliminary_score = (
                0.35 * lexical
                + 0.25 * trigger_hits
                + 0.20 * task_match
                + 0.10 * scope_score
                + 0.10 * priority
            )

            always_apply = bool(
                rule.get("always_apply", False)
            )
            has_direct_match = (
                task_match > 0.0
                or trigger_hits > 0.0
                or lexical > 0.0
            )

            # Critical RC3 filter: priority and scope are tie-break/support
            # signals only. They must not retrieve an otherwise unrelated rule.
            if not always_apply and not has_direct_match:
                continue

            item = MemoryCandidate(
                memory_id=rule_id,
                memory_type=MemoryType.PROCEDURAL,
                text=text,
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
                    "preliminary_procedural_score": (
                        preliminary_score
                    ),
                    "scope_score": scope_score,
                    "task_match": task_match,
                    "lexical_match": lexical,
                    "trigger_hits": trigger_hits,
                    "matched_triggers": matched_triggers,
                    "direct_match": has_direct_match,
                    "always_apply": always_apply,
                },
            )
            output.append((preliminary_score, item))

        output.sort(
            key=lambda entry: (
                entry[0],
                float(entry[1].metadata.get("priority", 0)),
                entry[1].memory_id,
            ),
            reverse=True,
        )
        return [item for _, item in output[:top_k]]

    @staticmethod
    def _string_list(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            return [str(item) for item in value]
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