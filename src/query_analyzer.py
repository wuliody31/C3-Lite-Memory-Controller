from __future__ import annotations

import re
from typing import Any

from .schemas import QueryFeatures, QueryMode
from .text_utils import normalise_text, split_clauses, tokenize


class QueryAnalyzer:
    """Rule-based, training-free query analysis for C3-Lite.

    Entity extraction is deliberately separated from temporal/task concepts.
    Words such as ``timeline`` and ``current`` are query-mode signals rather
    than entities.
    """

    TEMPORAL_CURRENT = {
        "current", "currently", "now", "latest", "present", "today",
        "目前", "现在", "当前", "最新",
    }
    TEMPORAL_HISTORICAL = {
        "previously", "originally", "before", "earlier", "past", "last time",
        "过去", "之前", "最初", "原来", "以前", "上一次",
    }
    TEMPORAL_TIMELINE = {
        "over time", "change", "changed", "changes", "changing",
        "evolve", "evolved", "evolution", "timeline", "history",
        "变化", "改变", "演变", "过程", "时间线",
    }
    PROCEDURAL_SIGNALS = {
        "how should", "what should", "format", "write", "answer style",
        "step by step", "follow", "must", "instruction", "procedure",
        "怎么", "应该如何", "格式", "写成", "一步步", "必须", "遵循",
    }
    EXPLANATION_SIGNALS = {
        "explain", "why", "which memories", "evidence", "support",
        "解释", "为什么", "哪些记忆", "证据", "支持",
    }
    CONFLICT_SIGNALS = {
        "conflict", "outdated", "older", "more current", "instead of",
        "versus", " vs ", "冲突", "过时", "旧", "哪个更当前", "还是",
    }

    TASK_RULES = {
        "academic_writing": {
            "dissertation", "methodology", "project plan", "supervisor",
            "论文", "导师",
        },
        "cv_writing": {
            "cv", "resume", "interview", "hr", "简历", "面试",
        },
        "travel_planning": {
            "travel", "hotel", "train", "bus", "restaurant", "itinerary",
            "旅行", "酒店", "火车",
        },
        "code_generation": {
            "code", "python", "implementation", "function", "代码", "实现",
        },
    }

    ENTITY_PATTERN = re.compile(
        r"\b[A-Z][A-Za-z0-9_.+-]*(?:\s+[A-Z][A-Za-z0-9_.+-]*){0,3}\b"
    )

    QUESTION_WORDS = {
        "how", "what", "when", "where", "why", "which", "who", "whom",
        "whose", "can", "could", "would", "should", "is", "are", "was",
        "were", "do", "does", "did",
    }

    NON_ENTITY_ALIAS_KEYS = {
        "current", "historical", "timeline", "past", "present",
    }

    ENTITY_HEADS = {
        "project", "scope", "memory", "controller", "model", "backbone",
        "database", "dataset", "benchmark", "algorithm", "system", "agent",
        "neo4j", "ollama", "llama", "qwen", "locomo", "c3-lite", "c3",
        "论文", "项目", "范围", "记忆", "控制器", "模型", "数据库",
        "数据集", "算法", "系统",
    }

    def __init__(self, config: dict[str, Any]):
        self.config = config
        stopwords = config["query_analysis"].get("stopwords", [])
        self.stopwords = {str(item).lower() for item in stopwords}
        self.aliases = config["query_analysis"].get("aliases", {})

    def analyse(self, query: str) -> QueryFeatures:
        normalised = normalise_text(query)
        tokens = tokenize(normalised, self.stopwords)

        asks_current = self._contains_any(normalised, self.TEMPORAL_CURRENT)
        asks_historical = self._contains_any(normalised, self.TEMPORAL_HISTORICAL)
        asks_timeline = self._contains_any(normalised, self.TEMPORAL_TIMELINE)
        asks_procedure = self._contains_any(normalised, self.PROCEDURAL_SIGNALS)
        asks_explanation = self._contains_any(normalised, self.EXPLANATION_SIGNALS)
        asks_conflict = self._contains_any(normalised, self.CONFLICT_SIGNALS)

        query_mode = self._resolve_mode(
            asks_current=asks_current,
            asks_historical=asks_historical,
            asks_timeline=asks_timeline,
            asks_procedure=asks_procedure,
        )
        task_type = self._detect_task_type(normalised)
        entities = self._extract_entities(query, normalised, tokens)
        temporal_expressions = self._extract_temporal_expressions(query)
        information_needs = self._extract_information_needs(query, query_mode)

        return QueryFeatures(
            normalised_query=normalised,
            tokens=tokens,
            entities=entities,
            temporal_expressions=temporal_expressions,
            query_mode=query_mode,
            task_type=task_type,
            asks_current_state=asks_current,
            asks_historical_state=asks_historical,
            asks_timeline=asks_timeline,
            asks_procedure=asks_procedure,
            asks_explanation=asks_explanation,
            asks_conflict=asks_conflict,
            information_needs=information_needs,
        )

    @staticmethod
    def _contains_any(text: str, signals: set[str]) -> bool:
        return any(signal in text for signal in signals)

    @staticmethod
    def _resolve_mode(
        *,
        asks_current: bool,
        asks_historical: bool,
        asks_timeline: bool,
        asks_procedure: bool,
    ) -> QueryMode:
        if asks_timeline or (asks_current and asks_historical):
            return QueryMode.TIMELINE
        if asks_historical:
            return QueryMode.HISTORICAL
        if asks_current:
            return QueryMode.CURRENT
        if asks_procedure:
            return QueryMode.PROCEDURAL
        return QueryMode.ATEMPORAL

    def _detect_task_type(self, text: str) -> str | None:
        scores: dict[str, int] = {}
        for task_type, keywords in self.TASK_RULES.items():
            scores[task_type] = sum(1 for keyword in keywords if keyword in text)
        if not scores or max(scores.values()) == 0:
            return None
        return max(scores, key=scores.get)

    def _extract_entities(
        self,
        original_query: str,
        normalised_query: str,
        tokens: list[str],
    ) -> list[str]:
        entities: set[str] = set()

        # Capitalised named entities, excluding sentence-initial question words.
        for match in self.ENTITY_PATTERN.findall(original_query):
            cleaned = " ".join(match.strip().split())
            if cleaned.lower() not in self.QUESTION_WORDS:
                entities.add(cleaned)

        # Preserve the alias surface that appears in the query. This prevents
        # "project" from being rewritten as the artificial entity
        # "dissertation".
        for canonical, aliases in self.aliases.items():
            canonical_text = str(canonical).lower()
            if canonical_text in self.NON_ENTITY_ALIAS_KEYS:
                continue

            forms = [str(canonical), *[str(item) for item in aliases]]
            for form in forms:
                form_lower = form.lower()
                if self._contains_phrase(normalised_query, form_lower):
                    entities.add(form_lower)
                    break

        # Recover useful technical/domain noun phrases without a heavy NLP
        # dependency, e.g. "project scope" and "memory controller".
        raw_tokens = tokenize(normalised_query)
        for index, token in enumerate(raw_tokens):
            if token not in self.ENTITY_HEADS:
                continue

            entities.add(token)

            if index > 0:
                previous = raw_tokens[index - 1]
                if previous not in self.stopwords and previous not in self.QUESTION_WORDS:
                    entities.add(f"{previous} {token}")

            if index + 1 < len(raw_tokens):
                following = raw_tokens[index + 1]
                if following not in self.stopwords and following not in self.QUESTION_WORDS:
                    entities.add(f"{token} {following}")

        # Preserve explicit technical identifiers.
        for token in tokens:
            if (
                any(character.isdigit() for character in token)
                or "_" in token
                or token in {"neo4j", "ollama", "locomo", "c3", "c3-lite"}
            ):
                entities.add(token)

        filtered = {
            entity.strip()
            for entity in entities
            if entity.strip()
            and entity.strip().lower() not in self.QUESTION_WORDS
            and entity.strip().lower() not in self.NON_ENTITY_ALIAS_KEYS
        }

        return sorted(filtered, key=lambda item: (len(item.split()), item.lower()))

    @staticmethod
    def _contains_phrase(text: str, phrase: str) -> bool:
        if not phrase:
            return False
        if re.fullmatch(r"[a-z0-9_.+-]+", phrase):
            return re.search(rf"\b{re.escape(phrase)}\b", text) is not None
        return phrase in text

    @staticmethod
    def _extract_temporal_expressions(query: str) -> list[str]:
        patterns = [
            (
                r"\b\d{1,2}\s+(?:January|February|March|April|May|June|"
                r"July|August|September|October|November|December)\s+\d{4}\b"
            ),
            r"\b\d{4}-\d{2}-\d{2}\b",
            r"\b(?:yesterday|today|tomorrow|last week|last month|last year)\b",
            r"\b(?:before|after|during|since|until)\b[^,?.;]*",
            r"(?:昨天|今天|明天|上周|上个月|去年|之前|之后|期间)[^，。；？]*",
        ]
        matches: list[str] = []
        for pattern in patterns:
            matches.extend(re.findall(pattern, query, flags=re.IGNORECASE))
        return list(dict.fromkeys(match.strip() for match in matches if match.strip()))

    def _extract_information_needs(self, query: str, mode: QueryMode) -> list[str]:
        clauses = split_clauses(query)
        if not clauses:
            clauses = [query.strip()]

        needs: list[str] = []
        for clause in clauses:
            cleaned_tokens = tokenize(clause, self.stopwords)
            if cleaned_tokens:
                needs.append(" ".join(cleaned_tokens))

        if mode == QueryMode.TIMELINE:
            needs.extend(["earlier state", "current state"])
        elif mode == QueryMode.HISTORICAL:
            needs.append("historical event or state")
        elif mode == QueryMode.CURRENT:
            needs.append("current valid state")
        elif mode == QueryMode.PROCEDURAL:
            needs.append("applicable procedure or rule")

        return list(dict.fromkeys(needs))
