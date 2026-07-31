from __future__ import annotations

import re
from typing import Any

from .schemas import QueryFeatures, QueryMode
from .text_utils import normalise_text, split_clauses, tokenize


class QueryAnalyzer:
    """Training-free query analysis for C3-Lite.

    The analyser separates:
    - domain/task context;
    - temporal intent;
    - procedural intent;
    - explanation intent;
    - conflict/alternative intent.

    Task type is only contextual metadata. It must not by itself force
    procedural-memory routing.
    """

    TEMPORAL_CURRENT = {
        "current",
        "currently",
        "now",
        "latest",
        "present",
        "today",
    }

    TEMPORAL_HISTORICAL = {
        "previously",
        "originally",
        "before",
        "earlier",
        "past",
        "last time",
        "older",
    }

    TEMPORAL_TIMELINE = {
        "over time",
        "change",
        "changed",
        "changes",
        "changing",
        "evolve",
        "evolved",
        "evolution",
        "timeline",
        "history",
    }

    # Strong procedural phrases. Broad factual-selection requests such as
    # "what should I mention in my CV" are handled separately and do not
    # automatically imply procedural-memory use.
    PROCEDURAL_SIGNALS = {
        "how should",
        "how should you",
        "how should the system",
        "what should the system",
        "should i",
        "should you",
        "should not",
        "how do i",
        "what steps",
        "step by step",
        "answer style",
        "answer in",
        "friendly language",
        "hr-friendly",
        "format",
        "instruction",
        "procedure",
        "must",
        "follow",
        "prioritise",
        "prioritize",
        "choose between",
        "which should",
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
    )

    PROCEDURAL_POLICY_PATTERNS = (
        r"\bwhat\s+should\s+i\s+"
        r"(?:do|say|answer|write|claim|use|follow|handle|respond)\b",
        r"\bwhat\s+should\s+(?:you|the\s+system|the\s+answer|"
        r"the\s+response)\s+"
        r"(?:do|say|answer|write|claim|use|follow|handle|respond|invent)\b",
        r"\bhow\s+should\s+(?:i|you|we|the\s+system|the\s+answer|"
        r"the\s+response)\b",
        r"\bif\b.+\b(?:how\s+should|what\s+should|should\s+the\s+system|"
        r"should\s+you)\b",
        r"\banswer\s+in\b",
    )

    TEMPORAL_CONTINUITY_PATTERNS = (
        r"\bwas\b.+\balways\b",
        r"\bwere\b.+\balways\b",
        r"\bhave\b.+\balways\b",
        r"\bhas\b.+\balways\b",
    )


    EXPLANATION_SIGNALS = {
        "explain",
        "why",
        "which memories",
        "what evidence",
        "which evidence",
        "evidence supports",
        "evidence support",
        "supporting evidence",
        "what supports",
    }

    NEGATIVE_EVIDENCE_SIGNALS = {
        "no evidence",
        "without evidence",
        "without sufficient evidence",
        "insufficient evidence",
        "lack of evidence",
        "no stored evidence",
    }

    CONFLICT_SIGNALS = {
        "conflict",
        "conflicts",
        "conflicting",
        "outdated",
        "more current",
        "instead of",
        "versus",
        " vs ",
        "rather than",
        "no longer",
        "completely removed",
        "removed from",
    }

    HISTORICAL_DECISION_PATTERNS = (
        r"\bdid\s+i\s+(?:decide|choose|select|adopt|agree)\b",
        r"\bwhat\s+did\s+i\s+(?:decide|choose|select|adopt)\b",
        r"\bhave\s+i\s+(?:selected|chosen|decided|adopted)\b",
        r"\bwhat\b.+\bhave\s+i\s+(?:selected|chosen|adopted)\b",
        r"\bwhich\b.+\bhave\s+i\s+(?:selected|chosen|adopted)\b",
        r"\bdid\s+i\s+(?:ask|request|want|mention)\b",
        r"\bwhat\s+happened\b",
    )

    CURRENT_COMMITMENT_PATTERNS = (
        r"\bhave\s+i\s+(?:selected|chosen|adopted)\b",
        r"\bwhat\b.+\bhave\s+i\s+(?:selected|chosen|adopted)\b",
        r"\bwhich\b.+\bhave\s+i\s+(?:selected|chosen|adopted)\b",
    )

    # Questions asking for the members of a selected set are not
    # timeline questions. "Have I selected" describes the requested
    # set, rather than an earlier-to-current state transition.
    SELECTION_SET_PATTERNS = (
        r"\b(?:what|which)\s+"
        r"(?:baselines?|methods?|models?|tools?|datasets?|"
        r"metrics?|options?|components?)\s+"
        r"(?:have\s+i|did\s+i)\s+"
        r"(?:selected|select|chosen|choose|adopted|adopt)\b",
        r"\b(?:what|which)\s+"
        r"(?:baselines?|methods?|models?|tools?|datasets?|"
        r"metrics?|options?|components?)\s+"
        r"(?:are|were)\s+(?:selected|chosen|adopted)\b",
    )

    ALTERNATIVE_PATTERNS = (
        r"\b(?:is|are|was|were|should|did|do|does|can|could|would)\b.+\bor\b.+",
        r"\b(?:prioritise|prioritize|choose|select|prefer)\b.+\bor\b.+",
        r"\b(?:currently|now|still)\b.+\bor\b.+",
    )

    # Domain detection only. Avoid ambiguous bare words such as "train".
    TASK_RULES = {
        "academic_writing": {
            "dissertation",
            "methodology",
            "project plan",
            "supervisor",
        },
        "cv_writing": {
            "cv",
            "resume",
            "interview",
            "hr",
        },
        "travel_planning": {
            "travel",
            "hotel",
            "train station",
            "train ticket",
            "railway",
            "rail",
            "bus",
            "restaurant",
            "itinerary",
        },
        "code_generation": {
            "code",
            "python",
            "implementation",
            "function",
        },
    }

    ENTITY_PATTERN = re.compile(
        r"\b[A-Z][A-Za-z0-9_.+-]*(?:\s+[A-Z][A-Za-z0-9_.+-]*){0,3}\b"
    )

    QUESTION_WORDS = {
        "how",
        "what",
        "when",
        "where",
        "why",
        "which",
        "who",
        "whom",
        "whose",
        "can",
        "could",
        "would",
        "should",
        "is",
        "are",
        "was",
        "were",
        "do",
        "does",
        "did",
    }

    NON_ENTITY_ALIAS_KEYS = {
        "current",
        "historical",
        "timeline",
        "past",
        "present",
    }

    ENTITY_HEADS = {
        "project",
        "scope",
        "memory",
        "controller",
        "model",
        "backbone",
        "database",
        "dataset",
        "benchmark",
        "algorithm",
        "system",
        "agent",
        "evaluation",
        "baseline",
        "baselines",
        "neo4j",
        "ollama",
        "llama",
        "qwen",
        "locomo",
        "c3-lite",
        "c3",
        "cv",
    }

    def __init__(self, config: dict[str, Any]):
        self.config = config
        stopwords = config["query_analysis"].get("stopwords", [])
        self.stopwords = {str(item).lower() for item in stopwords}
        self.aliases = config["query_analysis"].get("aliases", {})

    def analyse(self, query: str) -> QueryFeatures:
        normalised = normalise_text(query)
        tokens = tokenize(normalised, self.stopwords)

        asks_current = self._contains_any(
            normalised,
            self.TEMPORAL_CURRENT,
        )
        asks_historical = (
            self._contains_any(normalised, self.TEMPORAL_HISTORICAL)
            or self._matches_any(normalised, self.HISTORICAL_DECISION_PATTERNS)
        )
        asks_current = (
            asks_current
            or self._matches_any(normalised, self.CURRENT_COMMITMENT_PATTERNS)
        )
        asks_timeline = (
            self._contains_any(
                normalised,
                self.TEMPORAL_TIMELINE,
            )
            or self._matches_any(
                normalised,
                self.TEMPORAL_CONTINUITY_PATTERNS,
            )
        )

        selection_set_query = self._matches_any(
            normalised,
            self.SELECTION_SET_PATTERNS,
        )

        if selection_set_query:
            asks_current = False
            asks_historical = False
            asks_timeline = False

        asks_procedure = self._detect_procedural_intent(normalised)
        asks_explanation = self._detect_explanation_intent(normalised)
        asks_conflict = self._detect_conflict_intent(normalised)
        procedural_policy_override = (
            self._is_procedural_policy_query(normalised)
        )

        # A present/current alternative normally needs historical comparison,
        # the current state, and a policy for resolving the alternatives.
        if asks_conflict and asks_current:
            asks_historical = True

        query_mode = self._resolve_mode(
            asks_current=asks_current,
            asks_historical=asks_historical,
            asks_timeline=asks_timeline,
            asks_procedure=asks_procedure,
            procedural_policy_override=procedural_policy_override,
        )

        task_type = self._detect_task_type(normalised)
        entities = self._extract_entities(query, normalised, tokens)
        temporal_expressions = self._extract_temporal_expressions(query)
        information_needs = self._extract_information_needs(
            query,
            query_mode,
            asks_conflict=asks_conflict,
            asks_explanation=asks_explanation,
            asks_procedure=asks_procedure,
        )

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

    def _detect_procedural_intent(self, text: str) -> bool:
        # Requests to select factual CV/project content are memory-selection
        # questions, even when they contain the modal word "should".
        if self._is_factual_selection_query(text):
            return False

        if self._contains_any(text, self.PROCEDURAL_SIGNALS):
            return True

        if self._matches_any(text, self.PROCEDURAL_POLICY_PATTERNS):
            return True

        # Normative alternatives: "Should A or B?" / "prioritise A or B".
        if self._matches_any(text, self.ALTERNATIVE_PATTERNS) and re.search(
            r"\b(?:should|prioritise|prioritize|choose|select|prefer)\b",
            text,
        ):
            return True

        # Methodological justification can require a stored evaluation rule.
        if (
            "evaluation dimension" in text
            and re.search(r"\b(?:use|using|include|adopt)\b", text)
        ):
            return True

        return False

    def _is_factual_selection_query(self, text: str) -> bool:
        # Alternatives and explicit conditions remain policy questions.
        if " or " in f" {text} " or " if " in f" {text} ":
            return False
        return self._matches_any(text, self.FACTUAL_SELECTION_PATTERNS)

    @staticmethod
    def _is_procedural_policy_query(text: str) -> bool:
        return (
            (
                text.startswith("if ")
                or " if an " in text
                or " if a " in text
                or " if the " in text
            )
            and re.search(
                r"\b(?:how should|what should|should the system|"
                r"should i say|should the answer|should you)\b",
                text,
            )
            is not None
        )

    def _detect_explanation_intent(self, text: str) -> bool:
        if "explain" in text or self._contains_phrase(text, "why"):
            return True

        if self._contains_any(text, self.NEGATIVE_EVIDENCE_SIGNALS):
            # "No evidence" is an answerability condition, not necessarily
            # a request to explain the controller's evidence.
            return False

        return self._contains_any(text, self.EXPLANATION_SIGNALS)

    def _detect_conflict_intent(self, text: str) -> bool:
        if self._contains_any(text, self.CONFLICT_SIGNALS):
            return True
        return self._matches_any(text, self.ALTERNATIVE_PATTERNS)

    @classmethod
    def _contains_any(cls, text: str, signals: set[str]) -> bool:
        return any(cls._contains_phrase(text, signal) for signal in signals)

    @staticmethod
    def _matches_any(text: str, patterns: tuple[str, ...]) -> bool:
        return any(re.search(pattern, text) is not None for pattern in patterns)

    @staticmethod
    def _resolve_mode(
        *,
        asks_current: bool,
        asks_historical: bool,
        asks_timeline: bool,
        asks_procedure: bool,
        procedural_policy_override: bool = False,
    ) -> QueryMode:
        # Conditional policy questions can mention old/new/conflict terms but
        # still primarily ask for an applicable response rule.
        if asks_procedure and procedural_policy_override:
            return QueryMode.PROCEDURAL
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
            scores[task_type] = sum(
                1
                for keyword in keywords
                if self._contains_phrase(text, keyword)
            )

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

        for match in self.ENTITY_PATTERN.findall(original_query):
            cleaned = " ".join(match.strip().split())
            if cleaned.lower() not in self.QUESTION_WORDS:
                entities.add(cleaned)

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

        raw_tokens = tokenize(normalised_query)
        for index, token in enumerate(raw_tokens):
            if token not in self.ENTITY_HEADS:
                continue

            entities.add(token)

            if index > 0:
                previous = raw_tokens[index - 1]
                if (
                    previous not in self.stopwords
                    and previous not in self.QUESTION_WORDS
                ):
                    entities.add(f"{previous} {token}")

            if index + 1 < len(raw_tokens):
                following = raw_tokens[index + 1]
                if (
                    following not in self.stopwords
                    and following not in self.QUESTION_WORDS
                ):
                    entities.add(f"{token} {following}")

        for token in tokens:
            if (
                any(character.isdigit() for character in token)
                or "_" in token
                or token in {
                    "neo4j",
                    "ollama",
                    "locomo",
                    "c3",
                    "c3-lite",
                }
            ):
                entities.add(token)

        filtered = {
            entity.strip()
            for entity in entities
            if entity.strip()
            and entity.strip().lower() not in self.QUESTION_WORDS
            and entity.strip().lower() not in self.NON_ENTITY_ALIAS_KEYS
        }

        return sorted(
            filtered,
            key=lambda item: (len(item.split()), item.lower()),
        )

    @staticmethod
    def _contains_phrase(text: str, phrase: str) -> bool:
        phrase = phrase.strip().lower()
        if not phrase:
            return False

        if re.fullmatch(r"[a-z0-9_.+-]+", phrase):
            return re.search(rf"\b{re.escape(phrase)}\b", text) is not None

        # Multiword English phrases get boundary protection at both ends.
        if re.fullmatch(r"[a-z0-9_.+\-\s]+", phrase):
            return (
                re.search(
                    rf"(?<!\w){re.escape(phrase)}(?!\w)",
                    text,
                )
                is not None
            )

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
        ]

        matches: list[str] = []
        for pattern in patterns:
            matches.extend(
                re.findall(
                    pattern,
                    query,
                    flags=re.IGNORECASE,
                )
            )

        return list(
            dict.fromkeys(
                match.strip()
                for match in matches
                if match.strip()
            )
        )

    def _extract_information_needs(
        self,
        query: str,
        mode: QueryMode,
        *,
        asks_conflict: bool,
        asks_explanation: bool,
        asks_procedure: bool,
    ) -> list[str]:
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

        if asks_conflict:
            needs.extend(
                [
                    "alternative or conflicting states",
                    "preferred or current resolution",
                ]
            )

        if asks_explanation:
            needs.append("supporting reason or evidence")

        if asks_procedure and "applicable procedure or rule" not in needs:
            needs.append("applicable procedure or rule")

        return list(dict.fromkeys(needs))
