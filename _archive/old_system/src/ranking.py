from __future__ import annotations

import math
import re
from collections import Counter
from datetime import date
from typing import Any


STOPWORDS = {
    "what", "which", "when", "where", "why", "how", "should", "would",
    "could", "about", "with", "from", "into", "that", "this", "there",
    "their", "your", "mine", "current", "currently", "project", "system",
    "answer", "question", "explain", "evidence", "memory", "user", "does",
    "have", "need", "want", "did", "my", "i", "the", "a", "an", "to",
    "of", "and", "or", "for", "in", "is", "it", "be", "using", "use",
    "own", "explicitly", "definitely", "please", "on", "if", "say", "do",
    "me", "was", "were", "am", "are", "at",
}

SPECIAL_NORMALISATIONS = {
    "originally": "initial",
    "initially": "initial",
    "earlier": "early",
    "previously": "early",
    "narrowing": "narrow",
    "narrowed": "narrow",
    "changed": "change",
    "changing": "change",
    "memories": "memory",
    "supports": "support",
    "supporting": "support",
    "supported": "support",
    "approved": "approve",
    "approval": "approve",
    "training": "train",
    "trained": "train",
    "decided": "decide",
    "decision": "decide",
    "discussing": "discuss",
    "discussion": "discuss",
    "centred": "center",
    "centered": "center",
    "comparison": "compare",
    "comparing": "compare",
    "evaluation": "evaluate",
    "evaluating": "evaluate",
    "fictional": "fiction",
    "baselines": "baseline",
    "smaller": "small",
    "larger": "large",
    "roles": "role",
    "prices": "price",
    "hours": "hour",
    "deployed": "deploy",
    "deployment": "deploy",
    "attractions": "attraction",
    "requirements": "require",
    "required": "require",
    "tuning": "tune",
}

QUERY_ALIASES = {
    "multi-agent": ["multi", "agent", "multiagent", "memory system"],
    "all-memory": ["all", "memory", "prompt"],
    "user study": ["user", "study", "evaluate"],
    "data analyst": ["data", "analyst", "career", "target"],
    "llm agent": ["llm", "agent", "engineer", "applied", "ai"],
    "city-centre": ["city", "centre", "hotel", "location"],
    "city center": ["city", "centre", "hotel", "location"],
    "train station": ["train", "station", "hotel", "location"],
    "fine-tuning": ["train", "tune", "model"],
    "production deployment": ["production", "deploy", "claim"],
    "random forest": ["random", "forest", "mlp"],
    "originally": ["initial", "early", "broad"],
    "before narrowing": ["initial", "early", "broad", "narrow"],
    "change over time": ["early", "later", "current", "shift", "narrow"],
    "over time": ["early", "later", "current", "shift", "change"],
    "which memories support": ["support", "source", "fact", "event"],
    "explain which memories": ["support", "source", "fact", "event"],
}


def normalise_token(token: str) -> str:
    token = token.lower()
    if token in SPECIAL_NORMALISATIONS:
        return SPECIAL_NORMALISATIONS[token]

    if len(token) > 5 and token.endswith("ing"):
        root = token[:-3]
        if len(root) >= 3 and root[-1] == root[-2]:
            root = root[:-1]
        return SPECIAL_NORMALISATIONS.get(root, root)

    if len(token) > 4 and token.endswith("ed"):
        root = token[:-2]
        if root.endswith("i"):
            root = root[:-1] + "y"
        return SPECIAL_NORMALISATIONS.get(root, root)

    if len(token) > 4 and token.endswith("ly"):
        root = token[:-2]
        return SPECIAL_NORMALISATIONS.get(root, root)

    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"

    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]

    return token


def tokenize(text: str, content_only: bool = False) -> list[str]:
    text = (
        str(text)
        .lower()
        .replace("_", " ")
        .replace("‑", "-")
        .replace("–", "-")
    )
    tokens = [
        normalise_token(token)
        for token in re.split(r"[^a-z0-9]+", text)
        if len(token) >= 2
    ]
    if content_only:
        tokens = [
            token
            for token in tokens
            if token not in STOPWORDS and len(token) >= 2
        ]
    return tokens


def expand_query(question: str) -> str:
    lower = question.lower()
    extras: list[str] = []

    for phrase, aliases in QUERY_ALIASES.items():
        if phrase in lower:
            extras.extend(aliases)

    return question + " " + " ".join(extras)


def query_coverage(question: str, documents: list[str]) -> float:
    query_tokens = set(tokenize(expand_query(question), content_only=True))
    if not query_tokens:
        return 1.0

    document_tokens: set[str] = set()
    for document in documents:
        document_tokens.update(tokenize(document, content_only=True))

    return len(query_tokens & document_tokens) / len(query_tokens)


def candidate_query_coverage(question: str, document: str) -> float:
    query_tokens = set(tokenize(expand_query(question), content_only=True))
    if not query_tokens:
        return 1.0

    document_tokens = set(tokenize(document, content_only=True))
    return len(query_tokens & document_tokens) / len(query_tokens)


def token_similarity(first: str, second: str) -> float:
    a = set(tokenize(first, content_only=True))
    b = set(tokenize(second, content_only=True))

    if not a or not b:
        return 0.0

    return len(a & b) / len(a | b)


def _bm25_scores(
    question: str,
    documents: list[str],
    k1: float = 1.5,
    b: float = 0.75,
) -> list[float]:
    if not documents:
        return []

    query_tokens = tokenize(expand_query(question), content_only=True)
    document_tokens = [
        tokenize(document, content_only=True)
        for document in documents
    ]

    number_of_documents = len(document_tokens)
    average_length = (
        sum(len(document) for document in document_tokens)
        / max(number_of_documents, 1)
    )

    document_frequency: Counter[str] = Counter()
    for document in document_tokens:
        for token in set(document):
            document_frequency[token] += 1

    scores: list[float] = []

    for document in document_tokens:
        term_frequency = Counter(document)
        score = 0.0

        for token in query_tokens:
            frequency = document_frequency.get(token, 0)
            if frequency == 0:
                continue

            inverse_document_frequency = math.log(
                1.0
                + (
                    number_of_documents
                    - frequency
                    + 0.5
                )
                / (frequency + 0.5)
            )

            denominator = (
                term_frequency[token]
                + k1
                * (
                    1.0
                    - b
                    + b
                    * len(document)
                    / max(average_length, 1e-9)
                )
            )

            score += (
                inverse_document_frequency
                * term_frequency[token]
                * (k1 + 1.0)
                / max(denominator, 1e-9)
            )

        scores.append(score)

    return scores


def _binary_cosine_overlap(question: str, document: str) -> float:
    query_tokens = set(
        tokenize(expand_query(question), content_only=True)
    )
    document_tokens = set(tokenize(document, content_only=True))

    if not query_tokens or not document_tokens:
        return 0.0

    return len(query_tokens & document_tokens) / math.sqrt(
        len(query_tokens) * len(document_tokens)
    )


def _parse_date(value: Any) -> date | None:
    if value is None:
        return None

    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _recency_scores(
    candidates: list[dict[str, Any]],
) -> dict[str, float]:
    dated: list[tuple[str, date]] = []

    for candidate in candidates:
        candidate_date = _parse_date(
            candidate.get("date")
            or candidate.get("last_updated")
        )
        if candidate_date is not None:
            dated.append((candidate["id"], candidate_date))

    if not dated:
        return {
            candidate["id"]: 0.5
            for candidate in candidates
        }

    minimum = min(candidate_date for _, candidate_date in dated)
    maximum = max(candidate_date for _, candidate_date in dated)
    span = max((maximum - minimum).days, 1)

    result = {
        candidate["id"]: 0.5
        for candidate in candidates
    }

    for memory_id, candidate_date in dated:
        result[memory_id] = (
            candidate_date - minimum
        ).days / span

    return result


def _temporal_orientation(question: str) -> str:
    q = question.lower()

    if any(
        phrase in q
        for phrase in [
            "originally",
            "before narrowing",
            "previously",
            "earlier",
            "initially",
            "first wanted",
        ]
    ):
        return "past"

    if any(
        phrase in q
        for phrase in [
            "current",
            "currently",
            "now",
            "main target",
            "current scope",
            "current focus",
        ]
    ):
        return "current"

    if any(
        phrase in q
        for phrase in [
            "change over time",
            "changed over time",
            "how did",
            "over time",
            "shift",
        ]
    ):
        return "change"

    return "neutral"


def rank_candidates(
    question: str,
    candidates: list[dict[str, Any]],
    memory_type: str,
    limit: int,
) -> list[dict[str, Any]]:
    """
    Shared ranker used by all retrieval-based methods.

    v2.1 changes:
    - lightweight lexical normalisation;
    - phrase expansion;
    - temporal orientation for episodic memory;
    - query coverage logging.
    """
    if not candidates:
        return []

    documents = [
        str(candidate.get("text", ""))
        for candidate in candidates
    ]

    raw_bm25 = _bm25_scores(question, documents)
    maximum_bm25 = max(raw_bm25) if raw_bm25 else 0.0

    recency = _recency_scores(candidates)
    temporal_orientation = _temporal_orientation(question)

    ranked: list[dict[str, Any]] = []

    for candidate, raw_score in zip(candidates, raw_bm25):
        bm25_normalised = (
            raw_score / maximum_bm25
            if maximum_bm25 > 0
            else 0.0
        )

        lexical_overlap = _binary_cosine_overlap(
            question,
            str(candidate.get("text", "")),
        )

        lexical_relevance = (
            0.78 * bm25_normalised
            + 0.22 * lexical_overlap
        )

        # Metadata must not rescue an irrelevant memory.
        if lexical_relevance < 0.025:
            continue

        if memory_type == "episodic":
            importance = (
                float(candidate.get("importance") or 0.0)
                / 5.0
            )

            if temporal_orientation == "past":
                temporal_prior = 1.0 - recency[candidate["id"]]
            elif temporal_orientation == "current":
                temporal_prior = recency[candidate["id"]]
            elif temporal_orientation == "change":
                temporal_prior = 0.5
            else:
                temporal_prior = recency[candidate["id"]]

            score = (
                0.66 * lexical_relevance
                + 0.16 * importance
                + 0.18 * temporal_prior
            )

        elif memory_type == "semantic":
            confidence = float(
                candidate.get("confidence") or 0.0
            )
            status_prior = (
                1.0
                if candidate.get("status") == "active"
                else 0.25
            )

            score = (
                0.70 * lexical_relevance
                + 0.14 * confidence
                + 0.10 * status_prior
                + 0.06 * recency[candidate["id"]]
            )

        elif memory_type == "procedural":
            priority = (
                float(candidate.get("priority") or 0.0)
                / 5.0
            )
            score = (
                0.84 * lexical_relevance
                + 0.16 * priority
            )

        else:
            score = lexical_relevance

        enriched = dict(candidate)
        enriched["bm25_score"] = round(
            bm25_normalised,
            6,
        )
        enriched["lexical_overlap"] = round(
            lexical_overlap,
            6,
        )
        enriched["relevance_score"] = round(
            lexical_relevance,
            6,
        )
        enriched["candidate_query_coverage"] = round(
            candidate_query_coverage(
                question,
                str(candidate.get("text", "")),
            ),
            6,
        )
        enriched["evidence_score"] = round(
            min(1.0, score),
            6,
        )
        ranked.append(enriched)

    ranked.sort(
        key=lambda candidate: (
            candidate["evidence_score"],
            candidate["relevance_score"],
        ),
        reverse=True,
    )

    return ranked[:limit]
