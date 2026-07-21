from __future__ import annotations

import math
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from .schemas import MemoryCandidate, MemoryType, QueryFeatures, QueryMode, RouteDecision
from .text_utils import min_max_normalise, overlap_ratio, tokenize


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.docs, self.k1, self.b = docs, k1, b
        self.tf = [Counter(x) for x in docs]
        self.lengths = [len(x) for x in docs]
        self.avg_len = sum(self.lengths) / len(self.lengths) if self.lengths else 0.0
        self.df: Counter[str] = Counter()
        for doc in docs:
            self.df.update(set(doc))

    def score(self, query: list[str], index: int) -> float:
        if not self.docs or not self.avg_len:
            return 0.0
        score, n, length = 0.0, len(self.docs), self.lengths[index]
        for token in query:
            freq = self.tf[index].get(token, 0)
            if not freq:
                continue
            df = self.df.get(token, 0)
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            denom = freq + self.k1 * (1 - self.b + self.b * length / self.avg_len)
            score += idf * (freq * (self.k1 + 1)) / denom
        return score


class SharedRanker:
    def __init__(self, config: dict[str, Any]):
        self.config, self.weights = config, config["ranking"]
        self.stopwords = {str(x).lower() for x in config["query_analysis"].get("stopwords", [])}
        conflict = config["conflict"]
        self.current = {x.lower() for x in conflict["current_status_values"]}
        self.outdated = {x.lower() for x in conflict["outdated_status_values"]}
        self.authority = {str(k): float(v) for k, v in conflict.get("authority_order", {}).items()}

    def rank(self, *, candidates: list[MemoryCandidate], features: QueryFeatures, route: RouteDecision, current_time: datetime) -> list[MemoryCandidate]:
        if not candidates:
            return []
        docs = [tokenize(self._text(x), self.stopwords) for x in candidates]
        q = tokenize(features.normalised_query, self.stopwords)
        bm25 = BM25(docs)
        raw = [bm25.score(q, i) + float(x.metadata.get("neo4j_fulltext_score", 0.0)) for i, x in enumerate(candidates)]
        lexical = min_max_normalise(raw)
        for i, item in enumerate(candidates):
            item.lexical_score = lexical[i]
            item.graph_entity_score = self._entity(item, features)
            item.temporal_task_score = self._temporal(item, features, current_time)
            item.validity_score = self._validity(item, features)
            item.source_confidence_score = max(0.0, min(1.0, 0.75 * item.confidence + 0.25 * self.authority.get(item.authority, 0.5)))
            item.route_compatibility_score = route.scores.get(item.memory_type.value, 0.0)
            item.final_score = max(0.0, min(1.0,
                float(self.weights["lexical_weight"]) * item.lexical_score
                + float(self.weights["graph_entity_weight"]) * item.graph_entity_score
                + float(self.weights["temporal_task_weight"]) * item.temporal_task_score
                + float(self.weights["validity_weight"]) * item.validity_score
                + float(self.weights["source_confidence_weight"]) * item.source_confidence_score
                + float(self.weights["route_compatibility_weight"]) * item.route_compatibility_score))
        gate = float(self.config["retrieval"].get("lexical_relevance_gate", 0.0))
        output = [x for x in candidates if x.lexical_score >= gate or x.graph_entity_score >= 0.2 or x.memory_type == MemoryType.PROCEDURAL]
        output.sort(key=lambda x: (x.final_score, x.confidence, x.timestamp or datetime.min, x.memory_id), reverse=True)
        return output

    @staticmethod
    def _text(item: MemoryCandidate) -> str:
        triggers = item.metadata.get("triggers", [])
        return " ".join(str(x) for x in [item.text, item.subject, item.predicate, item.object_value, item.metadata.get("task_type"), " ".join(triggers) if isinstance(triggers, list) else ""] if x)

    def _entity(self, item: MemoryCandidate, f: QueryFeatures) -> float:
        text = self._text(item).lower()
        if f.entities:
            return sum(1 for x in f.entities if x.lower() in text) / len(f.entities)
        return overlap_ratio(f.tokens, tokenize(text, self.stopwords))

    def _temporal(self, item: MemoryCandidate, f: QueryFeatures, now: datetime) -> float:
        if item.memory_type == MemoryType.PROCEDURAL:
            return min(1.0, 0.65 * float(item.metadata.get("task_match", 0.0)) + 0.35 * float(item.metadata.get("scope_score", 0.5)))
        if f.query_mode == QueryMode.TIMELINE:
            return 1.0 if item.timestamp else 0.55
        if f.query_mode == QueryMode.HISTORICAL:
            return 1.0 if item.memory_type == MemoryType.EPISODIC else 0.7
        if f.query_mode == QueryMode.CURRENT:
            return 1.0 if item.status.lower() in self.current else 0.15
        if item.timestamp:
            a = now if now.tzinfo else now.replace(tzinfo=timezone.utc)
            b = item.timestamp if item.timestamp.tzinfo else item.timestamp.replace(tzinfo=timezone.utc)
            days = max(0.0, (a - b).total_seconds() / 86400)
            return 0.5 + 0.5 * math.exp(-days / 365)
        return 0.6

    def _validity(self, item: MemoryCandidate, f: QueryFeatures) -> float:
        if item.memory_type == MemoryType.EPISODIC:
            return 1.0
        status = item.status.lower()
        if status in self.current:
            return 1.0
        if status in self.outdated:
            return 0.85 if f.query_mode in {QueryMode.HISTORICAL, QueryMode.TIMELINE} else 0.1
        return 0.55
