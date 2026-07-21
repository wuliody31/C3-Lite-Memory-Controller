from __future__ import annotations

from collections import Counter
from typing import Any
from .coverage_estimator import CoverageEstimator
from .schemas import ConflictGroup, MemoryCandidate, MemoryType, QueryFeatures, QueryMode, RouteDecision
from .text_utils import approximate_token_count, jaccard_similarity, tokenize


class EvidenceSelector:
    def __init__(self, config: dict[str, Any], coverage: CoverageEstimator):
        self.s = config["selection"]; self.coverage = coverage
        self.stopwords = {str(x).lower() for x in config["query_analysis"].get("stopwords", [])}

    def select(self, *, candidates: list[MemoryCandidate], features: QueryFeatures, route: RouteDecision, conflicts: list[ConflictGroup]) -> list[MemoryCandidate]:
        if not candidates: return []
        selected, covered, counts, used = [], set(), Counter(), 0
        max_items, budget, max_type = int(self.s["max_evidence"]), int(self.s["evidence_token_budget"]), int(self.s["max_per_memory_type"])

        def add(item: MemoryCandidate) -> None:
            nonlocal used
            selected.append(item); counts[item.memory_type] += 1; used += approximate_token_count(item.text)
            covered.update(self.coverage.candidate_coverage(item, features.information_needs))

        if self.s.get("guarantee_procedural_if_routed", True) and MemoryType.PROCEDURAL in route.selected_types:
            rules = [x for x in candidates if x.memory_type == MemoryType.PROCEDURAL]
            if rules: add(max(rules, key=lambda x: x.final_score))

        if features.query_mode == QueryMode.TIMELINE:
            temporal = sorted([x for x in candidates if x.timestamp and x.memory_type in {MemoryType.EPISODIC, MemoryType.SEMANTIC}], key=lambda x: x.timestamp)
            if len(temporal) >= 2:
                for item in [temporal[0], temporal[-1]]:
                    if item not in selected and len(selected) < max_items and used + approximate_token_count(item.text) <= budget: add(item)

        by_id = {x.memory_id: x for x in candidates}
        for conflict in conflicts:
            if conflict.unresolved:
                for memory_id in conflict.candidate_ids[:2]:
                    item = by_id.get(memory_id)
                    if item and item not in selected and len(selected) < max_items and used + approximate_token_count(item.text) <= budget: add(item)

        remaining = [x for x in candidates if x not in selected]
        while remaining and len(selected) < max_items:
            feasible = [x for x in remaining if counts[x.memory_type] < max_type and used + approximate_token_count(x.text) <= budget]
            if not feasible: break
            best = max(feasible, key=lambda x: self._score(x, selected, covered, features.information_needs))
            add(best); remaining.remove(best)
        return selected

    def _score(self, item: MemoryCandidate, selected: list[MemoryCandidate], covered: set[int], needs: list[str]) -> float:
        redundancy = max((jaccard_similarity(tokenize(item.text, self.stopwords), tokenize(x.text, self.stopwords)) for x in selected), default=0.0)
        gain = len(self.coverage.candidate_coverage(item, needs) - covered) / len(needs) if needs else 0.0
        return float(self.s["relevance_weight"]) * item.final_score - float(self.s["redundancy_weight"]) * redundancy + float(self.s["coverage_gain_weight"]) * gain
