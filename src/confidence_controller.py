from __future__ import annotations

from statistics import mean
from typing import Any
from .schemas import AnswerDecision, ConfidenceResult, ConflictGroup, MemoryCandidate, RouteDecision


class ConfidenceController:
    def __init__(self, config: dict[str, Any]):
        self.c, self.d = config["confidence"], config["decision"]

    def evaluate(self, *, selected: list[MemoryCandidate], route: RouteDecision, conflicts: list[ConflictGroup], coverage: float) -> ConfidenceResult:
        scores = sorted([max(0.0, min(1.0, x.final_score)) for x in selected], reverse=True)
        top, top3, agreement = (scores[0] if scores else 0.0), (mean(scores[:3]) if scores else 0.0), self._agreement(conflicts)
        adequacy = max(0.0, min(1.0, float(self.c["top_evidence_weight"]) * top + float(self.c["top3_mean_weight"]) * top3 + float(self.c["route_confidence_weight"]) * route.confidence + float(self.c["agreement_weight"]) * agreement))
        unresolved = any(x.unresolved for x in conflicts)
        if adequacy >= float(self.d["direct_adequacy_threshold"]) and coverage >= float(self.d["direct_coverage_threshold"]) and not unresolved:
            decision = AnswerDecision.DIRECT
        elif adequacy >= float(self.d["caveat_adequacy_threshold"]) and coverage >= float(self.d["caveat_coverage_threshold"]):
            decision = AnswerDecision.CAVEAT
        else:
            decision = AnswerDecision.ABSTAIN
        return ConfidenceResult(adequacy, coverage, agreement, decision, {"top_evidence": top, "top3_mean": top3, "route_confidence": route.confidence, "agreement": agreement})

    def _agreement(self, conflicts: list[ConflictGroup]) -> float:
        if not conflicts: return float(self.c["no_conflict_agreement"])
        if any(x.unresolved for x in conflicts): return float(self.c["unresolved_conflict_agreement"])
        return float(self.c["resolved_conflict_agreement"])
