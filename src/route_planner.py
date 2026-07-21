from __future__ import annotations

from typing import Any

from .schemas import MemoryType, QueryFeatures, QueryMode, RouteDecision


class RoutePlanner:
    TERMS = {
        MemoryType.EPISODIC: {"when", "happened", "event", "meeting", "originally", "previously", "before", "earlier", "experience", "did", "went", "decided", "什么时候", "发生", "最初", "之前", "经历", "决定"},
        MemoryType.SEMANTIC: {"current", "currently", "what is", "which", "uses", "preference", "focus", "role", "main", "now", "目前", "现在", "是什么", "使用", "偏好"},
        MemoryType.PROCEDURAL: {"how should", "what should", "format", "write", "follow", "must", "step", "style", "procedure", "怎么", "应该", "格式", "写", "遵循", "必须"},
    }

    def __init__(self, config: dict[str, Any]):
        routing = config["routing"]
        self.thresholds, self.weights, self.priors = routing["thresholds"], routing["utility_weights"], routing["priors"]
        self.gates = routing["structural_gates"]

    def plan(self, f: QueryFeatures) -> RouteDecision:
        raw: dict[MemoryType, float] = {}
        reasons = {x.value: [] for x in MemoryType}
        for t in MemoryType:
            signals = self._signals(t, f)
            raw[t] = min(1.0, max(0.0, sum(float(self.weights[k]) * v for k, v in signals.items())))
            reasons[t.value] += [k for k, v in signals.items() if k != "prior" and v >= 0.6]
        selected = [t for t, score in raw.items() if score >= float(self.thresholds[t.value])]
        self._gates(f, selected, reasons)
        if not selected and self.gates.get("fallback_top_route", True):
            selected = [max(raw, key=raw.get)]
            reasons[selected[0].value].append("fallback_top_route")
        selected = [x for x in MemoryType if x in set(selected)]
        return RouteDecision(selected, {x.value: round(raw[x], 6) for x in MemoryType}, reasons)

    def _signals(self, t: MemoryType, f: QueryFeatures) -> dict[str, float]:
        q, tokens = f.normalised_query, set(f.tokens)
        lexical = min(1.0, sum(int((term in q) if " " in term else (term in tokens)) for term in self.TERMS[t]) / 2.0)
        if t == MemoryType.EPISODIC:
            return {"intent": 1.0 if f.query_mode in {QueryMode.HISTORICAL, QueryMode.TIMELINE} else 0.15, "lexical": lexical, "entity": min(1.0, 0.2 + 0.15 * len(f.entities)) if f.entities else 0.15, "temporal_task": 1.0 if f.temporal_expressions or f.asks_historical_state or f.asks_timeline else 0.1, "prior": float(self.priors[t.value])}
        if t == MemoryType.SEMANTIC:
            return {"intent": 1.0 if f.query_mode in {QueryMode.CURRENT, QueryMode.TIMELINE} else 0.55, "lexical": lexical, "entity": min(1.0, 0.35 + 0.15 * len(f.entities)) if f.entities else 0.3, "temporal_task": 1.0 if f.asks_current_state or f.asks_conflict or f.asks_timeline else 0.45, "prior": float(self.priors[t.value])}
        return {"intent": 1.0 if f.asks_procedure or f.task_type else 0.05, "lexical": lexical, "entity": 0.65 if f.task_type else 0.1, "temporal_task": 1.0 if f.asks_procedure else (0.75 if f.task_type else 0.05), "prior": float(self.priors[t.value])}

    def _gates(self, f: QueryFeatures, selected: list[MemoryType], reasons: dict[str, list[str]]) -> None:
        def add(t: MemoryType, reason: str) -> None:
            if t not in selected:
                selected.append(t)
            reasons[t.value].append(reason)
        if f.asks_timeline and self.gates.get("force_timeline_route", True):
            add(MemoryType.EPISODIC, "timeline_gate"); add(MemoryType.SEMANTIC, "timeline_gate")
        if f.asks_conflict and self.gates.get("force_conflict_route", True):
            add(MemoryType.EPISODIC, "conflict_gate"); add(MemoryType.SEMANTIC, "conflict_gate")
        if f.asks_historical_state: add(MemoryType.EPISODIC, "historical_gate")
        if f.asks_current_state: add(MemoryType.SEMANTIC, "current_state_gate")
        if f.asks_procedure or f.task_type: add(MemoryType.PROCEDURAL, "procedural_gate")
        if f.asks_explanation:
            add(MemoryType.EPISODIC, "explanation_gate"); add(MemoryType.SEMANTIC, "explanation_gate")
