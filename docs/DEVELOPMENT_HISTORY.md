# C3 Development History

This document records the major research and engineering stages that led from the original C3-Lite controller to the frozen lifecycle-aware Phase II-B system.

The history is intentionally structured around **research questions, observed failures, repairs, and validation**, rather than presenting Git commits as an undifferentiated implementation log.

---

## 1. Initial Research Objective

The original project investigated whether a long-term LLM agent could benefit from explicitly separating and controlling multiple memory types:

- episodic;
- semantic;
- procedural.

The initial problem formulation focused on evidence orchestration:

> Given a query and multiple heterogeneous memories, which memory sources and which evidence items should be used?

This resulted in C3-Lite: a context-adaptive, conflict-aware, confidence-controlled multi-memory controller.

---

## 2. C3-Lite v2.2 Final Candidate

Frozen baseline tag:

```text
c3-lite-v2.2-final-candidate
```

Representative commit:

```text
df60a28
```

Core system:

- query analysis;
- multi-label routing;
- episodic retrieval;
- semantic retrieval;
- procedural retrieval;
- shared BM25 / metadata ranking;
- conflict detection;
- conflict resolution;
- evidence selection;
- confidence / abstention control;
- structured trace generation.

This stage established the core controller architecture but did not yet represent the final research system.

---

## 3. RC8.1–RC8.3: Routing and Retrieval Repair

The next development phase focused on errors exposed by targeted evaluation.

### RC8.1 — Topic-grounded selector

Tag:

```text
rc8-1-topic-grounded-selector
```

Goal:

Improve the evidence selector so that evidence relevance was grounded in actual query topics rather than only generic ranking signals.

### RC8.2 — Requirement calibration

Tag:

```text
rc8-2-requirement-calibration
```

Goal:

Calibrate evidence requirements and feasibility decisions so that the controller did not over-request or under-request evidence.

### RC8.3 — Route and procedural retrieval repair

Tags:

```text
rc8-3a-route-recall-repair
rc8-3b-procedural-retrieval
```

Goal:

Repair routing recall failures and field-aware procedural retrieval.

This stage improved retrieval coverage without abandoning the controller's bounded-memory design.

---

## 4. RC8.4: Pilot Repair and Qwen3 Integration

The project moved from mock-only validation toward a formal frozen LLM backbone.

Key goals:

- integrate Qwen3-8B;
- validate GPU inference;
- repair pilot-stage routing and evidence failures;
- preserve identical backbone settings across compared methods.

The mock backbone remained useful for deterministic pipeline tests, but it was not treated as an experimental language model.

---

## 5. RC8.5–RC8.6: Ablation and Conflict Validation

These stages tested whether C3's internal components were causally useful.

Ablations included:

- removing route control;
- removing conflict resolution;
- removing the evidence gate;
- removing the selector.

The resulting degradation patterns showed that routing and evidence selection contributed meaningfully to the final system behaviour.

Conflict-specific audits were also added to ensure that selected contradictory evidence was handled explicitly.

---

## 6. RC8.7: LoCoMo External Validation

Tags included:

```text
rc8-7a-locomo-adapter-mock-passed
rc8-7b-locomo-smoke-audited
rc8-7c-locomo-pilot200-passed
rc8-7d-locomo-formal-passed
```

The purpose of LoCoMo was to test whether observations from the custom Dataset A transferred to an external long-term conversational-memory benchmark.

LoCoMo revealed an important trade-off:

- C3 improved evidence quality;
- C3 used substantially fewer evidence items;
- C3 did not produce the highest final Answer F1.

This redirected the research away from claiming universal answer-level superiority and toward a more precise **evidence orchestration** contribution.

---

## 7. RC8.8: LoCoMo Evidence Repair

Tags:

```text
rc8-8a-locomo-gold-repair
rc8-8a-locomo-gold-repair-passed
rc8-8a2-locomo-gold-repair-final
```

Evidence annotations and source-equivalent matching were audited and repaired.

The final interpretation distinguished:

- strict evidence-ID matching;
- source-equivalent evidence matching;
- answer generation quality.

This prevented evaluation artifacts from being interpreted as controller improvements.

---

## 8. Mem0 Baseline Investigation

The next stage investigated Mem0 as an external long-term-memory baseline.

Tags included:

```text
rc8-8b0-mem0-protocol-frozen
rc8-8b2-mem0-conv30-smoke-passed
rc8-8b3-mem0-pilot200-passed
rc8-8b4-c3-mem0-overlap-passed
```

The initial Mem0 dense-retrieval experiments used a provenance-preserving `infer=False` path.

These experiments were treated as candidate-retrieval analyses rather than as a complete Mem0 lifecycle baseline.

The main finding was complementarity:

> Mem0 and C3 often retrieved different relevant candidates.

This motivated experiments with companion-memory admission rather than simply replacing one retriever with the other.

---

## 9. Hybrid Candidate Experiments

Tags:

```text
rc8-8b5-hybrid-controller-replay-passed
rc8-8b6-hybrid-drop-diagnosis-passed
rc8-8b7-route-budget-ablation-passed
rc8-8b8-qwen-route-upperbound-conditional-pass
rc8-8b9-confidence-route-conditional-pass
rc8-8b10-route-score-deconfounding-passed
```

These experiments examined whether Mem0 candidates could improve C3 when admitted as companion evidence.

The development sequence included:

1. hybrid replay;
2. drop diagnosis;
3. route/budget ablation;
4. route-expansion upper-bound testing;
5. confidence-aware route expansion;
6. route-score deconfounding.

The important conclusion was that more retrieval candidates did not automatically translate into better generated answers.

---

## 10. Soft Probe

Frozen development and validation tags:

```text
rc8-8b11-soft-route-probe-passed
rc8-8b12-soft-probe1-dev-selected
rc8-8b13-soft-probe1-heldout-evidence-passed
rc8-8b14-qwen-heldout-validation-passed
```

Soft Probe introduced bounded companion-memory admission.

The final interpretation was deliberately conservative:

> Soft Probe-1 significantly improved evidence selection without a statistically detectable degradation in answer quality.

It was not claimed that Answer F1 significantly improved.

This stage identified evidence utilisation by the generator as a downstream bottleneck.

---

## 11. Phase II-A: Mem0 Lifecycle Characterisation

The research question then changed from:

> Which memories should be retrieved?

to:

> What happens when a user's semantic state changes over time?

Relevant tags:

```text
phase2a-mem0-infer-true-smoke-passed
phase2a-mem0-additive-path-audited
```

The pinned Mem0 OSS 2.0.15 `infer=True` path was inspected and evaluated.

The observed automatic formation path behaved additively in the inspected configuration, while explicit update and delete APIs existed separately.

This stage did **not** claim that Mem0 lacks update/delete functionality.

Instead, it identified a specific lifecycle failure mode:

> retained state history can remain searchable without explicit current/superseded role discrimination.

This became the motivation for Phase II-B.

---

## 12. Phase II-B1: Explicit Lifecycle Model

C3 introduced explicit semantic lifecycle state.

Core lifecycle states:

```text
current
superseded
archived
```

Core operations:

```text
add_initial
noop_duplicate
supersede
reject_out_of_order
```

The implementation enforced:

- one current state per state slot;
- duplicate/replay suppression;
- explicit supersession;
- out-of-order rejection;
- historical retention;
- deterministic transition identity.

This established the mechanism-level lifecycle contract.

---

## 13. Phase II-B2: Atomic Neo4j Persistence

Tag:

```text
phase2b-neo4j-atomic-store-passed
```

The lifecycle mechanism was connected to Neo4j through an atomic write path.

A state transition updates the semantic state and stores the associated transition evidence within a single write transaction.

This reduced the gap between the in-memory lifecycle contract and the real storage backend.

---

## 14. Phase II-B3/B4: Real Neo4j Validation

Tags:

```text
phase2b-real-neo4j-lifecycle-passed
phase2b-schema-retrieval-unit-passed
phase2b-real-neo4j-schema-retrieval-passed
```

Validation covered:

- lifecycle invariants;
- schema constraints;
- current-state retrieval;
- historical-state retrieval;
- real Neo4j storage behaviour.

Scope limitation:

The real Neo4j validation was single-writer and did not establish distributed-concurrency or fault-injection guarantees.

---

## 15. Phase II-B5.1: Controlled Lifecycle Benchmark

Tags:

```text
phase2b-b5-benchmark-contract-audited
phase2b-b5-lifecycle-benchmark-v01
```

Benchmark design:

```text
5 lifecycle patterns
x 5 state keys
x 4 surface variants
= 100 observations
```

Patterns:

- replacement;
- duplicate;
- multi-step update;
- out-of-order arrival;
- reversion.

The 100 observations correspond to 25 canonical scenarios with four paraphrased surface variants each.

The surface variants are not treated as independent experimental units.

---

## 16. Phase II-B5.2: C3 Lifecycle Conformance

Tag:

```text
phase2b-b5-c3-lifecycle-conformance-passed
```

The internal lifecycle mechanism achieved deterministic conformance on the 100 benchmark cases:

- operation exactness: 1.0;
- current-state accuracy: 1.0;
- single-current invariant: 1.0;
- stale exposure: 0;
- previous-state recall: 1.0;
- history retention: 1.0;
- contradiction rate: 0.

This was treated as mechanism-level contract validation, not as a cross-system comparison.

---

## 17. Phase II-B5.3: Mem0 Lifecycle Baseline

Development sequence:

```text
phase2b-b5-mem0-comparison-contract-audited
phase2b-b5-mem0-lifecycle-smoke-passed
phase2b-b5-mem0-lifecycle-pilot25-passed
phase2b-b5-lifecycle-observable-evaluator-v02
phase2b-b5-mem0-lifecycle-formal100-passed
```

The 25-case pilot was a development subset of the 100-case benchmark and is not presented as an independent replication.

The final Mem0 Formal100 evaluation showed:

- strong historical retention;
- frequent stale-state competition during current retrieval;
- unreliable current/historical lifecycle-role discrimination.

A later metadata-only provenance correction was frozen under:

```text
phase2b-b5-mem0-formal100-provenance-corrected
```

This correction did not modify raw results or evaluation metrics.

---

## 18. Phase II-B5.4: Observable C3 vs Mem0 Comparison

C3 observable protocol development:

```text
phase2b-b5-c3-observable-protocol-v02
phase2b-b5-c3-observable-protocol-v03
phase2b-b5-c3-observable-formal100-passed
```

An initial full-agent comparison exposed an asymmetric routing problem: generic historical queries did not always enter semantic retrieval.

The final controlled protocol therefore compared directly observable retrieval interfaces:

```text
Mem0:
memory.search()

C3:
role-conditioned direct semantic retrieval
```

C3 used benchmark-provided CURRENT/HISTORICAL query role only.

The benchmark did not supply:

- gold state;
- gold value;
- gold answer.

Natural-language historical-role classification remains a separate limitation.

---

## 19. Final Paired Comparison

Frozen tag:

```text
phase2b-b5-c3-vs-mem0-comparison-passed
```

Primary inference:

```text
80 symmetric observations
20 canonical scenario clusters
4 surface variants per cluster
50,000 paired cluster-bootstrap samples
```

Headline findings:

- current-only Top-1: C3 1.00 vs Mem0 0.55;
- stale-only exposure: C3 0.00 vs Mem0 0.967;
- history recall: both 1.00;
- previous-state Top-1: C3 0.00 vs Mem0 0.45.

The experiment therefore supports a mechanism-specific claim:

> Explicit lifecycle state substantially improves current-state isolation while preserving historical information.

It does not support the claim that C3 dominates Mem0 on every retrieval objective.

---

## 20. Final Phase II-B Freeze

Final thesis-facing summary tag:

```text
phase2b-b5-final-summary-v01
```

Representative commit:

```text
62bfaaf
```

At this point Phase II-B was frozen.

No historical-ranker tuning was performed after observing the Formal100 previous-state Top-1 result.

---

## 21. Current Research Position

The final project has two complementary contributions.

### Evidence orchestration

C3 controls:

- memory routing;
- candidate ranking;
- evidence selection;
- conflict resolution;
- confidence and abstention.

### Lifecycle orchestration

C3 additionally controls:

- state validity;
- supersession;
- duplicate handling;
- out-of-order temporal updates;
- current vs retained historical state.

The resulting framing is broader than a simple memory router:

> **C3 is a memory orchestration framework that controls both memory use and memory validity.**
