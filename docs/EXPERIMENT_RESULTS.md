# Experimental Results

This document consolidates the major frozen experimental results for C3.

Results are grouped by research phase and interpreted according to the scope of the corresponding protocol.

No single metric should be interpreted as demonstrating universal superiority.

---

# 1. Evaluation Principles

The project follows several evaluation rules.

1. Development, pilot, held-out and formal runs are distinguished.
2. Frozen configurations and Git tags are used for formal runs.
3. Gold answers and supporting-memory labels are not exposed to the controller.
4. Surface paraphrases from one canonical lifecycle scenario are not treated as statistically independent samples.
5. Out-of-order lifecycle cases are treated as temporal stress cases rather than part of the primary symmetric comparison.
6. Evidence-quality gains and final-answer gains are reported separately.
7. Failed or diagnostic experiment variants are not promoted to final results.

---

# 2. Dataset A Formal Evaluation

The formal Dataset A experiment contains 240 predictions across the compared methods.

## 2.1 Answer F1

| Method | Answer F1 |
|---|---:|
| No memory | 0.244293 |
| All memory | 0.306653 |
| Simple retrieval | 0.370510 |
| **C3** | **0.418357** |

C3 achieved the highest Answer F1 among these reported Dataset A baselines.

Compared with simple retrieval:

```text
Absolute difference:
0.418357 - 0.370510 = +0.047847
```

The result should be interpreted together with evidence and routing metrics rather than as an isolated generation score.

---

## 2.2 C3 Evidence and Routing

C3 achieved:

| Metric | C3 |
|---|---:|
| Evidence F1 | **0.507108** |
| Route exact accuracy | **0.928571** |
| Route F1 | **0.978571** |

The formal bootstrap analysis reported detectable C3 improvements over simple/all-memory baselines for Answer F1 and evidence metrics after the planned multiple-comparison correction.

---

# 3. Dataset A Ablation Study

The ablation study measured how removing internal controller components affected performance.

| Variant | Answer F1 | Evidence F1 |
|---|---:|---:|
| **Full C3** | **0.41836** | **0.50711** |
| No route | 0.38989 | 0.38060 |
| No conflict | 0.41875 | 0.50433 |
| No gate | 0.42871 | 0.50711 |
| No selector | 0.35666 | 0.47427 |

## Interpretation

### Routing

Removing route control reduced:

- Answer F1;
- Evidence F1.

The especially large Evidence F1 drop supports the importance of memory-type routing.

### Selector

Removing the selector caused the largest Answer F1 degradation among the listed major removals:

```text
0.41836 -> 0.35666
```

This supports the role of bounded evidence selection.

### Conflict resolution

Removing conflict handling produced only a small aggregate change in this overall metric.

This does not prove conflict handling is unnecessary: its intended benefit is concentrated on conflict-specific cases rather than every evaluation item.

### Evidence gate

Removing the gate did not reduce aggregate Answer F1 in this ablation.

Therefore the gate should not be described as a universally beneficial component based on this result alone.

---

# 4. LoCoMo External Validation

The LoCoMo evaluation used:

```text
10 conversations
1986 questions
1542 answerable questions
444 unanswerable questions
5882 turns
```

## 4.1 Main Results

| Method | Answer F1 | Evidence F1 | Mean selected evidence |
|---|---:|---:|---:|
| No memory | 0.02759 | — | — |
| Simple retrieval | **0.14236** | 0.21617 | ~5 |
| All memory | 0.13542 | 0.07829 | ~19.87 |
| **C3** | 0.12362 | **0.27123** | **~2.03** |

## Interpretation

LoCoMo did not reproduce the Dataset A answer-level ordering.

C3 obtained:

- the strongest Evidence F1;
- the smallest selected evidence set;
- lower Answer F1 than simple retrieval and all-memory prompting.

This supports a **precision-context-efficiency** contribution, but not universal Answer F1 superiority.

The experiment exposed a downstream bottleneck:

> Higher-quality retrieved evidence is not automatically converted into a better generated answer.

---

# 5. Mem0 Dense Retrieval Analysis

An earlier Mem0 comparison used a provenance-preserving dense retrieval path with:

```text
Mem0 infer=False
MiniLM embeddings
384-dimensional embeddings
CPU retrieval
```

This experiment is **not** treated as the full Mem0 lifecycle baseline.

## 5.1 Pilot-200

Reported retrieval results included:

| Metric | Mem0 |
|---|---:|
| Top-5 Evidence F1 | 0.18054 |
| Top-5 hit rate | 0.495 |
| Top-20 Evidence F1 | 0.08192 |
| Top-20 hit rate | 0.675 |

For comparison:

```text
C3 raw-30 hit rate = 0.625
Mem0 top-20 hit rate = 0.675
C3/Mem0 union hit rate = 0.750
```

The union result motivated candidate-complementarity experiments.

---

## 5.2 Held-out Retrieval

Mem0 dense retrieval achieved approximately:

| Metric | Value |
|---|---:|
| Precision | 0.044605 |
| Recall | 0.640324 |
| Evidence F1 | 0.081481 |
| Hit rate | 0.696529 |

The result was interpreted as evidence of candidate complementarity, not as a complete Mem0 agent comparison.

---

# 6. Soft Probe Development

Soft Probe introduced bounded companion-memory admission.

## 6.1 Development Pilot

Hard routing:

```text
Answer F1   = 0.0567
Evidence F1 = 0.2302
```

Soft Probe-1:

```text
Answer F1   = 0.0622
Evidence F1 = 0.2675
```

Answer change versus hard routing:

```text
Delta = +0.005485
95% CI = [-0.001987, +0.013702]
p = 0.15488
```

Evidence change:

```text
Delta = +0.037333
p = 0.00052
```

Interpretation:

> Soft Probe-1 improved evidence selection, while the observed positive Answer F1 change was not statistically detectable.

---

# 7. Soft Probe Held-out Evidence Validation

Held-out comparison:

| Metric | Hard | Soft Probe-1 |
|---|---:|---:|
| Precision | 0.221015 | **0.246613** |
| Recall | 0.400905 | **0.410097** |
| Evidence F1 | 0.277407 | **0.297340** |
| Hit rate | 0.438970 | **0.450728** |

Evidence F1 difference:

```text
Delta = +0.019933
95% CI = [+0.013158, +0.026689]
p < 0.001
```

The companion probe expanded retrieval in approximately 94.6% of evaluated cases.

Therefore it should be described as:

> bounded/asymmetric companion admission

rather than strongly selective confidence-triggered probing.

---

# 8. Frozen Qwen3 Held-out Validation

With the frozen Qwen3 backbone:

| Metric | Hard | Soft Probe-1 |
|---|---:|---:|
| Answer F1 | 0.076643 | 0.077130 |
| Answerable Answer F1 | 0.099048 | 0.099678 |
| Evidence F1 | 0.277407 | **0.297340** |

Answer F1 difference:

```text
Delta = +0.000487
95% CI = [-0.001605, +0.002645]
p = 0.65104
```

Interpretation:

> Evidence quality improved significantly, while answer quality was non-regressive but not statistically improved.

This reinforced the conclusion that **evidence utilisation by generation** is a remaining bottleneck.

---

# 9. Phase II-A: Mem0 `infer=True` Lifecycle Characterisation

The lifecycle baseline used:

```text
Mem0 OSS 2.0.15
infer=True
Qwen/Qwen3-8B
```

The inspected automatic formation path was additive in the pinned implementation/configuration.

Explicit update and delete APIs exist separately.

Therefore the supported wording is:

> In the pinned Mem0 OSS 2.0.15 V3 automatic `infer=True` extraction path inspected in this study, memory formation is additive: new memories are emitted as ADD operations, while explicit update and delete APIs exist separately.

The experiment does **not** support saying that Mem0 has no lifecycle APIs.

---

# 10. Lifecycle Benchmark

The controlled benchmark contains:

```text
5 lifecycle patterns
x 5 semantic state keys
x 4 surface variants
= 100 observations
```

Lifecycle patterns:

1. replacement;
2. duplicate;
3. multi-step update;
4. out-of-order update;
5. reversion.

State keys:

- user residence;
- employer;
- preferred language;
- subscription plan;
- current project tool.

There are:

```text
25 canonical lifecycle scenarios
4 surface variants each
```

Surface variants are treated as correlated observations.

---

# 11. C3 Internal Lifecycle Conformance

C3 achieved deterministic conformance across the 100 lifecycle cases.

| Metric | C3 |
|---|---:|
| Operation exactness | 1.000 |
| Current-state accuracy | 1.000 |
| Single-current invariant | 1.000 |
| Stale-memory exposure | 0.000 |
| Previous-state recall | 1.000 |
| History retention | 1.000 |
| Contradiction rate | 0.000 |

Mean semantic lifecycle records:

```text
2.2
```

Mean current semantic records:

```text
1.0
```

Mean superseded records:

```text
1.2
```

This experiment validates C3's internal lifecycle contract.

It is not itself the Mem0 comparison.

---

# 12. Mem0 Formal100 Lifecycle Evaluation

The Mem0 Formal100 run completed:

```text
100 / 100 cases
260 lifecycle events
261 server requests
Qwen3-8B
infer=True
```

## 12.1 Overall 100 Cases

| Metric | Mem0 |
|---|---:|
| Current-only Top-1 | 0.510 |
| Stale-only Top-1 | 0.490 |
| Previous-state availability | 1.000 |
| Previous-state Top-1 | 0.475 |
| History recall | 1.000 |
| Mean final memory count | 2.08 |

---

## 12.2 Primary Symmetric 80 Cases

| Metric | Mem0 |
|---|---:|
| Current-only Top-1 | 0.550 |
| Stale-only Top-1 | 0.450 |
| Stale-only exposure | 0.9667 |
| Mixed-transition exposure | 0.100 |
| Historical-signal exposure during current retrieval | 1.000 |
| Previous-state availability | 1.000 |
| Previous-state Top-1 | 0.450 |
| History recall | 1.000 |
| Mean final memory count | 2.05 |

The lifecycle interpretation is:

> **retention without reliable lifecycle-role discrimination**

Mem0 retained historical values well, but stale values frequently remained competitive during current-state retrieval.

---

# 13. C3 Observable Formal100

The valid C3 formal observable run used protocol v03.

Evaluation level:

```text
direct Neo4j semantic retrieval
top_k = 20
benchmark-role-conditioned CURRENT/HISTORICAL mode
```

The benchmark supplied retrieval role only.

It did not provide:

- gold value;
- gold state;
- gold answer.

## 13.1 Symmetric 80 Cases

| Metric | C3 |
|---|---:|
| Current value availability | 1.000 |
| Current Top-1 accuracy | 1.000 |
| Current-only Top-1 | 1.000 |
| Stale-only Top-1 | 0.000 |
| Stale-only exposure | 0.000 |
| Historical-signal exposure during current retrieval | 0.000 |
| Previous-state availability | 1.000 |
| Previous-state Top-1 | 0.000 |
| History recall | 1.000 |
| Mean final semantic memory count | 2.25 |

The result shows perfect current-state isolation under the controlled role-conditioned interface.

However:

> Immediate previous state remained available but was not ranked Top-1.

Therefore lifecycle validity and historical ranking are distinct problems.

---

# 14. Final C3 vs Mem0 Paired Comparison

## 14.1 Primary Statistical Unit

Primary comparison:

```text
80 symmetric observations
20 canonical lifecycle-scenario clusters
4 surface variants per cluster
```

Statistical procedure:

```text
50,000 paired cluster-bootstrap resamples
fixed seed
canonical scenario = resampling unit
```

The 80 surface variants are not treated as 80 independent scenarios.

---

## 14.2 Main Results

| Metric | C3 | Mem0 | Effect favouring C3 | 95% cluster-bootstrap CI |
|---|---:|---:|---:|---:|
| Current-only Top-1 | **1.000** | 0.550 | **+0.450** | **[+0.275, +0.625]** |
| Stale-only Top-1 | **0.000** | 0.450 | **+0.450** | **[+0.275, +0.625]** |
| Stale-only exposure | **0.000** | 0.9667 | **+0.9667** | **[+0.900, +1.000]** |
| Mixed-transition exposure | 0.000 | 0.100 | +0.100 | [-0.000, +0.200] |
| Historical-signal exposure during current retrieval | **0.000** | 1.000 | **+1.000** | **[+1.000, +1.000]** |
| Previous-state availability | 1.000 | 1.000 | 0.000 | [0.000, 0.000] |
| Previous-state Top-1 | 0.000 | **0.450** | **-0.450** | **[-0.633, -0.267]** |
| History recall | 1.000 | 1.000 | 0.000 | [0.000, 0.000] |

---

# 15. Headline Improvement

## Current-state isolation

Mem0:

```text
Current-only Top-1 = 55%
```

C3:

```text
Current-only Top-1 = 100%
```

Absolute improvement:

```text
+45 percentage points
```

Relative increase with respect to the Mem0 rate:

```text
approximately +81.8%
```

The preferred academic wording is the absolute difference:

> **+45 percentage points**

rather than claiming that C3 is generally “81.8% better”.

---

## Stale-memory exposure

Mem0:

```text
96.7%
```

C3:

```text
0%
```

Absolute reduction:

```text
96.7 percentage points
```

Relative reduction within this evaluated benchmark:

```text
100%
```

This is one of the strongest Phase II-B results.

---

# 16. Historical Retrieval Limitation

Historical retention was not degraded:

```text
Previous-state availability:
C3   = 100%
Mem0 = 100%

History recall:
C3   = 100%
Mem0 = 100%
```

However:

```text
Immediate previous Top-1:
C3   = 0%
Mem0 = 45%
```

Therefore the correct interpretation is:

> C3 preserves historical states, but its evaluated historical retrieval policy does not prioritise the immediate previous state.

It would be incorrect to describe this result as historical forgetting.

---

# 17. Temporal Out-of-Order Stress Cases

The out-of-order pattern is not part of the primary symmetric inference.

Descriptive stress result:

```text
C3 current-only Top-1  = 1.00
Mem0 current-only Top-1 = 0.35
```

However, the upstream inputs are not fully symmetric:

- C3 receives structured temporal/lifecycle information;
- Mem0 receives raw text through the evaluated automatic formation interface.

Therefore this is reported as a:

> **temporal capability stress case**

rather than headline apples-to-apples evidence.

---

# 18. What the Results Support

The experiments support the following claims.

### Supported

1. C3 can improve multi-memory evidence orchestration on Dataset A.
2. C3 produces strong evidence precision/context efficiency on LoCoMo.
3. Soft Probe improves evidence selection under the frozen protocol.
4. Explicit lifecycle state can eliminate stale-state competition from CURRENT retrieval in the controlled lifecycle benchmark.
5. C3 can preserve historical state while isolating current state.
6. Current-state isolation and historical-state ranking are distinct problems.

---

# 19. What the Results Do Not Support

The experiments do **not** support the following claims.

1. C3 universally outperforms every baseline on Answer F1.
2. C3 universally outperforms Mem0.
3. Mem0 cannot update or delete memory.
4. C3 has solved arbitrary natural-language historical routing.
5. C3 has solved immediate-previous historical ranking.
6. Phase II-B proves distributed/concurrent lifecycle correctness.
7. The controlled lifecycle experiment is a fully identical end-to-end C3/Mem0 comparison.
8. The 100 lifecycle observations are 100 statistically independent scenarios.
9. Phase II-C realised-utility retention control has been implemented.

---

# 20. Final Interpretation

The complete experimental record supports a two-part view of long-term memory control.

## Phase I

The controller must decide:

> **Which memories should be used?**

## Phase II-B

The lifecycle mechanism must decide:

> **Which memories should still be considered current?**

C3's strongest final contribution is therefore not simply increasing retrieval volume.

It is explicit **memory orchestration** across:

- memory source;
- evidence relevance;
- conflict;
- lifecycle validity;
- confidence;
- answerability.

The final lifecycle result can be summarised as:

> **Explicit lifecycle management substantially improves current-state isolation while preserving historical information, but historical-state ranking remains a separate retrieval problem.**
