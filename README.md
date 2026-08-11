# C3: Lifecycle-Aware Multi-Memory Orchestration for Long-Term LLM Agents

C3 is a training-free, lifecycle-aware memory orchestration framework for long-term LLM agents.

The system coordinates **episodic**, **semantic**, and **procedural** memory through query-aware routing, shared evidence ranking, conflict resolution, lifecycle-aware semantic state management, evidence selection, confidence control, and auditable execution traces.

> **Project status:** research-grade and production-oriented.  
> The repository contains frozen experimental protocols, real Neo4j validation, external-memory baselines, formal evaluation artifacts, statistical analyses, and reproducibility checkpoints. It should not be interpreted as a production-scale deployment claim.

---

## 1. Research Problem

Long-term LLM agents must solve two related but distinct memory-control problems:

1. **Which memories should be used for the current query?**
2. **Which stored memories should still be treated as current?**

C3 addresses these through two major research phases:

### Phase I — Evidence Orchestration

Phase I focuses on selecting useful evidence across heterogeneous memory sources:

- episodic memory;
- semantic memory;
- procedural memory;
- query-aware multi-memory routing;
- shared ranking;
- conflict resolution;
- evidence selection;
- abstention and confidence control.

### Phase II — Memory Lifecycle Control

Phase II extends C3 from retrieval orchestration to semantic-state lifecycle management:

- current-state tracking;
- supersession;
- duplicate suppression;
- out-of-order update rejection;
- reversion handling;
- historical-state retention;
- lifecycle-aware current retrieval.

The central distinction is:

> **Phase I asks which memories should be used; Phase II-B asks which memories should still be considered current.**

---

## 2. System Architecture

```text
                         User Query
                             |
                             v
                    +----------------+
                    | Query Analyzer |
                    +--------+-------+
                             |
                             v
                    +----------------+
                    | Route Planner  |
                    +--------+-------+
                             |
             +---------------+---------------+
             |               |               |
             v               v               v
       Episodic        Semantic         Procedural
        Memory          Memory            Memory
             |               |               |
             +---------------+---------------+
                             |
                             v
                    Candidate Retrieval
                             |
                             v
                      Shared Ranker
                             |
                             v
                  Conflict Detection /
                      Resolution
                             |
                             v
                    Evidence Selector
                             |
                             v
                  Confidence Controller
                             |
                             v
                       LLM Backbone
```

Phase II adds lifecycle-aware semantic state:

```text
Incoming Semantic State
          |
          v
 +--------------------+
 | Lifecycle Service  |
 +---------+----------+
           |
     +-----+-----------------------+
     |             |               |
     v             v               v
  Initial       Duplicate       Newer State
    ADD           NO-OP          SUPERSEDE
                                   |
                                   v
                         previous -> superseded
                         latest   -> current
                                   |
                                   v
                          transition history
```

---

## 3. Core Capabilities

### Multi-memory orchestration

C3 separates memory by function rather than by storage technology:

- **Episodic memory** — what happened and when;
- **Semantic memory** — what is currently believed to be true;
- **Procedural memory** — how a task should be performed.

### Query-aware routing

The controller determines which memory types are relevant instead of placing all available memory into every prompt.

### Shared evidence ranking

Candidates from different memory sources are normalized through a shared ranking layer before final evidence selection.

### Conflict-aware retrieval

The framework supports:

- explicit graph conflicts;
- implicit state/version conflicts;
- current-state queries;
- timeline queries;
- historical evidence.

### Lifecycle-aware semantic memory

Semantic facts can explicitly transition between lifecycle states such as:

```text
current
superseded
archived
```

This allows current-state retrieval to distinguish active state from retained history.

### Confidence-controlled generation

The controller estimates evidence adequacy and chooses between:

```text
direct answer
caveated answer
abstention
```

### Auditable execution

Experimental runs preserve:

- routing decisions;
- candidate memories;
- ranked evidence;
- selected evidence;
- confidence state;
- lifecycle operations;
- frozen configuration;
- experiment artifacts.

---

## 4. Technology Stack

Core research implementation:

- Python
- PyTorch / Transformers
- Qwen3-8B
- Neo4j
- Mem0 OSS
- BM25 / metadata ranking
- pytest
- Slurm GPU execution

Formal GPU validation used a frozen Qwen3-8B backbone with no task-specific fine-tuning.

---

## 5. Headline Experimental Results

### 5.1 Dataset A

On the formal Dataset A evaluation:

| Method | Answer F1 |
|---|---:|
| No memory | 0.2443 |
| All memory | 0.3067 |
| Simple retrieval | 0.3705 |
| **C3** | **0.4184** |

C3 additionally achieved:

- Evidence F1: **0.5071**
- Route exact accuracy: **0.9286**
- Route F1: **0.9786**

See [`docs/EXPERIMENT_RESULTS.md`](docs/EXPERIMENT_RESULTS.md) for the complete analysis.

### 5.2 LoCoMo External Validation

LoCoMo exposed a precision-recall-generation trade-off.

| Method | Answer F1 | Evidence F1 | Mean selected evidence |
|---|---:|---:|---:|
| No memory | 0.0276 | — | — |
| Simple retrieval | **0.1424** | 0.2162 | ~5 |
| All memory | 0.1354 | 0.0783 | ~19.87 |
| **C3** | 0.1236 | **0.2712** | **~2.03** |

C3 produced the strongest evidence quality and smallest context, but did not achieve the highest LoCoMo Answer F1.

### 5.3 C3 vs Mem0 Lifecycle Retrieval

The primary controlled lifecycle comparison used:

- 80 symmetric observations;
- 20 canonical lifecycle scenarios;
- 4 surface variants per scenario;
- 50,000 paired cluster-bootstrap resamples;
- canonical lifecycle scenario as the inference/resampling unit.

| Metric | C3 | Mem0 |
|---|---:|---:|
| **Current-only Top-1** | **1.000** | 0.550 |
| **Stale-only exposure** | **0.000** | 0.967 |
| Previous-state availability | 1.000 | 1.000 |
| History value recall | 1.000 | 1.000 |
| Previous-state Top-1 | 0.000 | **0.450** |

Current-only Top-1 improved by **45 percentage points**:

```text
Effect favouring C3 = +0.450
95% paired cluster-bootstrap CI = [+0.275, +0.625]
```

Stale-only exposure decreased from **96.7% to 0%**:

```text
Effect favouring C3 = +0.967
95% paired cluster-bootstrap CI = [+0.900, +1.000]
```

Historical information remained available in both systems, but C3 did not rank the immediate previous state first under the evaluated historical retrieval policy.

This distinction is important:

> Explicit lifecycle control strongly improves current-state isolation, but lifecycle storage alone does not solve historical-state ranking.

---

## 6. Experimental Scope

The lifecycle experiment is a:

> **controlled observable lifecycle-retrieval comparison**

It is not claimed to be a perfectly symmetric end-to-end comparison.

Important boundaries include:

- C3 and Mem0 do not receive identical upstream memory-formation interfaces;
- C3 uses explicit lifecycle state;
- Mem0 uses its evaluated OSS `infer=True` formation path;
- out-of-order cases are treated as temporal capability stress tests;
- surface variants are not treated as independent statistical samples;
- historical routing generalisation in the full C3 agent remains a separate limitation.

See:

- [`docs/EXPERIMENT_RESULTS.md`](docs/EXPERIMENT_RESULTS.md)
- [`docs/DEVELOPMENT_HISTORY.md`](docs/DEVELOPMENT_HISTORY.md)

---

## 7. Repository Structure

```text
.
├── configs/           Frozen and experimental configuration
├── data/              Benchmark and adapted data
├── evaluation/        Evaluation metrics and evaluators
├── experiments/       Frozen experimental artifacts
├── scripts/           Experiment, analysis and Slurm runners
├── src/               Core C3 implementation
├── tests/             Unit and regression tests
├── docs/              Research and reproducibility documentation
├── run_query.py       Local query entry point
├── run_query_gpu.py   GPU query entry point
└── run_experiment.py  Experiment entry point
```

---

## 8. Development Lineage

The repository preserves the development trail through Git tags.

Major stages include:

```text
C3-Lite v2.2
        |
        v
RC8 routing / retrieval repair
        |
        v
Dataset A formal evaluation
        |
        v
LoCoMo external validation
        |
        v
Soft Probe companion admission
        |
        v
Mem0 baseline investigation
        |
        v
Phase II-A lifecycle diagnosis
        |
        v
Phase II-B explicit lifecycle control
        |
        v
Real Neo4j validation
        |
        v
Formal100 C3 / Mem0 evaluation
        |
        v
Paired cluster-bootstrap comparison
        |
        v
Repository governance baseline
        |
        v
Reproducible quality gates
        |
        v
Typed integration contracts
        |
        v
Safe telemetry and observability isolation
        |
        v
Structured failure contracts
        |
        v
Neo4j retrieval failure boundary
        |
        v
Configuration and resource lifecycle hardening
```

See [`docs/DEVELOPMENT_HISTORY.md`](docs/DEVELOPMENT_HISTORY.md).

---

## 9. Reproducibility

The repository records frozen experiment tags and artifacts for:

- Dataset A;
- LoCoMo;
- Soft Probe;
- Mem0 baseline analysis;
- C3 lifecycle conformance;
- real Neo4j lifecycle validation;
- Mem0 Formal100;
- C3 Formal100;
- C3 vs Mem0 paired comparison.

The final Phase II-B synthesis is tagged:

```text
phase2b-b5-final-summary-v01
```

The final paired C3 vs Mem0 lifecycle comparison is tagged:

```text
phase2b-b5-c3-vs-mem0-comparison-passed
```

Enterprise-hardening checkpoints are preserved separately:

```text
c3-enterprise-governance-v01
c3-enterprise-quality-v01
c3-enterprise-contracts-v01
c3-enterprise-reliability-v01
```

These tags record repository governance, reproducible quality gates,
integration contracts, and reliability/failure semantics without replacing
the frozen experimental checkpoints.

---

## 10. Engineering Quality and Reliability

The frozen research implementation is supplemented by a compatibility-focused
engineering layer intended to improve reproducibility, integration clarity,
and in-process reliability.

### Quality baseline

At the current enterprise-hardening checkpoint:

- Python **3.11+** is the supported development baseline;
- the complete regression and contract suite contains **191 passing tests**;
- critical Ruff checks pass;
- MyPy checks pass across the hardened service boundaries;
- Python compilation checks pass;
- GitHub Actions provides automated repository quality gates;
- branch-aware coverage is recorded as a diagnostic baseline rather than used
  as a target for mechanical coverage inflation.

See [`docs/QUALITY_GATES.md`](docs/QUALITY_GATES.md).

### Stable integration contracts

The repository exposes typed service boundaries for:

- memory retrieval through `MemoryStore`;
- language-model generation through `Backbone`;
- lifecycle persistence through `LifecycleStore`;
- observability through `TraceSink`.

The compatibility facade is documented in
[`docs/INTERFACES_AND_CONTRACTS.md`](docs/INTERFACES_AND_CONTRACTS.md).

### Observability and failure isolation

C3 supports a restricted JSONL telemetry projection that excludes query text,
answer text, prompts, evidence content, and full debug payloads.

User-provided observability sinks are isolated through `BestEffortTraceSink`,
so telemetry failures do not invalidate an otherwise successful answer path.

### Structured failure contracts

The public exception hierarchy includes:

- `C3Error`;
- `ConfigurationError`;
- `RetrievalError`;
- `GenerationError`;
- `ObservabilityError`;
- `ResourceCleanupError`;
- the existing lifecycle exception hierarchy.

Neo4j retrieval distinguishes a valid empty result from an operational backend
failure, while preserving the existing full-text-to-MATCH fallback behaviour.

### Configuration and resource lifecycle

Configuration validation fails early for malformed required sections, missing
weight keys, non-numeric weights, and invalid weight normalisation.

`C3Pipeline.close()` performs coordinated resource teardown, attempts remaining
resources even when an earlier close operation fails, and is idempotent at the
pipeline boundary.

Detailed failure semantics are documented in
[`docs/RELIABILITY.md`](docs/RELIABILITY.md).

### Claim boundary

These mechanisms strengthen software quality, reproducibility, integration
contracts, and in-process failure handling.

They do **not** establish:

- distributed fault tolerance;
- automatic retry orchestration;
- circuit breaking;
- high-availability deployment;
- multi-process telemetry guarantees;
- production-scale concurrency guarantees;
- service-level availability;
- production deployment certification.

---

## 11. Known Limitations

Current limitations include:

1. Immediate-previous historical ranking is not solved by the current C3 retrieval policy.
2. Full-agent historical routing generalisation remains separate from the controlled lifecycle retrieval evaluation.
3. The lifecycle comparison does not use identical upstream formation interfaces for C3 and Mem0.
4. Out-of-order evaluation is a temporal capability stress case rather than part of the primary symmetric comparison.
5. Real Neo4j lifecycle validation is single-writer; distributed concurrency and fault-injection guarantees have not been established.
6. The repository does not claim production-scale QPS, latency, or distributed-service validation.
7. Phase II-C realised-utility feedback and online retention optimisation are not part of the frozen implementation.

---

## 12. Research Positioning

C3 should be interpreted as a:

> **research-grade, production-oriented memory orchestration framework**

rather than a production-proven commercial platform.

The primary contribution is not simply storing or retrieving more memory. It is providing explicit mechanisms for deciding:

- which memory sources should be consulted;
- which evidence should be selected;
- which conflicting state should dominate;
- which semantic memories remain current;
- when available evidence is insufficient to answer reliably.

---

## 13. Citation and Academic Use

This repository accompanies an MSc dissertation on multi-memory orchestration and lifecycle-aware long-term memory for LLM agents.

When using experimental results, preserve the protocol boundaries and limitations documented in this repository.
