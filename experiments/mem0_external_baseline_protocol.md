# Mem0 External Baseline Protocol

## Objective

This experiment evaluates Mem0 OSS as a contemporary long-term memory
baseline and tests whether C3 can operate above an alternative memory
backend.

Two configurations are evaluated:

1. Mem0 OSS as an independent memory system.
2. C3 using Mem0 as its candidate-retrieval backend.

## Research questions

- Does Mem0 improve answer quality over simple retrieval and all-memory
  prompting?
- Does C3 retain its evidence-selection advantage when its candidates
  are supplied by a different backend?
- Can backend substitution improve recall without losing C3's evidence
  precision and context efficiency?
- What are the trade-offs in answer quality, evidence quality, context
  size and runtime?

## Experimental conditions

### Existing frozen methods

- no_memory
- simple_retrieval
- all_memory
- c3_current_backend

### New methods

- mem0_oss
- c3_mem0_backend

## Dataset stages

### Smoke

- One LoCoMo conversation
- Ten fixed questions
- Used only to validate ingestion, retrieval, provenance and generation

### Pilot

- Frozen LoCoMo Pilot-200
- Used for implementation and runtime validation
- Not used for controller or retrieval tuning

### Formal

- Full LoCoMo
- 1,986 questions
- Formal outputs frozen before metric computation

## Fairness controls

- The same LoCoMo adapter v0.2 is used.
- The same question texts and gold answers are used.
- Conversations are ingested chronologically.
- Memory stores are isolated by conversation.
- The same Qwen3-8B family is used for answer generation.
- The same maximum generated-token budget is used.
- Retrieval count and context-token usage are recorded.
- No parameters are selected using the full LoCoMo test results.
- Existing C3 predictions remain frozen.
- New methods are compared using paired question-level evaluation.

## Primary metrics

### Answer quality

- Answer Token F1

### Evidence quality

When exact source provenance is available:

- source-equivalent Evidence Precision
- source-equivalent Evidence Recall
- source-equivalent Evidence F1

### Efficiency

- mean retrieved memories
- mean memories supplied to generation
- prompt context tokens
- memory-construction time
- retrieval latency
- answer-generation latency

### Reliability

- abstention accuracy
- unanswerable abstain recall
- answerable response rate

## Statistical analysis

- 50,000 question-level paired bootstrap resamples
- two-sided 95% confidence intervals
- Holm-Bonferroni correction within predefined comparison families

## Interpretation constraint

Mem0 OSS is treated as a contemporary software baseline. The experiment
must not be described as an exact reproduction of every configuration
reported in the original Mem0 publication.
