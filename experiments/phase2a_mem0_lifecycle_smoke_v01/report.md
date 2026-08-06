# Phase II-A Mem0 infer=True Memory-Formation Smoke

## Experiment identity

- Slurm job: `37342`
- Mem0 OSS: `2.0.15`
- Mem0 source commit: `50bdaaea0c02744720ed374d88584fd01494eeb7`
- LLM: `Qwen/Qwen3-8B`
- Runtime: local Transformers OpenAI-compatible endpoint
- Quantisation: BitsAndBytes NF4 4-bit
- Vector store: embedded Qdrant
- Embedder: `multi-qa-MiniLM-L6-cos-v1`
- User scope: `phase2a_residence_smoke_v01`

## Pipeline result

The end-to-end `infer=True` pipeline completed successfully.

- Turns completed: 4
- Completed lifecycle LLM requests: 4
- Final memory count: 3
- Mean per-turn latency: 4.797 seconds

## Observed operations

- Explicit ADD: 3
- Explicit UPDATE: 0
- Explicit DELETE: 0
- Inferred duplicate NOOP: 1

## Behavioural findings

1. Initial fact extraction succeeded.
2. Exact duplicate suppression succeeded.
3. The move event was appended as a new memory.
4. The current London state was appended as a new memory.
5. The stale Nottingham state remained retrievable.
6. No UPDATE, DELETE or supersession event was observed.
7. The current-state query ranked London first.
8. The historical query did not rank Nottingham first.

## Final memories

- User currently lives in London
- User moved from Nottingham to London in July 2026
- User lives in Nottingham

## Interpretation

This run validates automatic memory extraction, exact-duplicate
suppression, persistence, history access and retrieval. It does not
validate complete lifecycle state management. In this scenario, Mem0
behaved as an additive memory-formation system: a historical transition
and a new current state were added without superseding the stale current
state.

The result should therefore be reported as a Mem0 OSS `infer=True`
memory-formation smoke rather than evidence of a complete automatic
memory lifecycle.

## Implication for C3

The unresolved coexistence of current and stale state provides a direct
motivation for explicit C3 lifecycle policies such as semantic state
supersession, episodic transition preservation, temporal query routing
and conflict-aware evidence selection.
