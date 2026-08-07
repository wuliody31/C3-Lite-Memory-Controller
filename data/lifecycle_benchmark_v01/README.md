# Controlled Lifecycle Benchmark v0.1

This benchmark isolates memory-lifecycle behaviour from broad
open-domain conversational QA.

## Design

The benchmark contains 100 cases:

- 5 state keys;
- 5 lifecycle transition patterns;
- 4 linguistic surface variants.

## Lifecycle patterns

1. replacement: A -> B
2. duplicate: A -> A
3. multi_step: A -> B -> C
4. out_of_order: A(t1) -> B(t3) -> A(t2)
5. reversion: A -> B -> A

Historical retrieval is treated as a query mode, not as a lifecycle
transition pattern.

## Primary metrics

- Current-State Accuracy
- Stale-Memory Exposure Rate
- Previous-State Recall
- Contradiction Rate
- History-Retention Recall

## Secondary metric

- Memory Growth

## Experimental role

This benchmark is intended for controlled comparison between
C3-Lifecycle and an external memory-formation baseline such as the
pinned Mem0 OSS infer=True configuration.

It is a mechanism-isolation benchmark and does not replace LoCoMo or
other long-term conversational-memory benchmarks.
