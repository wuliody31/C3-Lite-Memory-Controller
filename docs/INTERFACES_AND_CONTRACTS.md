# Interfaces and Contracts

## Purpose

C3 uses explicit Python protocols to separate orchestration logic from
concrete storage, retrieval, lifecycle, backbone, and observability
implementations.

The enterprise-hardening process follows a compatibility-first principle:
stable contracts are added without rewriting the frozen research
implementation or changing previously evaluated algorithm behaviour.

## Public Contract Facade

The stable contract entry point is `src.contracts`.

It currently exports:

- `Backbone`
- `GenerationResult`
- `LifecycleStore`
- `MemoryRetriever`
- `MemoryStore`
- `NullTraceSink`
- `TraceSink`

Historical import paths remain supported for research reproducibility.

The facade therefore provides an additive public API without invalidating
existing experiments or frozen provenance.

## Memory Retrieval Contract

`MemoryStore` defines the retrieval capability required by `C3Pipeline`.

Current implementations include:

- `InMemoryMemoryStore`
- `Neo4jMemoryStore`
- `ProceduralJsonStore`

The historical name `MemoryStore` is retained for backward compatibility.

`MemoryRetriever` is a semantic alias because the current protocol describes
retrieval and cleanup capability rather than complete CRUD storage semantics.

Procedural memory does not require a separate retrieval protocol because
procedural and non-procedural backends are invoked through the same
`retrieve(...)` contract.

## Backbone Contract

`Backbone` defines the generation boundary used by C3.

Known implementations include:

- `MockBackbone`
- `OllamaBackbone`
- `TransformersBackbone`

The pipeline therefore depends on generation capability rather than a
specific model-serving implementation.

## Lifecycle Contract

`LifecycleStore` defines the persistence operations required by the explicit
memory lifecycle controller.

Current implementations include:

- `InMemoryLifecycleStore`
- `Neo4jLifecycleStore`

Lifecycle state-transition semantics remain separate from the retrieval
backend abstraction.

## Pipeline Dependency Injection

`C3Pipeline` now declares its principal external dependencies through typed
contracts:

- `MemoryStore`
- `Backbone`
- `TraceSink`

The procedural backend also satisfies `MemoryStore`.

This reduces concrete backend coupling while preserving the existing
orchestration sequence:

1. query analysis;
2. route planning;
3. retrieval;
4. ranking;
5. candidate budgeting;
6. conflict handling;
7. evidence selection;
8. confidence control;
9. prompt construction;
10. generation or abstention;
11. `C3Result`.

## Canonical Result Contract

`C3Result` remains the canonical result returned by the orchestration
pipeline.

It contains algorithm outputs and research diagnostics including:

- answer decision;
- route and memory-type selection;
- raw retrieval identifiers;
- ranked candidate identifiers;
- selected evidence identifiers;
- conflict groups;
- coverage, adequacy, and agreement;
- latency and token usage;
- selected evidence;
- backward-compatible debug traces.

The observability layer does not replace or redefine `C3Result`.

## TraceSink Contract

`TraceSink` provides an optional observability side channel.

`NullTraceSink` is the default implementation and preserves historical
pipeline behaviour when no observability adapter is supplied.

For each successful pipeline execution:

1. one `C3Result` is constructed;
2. that result is emitted to the configured `TraceSink`;
3. the same result object is returned to the caller.

Pipeline cleanup also closes the configured trace sink.

## Safe Telemetry Projection

`TelemetryRecord` provides a restricted projection of a completed
`C3Result`.

The default telemetry representation records:

- schema version;
- event identifier;
- UTC timestamp;
- decision;
- query mode;
- selected memory types;
- retrieval and selection counts;
- selected memory identifiers;
- coverage;
- adequacy;
- agreement;
- latency;
- input token count;
- output token count.

The default telemetry representation intentionally excludes:

- raw query text;
- user identifier;
- generated answer;
- final LLM prompt;
- memory content;
- selected evidence content;
- debug payloads;
- full candidate text.

This is a privacy-conscious default, not a complete privacy or compliance
system.

## JSONL Trace Adapter

`JsonlTraceSink` persists the restricted telemetry representation as
UTF-8 JSON Lines.

It is a lightweight local observability adapter.

It does not establish distributed logging, concurrent-writer safety,
encrypted audit storage, or production observability certification.

## Compatibility Guarantees

The contract hardening preserves:

- historical C3 import paths;
- existing `C3Pipeline(...)` construction without a trace sink;
- existing `C3Result` semantics;
- previously frozen orchestration behaviour;
- research experiment provenance.

Enterprise-facing interfaces are additive.

## Current Validation

The contract layer is protected by:

- protocol facade compatibility tests;
- TraceSink facade tests;
- Pipeline TraceSink integration tests;
- safe telemetry projection tests;
- JSONL persistence tests;
- the existing algorithm regression suite;
- critical Ruff checks;
- MyPy contract checks;
- Python compile checks.

At the current contract freeze point, the repository passes 175 tests.

The contract hardening should therefore be interpreted as a software
architecture and reproducibility improvement, not evidence of improved
answer quality.

## Known Limitations

The current contract layer does not establish:

- distributed transaction guarantees;
- concurrent JSONL writer safety;
- backend failover;
- retry policies;
- circuit breaking;
- OpenTelemetry integration;
- PII detection;
- encrypted trace storage;
- high-availability guarantees.

These remain separate production-hardening concerns.
