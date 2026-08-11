# Interfaces and Contracts

## Purpose

C3 uses explicit Python protocols to separate orchestration logic from
concrete storage, retrieval, lifecycle, backbone, and observability
implementations.

The enterprise-hardening work follows a compatibility-first principle:

> Stable contracts are introduced without rewriting the frozen research
> implementation or changing previously evaluated algorithm behaviour.

## Public Contract Facade

The supported public contract entry point is:

```python
from src.contracts import (
    Backbone,
    GenerationResult,
    LifecycleStore,
    MemoryRetriever,
    MemoryStore,
    NullTraceSink,
    TraceSink,
)
Historical import paths remain supported for research reproducibility.

The facade therefore provides a stable external API without invalidating
existing experiments, tests, or frozen provenance.

Memory Retrieval Contract

MemoryStore defines the retrieval capability required by C3Pipeline.

Its implementations currently include:

InMemoryMemoryStore
Neo4jMemoryStore
ProceduralJsonStore

The historical name MemoryStore is preserved for backward compatibility.

MemoryRetriever is provided as a semantic alias because the current
protocol expresses retrieval and resource cleanup rather than full CRUD
storage semantics.

Procedural memory does not require a separate retrieval protocol because
the pipeline invokes procedural and non-procedural backends through the same
retrieve(...) contract.

Backbone Contract

Backbone defines the generation boundary used by C3.

Known implementations include:

MockBackbone
OllamaBackbone
TransformersBackbone

The pipeline therefore depends on generation capability rather than a
specific model-serving technology.

Lifecycle Contract

LifecycleStore defines the persistence operations required by the explicit
memory lifecycle controller.

Current implementations include:

InMemoryLifecycleStore
Neo4jLifecycleStore

Lifecycle state-transition semantics remain independent from the retrieval
backend abstraction.

Pipeline Dependency Injection

C3Pipeline now declares its principal external dependencies through typed
contracts:

C3Pipeline
    |
    +-- MemoryStore
    |
    +-- Backbone
    |
    +-- TraceSink

The procedural backend also satisfies MemoryStore.

This reduces concrete backend coupling while preserving the existing C3
orchestration sequence:

query analysis
    ->
route planning
    ->
retrieval
    ->
ranking
    ->
candidate budgeting
    ->
conflict handling
    ->
evidence selection
    ->
confidence control
    ->
prompt construction
    ->
generation / abstention
    ->
C3Result
Canonical Result Contract

C3Result remains the canonical result returned by the orchestration
pipeline.

It contains algorithm outputs and research diagnostics including:

answer decision;
route and memory-type selection;
raw retrieval identifiers;
ranked candidate identifiers;
selected evidence identifiers;
conflict groups;
coverage, adequacy, and agreement;
latency and token usage;
selected evidence objects;
backward-compatible debug traces.

The enterprise observability layer does not replace or redefine
C3Result.

TraceSink Contract

TraceSink is an optional observability side channel.

The default implementation is NullTraceSink, which preserves historical
pipeline behaviour when no observability adapter is supplied.

The pipeline:

constructs exactly one C3Result;
emits that result to the configured TraceSink;
returns the same result object to the caller.

Pipeline resource cleanup also closes the configured trace sink.

Safe Telemetry Projection

TelemetryRecord provides a restricted projection of a completed
C3Result.

The default telemetry record includes operational and algorithm-decision
metadata such as:

schema version;
event identifier;
UTC timestamp;
decision;
query mode;
selected memory types;
retrieval and selection counts;
selected memory identifiers;
coverage;
adequacy;
agreement;
latency;
token usage.

The default telemetry projection intentionally excludes:

raw query text;
user identifier;
generated answer text;
final LLM prompt;
memory content;
selected evidence content;
debug payloads;
full candidate text.

This is a privacy-conscious default rather than a complete privacy,
compliance, or PII-detection system.

JSONL Trace Adapter

JsonlTraceSink persists the restricted telemetry representation as
UTF-8 JSON Lines.

It is intended as a lightweight local observability adapter and not as a
claim of distributed logging, concurrent-writer safety, encrypted audit
storage, or production observability certification.

Compatibility Guarantees

The current contract hardening is designed to preserve:

historical C3 import paths;
existing C3Pipeline(...) construction without a trace sink;
existing C3Result semantics;
previously frozen orchestration behaviour;
research experiment provenance.

New enterprise-facing imports are additive.

Current Validation

The contract layer is protected by:

Protocol facade compatibility tests
TraceSink facade tests
Pipeline TraceSink integration tests
Safe telemetry projection tests
JSONL persistence tests
Existing algorithm regression suite
Critical Ruff checks
MyPy contract checks
Python compile checks

The contract hardening should therefore be interpreted as a software
architecture and reproducibility improvement, not evidence of improved
answer quality.

Known Limitations

The current contract layer does not establish:

distributed transaction guarantees;
concurrent JSONL writer safety;
backend failover;
retry policies;
circuit breaking;
OpenTelemetry integration;
PII detection;
encrypted trace storage;
high-availability guarantees.

These concerns remain separate production-hardening work.
