# Reliability and Failure Contracts

## Purpose

This document describes the reliability boundaries introduced during the
enterprise-hardening phase of C3.

The work is compatibility-focused. It strengthens failure handling,
configuration validation, backend isolation, and resource cleanup without
changing the frozen research algorithms for routing, ranking, conflict
resolution, evidence selection, confidence control, or generation.

## Reliability Scope

The current reliability layer covers four main areas:

1. observability failure isolation;
2. structured exception contracts;
3. Neo4j retrieval failure boundaries;
4. configuration and resource lifecycle contracts.

These mechanisms improve software robustness and diagnosability.

They do not establish distributed fault tolerance, high availability,
automatic failover, or production service certification.

## Observability Failure Isolation

Pipeline observability is optional.

User-provided trace sinks are wrapped with `BestEffortTraceSink`.

A telemetry failure therefore does not invalidate an otherwise successful
C3 answer.

The wrapper records:

- emit failure count;
- close failure count;
- the most recent trace error description.

This policy is intentionally limited to observability.

Failures from primary memory resources are not silently discarded.

## Structured Exception Hierarchy

C3 exposes stable exception categories through `src.contracts`.

The hierarchy includes:

- `C3Error`;
- `ConfigurationError`;
- `RetrievalError`;
- `GenerationError`;
- `ObservabilityError`;
- `ResourceCleanupError`;
- the existing lifecycle exception hierarchy.

Backward compatibility is preserved.

Configuration errors remain compatible with `ValueError`.

Retrieval, generation, observability, lifecycle, and resource-cleanup
failures remain compatible with `RuntimeError`.

This allows existing callers to retain broad exception handling while new
integrations can catch more precise C3-specific categories.

## Lifecycle Exceptions

The existing lifecycle exception hierarchy remains authoritative:

- `LifecycleError`;
- `LifecycleInvariantError`;
- `LifecycleReplayConflict`;
- `LifecycleStoreError`.

`LifecycleError` also participates in the common `C3Error` hierarchy.

The lifecycle implementation was not replaced or duplicated during
enterprise hardening.

## Neo4j Retrieval Failure Contract

`Neo4jMemoryStore` distinguishes valid empty retrieval from backend failure.

A valid query with no matching memories returns an empty list.

The existing full-text fallback behaviour is preserved:

1. attempt the configured Neo4j full-text query;
2. allow Neo4j `ClientError` to reach the existing fallback logic;
3. execute the fallback MATCH query.

Operational failures at the session or query-execution boundary are wrapped
as `RetrievalError`.

If the fallback query itself fails with `ClientError`, the public retrieval
boundary exposes that failure as `RetrievalError`.

This preserves fallback semantics while providing a stable backend failure
category to callers.

## Optional Neo4j Dependency

Neo4j remains an optional adapter dependency.

Importing the wider C3 package does not require a usable Neo4j connection.

When Neo4j is unavailable or the adapter is constructed incorrectly, the
failure remains explicit.

The optional import boundary is also compatible with static type checking.

## Configuration Validation

Configuration loading performs fail-fast validation before normal pipeline
execution.

The current validation checks:

- required top-level configuration sections;
- mapping type for required sections;
- required ranking weight keys;
- required confidence weight keys;
- required selection weight keys;
- numeric convertibility of configured weights;
- ranking weight normalisation;
- confidence weight normalisation;
- selection weight normalisation.

Invalid configuration is surfaced through `ConfigError`, which is a
specialisation of `ConfigurationError`.

The configuration layer does not currently provide a complete schema
language or automatic migration between historical configuration versions.

## Resource Cleanup Contract

`C3Pipeline.close()` performs a single coordinated teardown attempt.

Managed resources currently include:

- the primary memory store;
- the optional procedural store;
- the configured trace sink.

Cleanup attempts continue even when an earlier primary resource fails.

Failures are collected and exposed after all cleanup attempts through
`ResourceCleanupError`.

This avoids the previous behaviour in which one early close failure could
prevent later resources from being released.

## Idempotent Pipeline Cleanup

Pipeline cleanup is idempotent at the pipeline boundary.

After the first teardown attempt, repeated calls to `close()` do not repeat
resource close side effects.

This behaviour applies after both successful cleanup and a failed cleanup
attempt.

The contract describes one teardown attempt per pipeline instance; it is not
an automatic retry mechanism.

## Resource Ownership

Backend ownership semantics remain implementation-specific and preserve the
existing research behaviour.

For the Neo4j retrieval and lifecycle adapters, a driver created internally
by the adapter is owned and closed by that adapter.

A driver injected externally is not automatically closed by the adapter.

The pipeline continues to close the memory-store and procedural-store
objects supplied to it, preserving the historical pipeline contract.

## Validation Status

At the Reliability v1 freeze point:

- the complete regression suite passes 191 tests;
- targeted reliability contract tests pass;
- critical Ruff checks pass;
- MyPy checks for the hardened service boundaries pass;
- Python compilation checks pass;
- repository diff validation passes after whitespace normalisation.

The additional tests cover configuration validation, structured exception
compatibility, trace failure isolation, resource-cleanup behaviour, valid
empty Neo4j retrieval, and Neo4j backend failure handling.

## Claim Boundary

Reliability v1 demonstrates tested in-process failure semantics for the C3
research artifact.

It should not be described as proving:

- distributed fault tolerance;
- automatic retry orchestration;
- circuit breaking;
- multi-process JSONL safety;
- concurrent Neo4j transaction guarantees;
- service-level availability;
- encrypted observability storage;
- automatic PII detection;
- production deployment certification.

Those capabilities remain separate engineering work.
