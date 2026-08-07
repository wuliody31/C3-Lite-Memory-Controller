# Phase II-B4 Real Neo4j Schema and Retrieval Validation

## Result

The real Neo4j schema and lifecycle-aware retrieval pipeline passed.

- Successful run: `20260806T212155Z`
- Database: `neo4j`
- Pipeline passed: `true`
- Cleanup completed: `true`
- Deleted test nodes: `3`

## Test suites

- Focused Phase II-B4 tests: 9 passed
- Combined lifecycle tests: 28 passed
- Complete regression suite: 153 passed
- Failures: 0

## Schema validation

The real Neo4j database contained two lifecycle identity constraints,
two lifecycle lookup indexes and two full-text indexes. All required
indexes were verified as ONLINE.

## Current-state retrieval

The current-state query returned only the current London semantic
state and excluded the superseded Nottingham state.

## Historical-state retrieval

The historical-state query returned both the current London state and
the superseded Nottingham state.

## Relation preservation

The London semantic state preserved an explicit SUPERSEDES relation
to the previous Nottingham semantic state.

## Real-driver issue and repair

The first real integration attempt exposed a Neo4j Python driver
argument collision:

`Session.run() got multiple values for argument 'query'`

The Cypher full-text parameter was renamed from `query` to
`search_query`. A dedicated regression test was added. The focused
tests, lifecycle tests, full regression suite and real Neo4j
integration smoke subsequently passed.

## Scope boundary

This experiment validates deterministic lifecycle writes and
query-mode-aware retrieval on a real Neo4j backend in a controlled
single-writer setting.

It does not yet validate concurrent lifecycle writers, unrestricted
natural-language state extraction, benchmark-scale lifecycle
performance, answer-generation improvement, or realised utility
feedback.
