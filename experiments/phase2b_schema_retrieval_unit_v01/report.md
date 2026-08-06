# Phase II-B4 Schema and Retrieval Unit Validation

## Scope

This phase adds:

1. an idempotent Neo4j schema bootstrap;
2. semantic and episodic identity constraints;
3. lifecycle lookup and full-text indexes;
4. query-mode-aware semantic retrieval.

## Retrieval behaviour

Current-state queries include only semantic records whose status is
`current`, `active`, or `valid`.

Historical-state and timeline queries allow current and superseded
records to be retrieved together.

Explicit relation metadata such as `SUPERSEDES` is preserved in the
returned memory candidate.

## Validation

- focused Phase II-B4 tests: 8 passed;
- combined lifecycle tests: 27 passed;
- complete regression suite: 152 passed;
- failures: 0.

This validates the implementation at the unit and mocked-driver level.
Real Neo4j schema creation and retrieval remain to be tested.
