# Phase II-B Lifecycle Kernel and Neo4j Atomic Store

## Scope

This phase introduces a backend-neutral lifecycle controller and a
Neo4j persistence adapter.

## Lifecycle behaviour

The controller supports:

- initial semantic-state creation;
- duplicate and replay no-op handling;
- rejection of older contradictory observations;
- supersession of a current semantic state;
- preservation of the old state;
- creation of an episodic transition record;
- current-state and historical-state access.

## Neo4j transaction boundary

A supersession commit atomically performs:

1. update the old `SemanticFact` to `superseded`;
2. create the new `current` `SemanticFact`;
3. create the transition `Episode`;
4. create `(new)-[:SUPERSEDES]->(old)`;
5. create `(event)-[:FROM_STATE]->(old)`;
6. create `(event)-[:TO_STATE]->(new)`.

All operations are executed inside one Neo4j write transaction.

## Isolation

Lifecycle-managed records use:

`lifecycle_managed=true`

This prevents Phase II-B writes from being confused with existing
Dataset A and LoCoMo graph records.

## Validation

- lifecycle-kernel tests: 10 passed;
- mocked Neo4j transaction tests: 9 passed;
- combined lifecycle tests: 19 passed;
- complete regression suite: 144 passed;
- failures: 0.

The mocked transaction layer is validated. A real Neo4j integration
smoke remains required.
