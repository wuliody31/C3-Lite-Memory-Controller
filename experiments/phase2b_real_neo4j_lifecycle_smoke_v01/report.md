# Phase II-B3 Real Neo4j Lifecycle Smoke

## Result

The real Neo4j lifecycle integration smoke passed.

- Run ID: `20260806T210145Z`
- Database: `neo4j`
- Pipeline passed: `True`
- Cleanup completed: `True`
- Deleted test nodes: `3`

## Observed lifecycle operations

- Initial ingestion: `add_initial`
- Duplicate ingestion: `noop_duplicate`
- State transition: `supersede_state`

## Final state

- Current value: `London`
- Semantic states retained: `2`
- Current semantic states: `1`
- Historical records: `3`

## Graph relations

- SUPERSEDES: `1`
- FROM_STATE: `1`
- TO_STATE: `1`

## Interpretation

This experiment demonstrates that the C3 lifecycle controller can
execute a semantic state transition against a real Neo4j backend.

The old Nottingham state was preserved and marked as superseded. A new
London state became current. An episodic move event was created and
connected to both states through explicit graph relations.

The complete mutation was executed through one Neo4j write transaction.

## Warning interpretation

Neo4j emitted schema notifications before the first Episode node and
its properties existed. These were informational warnings rather than
transaction failures. All behavioural checks passed after the records
were created.

## Scope boundary

This smoke validates a single-writer lifecycle transition. It does not
yet establish correctness under concurrent ingestion, automatic
state extraction from unrestricted dialogue, or benchmark-scale
performance.
