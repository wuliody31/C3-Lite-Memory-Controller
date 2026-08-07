# C3 Lifecycle Benchmark v0.1

## Scope

This run evaluates deterministic lifecycle-contract conformance. It does not compare C3 against Mem0 and does not measure LLM answer quality.

## Overall results

| Metric | Value |
| --- | ---: |
| Operation exact-match rate | 1.0000 |
| Operation step accuracy | 1.0000 |
| Current-state accuracy | 1.0000 |
| Single-current invariant rate | 1.0000 |
| Stale-memory exposure rate | 0.0000 |
| Stale-memory exposure case rate | 0.0000 |
| Previous-state recall | 1.0000 |
| History-retention recall | 1.0000 |
| Contradiction rate | 0.0000 |

## Memory growth

- Mean semantic records: 2.20
- Mean current semantic records: 1.00
- Mean superseded semantic records: 1.20
- Mean episodic transition records: 1.20
- Mean total records: 3.40

## Operation counts

- add_initial: 100
- noop_duplicate: 20
- reject_out_of_order: 20
- supersede_state: 120

## Interpretation boundary

A perfect score here would show that the implemented lifecycle kernel conforms to the controlled benchmark contract. It would not establish superiority over an external memory system.
