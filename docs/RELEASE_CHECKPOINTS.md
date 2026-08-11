# Release Checkpoints

C3 preserves research and engineering milestones through immutable Git tags.

## Enterprise hardening checkpoints

### c3-enterprise-governance-v01

Commit: `f532aa5`

Repository governance and implementation-audit baseline.

### c3-enterprise-quality-v01

Commit: `1a356c5`

Reproducible quality gates, Python 3.11 development baseline, clean-room CI,
critical Ruff checks, MyPy checks, and branch-aware coverage baseline.

### c3-enterprise-contracts-v01

Commit: `83901e0`

Stable typed integration contracts for memory retrieval, generation,
lifecycle storage, and observability.

### c3-enterprise-reliability-v01

Commit: `4d5358f`

Reliability freeze covering observability failure isolation, structured
exceptions, Neo4j retrieval failure contracts, fail-fast configuration
validation, and coordinated idempotent resource cleanup.

## Final repository artifact

The final consolidated checkpoint is intended to be tagged:

`c3-enterprise-v01`

This checkpoint represents the final research-grade, production-oriented C3
repository artifact.

It does not constitute production deployment certification, distributed
fault-tolerance validation, high-availability validation, or service-level
guarantees.
