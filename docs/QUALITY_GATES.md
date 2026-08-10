# Quality Gates

## Purpose

C3 preserves a frozen research implementation while introducing
production-oriented software quality controls.

The enterprise-hardening process follows a compatibility-first principle:

> Quality tooling should detect regressions without silently changing
> frozen research behaviour.

## Frozen Regression Baseline

The enterprise-hardening branch was created from the following verified baseline:

```text
Python: 3.11.13
Test files: 25
Collected tests: 169
Passed tests: 169
Core import smoke: 10 / 10
Python compile check: passed
```

These results form the initial engineering regression contract.

## Continuous Integration

The initial CI pipeline performs:

```text
Checkout
   |
Python 3.11
   |
Install dependencies
   |
Compile check
   |
Critical Ruff checks
   |
169-test regression suite
   |
Branch coverage
```

## Ruff Policy

The initial blocking Ruff rules are:

```text
E9
F63
F7
F82
```

These rules focus on high-severity Python defects such as syntax errors,
undefined names and invalid control flow.

Historical research code is not mass-reformatted solely for cosmetic linting.

## Static Typing

MyPy is introduced initially as an advisory engineering tool.

Repository-wide strict typing is not yet a blocking CI requirement.

Typing will be strengthened incrementally at stable interfaces.

## Coverage

Branch coverage is measured through `pytest-cov`.

The first hardening stage establishes a measured coverage baseline rather
than imposing an arbitrary threshold.

## Regression Contract

Before accepting an engineering change, the repository should continue to pass:

```bash
python -m compileall -q src evaluation
python -m ruff check src evaluation tests --select E9,F63,F7,F82
python -m pytest -q
```

## Repository Separation

```text
src/
    reusable algorithm implementation

tests/
    behavioural and contract protection

evaluation/
    evaluation logic

experiments/
    frozen research artifacts

scripts/
    experiment and operational runners

docs/
    research and engineering documentation
```

Frozen experiment artifacts are not general-purpose lint targets.

## Future Hardening

Planned stages include stable interfaces, typed orchestration contracts,
structured exceptions, structured logging, request-level tracing,
lifecycle contract tests, integration-test separation, coverage thresholds,
security scanning, and service health checks.

## Claim Boundary

Passing these quality gates means that the tested implementation satisfies
the repository's current regression and software-quality controls.

It does not imply production certification, security certification,
high-availability guarantees or distributed consistency guarantees.

## Enterprise Quality Baseline v1

The first frozen enterprise-hardening baseline was validated in both the
research environment and a clean Python 3.11 environment.

```text
Python runtime:              3.11.13
Regression tests:            169 / 169 passed
Critical Ruff checks:        passed
Python compile check:        passed
Core import smoke:           10 / 10
Branch-aware source coverage: 81.84%
Clean-room dependency install: passed
Clean-room regression suite:  passed
