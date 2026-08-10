# Phase II-B Final Lifecycle Evaluation Summary

## Experimental scope

The primary controlled comparison contains 80 symmetric observations,
grouped into 20 canonical lifecycle scenarios with four surface variants
per scenario. Statistical uncertainty is estimated using 50,000 paired
cluster-bootstrap resamples at the canonical-scenario level.

## Main results

| Metric | C3 | Mem0 | Effect favouring C3 | 95% cluster-bootstrap CI |
|---|---:|---:|---:|---:|
| Current-only Top-1 | 1.000 | 0.550 | +0.450 | [+0.275, +0.625] |
| Stale-only exposure | 0.000 | 0.967 | +0.967 | [+0.900, +1.000] |
| Previous-state Top-1 | 0.000 | 0.450 | -0.450 | [-0.633, -0.267] |
| History value recall | 1.000 | 1.000 | +0.000 | [+0.000, +0.000] |

## Interpretation

C3 increased current-only Top-1 retrieval from 55% to 100%, an
absolute improvement of 45 percentage points. Stale-only exposure
during current-state retrieval decreased from 96.7% for Mem0 to 0%
for C3.

Both systems preserved historical information: previous-state
availability and history-value recall were 100% in both arms.

However, immediate-previous-state Top-1 accuracy was 0% for the
evaluated C3 historical retrieval policy versus 45% for Mem0. This
shows that lifecycle-role discrimination and historical-state ranking
are distinct retrieval problems.

## Supported conclusion

Explicit lifecycle management primarily improves which memory is
treated as current, rather than simply increasing memory retention.

## Scope limitation

This is a controlled observable lifecycle-retrieval comparison.
C3 and Mem0 do not receive fully identical upstream memory-formation
interfaces, and the out-of-order condition is treated separately as a
temporal capability stress test.
