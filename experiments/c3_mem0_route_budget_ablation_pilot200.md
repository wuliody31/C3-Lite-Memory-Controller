# Hybrid Route–Budget Ablation — LoCoMo Pilot-200

## Purpose

This controller-only ablation tests whether complementary Mem0 candidates are lost mainly because of memory-type routing or the fixed per-type candidate budget.

## Variants

- `base_hybrid`: current C3 route and current candidate budget.
- `all_route`: all memory types enabled; current budget retained.
- `expanded_budget`: current route retained; per-type top-k multiplied.
- `all_route_expanded_budget`: all memory types plus expanded budget.

The evidence selector, conflict handling, confidence controller, maximum selected evidence and token budget remain unchanged.

## Main results

| Variant | Precision | Recall | F1 | Hit | Mean selected | Mean ranked |
|---|---:|---:|---:|---:|---:|---:|
| base_hybrid | 0.1975 | 0.3098 | 0.2302 | 0.3800 | 2.02 | 13.16 |
| all_route | 0.2475 | 0.3446 | 0.2741 | 0.4100 | 2.02 | 16.80 |
| expanded_budget | 0.1967 | 0.3098 | 0.2297 | 0.3800 | 2.02 | 24.00 |
| all_route_expanded_budget | 0.2467 | 0.3446 | 0.2736 | 0.4100 | 2.02 | 27.52 |

## Paired outcomes versus base hybrid

| Variant | Base only | Variant only | Net rescues |
|---|---:|---:|---:|
| all_route | 6 | 12 | 6 |
| expanded_budget | 0 | 0 | 0 |
| all_route_expanded_budget | 6 | 12 | 6 |

## Recovery on the 25 backend-rescue questions

| Variant | Hit questions | Hit rate |
|---|---:|---:|
| base_hybrid | 1 | 0.0400 |
| all_route | 11 | 0.4400 |
| expanded_budget | 1 | 0.0400 |
| all_route_expanded_budget | 11 | 0.4400 |

## Decision rule

- If `all_route` produces most of the recovery, implement confidence-aware route expansion rather than permanently forcing all memory types.
- If `expanded_budget` produces most of the recovery, implement dynamic per-type budget allocation.
- If the combined variant is substantially better than either single change, implement both mechanisms jointly.
- This is a diagnostic upper-bound study. The final policy must be query-adaptive and must not use gold evidence.
