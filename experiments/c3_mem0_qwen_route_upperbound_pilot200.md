# Qwen3 Hybrid Route Upper-Bound — LoCoMo Pilot-200

## Protocol

- Backbone: Qwen/Qwen3-8B, 4-bit NF4, frozen.
- Temperature: 0.0.
- Candidate pool: frozen C3 raw candidates union Mem0 top-20.
- `base_hybrid`: current C3 route.
- `all_route_hybrid`: all memory types enabled.
- Ranker, candidate budget, conflict handling, evidence selector, confidence controller and generation settings otherwise unchanged.

## Main results

| Variant | Answer F1 | Answerable F1 | Evidence P | Evidence R | Evidence F1 | Hit | Mean selected |
|---|---:|---:|---:|---:|---:|---:|---:|
| base_hybrid | 0.0567 | 0.0709 | 0.1975 | 0.3098 | 0.2302 | 0.3800 | 2.02 |
| all_route_hybrid | 0.0642 | 0.0803 | 0.2475 | 0.3446 | 0.2741 | 0.4100 | 2.02 |

## Paired bootstrap: all-route minus base

| Metric | Delta | 95% CI | p-value |
|---|---:|---:|---:|
| Answer F1 | +0.0075 | [-0.0018, +0.0175] | 0.117640 |
| Evidence F1 | +0.0439 | [+0.0153, +0.0730] | 0.002840 |

## Interpretation

This is an upper-bound routing experiment. A positive all-route result supports implementing query-adaptive route expansion; it does not justify permanently retrieving all memory types in the final controller.
