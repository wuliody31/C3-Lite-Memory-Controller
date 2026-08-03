# Route-Type Access and Route-Score Deconfounding — Pilot-200

## Why this experiment is necessary

The prior `all_route` ablation simultaneously exposed all memory types and replaced every route score with 1.0. This experiment separates type-access effects from route-score effects.

## Variants

- `base_hybrid`: unchanged C3 route.
- `adaptive_preserve_scores`: B9 one-type expansion, original scores.
- `adaptive_threshold_floor`: same expansion, added type floored at its route threshold.
- `all_types_preserve_scores`: all types exposed, original scores retained.
- `all_route_unit_scores`: all types exposed and all scores set to 1.0.

## Results

| Variant | Precision | Recall | F1 | Hit | Selected | Ranked | Raw |
|---|---:|---:|---:|---:|---:|---:|---:|
| base_hybrid | 0.1975 | 0.3098 | 0.2302 | 0.3800 | 2.02 | 13.16 | 39.76 |
| adaptive_preserve_scores | 0.2000 | 0.3098 | 0.2318 | 0.3800 | 2.02 | 13.82 | 40.50 |
| adaptive_threshold_floor | 0.2000 | 0.3098 | 0.2318 | 0.3800 | 2.02 | 13.82 | 40.50 |
| all_types_preserve_scores | 0.2450 | 0.3396 | 0.2707 | 0.4050 | 2.02 | 16.80 | 43.64 |
| all_route_unit_scores | 0.2475 | 0.3446 | 0.2741 | 0.4100 | 2.02 | 16.80 | 43.64 |

## Deconfounded contrasts

| Contrast | Evidence F1 delta | Hit delta |
|---|---:|---:|
| type_access_effect | +0.0406 | +0.0250 |
| score_inflation_effect_given_all_types | +0.0033 | +0.0050 |
| adaptive_floor_effect | +0.0000 | +0.0000 |
| adaptive_total_effect | +0.0017 | +0.0000 |

## Interpretation rule

- A positive `type_access_effect` means excluded memory types contain useful evidence even without score inflation.
- A positive `score_inflation_effect` means the previous all-route upper bound was partly driven by replacing route scores with 1.0.
- A positive `adaptive_floor_effect` supports assigning an admitted expansion type at least its route threshold.
