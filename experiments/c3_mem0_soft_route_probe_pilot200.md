# Soft Route Admission with Asymmetric Probe Budget — Pilot-200

## Protocol

- `hard_route`: unchanged C3 type routing.
- `soft_probe_1/3/5`: full original-route pool plus 1, 3, or 5 candidates from one complementary type.
- `soft_companion_full`: complete complementary-type pool.
- `all_types_full`: all memory types with original route scores.
- No gold evidence is used by the runtime policy.
- Ranker, candidate budget, selector and 1200-token evidence budget remain unchanged.

## Results

| Variant | Precision | Recall | F1 | Hit | Selected | Ranked | Raw | Probe raw |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| hard_route | 0.1975 | 0.3098 | 0.2302 | 0.3800 | 2.02 | 13.16 | 39.76 | 0.00 |
| soft_probe_1 | 0.2412 | 0.3406 | 0.2675 | 0.4100 | 2.02 | 14.02 | 40.59 | 0.83 |
| soft_probe_3 | 0.2462 | 0.3446 | 0.2727 | 0.4100 | 2.02 | 15.18 | 41.74 | 1.99 |
| soft_probe_5 | 0.2450 | 0.3396 | 0.2707 | 0.4050 | 2.02 | 15.85 | 42.45 | 2.69 |
| soft_companion_full | 0.2450 | 0.3396 | 0.2707 | 0.4050 | 2.02 | 16.80 | 43.64 | 3.88 |
| all_types_full | 0.2450 | 0.3396 | 0.2707 | 0.4050 | 2.02 | 16.80 | 43.64 | 0.00 |

## Gain and rescue capture

| Variant | F1 capture | Hit capture | Rescue capture | Net rescues |
|---|---:|---:|---:|---:|
| soft_probe_1 | 0.9202 | 1.2000 | 0.7000 | 6 |
| soft_probe_3 | 1.0493 | 1.2000 | 1.0000 | 6 |
| soft_probe_5 | 1.0000 | 1.0000 | 1.0000 | 5 |
| soft_companion_full | 1.0000 | 1.0000 | 1.0000 | 5 |

## Decision rule

Prefer the smallest probe budget that captures most of the all-types Evidence-F1 gain, yields positive net rescues, keeps final selected evidence stable, and reduces raw/ranked candidates relative to full soft routing.
