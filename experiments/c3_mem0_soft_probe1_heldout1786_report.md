# Frozen Soft Probe-1 Controller Replay — LoCoMo Held-out-from-Pilot

- Split: 1,786 questions excluding Pilot-200.
- Candidate pool: frozen C3 raw union Mem0 top-20.
- Runtime policy: frozen Soft Probe-1; no gold used.
- Mock backbone: controller evidence only.

| Variant | Evidence P | Evidence R | Evidence F1 | Hit | Selected | Ranked | Raw | Probe raw |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| hard_route | 0.2210 | 0.4009 | 0.2774 | 0.4390 | 2.03 | 13.68 | 41.14 | 0.00 |
| soft_probe_1 | 0.2466 | 0.4101 | 0.2973 | 0.4507 | 2.03 | 14.48 | 41.93 | 0.79 |

| Metric | Delta | 95% CI | p-value |
|---|---:|---:|---:|
| evidence_precision | +0.0256 | [+0.0191, +0.0323] | 0.000000 |
| evidence_recall | +0.0092 | [+0.0005, +0.0180] | 0.038200 |
| evidence_f1 | +0.0199 | [+0.0132, +0.0267] | 0.000000 |
| evidence_hit | +0.0118 | [+0.0022, +0.0213] | 0.017360 |

Strict Qwen progression gate: `True`.

This split excludes Pilot-200 but is described as post-development held-out-from-pilot, not a pristine benchmark test set.
