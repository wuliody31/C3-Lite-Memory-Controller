# Qwen3 Soft-Route Probe Validation — LoCoMo Pilot-200

## Role of this experiment

This experiment tests whether the evidence gains from asymmetric soft memory admission translate into end-to-end answer quality under a frozen Qwen3-8B backbone.

## Protocol

- Backbone: Qwen/Qwen3-8B, frozen 4-bit NF4.
- Temperature: 0.0; thinking disabled; no LoRA adapter.
- Candidate pool: frozen C3 raw candidates union Mem0 top-20.
- `hard_route`: unchanged C3 hard memory-type routing.
- `soft_probe_1`: one candidate from a complementary memory type.
- `soft_probe_3`: three candidates from a complementary memory type.
- Ranker, candidate budget, selector, confidence gate, prompt and generation settings are otherwise unchanged.
- Pilot-200 is treated as a development/model-selection set.

## Main results

| Variant | Answer F1 | Answerable F1 | Evidence P | Evidence R | Evidence F1 | Hit | Selected | Raw | Probe raw | Input tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hard_route | 0.0567 | 0.0709 | 0.1975 | 0.3098 | 0.2302 | 0.3800 | 2.02 | 39.76 | 0.00 | 370.17 |
| soft_probe_1 | 0.0622 | 0.0777 | 0.2412 | 0.3406 | 0.2675 | 0.4100 | 2.02 | 40.59 | 0.83 | 372.01 |
| soft_probe_3 | 0.0607 | 0.0759 | 0.2462 | 0.3446 | 0.2727 | 0.4100 | 2.02 | 41.74 | 1.99 | 372.79 |

## Paired bootstrap

| Comparison | Metric | Delta | 95% CI | p-value |
|---|---|---:|---:|---:|
| soft_probe_1_minus_hard | answer_f1 | +0.0055 | [-0.0020, +0.0137] | 0.154880 |
| soft_probe_1_minus_hard | evidence_f1 | +0.0373 | [+0.0163, +0.0597] | 0.000520 |
| soft_probe_3_minus_hard | answer_f1 | +0.0040 | [-0.0042, +0.0128] | 0.347680 |
| soft_probe_3_minus_hard | evidence_f1 | +0.0426 | [+0.0162, +0.0701] | 0.001400 |
| soft_probe_3_minus_soft_probe_1 | answer_f1 | -0.0015 | [-0.0045, +0.0008] | 0.256480 |
| soft_probe_3_minus_soft_probe_1 | evidence_f1 | +0.0052 | [-0.0100, +0.0217] | 0.481680 |

## Development-set selection

- Chosen variant: `soft_probe_1`.
- Reason: Probe-1 is not significantly worse than hard route, and Probe-3 is not significantly better than Probe-1; the efficiency-first rule selects Probe-1.

## Interpretation boundary

The selected probe budget must be frozen after this development-set comparison. Generalisation claims require an untouched held-out LoCoMo evaluation.
