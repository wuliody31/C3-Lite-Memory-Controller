# Frozen Qwen3 Soft Probe-1 Validation — LoCoMo Held-out-from-Pilot

## Role of this experiment

This confirmatory experiment tests whether the controller-level evidence gains from the frozen Soft Probe-1 policy translate into end-to-end answer quality under the same frozen Qwen3-8B backbone.

## Protocol

- Evaluation split: 1,786 LoCoMo questions excluding Pilot-200.
- Dataset role: post-development held-out-from-pilot.
- Backbone: Qwen/Qwen3-8B, frozen 4-bit NF4.
- Temperature: 0.0; thinking disabled; no LoRA adapter.
- Candidate pool: frozen C3 raw candidates union Mem0 top-20.
- `hard_route`: unchanged C3 hard memory-type routing.
- `soft_probe_1`: frozen asymmetric companion probe with budget 1.
- Ranker, selector, confidence gate, prompt, generation settings and final evidence budget are otherwise unchanged.
- Runtime routing does not use gold evidence.

## Main results

| Variant | Answer F1 | Answerable F1 | Evidence P | Evidence R | Evidence F1 | Hit | Selected | Raw | Probe raw | Input tokens | Latency ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hard_route | 0.0766 | 0.0990 | 0.2210 | 0.4009 | 0.2774 | 0.4390 | 2.03 | 41.14 | 0.00 | 372.68 | 3400.11 |
| soft_probe_1 | 0.0771 | 0.0997 | 0.2466 | 0.4101 | 0.2973 | 0.4507 | 2.03 | 41.93 | 0.79 | 374.62 | 3469.06 |

## Paired bootstrap: Soft Probe-1 minus Hard Route

| Metric | Delta | 95% CI | p-value |
|---|---:|---:|---:|
| answer_f1 | +0.0005 | [-0.0016, +0.0026] | 0.651040 |
| evidence_f1 | +0.0199 | [+0.0133, +0.0267] | 0.000000 |

## Confirmatory interpretation

- Answer F1 significantly improved: `False`.
- Answer F1 significantly regressed: `False`.
- Evidence F1 significantly improved: `True`.
- Frozen controller evidence reproduced: `True`.

## Interpretation boundary

This split excludes Pilot-200 but is described as post-development held-out-from-pilot rather than a pristine benchmark test set.
