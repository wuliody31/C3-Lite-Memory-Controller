# Phase II-B5 C3 vs Mem0 Lifecycle Retrieval Comparison

Primary analysis: 80 symmetric observations grouped into 20 canonical lifecycle-scenario clusters.

| Metric | C3 | Mem0 | C3-Mem0 | Effect favouring C3 | Cluster-bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|
| current_only_top1_rate | 1.0000 | 0.5500 | +0.4500 | +0.4500 | [+0.2750, +0.6250] |
| stale_only_top1_rate | 0.0000 | 0.4500 | -0.4500 | +0.4500 | [+0.2750, +0.6250] |
| mixed_transition_top1_rate | 0.0000 | 0.0000 | +0.0000 | -0.0000 | [-0.0000, -0.0000] |
| stale_only_exposure_case_rate | 0.0000 | 0.9667 | -0.9667 | +0.9667 | [+0.9000, +1.0000] |
| mixed_transition_exposure_case_rate | 0.0000 | 0.1000 | -0.1000 | +0.1000 | [-0.0000, +0.2000] |
| historical_signal_exposure_case_rate | 0.0000 | 1.0000 | -1.0000 | +1.0000 | [+1.0000, +1.0000] |
| previous_value_availability | 1.0000 | 1.0000 | +0.0000 | +0.0000 | [+0.0000, +0.0000] |
| previous_top1_accuracy | 0.0000 | 0.4500 | -0.4500 | -0.4500 | [-0.6333, -0.2667] |
| history_value_recall | 1.0000 | 1.0000 | +0.0000 | +0.0000 | [+0.0000, +0.0000] |

Temporal out-of-order cases are excluded from the primary inferential comparison and retained as descriptive stress tests.
