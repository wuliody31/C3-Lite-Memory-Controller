# Confidence-Aware Route Expansion — LoCoMo Pilot-200

## Protocol

- Candidate pool: frozen C3 raw candidates union Mem0 top-20.
- Adaptive policy uses only query features and route scores.
- At most one excluded memory type may be added.
- Gold evidence is used only for evaluation.
- Ranker, candidate budget, selector and 1200-token budget are unchanged.

## Results

| Variant | Precision | Recall | Evidence F1 | Hit | Mean selected | Mean ranked |
|---|---:|---:|---:|---:|---:|---:|
| base_hybrid | 0.1975 | 0.3098 | 0.2302 | 0.3800 | 2.02 | 13.16 |
| adaptive_route_hybrid | 0.2000 | 0.3098 | 0.2318 | 0.3800 | 2.02 | 13.82 |
| all_route_hybrid | 0.2475 | 0.3446 | 0.2741 | 0.4100 | 2.02 | 16.80 |

## Adaptive behaviour

- Expanded: 17/200 (0.0850).
- Added types: {'episodic': 11, 'semantic': 6}.
- Evidence-F1 upper-bound gain captured: 0.0380.
- Hit-rate upper-bound gain captured: 0.0000.
