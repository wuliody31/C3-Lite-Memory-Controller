# C3 + Mem0 Hybrid Controller Replay — Pilot-200

## Protocol

- Fixed question set: corrected LoCoMo Pilot-200.
- Original candidate pool: frozen C3 raw retrieval IDs.
- Added candidate pool: provenance-preserving Mem0 top-20 IDs.
- The C3 route planner, shared ranker, candidate budget, conflict handling, evidence selector, confidence controller and prompt builder were reused without modification.
- Mock generation was used because this stage evaluates evidence selection only.

## Replay validation

- Selected-ID set match against frozen C3: 1.0000
- Selected-ID order match against frozen C3: 1.0000
- Route-type match against frozen C3: 1.0000

## Evidence results

| System | Precision | Recall | F1 | Hit | Mean selected |
|---|---:|---:|---:|---:|---:|
| Frozen C3 selected | 0.1967 | 0.3073 | 0.2280 | 0.3800 | 2.02 |
| C3 controller replay | 0.1967 | 0.3073 | 0.2280 | 0.3800 | 2.02 |
| C3 + Mem0 hybrid | 0.1975 | 0.3098 | 0.2302 | 0.3800 | 2.02 |

## Paired hit outcomes: hybrid versus C3 replay

- Both hit: 74
- C3 replay only: 2
- Hybrid only (rescued): 2
- Neither hit: 122
- Net rescued questions: 0

## Hybrid deltas over C3 replay

- Precision: +0.0008
- Recall: +0.0025
- F1: +0.0022
- Hit rate: +0.0000

## Selection provenance

- Selected from C3 only: 180
- Selected from Mem0 only: 14
- Selected by both retrievers: 209

## Decision gate

Proceed to Qwen3 generation only when replay validation is high, hybrid-only rescues exceed C3-only losses, and evidence F1 or recall improves without an unacceptable precision collapse.
