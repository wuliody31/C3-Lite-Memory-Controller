# C3 + Mem0 Hybrid Candidate-Drop Diagnosis

## Scope

This diagnosis examines the Pilot-200 questions for which Mem0 top-20 retrieved at least one gold-equivalent candidate that was absent from the frozen C3 raw pool.

## Main counts

- Backend-rescue questions: 25
- Questions with a Mem0-only gold candidate selected: 1
- Questions with no Mem0-only gold candidate selected: 24

## Candidate-level loss stages

| Stage | Gold candidates |
|---|---:|
| route_filtered | 17 |
| type_top_k | 13 |
| evidence_selector | 3 |
| selected | 1 |

## Question-level furthest stage

| Furthest stage reached | Questions |
|---|---:|
| route_filtered | 10 |
| type_top_k | 11 |
| evidence_selector | 3 |
| selected | 1 |

## Interpretation gate

- A high `route_filtered` count indicates that Mem0 retrieves useful evidence from memory types excluded by the current route planner.
- A high `ranker_gate` or `type_top_k` count indicates that the shared ranker or fixed per-type budget cannot exploit complementary Mem0 candidates.
- A high `evidence_selector` count indicates that useful candidates survive ranking but are removed by final evidence selection.
- Only after identifying the dominant stage should the utility estimator or dynamic budget be modified.
