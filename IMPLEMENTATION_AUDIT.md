# C3-Lite v2.2 Implementation Audit

| Module | Status | Code evidence | Frozen parameters |
|---|---|---|---|
| Multi-signal query analysis | Implemented | `src/query_analyzer.py` | aliases/rules |
| Multi-label routing | Implemented | `src/route_planner.py` | `routing.*` |
| Episodic retrieval | Implemented | `src/retrievers/*` | episodic top-k |
| Semantic retrieval | Implemented | `src/retrievers/*` | semantic top-k |
| Procedural retrieval | Implemented | `src/retrievers/procedural_json.py` | procedural top-k |
| Shared BM25 + metadata ranking | Implemented | `src/shared_ranker.py` | `ranking.*` |
| Explicit conflict detection | Implemented | `src/conflict_detector.py` | graph relations |
| Implicit version conflict | Implemented | `src/conflict_detector.py` | subject-predicate slot |
| Query-aware conflict resolution | Implemented | `src/conflict_resolver.py` | status/authority |
| Coverage-aware MMR | Implemented | `src/evidence_selector.py` | `selection.*` |
| Coverage estimation | Implemented | `src/coverage_estimator.py` | need coverage |
| Evidence adequacy | Implemented | `src/confidence_controller.py` | `confidence.*` |
| Direct/caveat/abstain | Implemented | `src/confidence_controller.py` | `decision.*` |
| Structured prompt | Implemented | `src/prompt_builder.py` | prompt template |
| Explanation trace | Implemented | `src/pipeline.py` | logging fields |
| Frozen backbone | Implemented | `src/backbones.py` | inference only |

## Freeze checklist

- [ ] Tune only on Dataset A v0.1
- [ ] Freeze YAML and Git commit before held-out
- [ ] Never pass gold labels into the controller
- [ ] Verify Neo4j schema mapping
- [ ] Run `pytest -q`
- [ ] Save complete predictions and configuration
- [ ] Use identical backbone and decoding settings within comparisons
