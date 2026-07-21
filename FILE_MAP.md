# File map

1. `configs/c3_lite_v2_2_final.yaml`: all tunable/frozen parameters.
2. `src/query_analyzer.py`: query mode, task, entity and information-need extraction.
3. `src/route_planner.py`: weighted utility plus structural gates.
4. `src/retrievers/`: episodic, semantic and procedural candidate generation.
5. `src/shared_ranker.py`: shared BM25 and metadata scoring.
6. `src/conflict_detector.py`: explicit graph and implicit semantic conflicts.
7. `src/conflict_resolver.py`: current, historical and timeline policies.
8. `src/evidence_selector.py`: coverage-aware MMR and context budget.
9. `src/confidence_controller.py`: evidence adequacy and answer state.
10. `src/pipeline.py`: end-to-end C3 execution and trace.
11. `src/baselines.py`: fair comparison methods.
12. `evaluation/`: Dataset A metrics and experiment outputs.
