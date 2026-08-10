# Repository File Map

## Core Algorithm

| Path | Purpose |
|---|---|
| `src/query_analyzer.py` | Query-mode and information-need analysis |
| `src/route_planner.py` | Multi-memory routing |
| `src/shared_ranker.py` | Shared cross-memory candidate ranking |
| `src/conflict_detector.py` | Explicit and implicit conflict detection |
| `src/conflict_resolver.py` | Query-aware conflict resolution |
| `src/evidence_selector.py` | Coverage-aware bounded evidence selection |
| `src/confidence_controller.py` | Evidence adequacy and answer-state control |
| `src/prompt_builder.py` | Structured generation prompt |
| `src/backbones.py` | Frozen LLM backbone adapters |
| `src/pipeline.py` | End-to-end C3 execution |
| `src/lifecycle.py` | Explicit semantic lifecycle state machine |
| `src/lifecycle_neo4j.py` | Neo4j lifecycle persistence |

## Retrieval

| Path | Purpose |
|---|---|
| `src/retrievers/` | Episodic, semantic and procedural retrieval |
| `src/retrievers/neo4j_store.py` | Neo4j memory retrieval adapter |

## Configuration

| Path | Purpose |
|---|---|
| `configs/` | Runtime and frozen experiment configuration |
| `configs/c3_lite_v2_2_final.yaml` | Historical/frozen C3 configuration |

## Evaluation

| Path | Purpose |
|---|---|
| `evaluation/` | Metrics and evaluation utilities |
| `data/lifecycle_benchmark_v01/` | Controlled lifecycle benchmark |
| `experiments/` | Frozen and diagnostic experiment artifacts |

## Phase II Scripts

| Path | Purpose |
|---|---|
| `scripts/phase2b/` | C3 lifecycle benchmark and comparison scripts |
| `scripts/mem0/phase2b/` | Mem0 lifecycle evaluation |
| `scripts/mem0/slurm/` | Slurm experiment runners |

## Tests

| Path | Purpose |
|---|---|
| `tests/` | Unit, regression and lifecycle contract tests |

## Documentation

| Path | Purpose |
|---|---|
| `README.md` | Project overview and headline results |
| `docs/DEVELOPMENT_HISTORY.md` | Full research-development history |
| `docs/SYSTEM_ARCHITECTURE.md` | Architecture and component responsibilities |
| `docs/EXPERIMENTAL_PROTOCOL.md` | Formal evaluation protocol |
| `docs/EXPERIMENT_RESULTS.md` | Consolidated result analysis |
| `docs/REPRODUCIBILITY.md` | Reproduction guide |
| `docs/LIMITATIONS.md` | Explicit claim boundaries |
| `docs/TAG_AND_COMMIT_INDEX.md` | Frozen checkpoint index |
| `IMPLEMENTATION_AUDIT.md` | Implementation-status audit |

## Entry Points

| Path | Purpose |
|---|---|
| `run_query.py` | Local query execution |
| `run_query_gpu.py` | GPU query execution |
| `run_experiment.py` | Experiment execution |

## Repository Principle

```text
src/
= reusable algorithm implementation

scripts/
= experiment and operational tooling

evaluation/
= measurement logic

experiments/
= frozen research artifacts

docs/
= research and engineering documentation
```
