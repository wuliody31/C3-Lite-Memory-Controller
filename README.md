# C3-Lite v2.2 Final

**Context-adaptive, Conflict-aware, Confidence-controlled Multi-Memory Controller**

This is a dissertation-ready, training-free implementation. The LLM backbone remains frozen. C3-Lite performs query analysis, multi-label routing, three-memory retrieval, shared BM25/metadata ranking, explicit and implicit conflict handling, coverage-aware MMR selection, evidence adequacy estimation, and direct/caveat/abstain control.

## Put it here

```text
DISSERTATION/
└── c3_lite_system/
    └── c3_lite_v2_2_final/
```

Keep the old `c3_lite_algorithm_v2_1...` folders unchanged for historical comparison.

## Install on Windows

```powershell
cd C:\Users\Lenovo\Desktop\Dissertation\c3_lite_system\c3_lite_v2_2_final
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Smoke test without Neo4j/Ollama

```powershell
python run_query.py `
  --query "How did my project scope change over time?" `
  --user-id user01 `
  --memory-json examples/demo_memories.json `
  --procedure-json examples/demo_procedures.json `
  --backbone mock `
  --show-trace
```

The mock backbone only verifies the pipeline. Do not report it as an experimental model.

## Run with Ollama

```powershell
ollama pull llama3.2:3b
ollama serve
python run_query.py `
  --query "What is my current MSc project focus?" `
  --user-id user01 `
  --memory-json examples/demo_memories.json `
  --procedure-json examples/demo_procedures.json `
  --backbone ollama `
  --model llama3.2:3b `
  --show-trace
```

## Neo4j schema expected by the adapter

- `(:Episode)` with `episode_id`, `user_id`, `text`, `timestamp`, `confidence`, `importance`.
- `(:SemanticFact)` with `fact_id`, `user_id`, `subject`, `predicate`, `object`, `text`, `status`, `valid_from`, `valid_to`, `confidence`.
- Optional relations: `SUPERSEDES`, `CONTRADICTS`, `INVALIDATES`.

Recommended indexes:

```cypher
CREATE FULLTEXT INDEX episode_text IF NOT EXISTS FOR (e:Episode) ON EACH [e.text];
CREATE FULLTEXT INDEX semantic_fact_text IF NOT EXISTS FOR (f:SemanticFact) ON EACH [f.text, f.subject, f.predicate, f.object];
```

Set credentials:

```powershell
$env:NEO4J_URI="bolt://localhost:7687"
$env:NEO4J_USER="neo4j"
$env:NEO4J_PASSWORD="your-password"
```

Then add `--neo4j` instead of `--memory-json`.

## Dataset A experiment

```powershell
python run_experiment.py `
  --dataset C:\path\to\eval_questions.csv `
  --memory-json C:\path\to\all_memories.json `
  --procedure-json C:\path\to\procedural_memories.json `
  --methods no_memory episodic_only semantic_only procedural_only simple_retrieval all_memory c3 `
  --backbone ollama `
  --model llama3.2:3b `
  --output outputs\dataset_a_dev
```

The output contains predictions, per-question metrics, a global summary, a question-type summary and the frozen configuration.

## Experimental rules

- Tune only on Dataset A v0.1.
- Freeze the YAML before held-out and LoCoMo runs.
- Never pass `question_type`, `supporting_memory_ids`, `should_abstain`, `expected_outdated_memory_ids`, or `gold_answer` into C3.
- Use the same frozen backbone, prompts and decoding settings within each method comparison.
- Adapt only `src/retrievers/neo4j_store.py` if your existing labels/property names differ.
