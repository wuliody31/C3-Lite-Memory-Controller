# c3_lite_system

A complete runnable MVP system for the C3-Lite memory controller project.

## What it does

```text
Dataset A → Neo4j episodic/semantic memory → JSON procedural rules
→ C3-Lite controller → baselines → evaluation outputs
```

## Expected folder layout

```text
Dissertation/
  Dataset_A_v0_1/
  Dataset_A_v0_2/
  c3_lite_system/
```

The system can also detect:

```text
Dataset_A_v0_1_Neo4j_Episodic_Engineer
Dataset_A_v0_2_Neo4j_Expanded
```

## Setup in VSCode

```powershell
cd c3_lite_system
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Edit `.env`:

```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_neo4j_password
DATASET_PATH=../Dataset_A_v0_1
ANSWER_MODE=mock
```

## Important: import Dataset A into Neo4j first

Use the dataset package's own `loader.py` or `neo4j_full_import.cypher` first.

Then inspect:

```powershell
python -m scripts.inspect_neo4j
```

## Run pilot

```powershell
python -m src.run_pilot --limit 10
```

Balanced pilot:

```powershell
python -m src.run_pilot --limit 12 --balanced
```

Specific dataset:

```powershell
python -m src.run_pilot --dataset-path ../Dataset_A_v0_1 --limit 10
```

Outputs:

```text
outputs/pilot/predictions.jsonl
outputs/pilot/scores.jsonl
outputs/pilot/scores.csv
outputs/pilot/summary.csv
outputs/pilot/run_info.json
```

## Run full experiment

```powershell
python -m src.run_full --dataset-path ../Dataset_A_v0_2_Neo4j_Expanded
```

## Methods compared

1. `no_memory`
2. `simple_retrieval`
3. `all_memory`
4. `c3_lite_controller`

## Optional manual scoring sheet

```powershell
python -m scripts.make_manual_scoring_sheet --predictions outputs/pilot/predictions.jsonl
```

## Optional OpenAI generation

Default is `ANSWER_MODE=mock`. To use OpenAI:

```env
ANSWER_MODE=openai
OPENAI_API_KEY=your_api_key
OPENAI_MODEL=gpt-4o-mini
```

## Research framing

Neo4j is the graph backend. The contribution is controlled memory selection, conflict-aware memory use, confidence-calibrated answering, and explainable evidence selection.
